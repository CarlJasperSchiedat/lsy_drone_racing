"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

import json
import csv
import os
from lsy_drone_racing.control.own_mpc_functions import create_ocp_solver_for_mpc, create_ocp_solver_for_recompute


class MPController(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""



    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self.freq = config.env.freq
        self._tick = 0


        self.y=[]
        self.y_mpc=[]


        self.prev_obstacle = np.array([ [1, 0, 1.4], [0.5, -1, 1.4], [0, 1.5, 1.4], [-0.5, 0.5, 1.4], ])
        self.prev_gates_quat = np.array([ [0.0, 0.0, 0.92268986, 0.38554308], [0.0, 0.0, -0.38018841, 0.92490906], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], ])
        self.prev_gates= np.array([ [0.45, -0.5, 0.56], [1.1, -1.05, 1.11], [0.0, 1.0, 0.56], [-0.5, 0.0, 1.11], ])




        # Lade optimierte Waypoints
        file_name = "10_sec__18_06_25_1"
        with open(f"plot/traj_data/prepared_{file_name}.json", "r") as f:
            data = json.load(f)
            self.waypoints = np.array(data["waypoints"])
            self.ticks = data["ticks"]
            self.gate_map = {int(k): v for k, v in data["gate_idx_map"].items()}

        tick_times = np.array(self.ticks) / self.freq
        self.dt_segments = np.diff(self.ticks)
        # segment_weights = self.dt_segments / np.sum(self.dt_segments)
        ts = []
        for i in range(len(tick_times) - 1):
            t_start = tick_times[i]
            t_end = tick_times[i + 1]
            n_points = self.dt_segments[i]

            # Gleichmäßige Zwischenpunkte für diesen Abschnitt
            ts_seg = np.linspace(t_start, t_end, n_points, endpoint=False)
            ts.extend(ts_seg)
        # Letzten Punkt anhängen
        ts.append(tick_times[-1])
        ts = np.array(ts)



        cs_x = CubicSpline(tick_times, self.waypoints[:, 0])
        cs_y = CubicSpline(tick_times, self.waypoints[:, 1])
        cs_z = CubicSpline(tick_times, self.waypoints[:, 2])

        self.x_des = cs_x(ts)
        self.y_des = cs_y(ts)
        self.z_des = cs_z(ts)



        # # For Visualising
        tvisual = np.linspace(0, tick_times[-1], 50)
        x = cs_x(tvisual)
        y = cs_y(tvisual)
        z = cs_z(tvisual)
        self.traj_vis=np.array([x,y,z])
        self.update_traj_vis=np.array([x,y,z])



        # acados solver for MPC
        self.N_mpc = 20
        self.T_HORIZON_mpc = 1
        self.dt_mpc = self.T_HORIZON_mpc / self.N_mpc
        self.acados_ocp_solver, self.ocp = create_ocp_solver_for_mpc(self.T_HORIZON_mpc, self.N_mpc, name="example_mpc")

        # acados solver for the Recompute
        self.N_recompute = 60
        self.acados_recompute_solver, self.ocp_recompute = create_ocp_solver_for_recompute(
            self.N_recompute / self.freq, self.N_recompute, name="recompute_ocp",
            verbose = False)

        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.config = config
        self.finished = False



        # Estimation-variable of the OCP variable corresponding to Mass
        self.params_acc_0_hat = 20.907574256269616 # params_acc[0] ≈ k_thrust / m_nominal ; nominal value given for nominal_mass = 0.027
        self.vz_prev = 0.0 # estimated velocity at start = 0







    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        
        # Update Mass-Estimation of Drone , bzw. update self.params_acc_0_hat
        self.mass_estimator(obs)




        # Update Gates ?
        updated_gate = self.check_for_update(obs)
        if updated_gate:
            self.update_traj(obs, updated_gate)
                
        


        i = min(self._tick, len(self.x_des) - 1)
        if self._tick > i:
            self.finished = True

  


        q = obs["quat"]
        r = R.from_quat(q)
        # Convert to Euler angles in XYZ order
        rpy = r.as_euler("xyz", degrees=False)  # Set degrees=False for radians

        xcurrent = np.concatenate(
            (
                obs["pos"],
                obs["vel"],
                rpy,
                np.array([self.last_f_collective, self.last_f_cmd]),
                self.last_rpy_cmd,
            )
        )
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)

        self.y=[]
        for j in range(self.N_mpc):


            yref = np.hstack([ # params = vertcat(p_obs1, p_obs2, p_obs3, p_obs4, p_ref, m)
                self.prev_obstacle[:, :2].flatten(),
                self.x_des[i + j],
                self.y_des[i + j],
                self.z_des[i + j],
                self.params_acc_0_hat,
                ])

            self.acados_ocp_solver.set(j, "p", yref)
            self.y.append(yref)

        yref_N = np.hstack([
            self.prev_obstacle[:, :2].flatten(),
            self.x_des[i + self.N_mpc],
            self.y_des[i + self.N_mpc],
            self.z_des[i + self.N_mpc],
            self.params_acc_0_hat,
        ])
        self.acados_ocp_solver.set(self.N_mpc, "p", yref_N)
        
        self.acados_ocp_solver.solve()
        self.y_mpc = []
        for j in range(self.N_mpc + 1):  # Include terminal state
            x_pred = self.acados_ocp_solver.get(j, "x")
            # Extract relevant states to match your y_ref format if needed
            # This depends on how you want to compare them
            y_mpc = x_pred[:len(yref_N)]  # Adjust this based on your state vector
            self.y_mpc.append(y_mpc)

        x1 = self.acados_ocp_solver.get(1, "x")
        w = 1 / self.freq / self.dt_mpc
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]

        cmd = x1[10:14]


        return cmd




    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        self._tick += 1

        return self.finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0




    def check_for_update(self, obs):
        """Check if any gate's position has changed significantly.
        Returns:
            - `None` if no gate moved beyond threshold
            - The **index (int)** of the first changed gate (row-wise comparison)
        """
        threshold = 0.1


        gate_index_return = None


        current_gates = np.asarray(obs["gates_pos"])
        
        for gate_idx in range(len(self.prev_gates)):
            prev_gate = np.asarray(self.prev_gates[gate_idx])
            current_gate = np.asarray(current_gates[gate_idx])
            
            if np.linalg.norm(prev_gate - current_gate) > threshold:
                print(f"Gate {gate_idx} moved significantly.")
                gate_index_return = gate_idx +1
                
        # update changed variables either way - even if no update is nessesary -> no secound check
        self.prev_gates = current_gates.copy()
        self.prev_gates_quat = np.asarray(obs["gates_quat"]).copy()

        for i, idx in self.gate_map.items(): # update the waypoints that correspond to a specific gate
            self.waypoints[idx] = self.prev_gates[i]

        return gate_index_return

    


    def update_traj(self, obs, updated_gate):
        """
        optimize a section around the gate
        """

        if self._tick == 0:
            print("Kein Update, Tick == 0")
            return




        gate_idx = int(updated_gate-1)
        gate_waypoint_idx = self.gate_map[int(gate_idx)]
        gate_tick = self.ticks[gate_waypoint_idx]

        if gate_idx == obs["target_gate"]:
            start_tick = self._tick
            target_gate_flag = 1
        else:
            start_tick = gate_tick - 20
            target_gate_flag = 0

        rel_gate_tick = gate_tick - start_tick
        gate_pos  = self.prev_gates[gate_idx]

        print(
            f"\n\n\n"
            f"[DBG] gate_idx={gate_idx:2d} \n waypoint_idx={gate_waypoint_idx:3d} \n "
            f"start_tick={start_tick:4d} \n gate_tick={gate_tick:4d} \n "
            f"rel_gate_tick={rel_gate_tick:3d} \n target_gate_flag={target_gate_flag}"
            f"\n\n\n" )




        #  -----  acados solver for Recompute  -----
        nominal_constrain_lbx = self.ocp_recompute.constraints.lbx.copy()
        nominal_constrain_ubx = self.ocp_recompute.constraints.ubx.copy()
        pos_idx = np.array([0,1,2], dtype=int)

        
        # Set the First State
        px0 = self.x_des[start_tick]
        py0 = self.y_des[start_tick]
        pz0 = self.z_des[start_tick]
        state_first_step = np.array([px0, py0, pz0])

        lbx0 = nominal_constrain_lbx.copy()
        ubx0 = nominal_constrain_ubx.copy()

        lbx0[pos_idx] = state_first_step
        ubx0[pos_idx] = state_first_step

        self.acados_recompute_solver.constraints_set(0, "lbx", lbx0)
        self.acados_recompute_solver.constraints_set(0, "ubx", ubx0)




        # Fix Gate position that is to be updated
        lbx_g = nominal_constrain_lbx.copy()
        ubx_g = nominal_constrain_ubx.copy()

        lbx_g[pos_idx] = np.array(gate_pos)
        ubx_g[pos_idx] = np.array(gate_pos)
        
        self.acados_recompute_solver.constraints_set(rel_gate_tick, "lbx", lbx_g)
        self.acados_recompute_solver.constraints_set(rel_gate_tick, "ubx", ubx_g)

    


        # Fix position of last step -> updated trajectory slice will go back on old trajectory
        px0 = self.x_des[start_tick + self.N_recompute]
        py0 = self.y_des[start_tick + self.N_recompute]
        pz0 = self.z_des[start_tick + self.N_recompute]
        state_last_step = np.array([px0, py0, pz0])

        lbx_e = nominal_constrain_lbx.copy()
        ubx_e = nominal_constrain_ubx.copy()

        lbx_e[pos_idx] = state_last_step
        ubx_e[pos_idx] = state_last_step

        self.acados_recompute_solver.constraints_set(self.N_recompute, "lbx", lbx_e)
        self.acados_recompute_solver.constraints_set(self.N_recompute, "ubx", ubx_e)


        '''
        # Parameters for costs on Gate-Distance
        param_ref = np.hstack([         # para_recompute = vertcat(p_gate, R_gate)
            up_gate_pos,
            up_gate_quat,
            ])
        
        for j in range(self.N_recompute):
            self.acados_recompute_solver.set(j, "p", param_ref)
        
        self.acados_recompute_solver.set(self.N_recompute, "p", param_ref)
        '''

        # Warm Start for better convergence
        pos = np.array(gate_pos)
        vel = (np.array(gate_pos) - state_first_step) / (rel_gate_tick / self.freq)
        rpy = [0.0, 0.0, 0.0]
        force_cur_cmd = [0.4, 0.4]
        rpy_cmd = [0.0, 0.0, 0.0]
        warm_start_states = np.concatenate([pos, vel, rpy, force_cur_cmd, rpy_cmd])
        self._warm_start_from_global(warm_start_states)


        # Solve the OCP
        solver_status = self.acados_recompute_solver.solve()


        # If Solver didnt find anything -> keep old traj
        t_us = self.acados_recompute_solver.get_stats("time_tot")
        if solver_status != 0:
            print(f"[WARN] Recompute failed (status {solver_status}) - keep old traj.")
            # return


        # Sammle die vorhergesagte Position (px, py, pz) für alle Knoten 0 … N
        pos_pred = []                          # → Liste mit N+1 Einträgen
        for k in range(self.N_recompute + 1):
            x_k = self.acados_recompute_solver.get(k, "x")
            pos_pred.append(x_k[:3])           # nur [px, py, pz]
        pos_pred = np.vstack(pos_pred)         # Shape: (N+1, 3)
        x_new = pos_pred[:, 0]                 # px
        y_new = pos_pred[:, 1]                 # py
        z_new = pos_pred[:, 2]                 # pz
        end_tick = start_tick + len(x_new)




        self.x_des[start_tick : end_tick]  = x_new
        self.y_des[start_tick : end_tick]  = y_new
        self.z_des[start_tick : end_tick]  = z_new




        # For Visulization of the updated Trajectory
        self.update_traj_vis=np.array([x_new,y_new,z_new])
        # Plot of points
        print("📐  Neue Spline-Punkte:")
        for k, (xi, yi, zi) in enumerate(zip(x_new, y_new, z_new)):
            print(f"   P{k:02d}: ({xi:+.3f}, {yi:+.3f}, {zi:+.3f})")



        print(f"✅ Neue Teiltrajektorie um Gate {gate_idx} aktualisiert.")
        print(f"   Solver: {t_us * 1e-3:6.2f} ms  |  Budget: {1/self.freq*1e3:5.2f} ms")




    def _warm_start_from_global(self, warm_start):
        """
        Copy the MPC-Cycle in the Recompute Solver
        staring from start_tick
        """
        N_r  = self.N_recompute
        solR = self.acados_recompute_solver
        
        u_hover = np.array([0.4, 0, 0, 0])


        for k in range(N_r):                 # 0 … N_r (inkl. Terminal)
            solR.set(k, "x", warm_start)
            solR.set(k, "u", u_hover)

        solR.set(self.N_recompute, "x", warm_start)








    def mass_estimator(self, obs):

        max_angle = max_angle=np.deg2rad(20)


        params_acc = [20.907574256269616, 3.653687545690674] # params_acc[0] ≈ k_thrust / m_nominal
        nominal_m = 0.027
        GRAVITY = 9.806


        # Messgrößen
        vz_dot   = (obs["vel"][2] - self.vz_prev) * self.freq
        self.vz_prev = obs["vel"][2] # update für nächsten Durchlauf

        roll, pitch, _ = R.from_quat(obs["quat"]).as_euler("xyz", degrees=False)
        cos_roll_pitch   = np.cos(roll) * np.cos(pitch)
        
        # Only update, whne Drone is upright
        if abs(roll) > max_angle or abs(pitch) > max_angle or cos_roll_pitch < 0.3:
            return # self.m_hat


        denominator = self.last_f_collective * cos_roll_pitch + 1e-6             # Schutz vor 0

        params_acc_0   = (vz_dot + GRAVITY) / denominator - params_acc[1]/denominator
        if params_acc_0 <= 0:                         # safety against numerial errors
            return # self.m_hat

        alpha    = 0.02                                     # Glättung
        self.params_acc_0_hat = (1 - alpha) * self.params_acc_0_hat + alpha * params_acc_0
        # self.m_hat = k_thrust / self.k_hat                  # neue Massen-Schätzung -> nicht nötig





