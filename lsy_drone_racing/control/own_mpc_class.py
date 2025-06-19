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


        self.prev_obstacle = [ [1, 0, 1.4], [0.5, -1, 1.4], [0, 1.5, 1.4], [-0.5, 0.5, 1.4], ]
        self.prev_gates_quat = [ [0.0, 0.0, 0.92268986, 0.38554308], [0.0, 0.0, -0.38018841, 0.92490906], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], ]
        self.prev_gates=[ [0.45, -0.5, 0.56], [1.1, -1.05, 1.11], [0.0, 1.0, 0.56], [-0.5, 0.0, 1.11], ]




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



        self.N = 30
        self.T_HORIZON = 1.5
        self.dt = self.T_HORIZON / self.N
        self.acados_ocp_solver, self.ocp = create_ocp_solver_for_mpc(self.T_HORIZON, self.N, name="example_mpc")

        self.N_recompute = 60
        self.acados_recompute_solver, self.ocp_recompute = create_ocp_solver_for_recompute(
            self.N_recompute * self.dt, self.N_recompute, name="recompute_ocp",
            verbose = False)

        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.config = config
        self.finished = False




        self.traj_update_log = []  # Hier werden alle Trajektorie-Vergleiche gespeichert
        self.traj_update_counter = 0







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
        



        update = self.check_for_update(obs)
        if update:
            if update ==2:
                print('Changes were detected, now we can update traj at:',self._tick)
                ticks_per_segment = int(self.freq * self.des_completion_time) // (len(self.waypoints) - 1)
                current_waypoint_idx = int( (self._tick-np.mod(self._tick,ticks_per_segment))  / ticks_per_segment )
                print('The current waypoint ist:', (self._tick-np.mod(self._tick,ticks_per_segment))  / ticks_per_segment  )
                print("ticks per segment:", ticks_per_segment, "\n")

                self.update_traj(obs, current_waypoint_idx)
            else:
                print('Changes were detected, obstacle:',self._tick)
                print("update prev_obstacle welche benutzte werden für NB", "\n")
                
        




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
        for j in range(self.N):
            yref = np.array(
                [
                    self.x_des[i + j],
                    self.y_des[i + j],
                    self.z_des[i + j],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.35,
                    0.35,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
            self.acados_ocp_solver.set(j, "yref", yref)  
            self.y.append(yref)
        yref_N = np.array(
            [
                self.x_des[i + self.N],
                self.y_des[i + self.N],
                self.z_des[i + self.N],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.35,
                0.35,
                0.0,
                0.0,
                0.0,
            ]
        )
        self.acados_ocp_solver.set(self.N, "yref", yref_N)

        self.acados_ocp_solver.solve()
        self.y_mpc = []
        for j in range(self.N + 1):  # Include terminal state
            x_pred = self.acados_ocp_solver.get(j, "x")
            # Extract relevant states to match your y_ref format if needed
            # This depends on how you want to compare them
            y_mpc = x_pred[:len(yref_N)]  # Adjust this based on your state vector
            self.y_mpc.append(y_mpc)

        x1 = self.acados_ocp_solver.get(1, "x")
        w = 1 / self.config.env.freq / self.dt
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



    def check_for_update(self,obs):
        """
        return: flag:
        0 = keine update
        1 = update obstale
        2 = update gate
        """
        flag=0
        if not np.array_equal(self.prev_obstacle,obs["obstacles_pos"]):
            print('Obstacle has changed:')  
            print(obs["obstacles_pos"])
            self.prev_obstacle=obs["obstacles_pos"]
            flag=1
        if not np.array_equal(self.prev_gates_quat,obs["gates_quat"]):
            # print('Gate_rotation has changed:')
            # print(obs['gates_quat'])
            self.prev_gates_quat=obs["gates_quat"]
            flag=2
        if not np.array_equal(self.prev_gates,obs["gates_pos"]):
            # print('Gate_position has changed:')
            # print(obs['gates_pos'])
            self.prev_gates=obs["gates_pos"]
            flag=2

        return flag

    
    def update_traj(self, obs, waypoint_idx):
        """
        Set the cubic splines new from the current position
        """
        def quat2rpy(q):
            """w,x,y,z -> roll,pitch,yaw (rad)"""
            w, x, y, z = q
            # Tait-Bryan (ZYX)-Konvention
            roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
            pitch = np.arcsin (2*(w*y - z*x))
            yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            return np.array([roll, pitch, yaw])
        
        def make_inf_bounds(nx):
            return -1e9*np.ones(nx), 1e9*np.ones(nx)

        # update the waypoints that correspond to a specific gates
        for i, idx in self.gate_map.items(): 
            self.waypoints[idx] = self.prev_gates[i]

        solver = self.acados_recompute_solver
        nx     = solver.acados_ocp.dims.nx
        current_tick = self._tick
        N_rec = self.N_recompute

        # Erstelle eine Map für Gates welche in der Zukunft liegen - Ticks n relation zu Jetzt (current_tick)
        gate_wp_idx   = np.fromiter(self.gate_map.values(), dtype=int)
        gate_ticks    = np.asarray(self.ticks)[gate_wp_idx]
        gate_ticks_rel = gate_ticks - current_tick
        visit_gate = gate_ticks_rel > 0
        gate_ticks_rel = gate_ticks_rel[visit_gate]
        gate_ids       = np.fromiter(self.gate_map.keys(), dtype=int)[visit_gate]
        gate_tick_map = dict(zip(gate_ids, gate_ticks_rel)) # map für 
        print(gate_tick_map)

        ticks_np = np.asarray(self.ticks)
        gate_tick_map = {
            gate_id: (rel_tick := ticks_np[wp_idx] - self._tick)        # relativer Tick zu current_tick
            for gate_id, wp_idx in self.gate_map.items()                # (gate → waypoint-idx) aus gegeben / geladener map
            if 0 < rel_tick <= N_rec                         # nur künftige Gates welche im Horizon liegen
        }
        print(gate_tick_map)




        # 1) Startzustand k=0 festnageln
        x0 = np.hstack([np.asarray(obs["pos"]), np.asarray(obs["vel"]), quat2rpy(obs["quat"]),
                        self.last_f_collective, self.last_f_cmd,
                        self.last_rpy_cmd])
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)

        # 2)  Gates auf den zugehörigen Ticks festnageln
        for gate_id, k_gate in gate_tick_map.items():
            gate_pos = self.prev_gates[gate_id]

            lbx, ubx = make_inf_bounds(nx)
            lbx[:3] = gate_pos
            ubx[:3] = gate_pos

            solver.set(k_gate, "lbx", lbx)
            solver.set(k_gate, "ubx", ubx)

        # 3)  Terminal-Position (k = N) festlegen
        terminal_idx = current_tick + N_rec
        term_pos = [self.x_des[terminal_idx], self.y_des[terminal_idx], self.z_des[terminal_idx]]
        solver.set("lbx_e", term_pos)
        solver.set("ubx_e", term_pos)

        # 4)  gleiche Gate-Parameter für alle Stufen
        p_vec = np.hstack([ self.prev_gates[obs["target_gate"]], self.prev_gates_quat[obs["target_gate"]].reshape(9) ])
        for k in range(N_rec + 1):
            solver.set(k, "p", p_vec)

        solver.solve()
        traj = np.vstack([solver.get(k, "x") for k in range(N_rec+1)])
        x_new, y_new, z_new = traj[:, 0], traj[:, 1], traj[:, 2]
        
        seg_len            = len(x_new)
        slice_idx          = slice(current_tick, current_tick + seg_len)       # t0 … t0+N
        self.x_des[slice_idx] = x_new
        self.y_des[slice_idx] = y_new
        self.z_des[slice_idx] = z_new

        print(f"✅ Trajektorie-Teil ab Index {waypoint_idx} erfolgreich ersetzt.")




        
    def save_traj_update_csv(self, tick: int, old_traj: np.ndarray, new_traj: np.ndarray):
        os.makedirs("logs", exist_ok=True)
        with open("logs/traj_update_log.csv", "a", newline="") as f:
        # with open("traj_update_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tick", tick])
            writer.writerow(["old_x", "old_y", "old_z"])
            writer.writerows(old_traj)
            writer.writerow(["new_x", "new_y", "new_z"])
            writer.writerows(new_traj)
            writer.writerow([])  # Leerzeile zur Trennung

        print(f"💾 Trajektorienupdate gespeichert bei Tick {tick}, Länge alt: {len(old_traj)}, neu: {len(new_traj)}")

        
