"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import json
import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, sin, vertcat
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

import csv
import os
# Workaround für acados auf Windows – sorgt dafür, dass Kompilierung klappt
os.environ["CC"] = "gcc"
os.environ["LD"] = "gcc"
os.environ["RM"] = "del"



def export_quadrotor_ode_model() -> AcadosModel:
    """Symbolic Quadrotor Model."""
    # Define name of solver to be used in script
    model_name = "lsy_example_mpc"

    # Define Gravitational Acceleration
    GRAVITY = 9.806


    # Sys ID Params
    params_pitch_rate = [-6.003842038081178, 6.213752925707588]
    params_roll_rate = [-3.960889336015948, 4.078293254657104]
    params_yaw_rate = [-0.005347588299390372, 0.0]
    params_acc = [20.907574256269616, 3.653687545690674]

    """Model setting"""
    # define basic variables in state and input vector
    px = MX.sym("px")  # 0
    py = MX.sym("py")  # 1
    pz = MX.sym("pz")  # 2
    vx = MX.sym("vx")  # 3
    vy = MX.sym("vy")  # 4
    vz = MX.sym("vz")  # 5
    roll = MX.sym("r")  # 6
    pitch = MX.sym("p")  # 7
    yaw = MX.sym("y")  # 8
    f_collective = MX.sym("f_collective")

    f_collective_cmd = MX.sym("f_collective_cmd")
    r_cmd = MX.sym("r_cmd")
    p_cmd = MX.sym("p_cmd")
    y_cmd = MX.sym("y_cmd")

    df_cmd = MX.sym("df_cmd")
    dr_cmd = MX.sym("dr_cmd")
    dp_cmd = MX.sym("dp_cmd")
    dy_cmd = MX.sym("dy_cmd")

    # define state and input vectors
    states = vertcat(
        px,
        py,
        pz,
        vx,
        vy,
        vz,
        roll,
        pitch,
        yaw,
        f_collective,
        f_collective_cmd,
        r_cmd,
        p_cmd,
        y_cmd,
    )
    inputs = vertcat(df_cmd, dr_cmd, dp_cmd, dy_cmd)

    # Define nonlinear system dynamics
    f = vertcat(
        vx,
        vy,
        vz,
        (params_acc[0] * f_collective + params_acc[1])
        * (cos(roll) * sin(pitch) * cos(yaw) + sin(roll) * sin(yaw)),
        (params_acc[0] * f_collective + params_acc[1])
        * (cos(roll) * sin(pitch) * sin(yaw) - sin(roll) * cos(yaw)),
        (params_acc[0] * f_collective + params_acc[1]) * cos(roll) * cos(pitch) - GRAVITY,
        params_roll_rate[0] * roll + params_roll_rate[1] * r_cmd,
        params_pitch_rate[0] * pitch + params_pitch_rate[1] * p_cmd,
        params_yaw_rate[0] * yaw + params_yaw_rate[1] * y_cmd,
        10.0 * (f_collective_cmd - f_collective),
        df_cmd,
        dr_cmd,
        dp_cmd,
        dy_cmd,
    )
    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = model_name
    model.f_expl_expr = f
    model.f_impl_expr = None
    model.x = states
    model.u = inputs


    return model


def create_ocp_solver(
    Tf: float, N: int, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # set model
    model = export_quadrotor_ode_model()
    ocp.model = model

    # Get Dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados: https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    Q = np.diag(
        [
            100.0, 100.0, 100.0, # Position
            0.01, 0.01, 0.01, # 10.0, 10.0, 10.0, # Velocity
            0.1, 0.1, 0.1, # rpy
            0.01, 0.01, # f_collective, f_collective_cmd
            0.01, 0.01, 0.01, # rpy_cmd
        ]
    )  # rpy_cmd

    R = np.diag([0.01, 0.01, 0.01, 0.01])

    Q_e = Q.copy()

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[:nx, :] = np.eye(nx)  # Only select position states
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:nx, :nx] = np.eye(nx)  # Only select position states
    ocp.cost.Vx_e = Vx_e

    # Set initial references (we will overwrite these later on to make the controller track the traj.)
    # ocp.cost.yref = np.zeros((ny, ))
    # ocp.cost.yref_e = np.zeros((ny_e, ))
    ocp.cost.yref = np.array(
        [1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    ocp.cost.yref_e = np.array(
        [1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0]
    )

    # Set State Constraints
    ocp.constraints.lbx = np.array([0.1, 0.1, -1.57, -1.57, -1.57])
    ocp.constraints.ubx = np.array([0.55, 0.55, 1.57, 1.57, 1.57])
    # ocp.constraints.ubx = np.array([0.7, 0.7, 1.57, 1.57, 1.57])
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

    # Set Input Constraints
    # ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0. -10.0])
    # ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0])
    # ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI
    ocp.solver_options.tol = 1e-5

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf



    # Set nonlinear constraints for obstacle avoidance
    ocp.model.con_h_expr = model.con_h_expr
    
    # Lower bound for nonlinear constraint (must be > 0 to avoid pole)
    ocp.constraints.lh = np.array([0.0])  # Must be ≥ 0
    ocp.constraints.uh = np.array([1e8])  # No upper bound
    ocp.constraints.constr_type = 'BGH'




    # Alte JSON löschen
    if os.path.exists("lsy_example_mpc.json"):
        os.remove("lsy_example_mpc.json")

    # Alte DLL löschen
    dll_path = "c_generated_code/acados_ocp_solver_lsy_example_mpc.dll"
    if os.path.exists(dll_path):
        os.remove(dll_path)

    # Workaround: Rename DLL if needed (Windows: gcc prepends "lib")
    dll_expected = "c_generated_code/acados_ocp_solver_lsy_example_mpc.dll"
    dll_actual = "c_generated_code/libacados_ocp_solver_lsy_example_mpc.dll"

    if os.path.exists(dll_actual) and not os.path.exists(dll_expected):
        os.rename(dll_actual, dll_expected)
        print(f"🛠️ DLL umbenannt: {dll_actual} ➝ {dll_expected}")



    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="lsy_example_mpc.json", verbose=verbose)

    return acados_ocp_solver, ocp





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

        '''
        # Geschwindigkeit ?
        self.vx_des = cs_x.derivative()(ts)
        self.vy_des = cs_y.derivative()(ts)
        self.vz_des = cs_z.derivative()(ts)
        '''




        self.des_completion_time = 6



        self.N = 30
        self.T_HORIZON = 1.5
        self.dt = self.T_HORIZON / self.N
        # self.x_des = np.concatenate((self.x_des, [self.x_des[-1]] * (2 * self.N + 1)))
        # self.y_des = np.concatenate((self.y_des, [self.y_des[-1]] * (2 * self.N + 1)))
        # self.z_des = np.concatenate((self.z_des, [self.z_des[-1]] * (2 * self.N + 1)))

        self.acados_ocp_solver, self.ocp = create_ocp_solver(self.T_HORIZON, self.N)

        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.config = config
        self.finished = False






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
                waypoint_tick = int( (self._tick - np.mod(self._tick,ticks_per_segment)) )
                current_waypoint_idx = int( waypoint_tick  / ticks_per_segment )

                #print(f"The current Waypoint Calc: {(self._tick-np.mod(self._tick,ticks_per_segment))  / ticks_per_segment}")
                #print(f"The ChatGPT Waypoint Calc: {self._tick // ticks_per_segment}")
                #print("ticks per segment:", ticks_per_segment, "\n")

                self.update_traj(obs, current_waypoint_idx, waypoint_tick)
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
            ) )
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)

        self.y=[]
        for j in range(self.N):
            yref = np.array([
            self.x_des[i + j],  self.y_des[i + j],  self.z_des[i + j], # Position
            0.0, 0.0, 0.0,
            # self.vx_des[i + j], self.vy_des[i + j], self.vz_des[i + j], # Geschwindigkeit
            0.0, 0.0, 0.0,      # r, p, y
            0.35, 0.35,         # f_collective / _cmd
            0.0, 0.0, 0.0,      # rpy_cmd
            0.0, 0.0, 0.0, 0.0,
        ])

            self.acados_ocp_solver.set(j, "yref", yref)  
            self.y.append(yref)
        yref_N = np.array([
            self.x_des[i + self.N],  self.y_des[i + self.N],  self.z_des[i + self.N], # Position
            0.0, 0.0, 0.0,
            # self.vx_des[i + self.N], self.vy_des[i + self.N], self.vz_des[i + self.N], # Geschwindigkeit
            0.0, 0.0, 0.0,      # r, p, y
            0.35, 0.35,         # f_collective / _cmd
            0.0, 0.0, 0.0,      # rpy_cmd
        ])
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
        # print(f"\n\nw = {w}\n\n")
        w = 1
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]

        cmd = x1[10:14]



        # Seve flown trajectory
        if not hasattr(self, "log_ref"):
            self.log_ref = []
            self.log_real = []
        # Referenzposition zum aktuellen Tick
        self.log_ref.append([self.x_des[i], self.y_des[i], self.z_des[i]])
        # Aktuelle Position
        self.log_real.append(obs["pos"].tolist())



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

        # Sicherer Speicherordner
        os.makedirs("logs", exist_ok=True)

        data = {
            "ref": self.log_ref,
            "real": self.log_real
        }

        path = f"logs/trajectory_log.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"📦 Trajektorienverlauf gespeichert unter: {path}")

        # Optional: Zurücksetzen für nächste Episode
        self.log_ref = []
        self.log_real = []


    def check_for_update(self,obs):
        """
        return: flag:
        0 = keine update
        1 = update obstale
        2 = update gate
        """
        flag=0
        if not np.array_equal(self.prev_obstacle,obs["obstacles_pos"]):
            # print('Obstacle has changed:')  
            # print(obs["obstacles_pos"])
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



    
    def update_traj(self, obs, waypoint_idx, waypoint_tick):
        """
        Set the cubic splines new from the current position
        """

        if self._tick == 0:
            print("Kein Update, Tick == 0")
            return
        

        for i, idx in self.gate_map.items(): # update the waypoints that correspond to a specific gate
            self.waypoints[idx] = self.prev_gates[i]

        gate_idx = obs["target_gate"]
        center_idx = self.gate_map[int(gate_idx)]

        # 1. Neue Sub-Waypoints auswählen
        rel_indices = [-2, 0, 2, 3, 4, 5]
        abs_indices = [
            center_idx + i for i in rel_indices
            if 0 <= center_idx + i < len(self.waypoints)
        ]
        if len(abs_indices) < 2:
            print("⚠️ Nicht genug gültige Punkte für Splines.")
            return


        wp_section = self.waypoints[abs_indices]
        tick_section = [self.ticks[i] for i in abs_indices]
        tick_times = np.array(tick_section) / self.freq
        dt_segments = np.diff(tick_section)
        

        print("rel_indices:      ", rel_indices)
        print("abs_indices:      ", abs_indices)
        print("tick_section:     ", tick_section)
        print("tick_times:       ", tick_times)
        print("dt_segments:      ", dt_segments)


        ts = []
        for i in range(len(dt_segments)):
            t_start = tick_times[i]
            t_end = tick_times[i + 1]
            n_points = max(2, dt_segments[i])  # mind. 2 Punkte pro Segment
            ts_seg = np.linspace(t_start, t_end, n_points, endpoint=False)
            ts.extend(ts_seg)

        ts.append(tick_times[-1])  # letzten Zeitpunkt ergänzen
        ts = np.array(ts)


        # --- 3. Neue Splines erstellen
        cs_x = CubicSpline(tick_times, wp_section[:, 0])
        cs_y = CubicSpline(tick_times, wp_section[:, 1])
        cs_z = CubicSpline(tick_times, wp_section[:, 2])

        x_new = cs_x(ts)
        y_new = cs_y(ts)
        z_new = cs_z(ts)
        vx_new = cs_x.derivative()(ts) # Geschindigkei ?
        vy_new = cs_y.derivative()(ts)
        vz_new = cs_z.derivative()(ts)

        # --- 4. Aktuelle Trajektorie ersetzen
        tick_min = tick_section[0]
        tick_max = tick_section[-1]
        print(f"🔁 Ersetze Trajektorie von Tick {tick_min} bis {tick_max} ({tick_max - tick_min} Punkte)")

        # self.x_des = np.concatenate((self.x_des[:tick_min], x_new, self.x_des[tick_max + 1:]))
        # self.y_des = np.concatenate((self.y_des[:tick_min], y_new, self.y_des[tick_max + 1:]))
        # self.z_des = np.concatenate((self.z_des[:tick_min], z_new, self.z_des[tick_max + 1:]))
        self.x_des[tick_min:tick_max + 1]  = x_new
        self.y_des[tick_min:tick_max + 1]  = y_new
        self.z_des[tick_min:tick_max + 1]  = z_new
        #self.vx_des[tick_min:tick_max + 1] = vx_new
        #self.vy_des[tick_min:tick_max + 1] = vy_new
        #self.vz_des[tick_min:tick_max + 1] = vz_new

        print(f"✅ Neue Teiltrajektorie (Spline) um Gate {gate_idx} aktualisiert.")





        
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