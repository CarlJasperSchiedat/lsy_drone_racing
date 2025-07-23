"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
ATTACKE #2
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


import os
import platform
from pathlib import Path

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, sin, vertcat
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

# from lsy_drone_racing.utils.utils import generate_nonuniform_ts

os.environ["CC"] = "gcc"
os.environ["LD"] = "gcc"
os.environ["RM"] = "del"
def rename_acados_dll(name: str):
    """Workaround für acados auf Windows - sorgt dafür, dass Kompilierung klappt.
            
    Args:
        name: Name of the .ddl that has to be changed.

    Returns:
        None
    """
    # Unter Linux/macOS ist kein Rename nötig
    if platform.system().lower() != "windows":
        return

    # ▸ 1. Alte Artefakte aufräumen
    json_path = Path(f"{name}.json")
    if json_path.exists():
        json_path.unlink(missing_ok=True)

    # Ziel-DLL, die wir gerne haben möchten
    dst = Path("c_generated_code") / f"acados_ocp_solver_{name}.dll"
    # Quell-DLL, wie acados sie normalerweise erzeugt
    src = Path("c_generated_code") / f"libacados_ocp_solver_{name}.dll"

    # ▸ 2. Wenn das Ziel bereits existiert, ist die DLL schon umbenannt oder im Einsatz → nichts tun
    if dst.exists():
        return

    # ▸ 3. Prüfen, ob die Quell-DLL wirklich vorliegt
    if not src.exists():
        raise FileNotFoundError(src)

    # ▸ 4. Rename versuchen – schlägt unter Windows fehl, wenn die Datei gerade geladen ist
    try:
        src.rename(dst)
        print(f"🛠️ DLL renamed: {src} ➝ {dst}")
    except PermissionError:
        # DLL ist bereits von Python/ctypes geladen – rename nicht möglich
        # → einfach still überspringen, damit der nächste Solver-Build nicht crasht
        print(f"⚠️  DLL bereits in Benutzung – rename übersprungen ({src.name})")


def export_quadrotor_ode_model() -> AcadosModel:
    """Symbolic Quadrotor Model."""
    # Define name of solver to be used in script
    model_name = "lsy_example_mpc_ext"

    # Define Gravitational Acceleration
    GRAVITY = 9.806

    # Sys ID Params
    params_pitch_rate = [-6.003842038081178, 6.213752925707588]
    params_roll_rate = [-3.960889336015948, 4.078293254657104]
    params_yaw_rate = [-0.005347588299390372, 0.0]
    params_acc = [20.907574256269616, 3.653687545690674]

    """Model setting"""
    # Position
    px, py, pz = MX.sym("px"), MX.sym("py"), MX.sym("pz")
    # Geschwindigkeit
    vx, vy, vz = MX.sym("vx"), MX.sym("vy"), MX.sym("vz")
    # Euler-Winkel
    roll, pitch, yaw = MX.sym("r"), MX.sym("p"), MX.sym("y")
    # interne Zustände & Befehle
    f_collective      = MX.sym("f_collective")
    f_collective_cmd  = MX.sym("f_collective_cmd")
    r_cmd, p_cmd, y_cmd = MX.sym("r_cmd"), MX.sym("p_cmd"), MX.sym("y_cmd")

    # Eingänge
    df_cmd = MX.sym("df_cmd")
    dr_cmd = MX.sym("dr_cmd")
    dp_cmd = MX.sym("dp_cmd")
    dy_cmd = MX.sym("dy_cmd")
    
    # Obstacles as symbolic parameters (4 obstacles in 2D)
    p_obs1 = MX.sym("p_obs1", 2)
    p_obs2 = MX.sym("p_obs2", 2)
    p_obs3 = MX.sym("p_obs3", 2)
    p_obs4 = MX.sym("p_obs4", 2)
    p_ref = MX.sym("p_ref", 3)
    
    # Update the Mass of the Drone online -> bzw. only the corresponding parameter of the model
    params_acc_0 = MX.sym("params_acc_0")

    # Prameters for the Tunnel-Constrains
    p_tun_tan = MX.sym("y_ref_d", 3)     # Tunnel Tangente (x,y,z)
    p_tun_r = MX.sym("tunnel_r")        # Breite (Radius)

    # define state and input vector
    states = vertcat(
        px, py, pz,
        vx, vy, vz,
        roll, pitch, yaw,
        f_collective, f_collective_cmd,
        r_cmd, p_cmd, y_cmd,
    )
    inputs = vertcat(df_cmd, dr_cmd, dp_cmd, dy_cmd)

    # Define nonlinear system dynamics
    acc_term = (params_acc_0 * f_collective + params_acc[1])
    f = vertcat(
        vx, vy, vz,
        acc_term * (cos(roll) * sin(pitch) * cos(yaw) + sin(roll) * sin(yaw)),
        acc_term * (cos(roll) * sin(pitch) * sin(yaw) - sin(roll) * cos(yaw)),
        acc_term * cos(roll) * cos(pitch) - GRAVITY,
        params_roll_rate[0] * roll + params_roll_rate[1] * r_cmd,
        params_pitch_rate[0] * pitch + params_pitch_rate[1] * p_cmd,
        params_yaw_rate[0] * yaw + params_yaw_rate[1] * y_cmd,
        10.0 * (f_collective_cmd - f_collective),
        df_cmd,
        dr_cmd, dp_cmd, dy_cmd,
    )


    #Define params necessary for external cost function
    params = vertcat(p_obs1, p_obs2, p_obs3, p_obs4, p_ref,params_acc_0,p_tun_tan,p_tun_r)

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = model_name
    model.f_expl_expr = f
    model.f_impl_expr = None
    model.x = states
    model.u = inputs
    model.p = params




    # # # # # # Tunnel Constaints nach MPCC # # # # # # 
    err = vertcat(px, py, pz) - p_ref
    err_par = (p_tun_tan.T @ err) * p_tun_tan
    err_senk = err - err_par
    h_tunnel  = (err_senk.T @ err_senk) - p_tun_r**2

    model.con_h_expr = vertcat(h_tunnel)
   
    

    # # # # # # Cost Funktion # # # # # # 
    # Penalize aggressive commands (smoother control)
    Q_control = 0.05
    control_penalty = df_cmd**2 + dr_cmd**2 + dp_cmd**2 + dy_cmd**2

    # Penalize large angles (prevents flips)
    Q_angle = 0.05
    angle_penalty = roll**2 + pitch**2  # Yaw penalty optional

    sharpness=2
    #Penalising proximity to obstacles
    d1 = (px - p_obs1[0])**sharpness + (py - p_obs1[1])**sharpness
    d2 = (px - p_obs2[0])**sharpness + (py - p_obs2[1])**sharpness
    d3 = (px - p_obs3[0])**sharpness + (py - p_obs3[1])**sharpness
    d4 = (px - p_obs4[0])**sharpness + (py - p_obs4[1])**sharpness
    safety_margin = 0.015 # Min allowed distance squared
    Q_obs=50 
    obs_cost = (0*np.exp(-d1/(safety_margin)) + np.exp(-d2/safety_margin) + 
           0*np.exp(-d3/safety_margin) + np.exp(-d4/safety_margin))

    #Penalising deviation from Reference trajectory #1
    Q_pos = 10.0
    Q_pos_e = 10.0
    pos_error = (px - p_ref[0])**2 + (py - p_ref[1])**2 + (pz - p_ref[2])**2


    total_cost = (
        Q_pos * pos_error +
        Q_control * control_penalty +
        Q_angle * angle_penalty
        +Q_obs*obs_cost)

    model.cost_expr_ext_cost = total_cost
    model.cost_expr_ext_cost_e = Q_pos_e * pos_error 


    return model




def create_ocp_solver(
    Tf: float, N: int, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # set model
    model = export_quadrotor_ode_model()
    ocp.model = model
    

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ocp.dims.np = model.p.rows()


    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados: https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

    # Cost Type
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    default_params = np.zeros(ocp.dims.np)
    ocp.parameter_values = default_params  # Add this line
   

    # Set State Constraints
    ocp.constraints.lbx = np.array([0.1, 0.1, -1.57, -1.57, -1.57])
    ocp.constraints.ubx = np.array([0.55, 0.55, 1.57, 1.57, 1.57])
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

    nx = model.x.rows()
    ocp.constraints.x0 = np.zeros((nx))

    
    # Non-linear Tunnel Constraints
    ocp.dims.nh = 1
    ocp.constraints.lh = np.array([-100000])
    ocp.constraints.uh = np.array([0.0])



    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI
    ocp.solver_options.tol = 1e-5

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 40
    ocp.solver_options.nlp_solver_max_iter = 100

    # set prediction horizon
    ocp.solver_options.tf = Tf


    rename_acados_dll("lsy_example_mpc_ext")

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="lsy_example_mpc_ext.json", verbose=verbose)
    

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


        # Waypoints as in the trajectory controller but tuned and extended by trial and error.
        self.waypoints= np.array([
                [1.0, 1.5, 0.05],  # Original Punkt 0
                [0.95, 1.0, 0.2],   # Original Punkt 1
                [0.8, 0.3, 0.35], # Neu (Mitte zwischen 1 und 2)
                [0.7, -0.2, 0.5],#[0.65, -0.2, 0.5], # Original Punkt 2 (gate 0)
                [0.12, -0.9, 0.575], # Neu (Mitte zwischen 2 und 3)
                [0.1, -1.5, 0.65],  # Original Punkt 3
                [0.9, -1.4, 0.9],#[0.8, -1.35, 0.9],#[0.75, -1.3, 0.9], # Neu (Mitte zwischen 3 und 4)
                 [1.2, -0.8, 1.15],#[1.15, -0.8, 1.15],#[1.1, -0.85, 1.15], # Original Punkt 4 (gate 1)
                [0.65, -0.175, 0.85], # Neu (Mitte zwischen 4 und 5)
                [0.0, 0.4, 0.45],#[0.1, 0.45, 0.45],#[0.1, 0.45, 0.55],   
                [0.0, 1.32, 0.375],#[0.0, 1.28, 0.375],#[0.0, 1.2, 0.375],#[0.0, 1.2, 0.425],  # Original Punkt 6 (gate 2)
                [0.0, 1.32, 1.1],#[0.0, 1.28, 1.1], #[0.0, 1.2, 1.1],    # Original Punkt 7
                [-0.15, 0.6, 1.1],  # Neu (Mitte zwischen 7 und 8)
                [-0.5, 0.0, 1.1],   # Original Punkt 8 (gate 3)
                [-0.92, -0.5, 1.1],#[-0.9, -0.5, 1.1],#[-0.8, -0.5, 1.1],  # Original Punkt 9
                [-1.6, -1.0, 1.1],#[-1.4, -1.0, 1.1],#[-1.1, -1.0, 1.1],  # Original Punkt 10
            ])
        
        self.gate_map = {
            0 : 3,
            1 : 7,
            2 : 10,
            3 : 13
        }


        self.init_gates=[ [0.45, -0.5, 0.56], [1.0, -1.05, 1.11], [0.0, 1.0, 0.56], [-0.5, 0.0, 1.11], ]

        self.prev_obstacle = np.array([ [1, 0, 1.4], [0.5, -1, 1.4], [0, 1.5, 1.4], [-0.5, 0.5, 1.4], ])
        self.prev_gates_quat = [ [0.0, 0.0, 0.92268986, 0.38554308], [0.0, 0.0, -0.38018841, 0.92490906], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], ]
        self.prev_gates=[ [0.45, -0.5, 0.56], [1.0, -1.05, 1.11], [0.0, 1.0, 0.56], [-0.5, 0.0, 1.11], ]


        # Scale trajectory between 0 and 1
        ts = np.linspace(0, 1, np.shape(self.waypoints)[0])
        cs_x = CubicSpline(ts, self.waypoints[:, 0])
        cs_y = CubicSpline(ts, self.waypoints[:, 1])
        cs_z = CubicSpline(ts, self.waypoints[:, 2])

        #visualising traj. Needed for visualisiing draw line##
        tvisual = np.linspace(0, 1, 50)
        x = cs_x(tvisual)
        y = cs_y(tvisual)
        z = cs_z(tvisual)
        self.traj_vis=np.array([x,y,z])
        self.update_traj_vis=np.array([x,y,z])
        


        self.des_completion_time = config.controller.get("COMPLETION_TIME", 5.12)
        self.N = 20
        self.T_HORIZON = 1
        self.dt = self.T_HORIZON / self.N


        ts = np.linspace(0, 1, int(self.freq * self.des_completion_time))
        #ts = generate_nonuniform_ts(self.freq, self.des_completion_time)

        ticks_per_segment = int(self.freq * self.des_completion_time) / (len(self.waypoints) - 1)
        self.ticks = np.round(np.arange(0, len(self.waypoints)) * ticks_per_segment).astype(int)




        self.x_des = cs_x(ts)
        self.y_des = cs_y(ts)
        self.z_des = cs_z(ts)

        # Append points after trajectory for MPC
        self.x_des = np.concatenate((self.x_des, [self.x_des[-1]] * (2 * self.N + 1)))
        self.y_des = np.concatenate((self.y_des, [self.y_des[-1]] * (2 * self.N + 1)))
        self.z_des = np.concatenate((self.z_des, [self.z_des[-1]] * (2 * self.N + 1)))
    
        self.acados_ocp_solver, self.ocp = create_ocp_solver(self.T_HORIZON, self.N)

        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.config = config
        self.finished = False
        self.params_acc_0_hat = 20.907574256269616 # params_acc[0] ≈ k_thrust / m_nominal ; nominal value given for nominal_mass = 0.027
        self.vz_prev = 0.0 # estimated velocity at start = 0

        self.tunnel_width = 0.4 # Tunnel width (radius) for the MPC Tunnel constraintsAdd commentMore actions
        self.tunnel_w_gate = 0.2 # Tunnel width at the gate
        self.tunnel_trans = 0.6 # Distance at which the tunnel width transitions from gate width to far width


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
        self.mass_estimator(obs)

        # Check if "Update Gates" is nessessary
        updated_gate = self.check_for_update(obs)
        if updated_gate:
            self.update_traj(obs,updated_gate)
            
        if not np.array_equal(self.prev_obstacle,obs["obstacles_pos"]):
            print('Obstacle has changed:')  
            self.prev_obstacle=obs["obstacles_pos"]


        i = min(self._tick, len(self.x_des) - 1)
        if self._tick > i:
            self.finished = True


        q = obs["quat"]
        r = R.from_quat(q)
        rpy = r.as_euler("xyz", degrees=False)

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

        
        # Set parameters for the MPC
        self.y=[] # self.y for debug visulization
        for j in range(self.N):

            # # Help-Parameters for the Tunnel Constraints
            y_ref = np.array([ self.x_des[i + j], self.y_des[i + j], self.z_des[i + j] ])
            y_ref_p1 = np.array([ self.x_des[i + j + 1], self.y_des[i + j + 1], self.z_des[i + j + 1] ])
            delta = np.array(y_ref_p1 - y_ref)
            tangent_norm = delta / ( np.linalg.norm(delta) + 1e-6 )

            tunnel_width = self._tunnel_radius(y_ref)

            yref = np.hstack([ # params = vertcat(p_obs1, p_obs2, p_obs3, p_obs4, p_ref, params_acc_0, p_tun_tan, p_tun_r)
                self.prev_obstacle[:, :2].flatten(),
                y_ref, 
                self.params_acc_0_hat,
                tangent_norm,
                tunnel_width,
                ])

            self.acados_ocp_solver.set(j, "p", yref)
            self.y.append(yref) # self.y for debug visulization


        # # Help-Parameters for the Tunnel Constraints
        y_ref = np.array([ self.x_des[i + self.N], self.y_des[i + self.N], self.z_des[i + self.N] ])
        y_ref_p1 = np.array([ self.x_des[i + self.N + 1], self.y_des[i + self.N + 1], self.z_des[i + self.N + 1] ])
        delta = np.array(y_ref_p1 - y_ref)
        tangent_norm = delta / ( np.linalg.norm(delta) + 1e-6 )
        tunnel_width = self._tunnel_radius(y_ref)

        yref_N = np.hstack([
            self.prev_obstacle[:, :2].flatten(),
            self.x_des[i + self.N],
            self.y_des[i + self.N],
            self.z_des[i + self.N],
            self.params_acc_0_hat,
            tangent_norm,
            tunnel_width,
        ])
        self.acados_ocp_solver.set(self.N, "p", yref_N)

        
        self.acados_ocp_solver.solve()


        self.y_mpc = [] # for Debug: Plot the prediction of the MPC
        for j in range(self.N + 1):
            x_pred = self.acados_ocp_solver.get(j, "x")
            y_mpc = x_pred[:len(yref_N)] # [:len(yref_N)] # only take the nessessary states
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
    
    def check_for_update(self, obs: dict[str, NDArray[np.floating]]) -> int | None:
        """Check if any gate's position has changed significantly.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.

        Returns:
            - `None` if no gate moved beyond threshold
            - The **index (int)** of the first changed gate (row-wise comparison).
        """
        threshold = 0.05

        current_gates = np.asarray(obs["gates_pos"])
        for gate_idx in range(len(self.prev_gates)):  # Compare each gate (row) individually
            prev_gate = np.asarray(self.prev_gates[gate_idx])
            current_gate = np.asarray(current_gates[gate_idx])
            
            if np.linalg.norm(prev_gate - current_gate) > threshold:
                self.prev_gates = current_gates.copy()  # Update stored positions
                print(f"Gate {gate_idx} moved significantly.")
                print(self.prev_gates[gate_idx])
                return gate_idx+1  # Add one, so that we can check update for gate 0 with if statement. 
        
        return None

    def update_traj(self, obs: dict[str, NDArray[np.floating]], updated_gate: int):
        """Set the cubic splines new from the current position.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            updated_gate: The (index+1) of the gate that has moved beyond the threshold.

        Returns:
            None
        """
        if self._tick == 0:
            print("Kein Update, Tick == 0")
            return
        
        # update the waypoints that correspond to a specific gate
        for i, idx in self.gate_map.items(): # update the waypoints that correspond to a specific gate
            diff=self.prev_gates[i]-self.init_gates[i]
            self.waypoints[idx] += diff*1.2

        gate_idx = updated_gate-1 # Subtract the one we added in check_for_update because of if statement
        center_idx = self.gate_map[int(gate_idx)]

        # 1. Neue Sub-Waypoints auswählen
        rel_indices = [-1, 0, 1]
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
        

        # 2. Preparation for new segment
        ts = []
        for i in range(len(dt_segments)):
            t_start = tick_times[i]
            t_end = tick_times[i + 1]
            n_points = max(2, dt_segments[i])  # mind. 2 Punkte pro Segment
            ts_seg = np.linspace(t_start, t_end, n_points, endpoint=False)
            ts.extend(ts_seg)
        ts.append(tick_times[-1])  # letzten Zeitpunkt ergänzen
        ts = np.array(ts)


        # 3. Neue Splines erstellen
        cs_x = CubicSpline(tick_times, wp_section[:, 0])
        cs_y = CubicSpline(tick_times, wp_section[:, 1])
        cs_z = CubicSpline(tick_times, wp_section[:, 2])
        x_new = cs_x(ts)
        y_new = cs_y(ts)
        z_new = cs_z(ts)
        # For Visulization of the updated Trajectory
        self.update_traj_vis=np.array([x_new,y_new,z_new])

        # 4. Aktuelle Trajektorie ersetzen
        tick_min = tick_section[0]
        tick_max = tick_section[-1]
       
        self.x_des[tick_min:tick_max + 1]  = x_new
        self.y_des[tick_min:tick_max + 1]  = y_new
        self.z_des[tick_min:tick_max + 1]  = z_new


        print(f"✅ Neue Teiltrajektorie um Gate {gate_idx} aktualisiert.")
        

    def mass_estimator(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Updates the Acceleration Parameter in the MPC-Solver corresponding to the drone mass.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.

        Returns:
            None
        """
        max_angle = max_angle=np.deg2rad(20)


        params_acc = [20.907574256269616, 3.653687545690674] # params_acc[0] ≈ k_thrust / m_nominal
        GRAVITY = 9.806


        # Messgrößen
        vz_dot   = (obs["vel"][2] - self.vz_prev) / self.dt
        self.vz_prev = obs["vel"][2] # update für nächsten Durchlauf

        roll, pitch, _ = R.from_quat(obs["quat"]).as_euler("xyz", degrees=False)
        cos_roll_pitch   = np.cos(roll) * np.cos(pitch)
        
        # Only update, when Drone is upright
        if abs(roll) > max_angle or abs(pitch) > max_angle or cos_roll_pitch < 0.3:
            return


        denominator = self.last_f_collective * cos_roll_pitch + 1e-6    # safety against numerial errors

        params_acc_0   = (vz_dot + GRAVITY) / denominator - params_acc[1]/denominator
        if params_acc_0 <= 0:                         # safety against numerial errors
            return

        alpha    = 0.02                               # Glättung
        self.params_acc_0_hat = (1 - alpha) * self.params_acc_0_hat + alpha * params_acc_0


    def _tunnel_radius(self, p_ref: np.ndarray) -> float:
        """ref_pt: np.array([x,y,z]) eines MPC-Knotens.
        
        Args:
            p_ref: 3D-position of the reference point in the MPC trajectory.

        Returns:
            The radius of the tunnel at the given reference point.
        """
        # Entfernung zum nächsten Gate-Zentrum
        d_gate = np.min(np.linalg.norm(self.prev_gates - p_ref, axis=1))

        # lineare Interpolation zwischen R_gate und R_far 
        alpha = np.clip(d_gate / self.tunnel_trans, 0.0, 1.0)

        return self.tunnel_w_gate + (self.tunnel_width - self.tunnel_w_gate) * alpha