"""This module implements various functions for a MPC using attitude control for a quadrotor."""

import os
import platform
from pathlib import Path

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, sin, vertcat

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




def export_quadrotor_ode_model(Q_all: np.ndarray, set_tunnel: bool = True) -> AcadosModel:
    """Symbolic Quadrotor Model.
     
    Args:
        Q_all: All weights for the cost function in an array.
        set_tunnel: If True, the model will include tunnel constraints.

    Returns:
        An AcadosModel
    """
    # Define name of solver to be used in script
    model_name = "mpc_universal"

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
    if set_tunnel:
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

    # Define params necessary for external cost function
    if set_tunnel:
        params = vertcat(p_obs1, p_obs2, p_obs3, p_obs4, p_ref,params_acc_0, p_tun_tan,p_tun_r)
    else:
        params = vertcat(p_obs1, p_obs2, p_obs3, p_obs4, p_ref,params_acc_0)

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = model_name
    model.f_expl_expr = f
    model.f_impl_expr = None
    model.x = states
    model.u = inputs
    model.p = params




    # # # # # # Tunnel Constaints nach MPCC # # # # # # 
    if set_tunnel:
        err = vertcat(px, py, pz) - p_ref
        err_par = (p_tun_tan.T @ err) * p_tun_tan
        err_senk = err - err_par
        h_tunnel  = (err_senk.T @ err_senk) - p_tun_r**2

        model.con_h_expr = vertcat(h_tunnel)
   
    

    # # # # # # Cost Funktion # # # # # #
    # Penalize aggressive commands (smoother control)
    control_penalty = df_cmd**2 + dr_cmd**2 + dp_cmd**2 + dy_cmd**2

    # Penalize large angles (prevents flips)
    angle_penalty = roll**2 + pitch**2  # Yaw penalty optional

    #Penalising proximity to obstacles
    sharpness=2
    d1 = (px - p_obs1[0])**sharpness + (py - p_obs1[1])**sharpness
    d2 = (px - p_obs2[0])**sharpness + (py - p_obs2[1])**sharpness
    d3 = (px - p_obs3[0])**sharpness + (py - p_obs3[1])**sharpness
    d4 = (px - p_obs4[0])**sharpness + (py - p_obs4[1])**sharpness
    safety_margin = 0.015 # Min allowed distance squared
    obs_cost = (0.01*np.exp(-d1/(safety_margin)) + np.exp(-d2/safety_margin) + 
           0.01*np.exp(-d3/safety_margin) + np.exp(-d4/safety_margin))

    #Penalising deviation from Reference trajectory #1
    pos_error = (px - p_ref[0])**2 + (py - p_ref[1])**2 + (pz - p_ref[2])**2


    total_cost = (
        Q_all[0] * control_penalty +
        Q_all[1] * angle_penalty +
        Q_all[2] * obs_cost +
        Q_all[3] * pos_error
        )
    model.cost_expr_ext_cost = total_cost
    model.cost_expr_ext_cost_e = Q_all[4] * pos_error


    return model


def create_ocp_solver(
    Tf: float, N: int, Q_all: np.ndarray, set_tunnel: bool = True, windows_workaround: bool = True, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver.
     
    Args:
        Tf: Time-Horizon of the MPC prediction in seconds.
        N: Number of steps in the prediction horizon.
        Q_all: All weights for the cost function in an array.
        set_tunnel: If True, the model will include tunnel constraints.
        windows_workaround: Annoying windows workaround for my windows PC that hates acados.
        verbose: If True, the solver will print additional information.

    Returns:
        - An AcadosOcpSolver instance for solving the MPC problem.
        - An AcadosOcp instance representing the OCP.
    """
    ocp = AcadosOcp()

    # set model
    model = export_quadrotor_ode_model(Q_all, set_tunnel=set_tunnel)
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

    if set_tunnel:
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

    if windows_workaround:
        rename_acados_dll("mpc_universal")

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="mpc_universal.json", verbose=verbose)

    return acados_ocp_solver, ocp