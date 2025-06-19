"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""


from __future__ import annotations  # Python 3.10 type hints

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, sin, vertcat, mtimes, reshape, fabs, exp
from scipy.spatial.transform import Rotation as R



import os
import platform
# Workaround für acados auf Windows – sorgt dafür, dass Kompilierung klappt
os.environ["CC"] = "gcc"
os.environ["LD"] = "gcc"
os.environ["RM"] = "del"


def rename_acados_dll(name: str):
    """Rename the acados DLL on Windows if needed."""
    if platform.system().lower() != "windows":
        return  # Nur unter Windows notwendig

    # print(f"🛠️ In 'rename_acados_dll' with name '{name}'")

    # Alte JSON löschen
    json_path = f"{name}.json"
    if os.path.exists(json_path):
        os.remove(json_path)

    # Alte DLL löschen
    dll_path = f"c_generated_code/acados_ocp_solver_{name}.dll"
    if os.path.exists(dll_path):
        os.remove(dll_path)

    expected = f"c_generated_code/acados_ocp_solver_{name}.dll"
    actual = f"c_generated_code/libacados_ocp_solver_{name}.dll"
    if os.path.exists(actual) and not os.path.exists(expected):
        os.rename(actual, expected)
        print(f"🛠️ DLL renamed: {actual} ➝ {expected}")

        

def export_quadrotor_ode_model_for_mpc() -> AcadosModel:
    """Symbolic Quadrotor Model."""
    # Define name of solver to be used in script
    model_name = "example_mpc"

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


def export_quadrotor_ode_model_for_recompute() -> AcadosModel:
    """Symbolic Quadrotor Model."""
    # Define name of solver to be used in script
    model_name = "recompute_ocp"

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



    # define state and input vectors
    states = vertcat(px, py, pz,
                vx, vy, vz,
                roll, pitch, yaw,
                f_collective,
                f_collective_cmd, r_cmd, p_cmd, y_cmd)
    
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

    # Parameter-Vektor für das Gate welches Durchflogen werden soll  (Gate-Mitte + Rotation)
    p_gate = MX.sym("p_gate", 3)
    R_gate = MX.sym("R_gate", 9)
    p      = vertcat(p_gate, R_gate)

    # EXTERNAL-Stage-Kosten
    rel = mtimes(reshape(R_gate, 3, 3).T,
                 vertcat(px, py, pz) - p_gate)

    border = 0.225          # halbe Öffnung
    w_g    = 2e3            # Gewicht „Mitte halten“

    y_weight = exp(-((rel[1] / 0.7) ** 4) * 3000)
    x_borderdist_cost = ( - 1500 * ((fabs(rel[0]) - border) / 0.4 )**6 )
    z_borderdist_cost = ( - 1500 * ((fabs(rel[2]) - border) / 0.4 )**6 )

    gate_pen = w_g * y_weight * (x_borderdist_cost + z_borderdist_cost)

    R_u = np.diag([1e-3, 1e-3, 1e-3, 1e-3])
    input_pen = mtimes([inputs.T, R_u, inputs])

    ext_cost_stage    = gate_pen     +     input_pen  # Gate Penalty   +   Standard Input Penalty
    ext_cost_terminal = MX(0)                               # Terminal Penalty = 0





    # Initialize the nonlinear model for NMPC formulation
    model             = AcadosModel()
    model.name        = model_name
    model.x           = states
    model.u           = inputs
    model.p           = p
    model.f_expl_expr = f
    model.f_impl_expr = None
    model.ext_cost_expr   = ext_cost_stage
    model.ext_cost_expr_e = ext_cost_terminal


    return model




def create_ocp_solver_for_mpc(
    Tf: float,
    N: int,
    name: str,
    verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # set model
    model = export_quadrotor_ode_model_for_mpc()
    model.name = name
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
    Q = np.diag( [ 10.0, 10.0, 10.0,  # Position
            0.01, 0.01, 0.01,  # Velocity
            0.1, 0.1, 0.1,  # rpy
            0.01, 0.01,  # f_collective, f_collective_cmd
            0.01, 0.01, 0.01, ] )  # rpy_cmd

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


    # ACADOS WORKAROUND BEI WINDOWS:
    rename_acados_dll(name)



    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="example_mpc.json", verbose=verbose)

    return acados_ocp_solver, ocp






def create_ocp_solver_for_recompute(
    Tf: float,
    N: int,
    name: str,
    verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # set model
    model = export_quadrotor_ode_model_for_recompute()
    model.name = name
    ocp.model = model

    # Get Dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    np_param = model.p.rows()
    ocp.dims.np = np_param
    ocp.parameter_values = np.zeros(np_param)


    # Horizont- & Solver-Settings
    ocp.dims.N            = N          # Anzahl Shooting-Intervals
    ocp.solver_options.tf = Tf         # Horizontlänge in s

    ocp.solver_options.qp_solver        = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx   = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type  = "ERK"
    ocp.solver_options.nlp_solver_type  = "SQP"
    ocp.solver_options.tol              = 1e-5
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max   = 20
    ocp.solver_options.nlp_solver_max_iter  = 50

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados: https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

    # Cost Type
    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"




    # All states of all ticks can be later be bound
    ocp.constraints.idxbx = np.arange(nx)    

    # initilise infinite constrains
    lbx = -1e9 * np.ones(nx)
    ubx =  1e9 * np.ones(nx)

    # Set specific State Constraints - fixed
    lbx[[9,10,11,12,13]] = [0.1, 0.1, -1.57, -1.57, -1.57]
    ubx[[9,10,11,12,13]] = [0.55, 0.55,  1.57,  1.57,  1.57]

    ocp.constraints.lbx = lbx
    ocp.constraints.ubx = ubx

    # Set Input Constraints
    # ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0. -10.0])
    # ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0])
    # ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # Set Terminal Constrains
    ocp.constraints.idxbx_e = np.array([0,1,2])   # px,py,pz
    ocp.constraints.lbx_e   = np.zeros(3)
    ocp.constraints.ubx_e   = np.zeros(3)



    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))


    '''
    # -------------------------------------------------- 7) optionale BGH-Constraints (Hindernisse)
    if model.con_h_expr is not None:
        nh = model.con_h_expr.shape[0]
        ocp.constraints.constr_type = "BGH"
        ocp.constraints.lh = np.zeros(nh)          # ≥ 0
        ocp.constraints.uh = 1e8 * np.ones(nh)     # keine Obergrenze
    '''



    # ACADOS WORKAROUND BEI WINDOWS:
    rename_acados_dll(name)



    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="recompute_ocp.json", verbose=verbose)

    return acados_ocp_solver, ocp


