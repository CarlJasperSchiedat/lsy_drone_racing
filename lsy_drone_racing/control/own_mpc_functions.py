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
    model_name = "own_mpc"

    # Define Gravitational Acceleration
    GRAVITY = 9.806

    # Sys ID Params
    params_pitch_rate = [-6.003842038081178, 6.213752925707588]
    params_roll_rate = [-3.960889336015948, 4.078293254657104]
    params_yaw_rate = [-0.005347588299390372, 0.0]
    params_acc = [20.907574256269616, 3.653687545690674] # params_acc[0] ≈ k_thrust / m_nominal



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
    p_ref = MX.sym("p_ref", 3)#

    # Update the Mass of the Drone online -> bzw. only the corresponding parameter of the model
    params_acc_0 = MX.sym("params_acc_0")

    #Define params necessary for external cost function
    params = vertcat(p_obs1, p_obs2, p_obs3, p_obs4, p_ref, params_acc_0)




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
        (params_acc_0 * f_collective + params_acc[1]) * cos(roll) * cos(pitch) - GRAVITY,
        # (params_acc[0] * f_collective + params_acc[1]) * cos(roll) * cos(pitch) - GRAVITY,  # params_acc[0] ≈ k_thrust / m_nominal
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
    model.p = params

    
    # # # #  Werte für 9 Sekunden optimiert # # # # # #
    # Penalize aggressive commands (smoother control)
    Q_control = 0.05 #0.01
    control_penalty = df_cmd**2 + dr_cmd**2 + dp_cmd**2 + dy_cmd**2

    # Penalize large angles (prevents flips)
    Q_angle = 0.05 #0.01
    angle_penalty = roll**2 + pitch**2  # Yaw penalty optional

    sharpness=8
    #Penalising proximity to obstacles
    d1 = (px - p_obs1[0])**sharpness + (py - p_obs1[1])**sharpness
    d2 = (px - p_obs2[0])**sharpness + (py - p_obs2[1])**sharpness
    d3 = (px - p_obs3[0])**sharpness + (py - p_obs3[1])**sharpness
    d4 = (px - p_obs4[0])**sharpness + (py - p_obs4[1])**sharpness
    safety_margin = 0.000002 # Min allowed distance squared
    Q_obs = 0 #2
    obs_cost = (0.25*np.exp(-d1/(safety_margin)) + np.exp(-d2/safety_margin) + 
           np.exp(-d3/safety_margin) + 0.5*np.exp(-d4/safety_margin))

    #Penalising deviation from Reference trajectory #1
    Q_pos = 10
    pos_error = (px - p_ref[0])**2 + (py - p_ref[1])**2 + (pz - p_ref[2])**2


    total_cost = (
        Q_pos * pos_error +
        Q_control * control_penalty +
        Q_angle * angle_penalty +
        Q_obs*obs_cost)

    model.cost_expr_ext_cost = total_cost
    model.cost_expr_ext_cost_e = Q_pos * 0.01 * pos_error 


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

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ocp.dims.np = model.p.rows()



    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados: https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

    # Cost Type
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    default_params = np.zeros(ocp.dims.np)
    ocp.parameter_values = default_params
   

    # Set State Constraints
    ocp.constraints.lbx = np.array([0.1, 0.1, -1.57, -1.57, -1.57])
    ocp.constraints.ubx = np.array([0.55, 0.55, 1.57, 1.57, 1.57])
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

    nx = model.x.rows()
    ocp.constraints.x0 = np.zeros((nx))




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


    # ACADOS WORKAROUND BEI WINDOWS:
    rename_acados_dll(name)




    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="own_mpc.json", verbose=verbose)

    return acados_ocp_solver, ocp








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

    # Input, Angle Penalty for smooth control / trajectory
    Q_control = 0.05 # 0.01
    control_penalty = df_cmd**2 + dr_cmd**2 + dp_cmd**2 + dy_cmd**2
    Q_angle = 0.05 # 0.01
    angle_penalty = roll**2 + pitch**2  # Yaw penalty optional



    # Parameter-Vektor für das Gate welches Durchflogen werden soll  (Gate-Mitte + Rotation)
    p_gate = MX.sym("p_gate", 3,1)
    R_gate = MX.sym("R_gate", 3,3)
    para_recompute = vertcat(p_gate, reshape(R_gate, 9, 1))

    '''
    # EXTERNAL-Stage-Kosten
    rel = mtimes(R_gate, vertcat(px, py, pz) - p_gate)

    border = 0.225          # halbe Öffnung
    Q_gate = 10            # Gewicht „Mitte halten“

    y_weight = exp( - 3000 * ( (rel[1] / 0.7) ** 4 ) )
    x_borderdist_cost = exp( - 1500 * ( (fabs(rel[0]) - border) / 0.5 )**6 )
    z_borderdist_cost = exp( - 1500 * ( (fabs(rel[2]) - border) / 0.5 )**6 )

    gate_penalty = y_weight * (x_borderdist_cost + z_borderdist_cost)

    ext_cost_stage    = ( Q_gate * gate_penalty + 
        Q_control * control_penalty + Q_angle * angle_penalty )
    '''


    ext_cost_stage    = Q_control * control_penalty + Q_angle * angle_penalty
    
    ext_cost_terminal = 0
    #'''




    # Initialize the nonlinear model for NMPC formulation
    model             = AcadosModel()
    model.name        = model_name
    model.x           = states
    model.u           = inputs
    model.p           = para_recompute
    model.f_expl_expr = f
    model.f_impl_expr = None
    model.cost_expr_ext_cost = ext_cost_stage
    model.cost_expr_ext_cost_e = ext_cost_terminal 


    return model




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


    ocp.solver_options.N_horizon = N        # Anzahl Shooting-Knoten = gewünschte Sim-Schritte
    ocp.solver_options.tf        = Tf       # Tf = N * dt


    # Get Dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    np_param = model.p.rows()
    ocp.dims.np = np_param
    ocp.parameter_values = np.zeros(np_param)




    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados: https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

    # Cost Type
    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"


    # Set State Constraints
    # # normal constrains for 9 to 13
    # # + contraining the positon and velocity to inf so that gate pos  and first can be set anywhere.
    ocp.constraints.idxbx   = np.array([0,1,2,3,4,5,6,7,8,    9,10,11,12,13])
    ocp.constraints.lbx     = np.array([-1.00e+05]*9 +          [0.1,0.1,-1.57,-1.57,-1.57])
    ocp.constraints.ubx     = np.array([ 1.00e+05]*9 +          [0.8,0.8, 1.57, 1.57, 1.57])
    
    # # contrains for first state
    ocp.constraints.idxbx_0 = ocp.constraints.idxbx.copy()
    ocp.constraints.lbx_0   = ocp.constraints.lbx.copy()
    ocp.constraints.ubx_0   = ocp.constraints.ubx.copy()

    # # Constrain the terminal positon+velocity to the old trajectory
    ocp.constraints.idxbx_e = np.array([0,1,2,3,4,5,6,7,8,    9,10,11,12,13])
    ocp.constraints.lbx_e   = np.array([-1.00e+05]*9 +          [0.1,0.1,-1.57,-1.57,-1.57])
    ocp.constraints.ubx_e   = np.array([ 1.00e+05]*9 +          [0.8,0.8, 1.57, 1.57, 1.57])
    


    '''
    pos_idx = np.array([0,1,2], dtype=int)
    ns = pos_idx.size
    ocp.constraints.idxsbx = pos_idx
    ocp.cost.zl = np.full(ns, 1e4)                # quadratisch
    ocp.cost.zu = np.full(ns, 1e4)
    ocp.cost.Zl = np.zeros(ns)                    # linear (meist 0)
    ocp.cost.Zu = np.zeros(ns)
    '''



    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" # FULL_CONDENSING_QPOASES # FULL_CONDENSING_HPIPM
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP # SQP_RTI
    ocp.solver_options.tol = 1e-5

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 80 # 40 # 20
    ocp.solver_options.nlp_solver_max_iter = 100 # 50

    ocp.solver_options.tol                 = 1e-3




    # ACADOS WORKAROUND BEI WINDOWS:
    rename_acados_dll(name)




    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="recompute_ocp.json", verbose=verbose)

    return acados_ocp_solver, ocp


