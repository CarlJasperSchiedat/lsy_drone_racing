"""
This module contains the helping functions for my_mpc_controller.py .

0) export_quadrotor_ode_model() -> drone model for solving the OCP in trajectory optimization and MPC

1) recompute_trajectory

2) OCP-Funtion for the MPC

"""

import numpy as np
import scipy
import os

# from scipy.interpolate import CubicSpline
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from scipy.spatial.transform import Rotation as R
from casadi import MX, Function, DM, cos, sin, vertcat, sumsqr, exp, fmin, fabs, dot, inf, nlpsol





def export_quadrotor_ode_model(acados: bool = False):
    # Define Drone Model

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
    f_collective = MX.sym("f_collective") # 9

    f_collective_cmd = MX.sym("f_collective_cmd") # 10
    r_cmd = MX.sym("r_cmd") # 11
    p_cmd = MX.sym("p_cmd") # 12
    y_cmd = MX.sym("y_cmd") # 13

    df_cmd = MX.sym("df_cmd")
    dr_cmd = MX.sym("dr_cmd")
    dp_cmd = MX.sym("dp_cmd")
    dy_cmd = MX.sym("dy_cmd")



    # define state and input vector
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

    # Define nonlinear system dynamics -> dot(states)
    f = vertcat(
        vx, # 1
        vy, # 2
        vz, # 3
        (params_acc[0] * f_collective + params_acc[1])
        * (cos(roll) * sin(pitch) * cos(yaw) + sin(roll) * sin(yaw)), # 4
        (params_acc[0] * f_collective + params_acc[1])
        * (cos(roll) * sin(pitch) * sin(yaw) - sin(roll) * cos(yaw)), # 1
        (params_acc[0] * f_collective + params_acc[1]) * cos(roll) * cos(pitch) - GRAVITY, # 5
        params_roll_rate[0] * roll + params_roll_rate[1] * r_cmd, # 6
        params_pitch_rate[0] * pitch + params_pitch_rate[1] * p_cmd, # 7
        params_yaw_rate[0] * yaw + params_yaw_rate[1] * y_cmd, # 8
        10.0 * (f_collective_cmd - f_collective), # 9
        df_cmd, # 10
        dr_cmd, # 11
        dp_cmd, # 12
        dy_cmd, # 13
    )

    if acados:          # return AcadosModel for OCP - MPC-Steuerung
        model = AcadosModel()
        model.name = "my_mpc_controller"
        model.x = states
        model.u = inputs
        model.f_expl_expr = f
        model.f_impl_expr = None  # Explizite Form
        return model
    else:               # return CasADi Function for recomputing trajectory
        f_func = Function("f", [states, inputs], [f])
        return f_func, states.size1(), inputs.size1()





def recompute_trajectory(trajectory, gate_pos, gate_quat, obstacles, N_list, current_tick, start_state, dt=1/50):
    """
    Recompute the trajectory based on the current gate positions, orientations, and obstacles.
    
    Args:
        gate_pos: Positions of the gates.
        gate_quat: Orientations of the gates in quaternion format.
        obstacles: Positions of the obstacles.
        N_list: List of section ticks.
        current_tick: Current tick of the simulation.
        current_vel: Current velocity of the drone.
        dt: Time step for the trajectory computation.

    Returns:
        traj_section: The computed trajectory section.
        ab_tick: The starting tick for the trajectory section.
    """

    # drone model
    f_func, nx, nu = export_quadrotor_ode_model(acados=False)

    # Define the horizon of the remaining trajectory
    total_N = sum(N_list) - current_tick
    X = [MX.sym(f"x_{i}", nx) for i in range(total_N+1)]
    U = [MX.sym(f"u_{i}", nu) for i in range(total_N)]    

    # Optimierungsvariablen und initial guess + Untere und obere Schranken für jede Optimierungsvariable
    w, w0, lbw, ubw = [], [], [], []
    # Gleichungs-NB
    g, lbg, ubg = [], [], []
    # Cost function
    cost = 0.0

    # Bounds for inputs
    input_lower_bound = [-5, -5, -5, -5]
    input_upper_bound = [ 5,  5,  5,  5]

    # Bounds for states
    # # position bounds -> 0 - 2
    position_lower_bound = [-1.5, -2.0, 0.0]
    position_upper_bound = [ 1.5,  2.0, 2.0]
    # # velocity bounds -> 3 - 5
    velocity_lower_bound = [-2.0, -2.0, -2.0]
    velocity_upper_bound = [ 2.0,  2.0,  2.0]
    # # roll, pitch, yaw bounds -> 6 - 8
    roll_lower_bound = [-np.pi/4, -np.pi/4, -np.pi]
    roll_upper_bound = [ np.pi/4,  np.pi/4,  np.pi]
    # # collective force bounds -> 9
    # collective_force_lower_bound = [0]  # alte Werte
    # collective_force_upper_bound = [1.0]
    collective_force_lower_bound = [0.1]  # Werte aus gegebenem MPC
    collective_force_upper_bound = [0.55]
    # # command bounds -> 10 - 13
    # command_lower_bound = [-1.0, -np.pi/2, -np.pi/2, -2*np.pi] # alte Werte
    # command_upper_bound = [ 1.0,  np.pi/2,  np.pi/2,  2*np.pi]
    command_lower_bound = [ 0.10, -np.pi/2, -np.pi/2, -2*np.pi] # alte Werte
    command_upper_bound = [ 0.55,  np.pi/2,  np.pi/2,  2*np.pi]

    idx = 0
    gate_idx = 1
    for seg_idx, N_seg in enumerate(N_list):

        for j in range(N_seg):
            xi = X[idx]
            ui = U[idx]
            xi_next = X[idx + 1]

            w += [xi, ui]
            # nehme die alte trajectory als Ausgangspunkt der Optimierung
            # w0 += list(np.concatenate([np.array(trajectory[current_tick+idx]), np.zeros(nx - 3)])) + [0]*nu # trajectory hat nur die Positonswerte
            w0 += list(trajectory[current_tick + idx]) + [0] * nu # trajectory mit vollen states
            lbw += position_lower_bound + velocity_lower_bound + roll_lower_bound + collective_force_lower_bound + command_lower_bound + input_lower_bound
            ubw += position_upper_bound + velocity_upper_bound + roll_upper_bound + collective_force_upper_bound + command_upper_bound + input_upper_bound

            # Dynamik
            g += [xi_next - (xi + dt * f_func(xi, ui))]
            lbg += [0]*nx
            ubg += [0]*nx




            # --------------------------------Cost------------------------------------
            cost += 1e-1 * sumsqr(ui) # Control effort minimization # 1e-0
            cost += 1e-1 * sumsqr(xi_next[:3] - xi[:3]) # Position change minimization # 1e-0

            # Obstacle penalty
            if obstacles: 
                for idx_obs, obs in enumerate(obstacles):
                    dist = sumsqr(xi[0:2] - obs[0:2])
                    cost += 1.0 * exp(-100 * dist)

            # Gate-Boundries penatly: current and previous gate
            for gate_offset in [0, -1]:
                gate_ref_idx = gate_idx + gate_offset
                if 0 < gate_ref_idx < len(gate_pos): # ignoriere Startposition - 1. Gate ist Startposition
                    gate_position = np.array(gate_pos[gate_ref_idx])
                    gate_quaterunion = np.asarray(gate_quat[gate_ref_idx], dtype=float)
                    gate_rotation = R.from_quat(gate_quaterunion).as_matrix()

                    target_dist = 0.225 + 0.05/2    # Mitte von Rahmen: Innenkante=0.225m und Rahmendicke=0.05m
                    falloff = 0.25                  # Wirkradius nach innen und außen
                    faktor = 200.0                  # Faktor für die Exponentialstrafe
                    y_range = 0.05                  # Toleranzbereich in y-Richtung

                    # Transform drone position into gate frame
                    rel_pos = gate_rotation.T @ (xi[0:3] - gate_position)

                    # minimal Distance to centre of gate frame ( in x or z direction )
                    penalty_dist = fmin(fabs(fabs(rel_pos[0]) - target_dist), fabs(fabs(rel_pos[2]) - target_dist))

                    # if abs(rel_pos[2]) < y_range: # Wenn innerhalb des Toleranzbereichs in y-Richtung
                    if_condition = fabs(rel_pos[1]) < y_range
                    cost += if_condition * (1.0 * exp(-((penalty_dist / falloff)**4) * faktor))



            idx += 1

        # Zwischen-Gates: Position (fixieren) - als UGB + Velocity frei
        if  0 < gate_idx < len(gate_pos) - 1:
            # UGB für Position
            gate_position = np.array(gate_pos[gate_idx])
            gate_rotation = R.from_quat(gate_quat[gate_idx]).as_matrix()
            rel_pos = gate_rotation.T @ (xi[0:3] - gate_position)
            g += [rel_pos[0], rel_pos[2], rel_pos[1]] # x und z als UGB; y festlegen
            lbg += [-0.15, -0.15, 0.0]
            ubg += [ 0.15,  0.15, 0.0]

            '''
            # UGB für Velocity -> minimale Velocity in Gate Richtung wird forciert - NÖTIG ?????
            v_drone = X[idx][3:6]
            gate_dir = DM(R.from_quat(gate_quat[gate_idx]).apply([0, 1, 0]))
            g += [dot(v_drone, gate_dir)]
            lbg += [0.5]
            ubg += [inf]
            '''
            
            # ----------------------------------Cost--------------------------------------
            # cost korresponierend zu den UBGs des Gate-Durchflugs - damit robust: "belohen" für Durchflug nahe Mitte
            cost += 10.0 * sumsqr(rel_pos[0:3])


        gate_idx += 1



    # Start-Gate: Position + Velocity fix
    # g += [X[0][0:6] - DM(np.concatenate([trajectory[current_tick], current_vel]))] # Startposition und -geschwindigkeit nur fest
    g += [X[0] - DM(start_state)] # gesamter Startzustand festgelegt - durch Observations am besten
    lbg += [0]*6
    ubg += [0]*6


    # Ziel-Gate: Position als UGB + Velocity frei
    gate_position = np.array(gate_pos[-1])
    gate_rotation = R.from_quat(gate_quat[-1]).as_matrix()
    rel_pos = gate_rotation.T @ (X[total_N][0:3] - gate_position)
    g += [rel_pos[0], rel_pos[2], rel_pos[1]] # x und z als UGB; y festlegen
    lbg += [-0.15, -0.15, 0.0]
    ubg += [ 0.15,  0.15, 0.0]

    '''
    # UGB für Velocity -> minimale Velocity in Gate Richtung wird forciert - NÖTIG ?????
    v_drone = X[total_N][3:6]
    gate_dir = DM(R.from_quat(gate_quat[-1]).apply([0, 1, 0]))
    g += [dot(v_drone, gate_dir)]
    lbg += [0.5]
    ubg += [inf]
    '''

    # Letztes X anhängen
    w += [X[total_N]]
    w0 += list(np.concatenate([np.array(gate_pos[-1]).flatten(), np.zeros(nx - 3)]))
    lbw += position_lower_bound + velocity_lower_bound + roll_lower_bound + collective_force_lower_bound + command_lower_bound
    ubw += position_upper_bound + velocity_upper_bound + roll_upper_bound + collective_force_upper_bound + command_upper_bound



    # --------------------------------Cost------------------------------------
    # cost for time / points used:
    cost += (total_N * dt) * 2
    cost += 10.0 * sumsqr(rel_pos[0:3]) # cost korresponierend zu den UBGs des Gate-Durchflugs - damit robust: "belohen" für Durchflug nahe Mitte



    # 4. Solve
    prob = {"x": vertcat(*w), "f": cost, "g": vertcat(*g)}
    solver = nlpsol("solver", "ipopt", prob, {
        "ipopt.print_level": 0,
        "print_time": 0,
        "verbose": False,
        "ipopt.max_cpu_time": 5.0,  # Zeitlimit in Sekunden bis OPtimierung abgebrochen wird
    })
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    if solver.stats()["success"]:
        print("✅ Optimierung erfolgreich.")
    else:
        print("❌ Optimierung fehlgeschlagen.")


    # 5. Extract solution
    w_opt = np.array(sol["x"]).flatten()
    X_opt = []
    for i in range(total_N + 1):
        base = i * (nx + nu) if i < total_N else total_N * (nx + nu)
        x_i = w_opt[base : base + nx]
        X_opt.append(x_i)

    return np.array(X_opt), current_tick
    return #traj_section, ab_tick





def create_ocp_solver(
    Tf: float, N: int, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """
    Creates an acados Optimal Control Problem and Solver.
    
    Args:
        Tf: Total prediction horizon in seconds - optimal: N * dt
        N: Number of discretization steps over the horizon
        verbose: If True, enables verbose output during solver creation

    Returns:
        AcadosOcpSolver: Configured MPC solver instance for online use
        AcadosOcp: Underlying OCP configuration object
    """

    ocp = AcadosOcp()

    # set model
    model = export_quadrotor_ode_model(acados=True)
    ocp.model = model

    # Get Dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # Cost function (LINEAR_LS) in acados:
    # ℓ(x, u) = 0.5 * (Vx @ x + Vu @ u - yref).T @ W @ (Vx @ x + Vu @ u - yref)
    # Interpretation:
    # - Vx:  Maps the state vector x to the cost vector.
    # - Vu:  Maps the input vector u to the cost vector.
    # - yref: Desired target/reference for the combined (Vx*x + Vu*u) vector.
    # - W:   Weighting matrix that penalizes the deviation from yref.


    # Cost Type 
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    Q = np.diag(    # cost weights for the States
        [
            10.0,
            10.0,
            10.0,  # Position
            0.01,
            0.01,
            0.01,  # Velocity
            0.1,
            0.1,
            0.1,  # rpy
            0.01,  # f_collective
            0.01,  # f_collective_cmd
            0.01,  
            0.01,
            0.01,  # r_cmd, p_cmd, y_cmd; Sollwerte für Roll, Pitch, Yaw
        ]
    )

    R = np.diag([0.01, 0.01, 0.01, 0.01])   # cost weights for the Inputs

    Q_e = Q.copy() # cost weights for the Terminal-Kosten

    # Set Cost Weights in the solver
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


    # Set references to any -> will be overwritten by the controller
    ocp.cost.yref = np.zeros((ny, ))
    ocp.cost.yref_e = np.zeros((ny_e, ))
    # ocp.cost.yref = np.array( [1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] )
    # ocp.cost.yref_e = np.array( [1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0] )






    # Constraints for States and Inputs
    # # Set State Constraints                                                                                                   -----> ABSPRECHEN MIT DER TRAJEKTORIEN PLANUNG
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13]) # welche IDs der States werden im folgenden constrained?              -----> ABSPRECHEN MIT DER TRAJEKTORIEN PLANUNG
    ocp.constraints.lbx = np.array([0.1, 0.1, -1.57, -1.57, -1.57]) # -> f_collective, f_collective_cmd, r_cmd, p_cmd, y_cmd    -----> ABSPRECHEN MIT DER TRAJEKTORIEN PLANUNG
    ocp.constraints.ubx = np.array([0.55, 0.55, 1.57, 1.57, 1.57])
    

    # # Set Input Constraints                                                                                                   -----> ABSPRECHEN MIT DER TRAJEKTORIEN PLANUNG
    # ocp.constraints.idxbu = np.array([0, 1, 2, 3]) # welche IDs der Inputs werden im folgenden constrained?                   -----> ABSPRECHEN MIT DER TRAJEKTORIEN PLANUNG
    # ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0. -10.0]) # -> df_cmd, dr_cmd, dp_cmd, dy_cmd                          -----> ABSPRECHEN MIT DER TRAJEKTORIEN PLANUNG
    # ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0])
    

    # # Set x0 to any -> will be overwritten by the controller
    ocp.constraints.x0 = np.zeros((nx))





    # Solver Options -> Optimierungsalgorithmen und Numerik
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI # Für schnelle Online-Anwendung evtl. SQP_RTI ausprobieren statt SQP
    ocp.solver_options.tol = 1e-5

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf



    if os.path.exists("my_mpc_controller.json"):
        os.remove("my_mpc_controller.json")
        print("🗑️ Alte my_mpc_controller.json gelöscht und neu erzeugt.")
    assert not os.path.exists("my_mpc_controller.json")



    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="my_mpc_controller.json", verbose=verbose) # speichere die Konfiguration in einer JSON-Datei

    return acados_ocp_solver, ocp


