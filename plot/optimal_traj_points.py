import numpy as np
from casadi import MX, Function, DM, vertcat, cos, sin, exp, inf, sumsqr, nlpsol
# from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def export_quadrotor_ode_model():
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
    f_collective = MX.sym("f_collective")

    f_collective_cmd = MX.sym("f_collective_cmd")
    r_cmd = MX.sym("r_cmd")
    p_cmd = MX.sym("p_cmd")
    y_cmd = MX.sym("y_cmd")

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
    f_func = Function("f", [states, inputs], [f])

    return f_func, states.size1(), inputs.size1()




def optimize_waypoint_positions(gate_1, gate_2, f_func, nx, nu, N=10, tmax=10, obstacles=None):

    # Define symbolic variables for states and inputs for each time step
    X = [MX.sym(f"x_{i}", nx) for i in range(N+1)]
    U = [MX.sym(f"u_{i}", nu) for i in range(N)]
    DT = [MX.sym(f"dt_{i}") for i in range(N)]

    # initialen Wert für jede Variable
    w = []
    w0 = []

    # Untere und obere Schranken für jede Optimierungsvariable
    lbw = []
    ubw = []

    # Gleichungs- und Ungleichungsrestriktionen
    g = []
    lbg = []
    ubg = []

    # Cost function
    cost = 0

    # Bounds for time step
    dt_lower_bound = 0.01
    dt_upper_bound = tmax/N

    # Bounds for inputs
    input_lower_bound = [-5, -5, -5, -5]
    input_upper_bound = [ 5,  5,  5,  5]

    # Bounds for states
    # # position bounds -> 0 - 2
    position_lower_bound = [-1.5, -2.0, 0.0]
    position_upper_bound = [ 1.5,  2.0, 2.0]
    # # velocity bounds -> 3 - 5
    velocity_lower_bound = [-inf, -inf, -inf]
    velocity_upper_bound = [ inf,  inf,  inf]
    # # roll, pitch, yaw bounds -> 6 - 8
    roll_lower_bound = [-inf, -inf, -inf]
    roll_upper_bound = [ inf,  inf,  inf]
    # # collective force bounds -> 9
    collective_force_lower_bound = [-inf]
    collective_force_upper_bound = [inf]
    # # command bounds -> 10 - 13
    command_lower_bound = [-inf, -inf, -inf, -inf]
    command_upper_bound = [ inf,  inf,  inf,  inf]

    # Set bounds for states
    for i in range(N):

        # optimisation variables and their initial values for solver
        w += [X[i], U[i], DT[i]]
        w0 += list(gate_1[:3].full().flatten()) + [0]*nu + [tmax/(2*N)]

        # Set bounds for states
        lbx = position_lower_bound + velocity_lower_bound + roll_lower_bound + collective_force_lower_bound + command_lower_bound
        ubx = position_upper_bound + velocity_upper_bound + roll_upper_bound + collective_force_upper_bound + command_upper_bound

        lbw += lbx + input_lower_bound + dt_lower_bound
        ubw += ubx + input_upper_bound + dt_upper_bound

        # Dynamics constraint via Euler integration (genauer: RK4) -> Gleichungs-NB
        x_next = X[i] + DT[i] * f_func(X[i], U[i])
        g += [X[i+1] - x_next]
        lbg += [0]*nx
        ubg += [0]*nx

        # Cost: minimize squared control effort and position change
        cost += sumsqr(U[i]) # Control effort minimization
        cost += 1e-2 * sumsqr(X[i+1][:3] - X[i][:3]) # Position change minimization
        cost += DT[i]  # Time minimization

        # Obstacle penalty
        if obstacles:
            for obs in obstacles:
                dist = sumsqr(X[i][0:2] - obs[0:2])
                cost += exp(-5 * dist)


    # First and terminal state fixed
    # # desired velocity
    v_des = 1.0

    # # extract gate positions and orientations
    gatepos_1 = gate_1[:3]
    gatequat_1 = gate_1[3:7]
    gatepos_2 = gate_2[:3]
    gatequat_2 = gate_2[3:7]
    # # initial state constrained by pos and vel.
    dir = (R.from_quat(gatequat_1)).apply([0, 1, 0])
    velocity_1 = dir / np.linalg.norm(dir) * v_des
    x0_target = np.concatenate([gatepos_1, velocity_1])
    g += [X[0][0:6] - DM(x0_target)]
    lbg += [0]*6
    ubg += [0]*6
    # # terminal state
    # # # Add X[N] as variable
    w += [X[N]]
    w0 += list(gate_2[:3].full().flatten())
    lbx = position_lower_bound + velocity_lower_bound + roll_lower_bound + collective_force_lower_bound + command_lower_bound
    ubx = position_upper_bound + velocity_upper_bound + roll_upper_bound + collective_force_upper_bound + command_upper_bound
    lbw += lbx
    ubw += ubx
    # # # Terminal constraint (position + velocity)
    dir = (R.from_quat(gatequat_2)).apply([0, 1, 0])
    velocity_2 = dir / np.linalg.norm(dir) * v_des
    xN_target = np.concatenate([gatepos_2, velocity_2])
    g += [X[N][0:6] - DM(xN_target)]
    lbg += [0]*6
    ubg += [0]*6

    # Problem definition
    prob = {"x": vertcat(*w), "f": cost, "g": vertcat(*g)}
    solver = nlpsol("solver", "ipopt", prob)
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)


    # Extract solution -> X_opt and DT_opt
    X_opt = []
    DT_opt = []
    for i in range(N):
        base = i * (nx + nu + 1)
        x_i = sol[base : base + nx]
        dt_i = sol[base + nx + nu]  # dt liegt am Ende des Blocks
        X_opt.append(x_i)
        DT_opt.append(dt_i)
    # Letzten Zustand anhängen
    X_opt.append(sol[-nx:])

    return np.array(X_opt), np.array(DT_opt)

