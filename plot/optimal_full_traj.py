import numpy as np
from casadi import MX, Function, DM, vertcat, cos, sin, exp, inf, sumsqr, nlpsol, dot, norm_2, fmax, fabs, fmin
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


def distribute_timesteps_by_distance(gates, N_total):
    """
    gates: list of gate poses, each [x, y, z, quat]
    N_total: total number of optimization steps to distribute across segments

    returns: list of ints N_list such that sum(N_list) = N_total
    """
    import numpy as np

    # Extrahiere nur die Positionen
    positions = [np.array(g[:3]).flatten() for g in gates]

    # Berechne Längen aller Segmente
    distances = [np.linalg.norm(positions[i+1] - positions[i]) for i in range(len(positions) - 1)]
    total_dist = sum(distances)

    # Verteile N proportional zur Strecke (mindestens 1 pro Segment)
    N_raw = [max(1, d / total_dist * N_total) for d in distances]
    N_int = [int(round(n)) for n in N_raw]

    # Korrektur, falls Rundungsfehler
    diff = sum(N_int) - N_total
    while diff != 0:
        i = np.argmax(N_raw) if diff > 0 else np.argmin(N_raw)
        N_int[i] -= np.sign(diff)
        diff = sum(N_int) - N_total

    return N_int

def apply_segment_tendency(N_list, shorten=[0, 2], lengthen=[1, 3], max_shift=0.1):
    """
    Verändert N_list durch gezielte Verlängerung/Verkürzung einzelner Segmente.
    
    shorten: Indizes, die leicht verkürzt werden
    lengthen: Indizes, die leicht verlängert werden
    max_shift: max. prozentuale Änderung je Segment (z.B. 0.1 = 10%)
    """

    N_array = np.array(N_list, dtype=float)
    N_total = int(np.sum(N_array))

    # Erzeuge Änderungsfaktoren
    deltas = np.zeros_like(N_array)
    
    for i in shorten:
        deltas[i] -= np.random.uniform(0, max_shift)
    for i in lengthen:
        deltas[i] += np.random.uniform(0, max_shift)

    # Wende relative Änderung an
    N_var = N_array * (1 + deltas)
    N_var = np.maximum(N_var, 1.0)

    # print("Narray:", N_array)
    # print("N_var:", N_var)

    # Normiere zurück auf N_total
    N_int = [int(round(x)) for x in N_var]
    diff = sum(N_int) - N_total
    while diff != 0:
        i = np.argmax(N_int) if diff > 0 else np.argmin(N_int)
        N_int[i] -= int(np.sign(diff))
        diff = sum(N_int) - N_total

    return N_int




def optimize_waypoint_positions(gates, gates_quat, N_list, obstacles, v_start, v_end, dt=1/50):

    # drone model
    f_func, nx, nu = export_quadrotor_ode_model()

    # Define symbolic variables for states and inputs for each time step
    total_N = sum(N_list)
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
        gate_from = gates[seg_idx]
        # gate_to = gates[seg_idx + 1]

        for j in range(N_seg):
            xi = X[idx]
            ui = U[idx]
            xi_next = X[idx + 1]

            w += [xi, ui]
            w0 += list(np.concatenate([np.array(gate_from).flatten(), np.zeros(nx - 3)])) + [0]*nu
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
                    # cost += exp(-100 * dist)


                    if idx_obs == 3:
                        # Härtere Bestrafung für Obstacle 4
                        cost += 1.0 * exp(-100 * dist)
                    else:
                        # Standardbestrafung für alle anderen
                        cost += 1.0 * exp(-100 * dist)

            # Gate-Boundries penatly: current and previous gate
            for gate_offset in [0, -1]:
                gate_ref_idx = gate_idx + gate_offset
                if 0 < gate_ref_idx < len(gates): # ignoriere Startposition - 1. Gate ist Startposition
                    gate_pos = np.array(gates[gate_ref_idx])
                    gate_quat = np.asarray(gates_quat[gate_ref_idx], dtype=float)
                    # gate_rot = R.from_quat(gate_quat)
                    gate_rot = R.from_quat(gate_quat).as_matrix()

                    target_dist = 0.225 + 0.05/2    # Mitte von Rahmen: Innenkante=0.225m und Rahmendicke=0.05m
                    falloff = 0.25                  # Wirkradius nach innen und außen
                    faktor = 200.0                  # Faktor für die Exponentialstrafe
                    y_range = 0.05                  # Toleranzbereich in y-Richtung

                    # Transform drone position into gate frame
                    # rel_pos = gate_rot.apply(np.array(xi[:3]) - gate_pos, inverse=True)
                    rel_pos = gate_rot.T @ (xi[0:3] - gate_pos)

                    # minimal Distance to centre of gate frame ( in x or z direction )
                    # penalty_dist = max(abs(abs(rel_pos[0]) - target_dist), abs(abs(rel_pos[2]) - target_dist))
                    penalty_dist = fmin(fabs(fabs(rel_pos[0]) - target_dist), fabs(fabs(rel_pos[2]) - target_dist))

                    # if abs(rel_pos[2]) < y_range: # Wenn innerhalb des Toleranzbereichs in y-Richtung
                    #     cost += exp(-((penalty_dist / falloff) ** 4) * 200)
                    if_condition = fabs(rel_pos[1]) < y_range
                    cost += if_condition * (1.0 * exp(-((penalty_dist / falloff)**4) * faktor))



            idx += 1
        
        # Zwischen-Gates: Position (fixieren) - als UGB, Velocity frei
        if  0 < gate_idx < len(gates) - 1:
            '''
            # gate Durchflug-Punkt als Fixpunkt setzen
            pos_i = gates[gate_idx]
            g += [X[idx][:3] - DM(np.array(pos_i).flatten())]
            lbg += [0]*3
            ubg += [0]*3
            '''
            # Gate Durchflug-Punkt als UGB setzen - nicht als Fixpunkt
            gate_pos = np.array(gates[gate_idx])
            gate_quat = gates_quat[gate_idx]
            R_gate = R.from_quat(gate_quat).as_matrix()
            rel_pos = R_gate.T @ (xi[0:3] - gate_pos)
            g += [rel_pos[0], rel_pos[2], rel_pos[1]] # x und z als UGB; y festlegen
            lbg += [-0.15, -0.15, 0.0]
            ubg += [ 0.15,  0.15, 0.0]

            ''' # Velocity ignorieren ? 
            v_drone = X[idx][3:6]
            gate_dir = DM(R.from_quat(gates_quat[gate_idx]).apply([0, 1, 0]))
            g += [dot(v_drone, gate_dir)]
            lbg += [0.5]
            ubg += [inf]
            '''


            # ----------------------------------Cost--------------------------------------
            '''
            # Flugrichtung (/Geschwindigkeit) am Gate-Durchflug
            gate_dir = DM(R.from_quat(gates_quat[gate_idx]).apply([0, 1, 0]))
            # print(f"Gate Direction ({gate_idx}): {gate_dir}")
            v_drone = X[idx][3:6]
            v_norm = fmax(norm_2(v_drone), 1e-3)
            cost_ = v_drone / (1e-6 + norm_2(v_drone)) - gate_dir # absolute Vektor-Diff.
            cost += 5.0 * sumsqr(cost_)
            safe_cos = dot(v_drone, gate_dir) / v_norm # Diff mit cos()
            # print(f"Gate Direction cos-Cost: {safe_cos}")
            cost += 5.0 * (1.0 - safe_cos)**2
            cost -= 3 * dot(v_drone, gate_dir)
            v_norm = v_drone / norm_2((v_drone + 1e-3))
            cost += 10 * norm_2(v_norm - gate_dir)
            '''

            # cost korresponierend zu den UBGs des Gate-Durchflugs
            cost += 10.0 * sumsqr(rel_pos[0:3])


        gate_idx += 1



    # Start-Gate: Position + Velocity fix
    g += [X[0][0:6] - DM(np.concatenate([gates[0], v_start]))]
    lbg += [0]*6
    ubg += [0]*6

    # Ziel-Gate: Position (fix) als UGB + Velocity-Richtung nicht fix
    '''
    g += [X[total_N][0:3] - DM(gates[-1][:3])]
    lbg += [0]*3
    ubg += [0]*3
    '''
    gate_pos = np.array(gates[-1])
    gate_quat = gates_quat[-1]
    R_gate = R.from_quat(gate_quat).as_matrix()
    rel_pos = R_gate.T @ (X[total_N][0:3] - gate_pos)
    g += [rel_pos[0], rel_pos[2], rel_pos[1]] # x und z als UGB; y festlegen
    lbg += [-0.15, -0.15, 0.0]
    ubg += [ 0.15,  0.15, 0.0]

    '''
    v_drone = X[total_N][3:6]
    gate_dir = DM(R.from_quat(gates_quat[-1]).apply([0, 1, 0]))
    g += [dot(v_drone, gate_dir)]
    lbg += [0.5]
    ubg += [inf]
    '''

    # Letztes X anhängen
    w += [X[total_N]]
    w0 += list(np.concatenate([np.array(gates[-1]).flatten(), np.zeros(nx - 3)]))
    lbw += position_lower_bound + velocity_lower_bound + roll_lower_bound + collective_force_lower_bound + command_lower_bound
    ubw += position_upper_bound + velocity_upper_bound + roll_upper_bound + collective_force_upper_bound + command_upper_bound



    # --------------------------------Cost------------------------------------
    # cost for time / points used:
    cost += (total_N * dt) * 2
    cost += 10.0 * sumsqr(rel_pos[0:3])



    # 4. Solve
    prob = {"x": vertcat(*w), "f": cost, "g": vertcat(*g)}
    solver = nlpsol("solver", "ipopt", prob, {
        "ipopt.print_level": 0,
        "print_time": 0,
        "verbose": False,
        "ipopt.max_cpu_time": 5.0,  # Zeitlimit in Sekunden
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

    return np.array(X_opt), float(sol["f"])





def optimize_full_trajectory(gates, gates_quat, obstacles, v_start, v_end, t_min, t_max, step=1, random_iteraitions=5, dt=1/50):
    # step wird hier in sekunden gezählt

    N_min = int(t_min * 1/dt)
    N_max = int(t_max * 1/dt)
    step_it = int(step * 1/dt)

    best_cost = np.inf
    best_X_opt = None
    best_N = None


    for N in range(N_min, N_max + 1, step_it):
        N_list = distribute_timesteps_by_distance(gates, N)
        print("ÄUßERE KLAMMER :", N_list)
        for _ in range(random_iteraitions):
            
            N_list_innen = apply_segment_tendency(N_list, shorten=[0,2], lengthen=[1,3], max_shift=0.1)
            print("ERGBENISS INNEN:", N_list_innen)
            try:
                pos, cost = optimize_waypoint_positions(gates, gates_quat, N_list_innen, obstacles, v_start, v_end, dt=dt)
                if cost < best_cost:
                    best_cost = cost
                    best_X_opt = pos
                    best_N = N_list_innen

            except Exception as e:
                print(f"❌ Optimization failed for N={N}: {e}")
                continue

    return best_X_opt, best_N
