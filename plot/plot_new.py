import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from casadi import DM
import json
import os

# from optimal_traj_2gates import optimize_waypoint_positions
# from optimal_trajectory import optimize_waypoint_pos_and_num
from optimizer_help_func import optimize_full_trajectory_random, optimize_from_given_N_list_random
from optimizer import optimize_original, optimize_velocity_bounded


def plot_waypoints_and_environment(waypoints, obstacle_positions, gates_positions, gates_quat):

    def quaternion_to_rotation_matrix(q):
        x, y, z, w = q
        R = np.array([
            [1-2*(y**2+z**2),  2*(x*y - z*w),    2*(x*z + y*w)],
            [2*(x*y + z*w),    1-2*(x**2+z**2),  2*(y*z - x*w)],
            [2*(x*z - y*w),    2*(y*z + x*w),    1-2*(x**2+y**2)]
        ])
        return R
    
    def rotate_and_translate(square, R, gate):
        return (R @ square.T).T + gate



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Waypoints plotten
    waypoints = np.array(waypoints)
    ax.plot(waypoints[:,0], waypoints[:,1], waypoints[:,2], 'bo-', label='Waypoints', markersize=2)

    # Gates als rotierte Quadrate
    gate_size = 0.45 / 2
    gate_size_outer = gate_size + 0.05
    gate_color = (1, 0, 0, 0.5)
    gates_positions = np.array(gates_positions)

    for i, gate in enumerate(gates_positions):
        quat = gates_quat[i]
        R = quaternion_to_rotation_matrix(quat)

        # Inner gate (actual fly-through area)
        inner_square = np.array([
            [-gate_size, 0, -gate_size],
            [ gate_size, 0, -gate_size],
            [ gate_size, 0,  gate_size],
            [-gate_size, 0,  gate_size],
        ])
        inner_square = rotate_and_translate(inner_square, R, gate)
        poly = Poly3DCollection([inner_square], color=gate_color, label='Gate' if i == 0 else "")
        ax.add_collection3d(poly)

        outer_square = np.array([
            [-gate_size_outer, 0, -gate_size_outer],
            [ gate_size_outer, 0, -gate_size_outer],
            [ gate_size_outer, 0,  gate_size_outer],
            [-gate_size_outer, 0,  gate_size_outer],
        ])
        outer_transformed = rotate_and_translate(outer_square, R, gate)
        poly_outer = Poly3DCollection([outer_transformed], color=gate_color)
        ax.add_collection3d(poly_outer)

    # Gate-Zentren markieren
    ax.scatter(gates_positions[:,0], gates_positions[:,1], gates_positions[:,2], c='r', s=50, label=None)

    # Stäbe plotten (von z=0 bis z=1)
    for idx, point in enumerate(obstacle_positions):
        ax.plot([point[0], point[0]], [point[1], point[1]], [0, 1],
                linewidth=4, label='Staves' if idx == 0 else "")

    # Achsenbeschriftung und Limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 2)

    ax.legend()
    ax.set_title('3D Waypoints mit Stäben und rotierbaren Gates')
    plt.show()




waypoints = np.array(
    [
        [1.0, 1.5, 0.05],
        [0.8, 1.0, 0.2],
        [0.55, -0.3, 0.5], # gate 0
        [0.1, -1.5, 0.65],
        [1.1, -0.85, 1.15], # gate 1
        [0.2, 0.5, 0.65],
        [0.0, 1.2, 0.525], # gate 2
        [0.0, 1.2, 1.1],
        [-0.5, 0.0, 1.1], # gate 3
        [-0.5, -0.5, 1.1],
        [-0.5, -1.0, 1.1],
    ]
)

obstacles_positions = [
    [1.0, 0.0, 1.4],
    [0.5, -1.0, 1.4],
    [0., 1.5, 1.4],
    [-0.5, 0.5, 1.4],
]

gates_positions = [
    [0.45, -0.5, 0.56],
    [1.0, -1.05, 1.11],
    [0.0, 1.0, 0.56],
    [-0.5, 0.0, 1.11],
]

gates_quat = [   # sind als rpy gegeben und nicht als Quaternion ??????? -> z.B. siehe level0.toml
    [0.0, 0.0, 0.92388, 0.38268],
    [0.0, 0.0, -0.38268, 0.92388],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
]



plot_waypoints_and_environment(waypoints, obstacles_positions, gates_positions, gates_quat)


'''
gate_1 = DM(gates_positions[0] + gates_quat[0])  # [x, y, z, qx, qy, qz, qw]
gate_2 = DM(gates_positions[1] + gates_quat[1])  # [x, y, z, qx, qy, qz, qw]

X_opt, N_opt = optimize_waypoint_pos_and_num(gate_1, gate_2, obstacles_positions, 1.5, 2.5, step=10)

print("N_opt:", N_opt)
print("optimale Zeit:", N_opt * 1/50)

plot_waypoints_and_environment(X_opt, obstacles_positions, gates_positions, gates_quat)
'''




start_vel = [0, 0, 0.1]
velocity_gate4 = [0, -2, 0] # wird nicht verwendet - velocity in gate richtung wird betrachtet in UGB

start_pos = [1.0, 1.5, 0.0]

extended_gates = [start_pos] + gates_positions
extended_gates_quat = [[0, 0, 0, 0]] + gates_quat # der erste Eintrag wird nicht verwendet, da kein richtiges Gate

# gates = [start_pos, gates_positions[0], gates_positions[1]]


# X_opt, N_opt = optimize_full_trajectory_random(extended_gates, extended_gates_quat, obstacles_positions, start_vel, velocity_gate4, t_min=10, t_max=18, step=1, random_iteraitions=10)

# N_list = [72, 99, 73, 106]
# X_opt, N_opt = optimize_waypoint_positions(extended_gates, extended_gates_quat, N_list, obstacles_positions, start_vel, velocity_gate4, dt=1/50)

N_opt = [60, 80, 80, 80] # 6 sec - normal
N_opt = [63, 87, 71, 79] # 6 sec - optimized
N_opt = [100, 133, 133, 133] # 10 sec - normal
N_opt = [105, 145, 118, 132] # 10 sec - optimized 6 scaled
N_opt = [100, 150, 140, 110] # 10 sec - optimized
# N_opt = [35, 55, 45, 40] # 4 sec - optimized
# X_opt, cost = optimize_velocity_bounded(extended_gates, extended_gates_quat, N_opt, obstacles_positions, start_vel, velocity_gate4, dt=1/50)
X_opt, N_opt, cost = optimize_from_given_N_list_random(extended_gates, extended_gates_quat, obstacles_positions, start_vel, velocity_gate4, N_opt, iterations=10, max_shift=0.1, shorten=[], lengthen=[])












print("N_opt:", N_opt)
print("optimale Zeiten:", np.array(N_opt) * (1/50) )
print("optimale Gesamtzeit:", sum(N_opt) * (1/50))
print("optimale Kosten: ", cost)

plot_waypoints_and_environment(X_opt, obstacles_positions, gates_positions, gates_quat)

# Daten Speichern
output_data = {
    "N_opt": N_opt,
    "X_opt": np.array(X_opt).tolist()
}
output_dir = os.path.join("plot", "traj_data")
output_path = os.path.join(output_dir, "opt__test.json")
with open(output_path, "w") as f:
    json.dump(output_data, f, indent=2)
print(f"✅ Optimierungsdaten gespeichert in: {output_path}")