"""This module optimizes the trajectory and plots it."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

#from casadi import DM
#from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from optimizer_help_func import optimize_from_given_N_list_random

#from optimizer import optimize_original, optimize_velocity_bounded
#from optimal_traj_2gates import optimize_waypoint_positions
#from optimal_trajectory import optimize_waypoint_pos_and_num
#from optimizer_help_func import optimize_full_trajectory_random


def plot_waypoints_and_environment(waypoints: np.ndarray, obstacle_positions: np.ndarray, gates_positions: np.ndarray, gates_quat: np.ndarray) -> None:
    """Plots the waypoints, obstacles, and gates in a 3D environment."""

    def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """Convert a quaternion into a rotation matrix."""
        x, y, z, w = q
        R = np.array([
            [1-2*(y**2+z**2),  2*(x*y - z*w),    2*(x*z + y*w)],
            [2*(x*y + z*w),    1-2*(x**2+z**2),  2*(y*z - x*w)],
            [2*(x*z - y*w),    2*(y*z + x*w),    1-2*(x**2+y**2)]
        ])
        return R
    
    def rotate_and_translate(square: np.ndarray, R: np.ndarray, gate: np.ndarray) -> np.ndarray:
        """Rotate and translate a square defined in the XY-plane."""
        return (R @ square.T).T + gate



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Waypoints plotten
    waypoints = np.array(waypoints)
    ax.plot(waypoints[:,0], waypoints[:,1], waypoints[:,2], 'bo-', label='Waypoints', markersize=2)

    # Gates als rotierte Quadrate
    gate_size = 0.45 / 2
    gate_size_outer = gate_size + 0.1
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




def plot_waypoints_and_environment_extra(
    waypoints: np.ndarray,
    extra_waypoints: np.ndarray,
    obstacle_positions: np.ndarray,
    gates_positions: np.ndarray,
    gates_quat: np.ndarray,
) -> None:
    """Zeichnet Trajektorie, zusätzliche Wegpunkte, Obstacles und Gates in einer 3D-Szene.

    Parameter
    ---------
    waypoints : (N,3) array_like
        Haupt-Trajektorie (wird als Linie verbunden dargestellt).
    extra_waypoints : (M,3) array_like
        Zusatz-Wegpunkte, die hervorgehoben (Marker-Only) geplottet werden.
    obstacle_positions : (K,3) array_like
        Mittelpunkt-Koordinaten der Stäbe (werden als vertikale Linien gezeichnet).
    gates_positions : (G,3) array_like
        Zentren der Gates.
    gates_quat : (G,4) array_like
        Rotation der Gates als [x, y, z, w]-Quaternionen.
    """

    def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        return np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
            ]
        )

    def rotate_and_translate(square: np.ndarray, R: np.ndarray, gate: np.ndarray) -> np.ndarray:
        return (R @ square.T).T + gate

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # ───────────────────────────────────────────────────────────────── Trajektorie
    waypoints = np.asarray(waypoints)
    ax.plot(
        waypoints[:, 0],
        waypoints[:, 1],
        waypoints[:, 2],
        "bo-",
        label="Trajektorie",
        markersize=2,
    )

    # ───────────────────────────────────────────────────────────── Extra-Wegpunkte
    extra_waypoints = np.asarray(extra_waypoints)
    ax.scatter(
        extra_waypoints[:, 0],
        extra_waypoints[:, 1],
        extra_waypoints[:, 2],
        c="orange",
        s=40,
        marker="^",
        label="Extra-Waypoints",
    )

    # ───────────────────────────────────────────────────────────────────── Gates
    gate_size = 0.45 / 2
    gate_size_outer = gate_size + 0.1
    gate_color = (1, 0, 0, 0.5)
    gates_positions = np.asarray(gates_positions)

    for i, gate in enumerate(gates_positions):
        R = quaternion_to_rotation_matrix(gates_quat[i])

        inner_square = np.array(
            [
                [-gate_size, 0, -gate_size],
                [gate_size, 0, -gate_size],
                [gate_size, 0, gate_size],
                [-gate_size, 0, gate_size],
            ]
        )
        inner_square = rotate_and_translate(inner_square, R, gate)
        ax.add_collection3d(
            Poly3DCollection([inner_square], color=gate_color, label="Gate" if i == 0 else "")
        )

        outer_square = np.array(
            [
                [-gate_size_outer, 0, -gate_size_outer],
                [gate_size_outer, 0, -gate_size_outer],
                [gate_size_outer, 0, gate_size_outer],
                [-gate_size_outer, 0, gate_size_outer],
            ]
        )
        outer_square = rotate_and_translate(outer_square, R, gate)
        ax.add_collection3d(Poly3DCollection([outer_square], color=gate_color))

    # Zentren der Gates
    ax.scatter(gates_positions[:, 0], gates_positions[:, 1], gates_positions[:, 2], c="r", s=50)

    # ─────────────────────────────────────────────────────────────── Obstacles
    for idx, point in enumerate(obstacle_positions):
        ax.plot(
            [point[0], point[0]],
            [point[1], point[1]],
            [0, 1],
            linewidth=4,
            color="k",
            label="Stäbe" if idx == 0 else "",
        )

    # ─────────────────────────────────────────────────────────────── Deko
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 2)
    ax.set_title("3D-Waypoints mit Extra-Punkten & rotierbaren Gates")
    ax.legend()
    plt.show()






waypoints = np.array([                           # erster Versuch für die robuste Trajektorie - 14.07
    [1.0, 1.5, 0.05],    # Original Punkt 0
    [0.95, 1.0, 0.2],    # Original Punkt 1
    [0.8, 0.3, 0.35],    # Neu
    [0.65, -0.2, 0.5],   # Original Punkt 2 (gate 0)
    [0.12, -0.9, 0.575], # Neu
    [0.1, -1.5, 0.65],   # Original Punkt 3
    [0.75, -1.3, 0.9],   # Neu
    [1.1, -0.85, 1.15],  # Original Punkt 4 (gate 1)
    [0.65, -0.175, 0.9], # Neu
    [0.1, 0.45, 0.6],    # Original Punkt 5 ??
    [0.0, 1.2, 0.5],     # Original Punkt 6 (gate 2)
    [0.0, 1.2, 1.1],     # Original Punkt 7
    [-0.15, 0.6, 1.1],   # Neu
    [-0.5, 0.0, 1.1],    # Original Punkt 8 (gate 3)
    [-0.9, -0.5, 1.1],   # Original Punkt 9
    [-1.7, -1.0, 1.1],   # Original Punkt 10
])

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




start_vel = [0, 0, 0.1]
velocity_gate4 = [0, -2, 0] # wird nicht verwendet - velocity in gate richtung wird betrachtet in UGB

start_pos = [1.0, 1.5, 0.0]

extended_gates = [start_pos] + gates_positions
extended_gates_quat = [[0, 0, 0, 0]] + gates_quat # der erste Eintrag wird nicht verwendet, da kein richtiges Gate

# gates = [start_pos, gates_positions[0], gates_positions[1]]


# X_opt, N_opt = optimize_full_trajectory_random(extended_gates, extended_gates_quat, obstacles_positions, start_vel, velocity_gate4, t_min=10, t_max=18, random_iteraitions=10, step=1)

# N_list = [72, 99, 73, 106]
# X_opt, N_opt = optimize_waypoint_positions(extended_gates, extended_gates_quat, N_list, obstacles_positions, start_vel, velocity_gate4, dt=1/50)

# N_opt = [60, 80, 80, 80] # 6 sec - normal
# N_opt = [63, 87, 71, 79] # 6 sec - optimized
# N_opt = [75, 76, 75, 74]
# N_opt = [100, 133, 133, 133] # 10 sec - normal
# N_opt = [105, 145, 118, 132] # 10 sec - optimized 6 scaled
# N_opt = [100, 150, 140, 110] # 10 sec - optimized
# N_opt = [35, 55, 45, 40] # 4 sec - optimized
# X_opt, cost = optimize_velocity_bounded(extended_gates, extended_gates_quat, N_opt, obstacles_positions, start_vel, velocity_gate4, dt=1/50)

N_opt = [63, 90, 71, 80]
#N_opt = [100, 150, 140, 110]
X_opt, N_opt, cost = optimize_from_given_N_list_random(extended_gates, extended_gates_quat, obstacles_positions, start_vel, velocity_gate4, N_opt, iterations=1, max_shift=0.1, shorten=[], lengthen=[])












print("N_opt:", N_opt)
print("optimale Zeiten:", np.array(N_opt) * (1/50) )
print("optimale Gesamtzeit:", sum(N_opt) * (1/50))
print("optimale Kosten: ", cost)


plot_waypoints_and_environment_extra(X_opt, waypoints, obstacles_positions, gates_positions, gates_quat)
#plot_waypoints_and_environment(X_opt, obstacles_positions, gates_positions, gates_quat)

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