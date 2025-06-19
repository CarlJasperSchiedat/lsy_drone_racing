import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
import json

def quaternion_to_rotmat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y**2 + z**2),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),       1 - 2 * (x**2 + z**2),   2 * (y * z - x * w)],
        [2 * (x * z - y * w),       2 * (y * z + x * w),     1 - 2 * (x**2 + y**2)]
    ])
def transform_square(square, R, t):
    return (R @ square.T).T + t


def sample_trajectory(traj, step=20):
    traj = np.asarray(traj)
    return traj[::step]




def plot_waypoints_and_environment(waypoints, obstacle_positions, gates_positions, gates_quat):
    """Plottet Waypoints als Linie, rotierte Gates und vertikale Hindernisstangen in 3D."""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    

    # Waypoints als Linie zeichnen
    waypoints = np.asarray(waypoints)
    ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], "b-", label="Trajectory")  # nur Linie, kein Marker

    # Gates
    gate_size = 0.225
    gate_outer = gate_size + 0.05
    gate_color = (1, 0, 0, 0.5)

    for i, (gate, quat) in enumerate(zip(gates_positions, gates_quat)):
        R = quaternion_to_rotmat(quat)
        '''
        for size, alpha in [(gate_size, 1.0), (gate_outer, 0.4)]:
            square = np.array([
                [-size, 0, -size],
                [ size, 0, -size],
                [ size, 0,  size],
                [-size, 0,  size],
            ])
            transformed = transform_square(square, R, gate)
            poly = Poly3DCollection([transformed], color=gate_color[:3], alpha=alpha)
            if i == 0 and alpha == 1.0:
                poly.set_label("Gate")
            ax.add_collection3d(poly)
        '''
        square = np.array([
            [-gate_size, 0, -gate_size],
            [ gate_size, 0, -gate_size],
            [ gate_size, 0,  gate_size],
            [-gate_size, 0,  gate_size],
        ])
        transformed = transform_square(square, R, gate)
        poly = Poly3DCollection([transformed], color=gate_color[:3], alpha=gate_color[3])
        if i == 0:
            poly.set_label("Gate")
        ax.add_collection3d(poly)

    # Gate-Zentren
    ax.scatter(*np.array(gates_positions).T, c="r", s=50, label=None)

    # Obstacles
    for idx, point in enumerate(obstacle_positions):
        ax.plot([point[0], point[0]], [point[1], point[1]], [0, 1],
                linewidth=4, label='Staves' if idx == 0 else "")

    # Achsen und Anzeige
    # ax.set_title("optimized trajectory")
    ax.view_init(elev=50, azim=310) # 225 # 310
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    '''
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 2)
    '''
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-1.6, 0.6)
    ax.set_zlim(0, 1.5)
    # '''
    # ax.legend()
    plt.tight_layout()
    plt.show()


def plot_waypoints_and_uncertainties(waypoints, obstacle_positions, gates_positions, gates_quat):    

    def quaternion_to_rotmat(q):
        return R.from_quat(q).as_matrix()

    def transform_square(square, R, t):
        return (R @ square.T).T + t



    def draw_gate(ax, center, rot, alpha=0.15, label=None):
        gate_size = 0.225
        square = np.array([
            [-gate_size, 0, -gate_size],
            [ gate_size, 0, -gate_size],
            [ gate_size, 0,  gate_size],
            [-gate_size, 0,  gate_size],
        ])
        transformed = transform_square(square, rot, center)
        poly = Poly3DCollection([transformed], color=(1, 0, 0), alpha=alpha)
        if label:
            poly.set_label(label)
        ax.add_collection3d(poly)



    def draw_cylinder(ax, center, radius, height, alpha=0.1, color="gray"):
        x0, y0 = center[:2]
        z = np.linspace(0, height, 10)
        theta = np.linspace(0, 2 * np.pi, 30)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = x0 + radius * np.cos(theta_grid)
        y_grid = y0 + radius * np.sin(theta_grid)
        ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, linewidth=0)



    # Plot vorbereiten
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Trajektorie
    waypoints = np.asarray(waypoints)
    ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], "b-", label="Trajectory")

    # Gates
    for i, (gate_pos, gate_quat) in enumerate(zip(gates_positions, gates_quat)):
        rot = quaternion_to_rotmat(gate_quat)

        # Nominal Gate
        # draw_gate(ax, gate_pos, rot, alpha=0.5, label="Nominal Gate" if i == 0 else None)

        # Unsicherheiten ±15cm in globalem X/Y
        for dx in [-0.15, 0.15]:
            for dy in [-0.15, 0.15]:
                offset = np.array([dx, dy, 0.0])
                draw_gate(ax, gate_pos + offset, rot, alpha=0.12)

    # Obstacles + Unsicherheitszylinder
    for idx, obs in enumerate(obstacle_positions):
        x, y = obs[:2]
        ax.plot([x, x], [y, y], [0, 1.0], linewidth=4, label="Obstacle" if idx == 0 else "")
        draw_cylinder(ax, [x, y], radius=0.15, height=1.0, alpha=0.12, color="gray")

    # Anzeige
    ax.set_title("Trajectory mit Unsicherheiten (Gates & Obstacles)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 2)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_waypoints_comparison(old_traj, new_traj, obstacle_positions, gates_positions, gates_quat):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=50, azim=225)

    # Plot old and new trajectories
    old_traj = np.asarray(old_traj)
    new_traj = np.asarray(new_traj)
    ax.plot(old_traj[:, 0], old_traj[:, 1], old_traj[:, 2], "r--", label="Old Trajectory")
    ax.plot(new_traj[:, 0], new_traj[:, 1], new_traj[:, 2], "b-", label="New Trajectory")

    # Gates
    gate_size = 0.225
    gate_outer = gate_size + 0.05
    gate_color = (1, 0, 0, 0.5)

    for i, (gate, quat) in enumerate(zip(gates_positions, gates_quat)):
        R = quaternion_to_rotmat(quat)
        square = np.array([
            [-gate_size, 0, -gate_size],
            [ gate_size, 0, -gate_size],
            [ gate_size, 0,  gate_size],
            [-gate_size, 0,  gate_size],
        ])
        transformed = transform_square(square, R, gate)
        poly = Poly3DCollection([transformed], color=gate_color[:3], alpha=gate_color[3])
        if i == 0:
            poly.set_label("Gate")
        ax.add_collection3d(poly)

    # Gate centers
    ax.scatter(*np.array(gates_positions).T, c="r", s=50)

    # Obstacles
    for idx, point in enumerate(obstacle_positions):
        ax.plot([point[0], point[0]], [point[1], point[1]], [0, 1],
                linewidth=4, label='Obstacle' if idx == 0 else "")

    # Axis settings
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 2)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_waypoints_with_sampling(full_traj, step, obstacle_positions, gates_positions, gates_quat):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=50, azim=225)

    full_traj = np.asarray(full_traj)
    sampled = sample_trajectory(full_traj, step=step)

    # Original trajectory (transparent)
    ax.plot(full_traj[:, 0], full_traj[:, 1], full_traj[:, 2], "b-", alpha=0.3, label="Original Trajectory")

    # Sampled points (highlighted)
    ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], c="blue", s=20, label="Sampled Waypoints")

    # Gates
    gate_size = 0.225
    gate_color = (1, 0, 0, 0.5)

    for i, (gate, quat) in enumerate(zip(gates_positions, gates_quat)):
        R = quaternion_to_rotmat(quat)
        square = np.array([
            [-gate_size, 0, -gate_size],
            [ gate_size, 0, -gate_size],
            [ gate_size, 0,  gate_size],
            [-gate_size, 0,  gate_size],
        ])
        transformed = transform_square(square, R, gate)
        poly = Poly3DCollection([transformed], color=gate_color[:3], alpha=gate_color[3])
        if i == 0:
            poly.set_label("Gate")
        ax.add_collection3d(poly)

    # Gate centers
    ax.scatter(*np.array(gates_positions).T, c="r", s=50)

    # Obstacles
    for idx, point in enumerate(obstacle_positions):
        ax.plot([point[0], point[0]], [point[1], point[1]], [0, 1],
                linewidth=4, label='Obstacle' if idx == 0 else "")

    ax.view_init(elev=50, azim=300) # 225 # 310
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    '''
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 2)
    '''
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-1.6, 0.6)
    ax.set_zlim(0, 1.5)
    # ax.legend()
    plt.tight_layout()
    plt.show()


def plot_sampled_waypoints_with_highlight(traj, step, highlight_range, obstacle_positions, gates_positions, gates_quat):
    traj = np.asarray(traj)
    sampled = traj[::step]

    i1, i2 = highlight_range
    highlighted = sampled[i1:i2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=50, azim=225)

    # Original trajectory (transparent)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "b-", alpha=0.3, label="Original Trajectory")

    # Alle gesampleten Punkte (blau)
    ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], c="blue", s=20, label="Sampled")

    # Hervorgehobene Punkte (grün)
    ax.scatter(highlighted[:, 0], highlighted[:, 1], highlighted[:, 2], c="darkgreen", s=60, label="Highlight", depthshade=False)



    # Gates
    gate_size = 0.225
    gate_color = (1, 0, 0, 0.5)

    for i, (gate, quat) in enumerate(zip(gates_positions, gates_quat)):
        R = quaternion_to_rotmat(quat)
        square = np.array([
            [-gate_size, 0, -gate_size],
            [ gate_size, 0, -gate_size],
            [ gate_size, 0,  gate_size],
            [-gate_size, 0,  gate_size],
        ])
        transformed = transform_square(square, R, gate)
        poly = Poly3DCollection([transformed], color=gate_color[:3], alpha=gate_color[3])
        if i == 0:
            poly.set_label("Gate")
        ax.add_collection3d(poly)

    # Gate centers
    ax.scatter(*np.array(gates_positions).T, c="r", s=50)

    # Obstacles
    for idx, point in enumerate(obstacle_positions):
        ax.plot([point[0], point[0]], [point[1], point[1]], [0, 1],
                linewidth=4, label='Obstacle' if idx == 0 else "")

    ax.view_init(elev=50, azim=300) # 225 # 310
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    '''
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 2)
    '''
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-1.6, 0.6)
    ax.set_zlim(0, 1.5)
    # ax.legend()
    plt.tight_layout()
    plt.show()





# Enviroment
obstacles_positions = [[1.0, 0.0, 1.4], [0.5, -1.0, 1.4], [0., 1.5, 1.4], [-0.5, 0.5, 1.4], ]
gates_positions = [ #[ 0.57457256, -0.56958073,  0.5334348 ],
    [0.45, -0.5, 0.56], [1.0, -1.05, 1.11], [0.0, 1.0, 0.56], [-0.5, 0.0, 1.11], ]
gates_quat = [   # sind als rpy gegeben und nicht als Quaternion ??????? -> z.B. siehe level0.toml
    [0.0, 0.0, 0.92388, 0.38268], [0.0, 0.0, -0.38268, 0.92388], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], ]


obstacle_positions_level2 = np.array([ [ 0.8585742 ,  0.07062764, 1.430361  ], [ 0.4857328 , -0.8829206 , 1.4369684 ], [-0.06284191,  1.4103925 , 1.4420558 ], [-0.60142666,  0.40711257, 1.3848282 ] ])
gates_positions_level2 = np.array([ [ 0.57457256, -0.56958073,  0.5334348 ], [ 1.0139579,  -0.9989303,   1.16429   ], [ 0.1489047,   1.0194494,   0.49401802], [-0.43796134, -0.03196271,  1.0649462 ] ])
gates_quat_level2 = np.array([ [ 0.        ,  0.        ,  0.9046965 ,  0.4260566 ], [ 0.        ,  0.        , -0.36786887,  0.9298777 ], [ 0.        ,  0.        ,  0.0308489 ,  0.99952406], [ 0.        ,  0.        ,  0.99977046, -0.02142496] ])



# file = "plot/traj_data/opt_sub_12_3sec.json"
file = "plot/traj_data/opt_12_6sec.json"
with open(file, "r") as f:
    data = json.load(f)

# Extrahiere Positionen aus X_opt
waypoints = [x[:3] for x in data["X_opt"]]
# old_way = [x[:3] for x in data["old"]]
# new_way = [x[:3] for x in data["new"]]



# Plotten
plot_waypoints_and_environment(waypoints, obstacles_positions, gates_positions, gates_quat)
# plot_waypoints_and_environment(waypoints, obstacle_positions_level2, gates_positions_level2, gates_quat_level2)

# plot_waypoints_and_uncertainties(waypoints, obstacles_positions, gates_positions, gates_quat)

# plot_waypoints_comparison(old_way, new_way, obstacles_positions, gates_positions, gates_quat)

# plot_waypoints_with_sampling(waypoints, 40, obstacles_positions, gates_positions, gates_quat)
# plot_sampled_waypoints_with_highlight(waypoints, 40, (3, 8), obstacles_positions, gates_positions, gates_quat=gates_quat)