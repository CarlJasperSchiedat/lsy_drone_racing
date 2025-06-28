import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import colors as mcolors



def gate_cost(rel_x, rel_y, rel_z, border=0.225):
    y_weight = np.exp(-100 * ((rel_y / 1.0) ** 2))
    # x_borderdist_cost = np.exp(-150 * (((np.abs(rel_x) - border - 0.05) / 2.0) ** 2))
    # z_borderdist_cost = np.exp(-150 * (((np.abs(rel_z) - border - 0.05) / 2.0) ** 2))

    # border_cost = np.maximum(x_borderdist_cost, z_borderdist_cost)

    # border_cost = x_borderdist_cost + z_borderdist_cost

    rel = np.maximum(np.abs(rel_x), np.abs(rel_z))
    border_cost = np.exp(-150 * (((rel - border - 0.05) / 2.0) ** 2))

    return y_weight * border_cost




def random_points(num_samples, x_range, y_range, z_range):
    """Erzeugt Punkte in jeweils eigenem Intervall für x, y, z."""
    x = np.random.uniform(*x_range, size=num_samples)
    y = np.random.uniform(*y_range, size=num_samples)
    z = np.random.uniform(*z_range, size=num_samples)
    return np.column_stack((x, y, z))




def regular_grid_points(n_per_axis: int, x_range: tuple[float, float], y_range: tuple[float, float], z_range: tuple[float, float]) -> np.ndarray:
    xs = np.linspace(*x_range, n_per_axis)
    ys = np.linspace(*y_range, n_per_axis)
    zs = np.linspace(*z_range, n_per_axis)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))




def square_corners(b):
    return np.array(
        [
            [-b, 0.0, -b],
            [b, 0.0, -b],
            [b, 0.0, b],
            [-b, 0.0, b],
            [-b, 0.0, -b],
        ]
    )




def plot_gate_cost_alpha_gate(
    num_samples=None,
    n_per_axis=None,
    border=0.225,
    x_range=(-0.7, 0.7),
    y_range=(-0.5, 0.25),
    z_range=(-0.7, 0.7),
):
    # Zufällige Punkte in einem Würfel
    if num_samples is not None and n_per_axis is not None:
        print("Warnung: Es wurden sowohl num_samples als auch n_per_axis angegeben - Verwende Gitter.")
        num_samples = None

    if num_samples is not None:
        pts = random_points(num_samples, x_range, y_range, z_range)
    if n_per_axis is not None:
        pts = regular_grid_points(n_per_axis, x_range, y_range, z_range)

    if num_samples is None and n_per_axis is None:
        raise ValueError("Entweder num_samples oder n_per_axis muss angegeben werden.")
    
    cost = gate_cost(pts[:, 0], pts[:, 1], pts[:, 2], border + 0.05)

    # Normiere Kosten 0…1 und berechne Alpha: hohe Kosten -> geringe Alpha
    cost_norm = (cost - cost.min()) / (cost.max() - cost.min() + 1e-9)
    alpha = 1 - 0.95 * (1-cost_norm)  # 0.05…1.0

    # Erzeuge RGBA‑Farben (konstantes Blau)
    base_rgb = np.array(mcolors.to_rgba("tab:blue"))[:3]
    colors = np.hstack([np.tile(base_rgb, (pts.shape[0], 1)), alpha[:, None]])

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=6, depthshade=False)


    # ─ Gate‑Rahmen einzeichnen (Quadrat im x‑z, y=0) ─
    gate = square_corners(border)
    conners_au = square_corners(border + 0.1)
    ax.plot(gate[:, 0], gate[:, 1], gate[:, 2], lw=2.0, color="black")
    ax.plot(conners_au[:, 0], conners_au[:, 1], conners_au[:, 2], lw=2.0, color="red")
    

    ax.set_xlabel("rel_x [m]")
    ax.set_ylabel("rel_y [m]")
    ax.set_zlabel("rel_z [m]")
    ax.set_title("Gate‑Kosten – Alpha ∝ (1/Kosten)")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)

    # Dezente Gitterlinien
    ax.grid(True, alpha=0.2, linestyle='--')
    plt.show()




# Beispiel‑Aufruf mit individuellen Grenzen
plot_gate_cost_alpha_gate(
    num_samples=2000,
    n_per_axis=None,
    border=0.225,
    x_range=(-0.7, 0.7),
    y_range=(-0.05, 0.2),
    z_range=(-0.7, 0.7),
)



