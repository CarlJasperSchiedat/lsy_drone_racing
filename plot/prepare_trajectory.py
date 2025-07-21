"""Prepares waypoints as a sampled trajectory from an optimized solution and plots it."""
import json

import matplotlib.pyplot as plt
import numpy as np


def prepare_trajectory_from_solution(X_opt: np.ndarray, N_list: np.ndarray, stepsize: int=10, exclude_gate_window: int=5) -> tuple:
    """Erstellt aus einer optimierten Trajektorie eine geglättete Waypoint-Trajektorie mit festen Abständen.

    Args:
        X_opt (np.ndarray): Optimierte Zustände der Trajektorie, Form = (N_total + 1, state_dim)
        N_list (list[int]): Schrittanzahl pro Segment zwischen Gates
        stepsize (int): Schrittgröße für gleichmäßige Abtastung (z. B. alle 10 Ticks)
        exclude_gate_window: Fenstergröße, um Ticks in der Nähe von Gates zu vermeiden
        
    Returns:
        Tuple:
            - Abgetastete Waypoints (M x 3)
            - Ticks/Indizes der Waypoints (Liste von ints)
    """
    X_opt = np.asarray(X_opt)
    positions = X_opt[:, :3]  # Nur Positionen extrahieren
    ticks = []

    # Schrittweise abtasten
    total_steps = len(positions)
    for i in range(0, total_steps, stepsize):
        ticks.append(i)


    # Filtere Ticks nahe bei Gates
    gate_ticks = np.cumsum(N_list)
    filtered_ticks = []
    for t in ticks:
        if all(abs(t - g) > exclude_gate_window for g in gate_ticks):
            filtered_ticks.append(t)


    # Gate-Positionen hinzufügen
    gate_tick = 0
    for N in N_list:
        gate_tick += N
        if gate_tick < total_steps and gate_tick not in filtered_ticks:
            filtered_ticks.append(gate_tick)


    ticks = sorted(filtered_ticks)
    waypoints = positions[ticks]

    # Mapping: Gate-Index → Waypoint-Index im ticks-Array
    gate_idx_map = {}
    for gate_id, g_tick in enumerate(gate_ticks):
        matches = np.where(ticks == g_tick)[0]
        if len(matches) > 0:
            gate_idx_map[gate_id] = int(matches[0])
    # print(gate_idx_map)

    return waypoints, ticks, gate_idx_map


def extend_trajectory(waypoints: np.ndarray, ticks: np.ndarray, X_opt: np.ndarray, n_extra: int=5, spacing: int=10, dt: float=1/50) -> tuple:
    """Verlängert die Trajektorie nach dem letzten Waypoint in Richtung der Endgeschwindigkeit.

    Args:
        waypoints (np.ndarray): Alle Waypoints der Trajektorie.
        ticks (list[int]): Indizes der vorhandenen Waypoints.
        X_opt (np.ndarray): Vollständige optimierte Zustände.
        n_extra (int): Anzahl zusätzlicher Waypoints.
        dt (float): Zeitschrittgröße.
        spacing (int): Schrittweite für Abstand zwischen neuen Punkten (in Ticks).

    Returns:
        Tuple[np.ndarray, list[int]]: Erweiterte Waypoints und Ticks.
    """
    last_tick = ticks[-1]
    last_vel = X_opt[last_tick][3:6]

    last_pos = waypoints[-1]

    v_norm = np.linalg.norm(last_vel)
    if v_norm < 1e-3:
        print("v_norm < 1e-3")
        return waypoints, ticks  # nichts tun



    # nicht so hart steigen: Skaliere z-Komponente um z_scale
    z_scale = 0.5
    last_vel = last_vel.copy()
    last_vel[2] *= z_scale
    last_vel *= v_norm / np.linalg.norm(last_vel)
    


    # Neue Waypoints generieren
    extended_points = []
    extended_ticks = []
    for i in range(1, n_extra + 1):
        delta_t = i * spacing * dt
        new_pos = last_pos + delta_t * last_vel
        extended_points.append(new_pos)
        extended_ticks.append(last_tick + i * spacing)

    return np.vstack([waypoints, extended_points]), list(ticks) + extended_ticks




file_name = "traj_2"



with open(f"plot/traj_data/opt_{file_name}.json", "r") as f:
    data = json.load(f)

N_list = data["N_opt"]
X_raw = data["X_opt"]
X_opt = np.array(X_raw)  # Shape: (N+1, nx)




waypoints, ticks, gate_idx_map = prepare_trajectory_from_solution(X_opt, N_list, stepsize=5, exclude_gate_window=1)

waypoints, ticks = extend_trajectory(waypoints, ticks, X_opt, n_extra=1, spacing=10)


# print(ticks)


save_dict = {
    "waypoints": waypoints.tolist(),
    "ticks": ticks,
    "gate_idx_map": {str(k): v for k, v in gate_idx_map.items()}
}
with open(f"plot/traj_data/prepared_{file_name}.json", "w") as f:
    json.dump(save_dict, f, indent=2)






positions = X_opt[:, :3]  # Nur x, y, z

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Ganze Trajektorie
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="Trajektorie", color="gray", alpha=0.5)

# Waypoints
ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], color="red", label="Waypoints", s=40)

# Gate-Waypoints hervorheben
gate_ticks = np.cumsum(N_list)
gate_positions = positions[gate_ticks]
ax.scatter(gate_positions[:, 0], gate_positions[:, 1], gate_positions[:, 2], color="green", s=60, marker="^", label="Gates")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Abgetastete MPC-Trajektorie")
ax.legend()
plt.tight_layout()
plt.show()

