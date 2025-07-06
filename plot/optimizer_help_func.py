"""Helpers for trajectory optimization in drone racing environments."""
import numpy as np
from optimizer import optimize_velocity_bounded

#from optimizer import optimize_original


def distribute_timesteps_by_distance(gates: np.ndarray, N_total: int) -> list:
    """Distributes a total number of optimization steps across segments based on the distance between gate poses.

    Args:
        gates: list of gate poses, each [x, y, z, quat]
        N_total: total number of optimization steps to distribute across segments

    Returns:
        list of ints N_list such that sum(N_list) = N_total
    """
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
 

def apply_segment_tendency(N_list: np.ndarray, shorten: np.ndarray=[0, 2], lengthen: np.ndarray=[1, 3], max_shift: float=0.1) -> list:
    """Verändert N_list durch gezielte Verlängerung/Verkürzung einzelner Segmente.
    
    Args:
        N_list: Startliste der Segmentlängen.
        shorten: Indizes, die leicht verkürzt werden
        lengthen: Indizes, die leicht verlängert werden
        max_shift: max. prozentuale Änderung je Segment (z.B. 0.1 = 10%)

    Returns:
        Angepasstes N_list mit gleichem Gesamtsummenwert.
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


def apply_tendency_to_N_list(N_list: np.ndarray, max_shift: float=0.2 , shorten: np.ndarray=[0, 2], lengthen: np.ndarray=[1, 3]) -> list:
    """Modifiziert ein gegebenes N_list durch gezielte Verlängerung/Verkürzung einzelner Segmente.

    Args:
        N_list: Startliste der Segmentlängen.
        max_shift: max. prozentuale Änderung je Segment (z.B. 0.2 = 20%)
        shorten: Indizes, die leicht verkürzt werden
        lengthen: Indizes, die leicht verlängert werden
        
    Returns:
        Angepasstes N_list mit gleichem Gesamtsummenwert.
    """
    N_array = np.array(N_list, dtype=float)
    N_total = int(np.sum(N_array))

    deltas = np.zeros_like(N_array)

    if not shorten and not lengthen:
        # Wenn beide leer sind: alle zufällig anpassen (mit ±max_shift)
        deltas = np.random.uniform(-max_shift, max_shift, size=len(N_array))
    else:
        deltas = np.zeros_like(N_array)
        for i in shorten:
            if 0 <= i < len(deltas):
                deltas[i] -= np.random.uniform(0.0, max_shift)
        for i in lengthen:
            if 0 <= i < len(deltas):
                deltas[i] += np.random.uniform(0.0, max_shift)


    N_var = N_array * (1 + deltas)
    N_var = np.maximum(N_var, 1.0)  # min 1 Schritt pro Segment


    # Normalisiere zurück zur ursprünglichen Gesamtlänge
    N_int = [int(round(x)) for x in N_var]
    diff = sum(N_int) - N_total

    while diff != 0:
        # Füge/Subtrahiere 1 an Stelle mit größtem/längstem Beitrag
        idx = np.argmax(N_int) if diff > 0 else np.argmin(N_int)
        N_int[idx] -= int(np.sign(diff))
        diff = sum(N_int) - N_total


    return N_int





def optimize_full_trajectory_random(
        gates: np.ndarray, gates_quat: np.ndarray, obstacles: np.ndarray, v_start: np.ndarray, v_end: np.ndarray, 
        t_min: float, t_max: float, step: int=1, random_iteraitions: int=5, dt: float=1/50
    ) -> tuple:
    """Optimiert die Trajektorie in dem er alle Env-Parameter beachtet. Dabei wird Die Zeit der Trajektorie variiert wobei bei jeder versuchten Zeit die Anzahl der Steps pro Segment zwischen Gates randomisiert varriiert.

    Args:
        gates: Liste der Gate-Positionen, jedes [x, y, z].
        gates_quat: Liste der Gate-Orientierungen als Quaternionen.
        obstacles: Liste der Hindernis-Positionen, jedes [x, y, z].
        v_start: Startgeschwindigkeit des Drohnenfluges.
        v_end: Endgeschwindigkeit des Drohnenfluges.
        t_min: Minimalzeit für die Trajektorie in Sekunden.
        t_max: Maximalzeit für die Trajektorie in Sekunden.
        step: Schrittweite in dt-Schritten, die zwischen den Optimierten Zeiten leigen.
        random_iteraitions: Anzahl der zufälligen Varianten, die getestet werden.
        dt: Zeit pro Schritt in Sekunden.
        
    Returns:
        Angepasstes N_list mit gleichem Gesamtsummenwert.
    """
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
            
            N_list_innen = apply_segment_tendency(N_list, shorten=[0,2], lengthen=[1,3], max_shift=0.3)
            print("ERGBENISS INNEN:", N_list_innen)
            try:
                pos, cost = optimize_velocity_bounded( gates, gates_quat, N_list_innen, obstacles, v_start, v_end, dt=dt )
                #pos, cost = optimize_original( gates, gates_quat, N_list_innen, obstacles, v_start, v_end, dt=dt )
                if cost < best_cost:
                    best_cost = cost
                    best_X_opt = pos
                    best_N = N_list_innen

            except Exception as e:
                print(f"❌ Optimization failed for N={N}: {e}")
                continue

    return best_X_opt, best_N




def optimize_from_given_N_list_random(
        gates: np.ndarray, gates_quat: np.ndarray, obstacles: np.ndarray, v_start: np.ndarray, v_end: np.ndarray, 
        N_list_start: np.ndarray, iterations: int=10, max_shift: float=0.2, shorten: np.ndarray=[0, 2], lengthen: np.ndarray=[1, 3], dt: float=1/50) -> tuple:
    """Optimiert die Trajektorie in dem er alle Env-Parameter beachtet. Dabei gibt es ein gegebenes N_list_start, welches variiert wird. Die Anzahl der Schritte pro Segment wird dabei randomisiert variiert.

    Args:
        gates: Liste der Gate-Positionen, jedes [x, y, z].
        gates_quat: Liste der Gate-Orientierungen als Quaternionen.
        obstacles: Liste der Hindernis-Positionen, jedes [x, y, z].
        v_start: Startgeschwindigkeit des Drohnenfluges.
        v_end: Endgeschwindigkeit des Drohnenfluges.
        N_list_start: Startliste der Segmentlängen.
        iterations: Anzahl der zufälligen Varianten, die getestet werden.
        max_shift: max. prozentuale Änderung je Segment (z.B. 0.2 = 20%).
        shorten: Indizes, die leicht verkürzt werden
        lengthen: Indizes, die leicht verlängert werden
        dt: Zeit pro Schritt in Sekunden.
        
    Returns:
        Angepasstes N_list mit gleichem Gesamtsummenwert.
    """
    best_cost = np.inf
    best_X_opt = None
    best_N = None


    for _ in range(iterations):

        N_list_var = apply_tendency_to_N_list( N_list_start, max_shift=max_shift , shorten=shorten, lengthen=lengthen )

        print("🔁 Variante:", N_list_var)

        try:
            pos, cost = optimize_velocity_bounded( gates, gates_quat, N_list_var, obstacles, v_start, v_end, dt=dt )
            #pos, cost = optimize_original( gates, gates_quat, N_list_var, obstacles, v_start, v_end, dt=dt )
            if cost < best_cost:
                best_cost = cost
                best_X_opt = pos
                best_N = N_list_var
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            continue

    return best_X_opt, best_N, best_cost
