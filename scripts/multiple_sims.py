"""This module runs multiple simulations of the drone racing environment."""
import numbers
import platform
import shutil
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sim import simulate
from tqdm.auto import tqdm

from lsy_drone_racing.utils import load_config

# Define the simulation parameters
T_list = 8                              # desired completion time - actual completion time will be up to a second faster

how_many = 10                           # How many runs do you want to simulate per timestep ?

gui_enabled = True                      # Should the GUI be enabled ?

cfg_path = Path("config/level2.toml")   # Which enviroment configuration should be run ?



# Define the simulation parameters
tunnel = [False]

Q_CONTROL       =  0.05
Q_ANGLE         =  0.05
Q_OBSTACLE      = 50.0
Q_POSITION      = 10.0
Q_POSITION_END  = 10.0
Q_ALL = np.array([Q_CONTROL, Q_ANGLE, Q_OBSTACLE, Q_POSITION, Q_POSITION_END], dtype=float)

Q_50 = Q_ALL.copy()
Q_500 = np.where(Q_ALL == Q_OBSTACLE, 500, Q_ALL)
Q_10 = np.where(Q_ALL == Q_OBSTACLE, 10, Q_ALL)
Q_00 = np.where(Q_ALL == Q_OBSTACLE, 0, Q_ALL)

mpc_settings = [(30, 1.0, Q_50)]    #[(mpc_horizon_steps, mpc_horizon_time, Q_all), (...,....)]


def run_multiple_simulations():
    """Runs multiple simulations at given Level for different times."""
    # Check if T_list is iterable
    T_iter = [float(T_list)] if not isinstance(T_list, Iterable) else T_list

    # saves data
    stats = defaultdict(lambda: {"T": [], "avg": [], "std": [], "succ": []})
    
    # number of total runs for progress-bar
    total_runs = len(tunnel) * len(mpc_settings) * len(T_iter)


    with tqdm(total=total_runs, desc="Simulations", unit="run") as pbar:
        for set_tunnel in tunnel:
            for N, Tf, Q_all in mpc_settings:
                
                # Workaround for my Windows Computer who doesnt like acados at all
                clean_acados_folder(N, Tf, Q_all, set_tunnel)

                # O_obstacle - is tested in the most part so it is written into the plot
                q_obst = float(Q_all[2])

                for T in T_iter:
                
                    # Load and modify given config
                    cfg = load_config(cfg_path)
                    cfg.controller.COMPLETION_TIME        = float(T)
                    cfg.controller.MPC_HORIZON_STEPS      = N
                    cfg.controller.MPC_HORIZON_TIME       = Tf
                    cfg.controller.Q_ALL                  = Q_all
                    cfg.controller.SET_TUNNEL             = set_tunnel

                    # Simulate
                    ep_times = simulate(config=cfg, n_runs=how_many, gui=gui_enabled)

                    # Calculate statistics
                    passed = [t for t in ep_times if t is not None]
                    n_success = len(passed)
                    mean_time = np.nan if not passed else np.mean(passed)
                    std_time   = np.nan if not passed else np.std(passed, ddof=1)

                    # Save simulation results
                    key = (set_tunnel, N, Tf, q_key(Q_all))
                    stats[key]["T"].append(T)
                    stats[key]["avg"].append(mean_time)
                    stats[key]["std"].append(std_time)
                    stats[key]["succ"].append(n_success)
                
                    post = OrderedDict([
                        ("T",       f"{T:0.1f}"),
                        ("avg-T",   f"{mean_time:4.1f}s"),
                        ("std-T",   f"{std_time:4.1f}s"),
                        ("succ",    f"{n_success}/{len(ep_times)}"),
                        ("tunnel",  set_tunnel),
                        ("N",       N),
                        ("Tf",      f"{Tf:0.1f}"),
                        ("Q_obst",  int(q_obst)),
                    ])

                    pbar.set_postfix(post)
                    
                    pbar.update(1)

    # if some timesteps have no completion -> fill these with interpolated values
    stats = fill_missing_avgs(stats)

    print("\n\nFertig\n\n", stats)

    plot_results(stats, how_many)




def q_key(Q_all: np.ndarray | list[float]) -> tuple:
    """Rundet und gibt IMMER ein Tupel zurück."""
    return tuple(np.round(np.asarray(Q_all, dtype=float), 8))




def clean_acados_folder(
        mpc_horizon_steps: int,
        mpc_horizon_time: float,
        Q_all: NDArray[np.floating[Any]],
        set_tunnel: bool,
        folder: str = "c_generated_code"
) -> None:
    """Entfernt den kompletten Inhalt von <folder>, sofern wir unter Windows laufen. Auf anderen Betriebssystemen macht die Funktion nichts."""
    if platform.system() != "Windows":
        return                      # Non-Windows: überspringen

    cg_path = Path(folder)
    if cg_path.is_dir():
        shutil.rmtree(cg_path, ignore_errors=True)

    # Dummy Aufruf von der sim aus gründen
    cfg = load_config(cfg_path)

    cfg.controller.windows_workaround = False

    cfg.controller.MPC_HORIZON_STEPS      = mpc_horizon_steps
    cfg.controller.MPC_HORIZON_TIME       = mpc_horizon_time
    cfg.controller.Q_ALL                  = Q_all
    cfg.controller.SET_TUNNEL             = set_tunnel
    try:
        simulate(config=cfg, n_runs=1, gui=False)
    except(FileNotFoundError, OSError):
        print("")



def fill_missing_avgs(stats: dict) -> dict:
    """Ersetzt NaNs in stats[flag]["avg"] durch lineare Schätzungen basierend auf den vorhandenen Ø-Zeiten."""
    for flag, data in stats.items():
        T   = np.asarray(data["T"],   dtype=float)
        avg = np.asarray(data["avg"], dtype=float)

        mask = ~np.isnan(avg)            # gültige Punkte

        if mask.sum() <= 1:
            # keinerlei echte Werte – hier können wir nichts fitten
            continue
        else:
            # klassisches lineares least-squares-Fit
            m, b = np.polyfit(T[mask], avg[mask], 1)

        # Nur dort überschreiben, wo avg NaN war:
        avg[np.isnan(avg)] = m * T[np.isnan(avg)] + b
        data["avg"] = avg.tolist()

    return stats


def plot_results(stats: dict, how_many: int) -> None:
    """Plot of stats.
    
    • X-Achse  = Ø-Zeit je COMPLETION_TIME  (NaNs bereits ersetzt)
    • Y-Achse  = # erfolgreiche Runs
    """
    fig, ax = plt.subplots()

    # optionale Stil-Maps: Tunnel-Flag → Linestyle, Q_obst → Marker
    linestyles = {True: "-", False: "--"}
    markers    = {50: "o", 25: "s", 10: "^"}   # Q_OBSTACLE-Wert → Marker

    for (flag, N, Tf, q_vec), data in stats.items():
        x_vals = data["avg"]        # Ø-Zeit
        y_vals = data["succ"]       # # erfolgreiche Läufe

        
        if isinstance(q_vec, numbers.Number):
            q_obst = float(q_vec)
        else:
            q_obst = float(q_vec[2])


        ax.plot(
            x_vals,
            y_vals,
            marker = markers.get(q_obst, "o"),
            linestyle = linestyles.get(flag, "-"),
            label = (f"Tunnel={flag} | N={N}, Tf={Tf}s | "
                     f"Q_obst={q_obst:g}")
        )

    ax.set_xlabel("Ø-Zeit erfolgreicher Läufe [s]")
    ax.set_ylabel("Anzahl erfolgreicher Läufe")
    ax.set_ylim(0, how_many)
    ax.margins(x=0.05, y=0.05)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize="x-small", loc="upper left")
    ax.set_title("Einstellungen für Max Sim")
    fig.tight_layout()
    plt.show()




if __name__ == "__main__":
    run_multiple_simulations()