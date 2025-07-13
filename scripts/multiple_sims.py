"""This module runs multiple simulations of the drone racing environment."""
import platform
import shutil
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sim import simulate
from tqdm.auto import tqdm

from lsy_drone_racing.utils import load_config

# Define the simulation parameters
T_list = np.arange(5.5, 7.0 + 0.5, 0.5)
#T_list = 5.12
tunnel = [True, False]
how_many = 5
gui_enabled = False


cfg_path = Path("config/level2.toml")




def run_multiple_simulations():
    """Runs multiple simulations at given Level for different times."""
    # Check if T_list is iterable
    T_iter = [float(T_list)] if not isinstance(T_list, Iterable) else T_list

    # container that is dynamically adapted to 'tunnel'
    stats   = {flag: {"T": [], "avg": [], "succ": []} for flag in tunnel}

    for set_tunnel in tunnel:
        clean_acados_folder(set_tunnel)

        tqdm.write(f"\n\n=====  SET_TUNNEL = {set_tunnel}  =====\n\n")

        for T in tqdm(T_iter, desc="COMPLETION_TIME-Sweep", unit="s", leave=False):
            # Load and modify given config
            cfg = load_config(cfg_path)
            cfg.controller.COMPLETION_TIME = float(T)
            cfg.controller.SET_TUNNEL      = set_tunnel

            # Simulate
            ep_times = simulate(config=cfg, n_runs=how_many, gui=gui_enabled)

            # Calculate statistics
            passed = [t for t in ep_times if t is not None]
            n_success = len(passed)
            success_rate = n_success / len(ep_times)
            mean_time = np.nan if not passed else np.mean(passed)

            # Save simulation results
            stats[set_tunnel]["T"].append(T)
            stats[set_tunnel]["avg"].append(mean_time)
            stats[set_tunnel]["succ"].append(n_success)
            
            tqdm.write(
                f"[Tunnel {set_tunnel}] T ={T:4.1f}s | "
                f"Ø ={mean_time:6.2f}s | "
                f"{n_success}/{len(ep_times)} passed ({success_rate:.0%})" 
            )


    print("\n\nFertig\n\n", stats)


    stats = fill_missing_avgs(stats)

    print("\n\nFertig\n\n", stats)

    plot_results(stats, how_many)



def clean_acados_folder(set_tunnel: bool, folder: str = "c_generated_code") -> None:
    """Entfernt den kompletten Inhalt von <folder>, sofern wir unter Windows laufen. Auf anderen Betriebssystemen macht die Funktion nichts."""
    if platform.system() != "Windows":
        return                      # Non-Windows: überspringen

    cg_path = Path(folder)
    if cg_path.is_dir():
        shutil.rmtree(cg_path, ignore_errors=True)

    # Dummy Aufruf von der sim aus gründen
    cfg = load_config(cfg_path)
    cfg.controller.windows_workaround = False
    cfg.controller.SET_TUNNEL      = set_tunnel
    try:
        simulate(config=cfg, n_runs=1, gui=False)
    except(FileNotFoundError, OSError):
        print("Dummy Aufruf fehlgeschlagen")


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


def plot_results(stats: dict,
                        how_many: int) -> None:
    """Plot of stats.
    
    • X-Achse  = Ø-Zeit je COMPLETION_TIME  (NaNs bereits ersetzt)
    • Y-Achse  = # erfolgreiche Runs
    """
    fig, ax = plt.subplots()

    markers = {True: "o", False: "s"}      # Tunnel an/aus
    linestyles = {True: "-", False: "--"}

    for flag, data in stats.items():
        x = data["avg"]            # Ø-Zeit  (bereits mit Schätzungen gefüllt)
        y = data["succ"]           # Erfolgs-Zahl

        # Plot als Linie + Marker
        ax.plot(x, y,
                marker=markers.get(flag, "o"),
                linestyle=linestyles.get(flag, "-"),
                label=f"Tunnel = {flag}")

    ax.set_xlabel("Ø-Zeit erfolgreicher Läufe [s]")
    ax.set_ylabel("Anzahl erfolgreicher Läufe")
    ax.set_ylim(0, how_many)
    ax.margins(x=0.05, y=0.05)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper left")
    plt.title("Erfolgs-Quote vs. Ø-Zeit")
    fig.tight_layout()
    plt.show()





if __name__ == "__main__":
    run_multiple_simulations()