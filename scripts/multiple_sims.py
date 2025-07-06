"""This module runs multiple simulations of the drone racing environment."""

import random
import time

from sim import simulate


def run_multiple_simulations():
    """Runs multiple simulations at Level 2."""
    # Run the simulation 10 times
    finished_counter = 0

    how_many = 2
    for run in range(1, how_many +1):
        print(f"\n=== Starting simulation run {run}/{how_many} ===")
        
        # Random wait to help ensure different seeds
        time.sleep(random.uniform(0.1, 0.5))
        
        # Run simulation with GUI disabled (gui=False) and random seed
        # You can set gui=True if you want to visualize each run
        ep_times = simulate(n_runs=1, gui=True)

        print(f"Run {run} completed with time: {ep_times[0] if ep_times[0] is not None else 'DNF'}")

        if ep_times[0] is not None:
            finished_counter += 1

    print(f"\n\n" \
    f"Total Attempts     = {how_many}\n" \
    f"Successful Attemps = {finished_counter}" \
    "\n\n")



if __name__ == "__main__":
    run_multiple_simulations()