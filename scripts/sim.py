"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import mujoco
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import draw_line, draw_tunnel, load_config, load_controller

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv

rgba_1 = np.array([1.0, 0, 0, 1.0])  # Red, fully opaque
rgba_2=np.array([0,0,1,1])
rgba_3=np.array([0,1,0,1])
rgba_4=np.array([1,1,0,1])

logger = logging.getLogger(__name__)


def simulate(
    config: str = "level2.toml",
    controller: str | None = None,
    n_runs: int = 1,
    gui: bool | None = True,
    
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
        n_runs: The number of episodes.
        gui: Enable/disable the simulation GUI.

    Returns:
        A list of episode times.
    """
    # Check if ConfigDict is given or a Path is given
    if isinstance(config, (str, Path)):
        # Load configuration and check if firmare should be used.
        config = load_config(Path(__file__).parents[1] / "config" / config)

    if gui is None:
        gui = config.sim.gui
    else:
        config.sim.gui = gui
    # Load the controller module
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)  # This returns a class, not an instance
    # Create the racing environment
    env: DroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=np.random.randint(0,1000), #seed=config.env.seed,
    )
    env = JaxToNumpy(env)

    ep_times = []
    for _ in range(n_runs):  # Run n_runs episodes with the controller
        obs, info = env.reset()
        controller: Controller = controller_cls(obs, info, config)
        i = 0
        fps = 60

        while True:
            curr_time = i / config.env.freq

            action = controller.compute_control(obs, info)
            y_ref = np.array([y[8:11] for y in controller.y])
            radii   = np.array([y[-1]   for y in controller.y])
            y_mpc=np.array([y[:3] for y in controller.y_mpc])
            #radii   = np.array([y[-1]   for y in controller.y])
            obs, reward, terminated, truncated, info = env.step(action)
            # Update the controller internal state and models.
            controller_finished = controller.step_callback(
                action, obs, reward, terminated, truncated, info
            )
            # Add up reward, collisions
            if terminated or truncated or controller_finished:
                break
            # Synchronize the GUI.
            if config.sim.gui:
                if ((i * fps) % config.env.freq) < fps:
                    draw_line(env=env,points=controller.traj_vis.T,rgba=rgba_2) # nominal trajectory
                    draw_line(env=env,points=controller.update_traj_vis.T,rgba=rgba_4) # updated trajectory segment
                    draw_line(env=env,points=y_mpc,rgba=rgba_3) # optimized MPC trajectory
                    
                    
                    try:
                        draw_line(env=env,points=y_ref,rgba=rgba_1, min_size=6.0,max_size=6.0) # green MPC line -> reference trajectory
                    except Exception as e:
                        logger.error(f"Error drawing MPC line: {e} \nExpend the refence trajectory")

                    if getattr(controller, "set_tunnel", False):  # only draw tunnel if it is set
                        draw_tunnel(env=env,centers=y_ref,radii=radii,rgba=rgba_1) # MPC tunnel
                    env.render()
                if i == 1:
                    viewer = env.unwrapped.sim.viewer.viewer
                    viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
            i += 1

        controller.episode_callback()  # Update the controller internal state and models.
        log_episode_stats(obs, info, config, curr_time)
        controller.episode_reset()
        ep_times.append(curr_time if obs["target_gate"] == -1 else None)

    # Close the environment
    env.close()
    return ep_times


def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:  # The drone has passed the final gate
        gates_passed = len(config.env.track.gates)
    finished = gates_passed == len(config.env.track.gates)
    logger.info(
        f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {gates_passed}\n"
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
