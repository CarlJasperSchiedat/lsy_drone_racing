"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are based on a optimal control optimization.
When a new obstacle or gate is detected, the trajectory waypoints are optimized again.
"""

import numpy as np
import threading
import time
import json

from scipy.spatial.transform import Rotation as R
from numpy.typing import NDArray

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.my_contoller_func import create_ocp_solver, recompute_trajectory




class MyMPCController(Controller):

    def __init__(self, obs, info, config):
        """
        Initializes the MPC controller with the given observation, info, and configuration.

        """
        super().__init__(obs, info, config)
        self.freq = config.env.freq
        self.dt = 1 / self.freq
        self.tick = 0
        self.finished = False


        # Statusanzeigen ob Update schon durchgeführt wurde
        self.gate_updated = [False] * 4         # wurde nach erkennen die trajektorie aktualisiert? 
        self.obstacle_updated = [False] * 4

        self.obs = None     # Aktuelle Beobachtung der Umgebung für den Hintergrundprozess


        # initial geplante Trajektorie
        traj_path = "plot/traj_data/opt_traj_290525.json"
        with open(traj_path, "r") as f:
            traj_data = json.load(f)
        # self.trajectory = np.array(traj_data["X_opt"])[:, :3] # Trajektorie - nur Positionen
        self.trajectory = np.array(traj_data["X_opt"]) # Trajektorie - vollständige Zustände
        self.section_ticks = np.array(traj_data["N_opt"])  # Anzahl der Ticks pro Abschnitt
        # self._planning_done = False
        

        # Hintergrundprozess zur Trajektorienplanung starten
        self._lock = threading.Lock() # Verhindert Probleme beim Lesen/Schreiben gemeinsam genutzter Variablen: self._lock.acquire() + self._lock.release() oder with self._lock: 
        self._thread = threading.Thread(target=self._plan_trajectory) # ruft self._plan_trajectory() im Hintergrund auf
        self._thread.start()


        # Initialisierung des Solvers für MPC
        self.N_Horizon = 30
        self.T_Horizon = self.N_Horizon * self.dt # Nehme die OCP Schrittweite wie die echte ??
        self.acados_ocp_solver, self.ocp = create_ocp_solver(self.T_Horizon, self.N_Horizon)

        # Zustandsvaribalen welche nicht gemessen werden können im Observations und die letzten Befehle
        self.last_f_collective = 0.3             # MÖGLICHERWEISE BESSER INITIALISIEREN, 0.3~=Schub für Schwebeflug-Balancepunkt, bei nominaler Masse
        self.last_f_cmd = 0.3                    # MÖGLICHERWEISE BESSER INITIALISIEREN
        self.last_rpy_cmd = np.zeros(3)





    def _plan_trajectory(self):
        """
        Hintergrundprozess zur Trajektorienplanung.
        Überprüfung ob neues Objekt erkannt wurde -> Trajektorie neu planen.
        """

        if self.obs is None: # Falls compute_control noch nicht aufgerufen wurde
            return

        update = False

        for idx in range(4):
            if self.obs["gates_visited"][idx] and not self.gate_updated[idx]:
                print("Gate erkannt:", idx+1)
                update = True
                self.gate_updated[idx] = True

            if self.obs["obstacles_visited"][idx] and not self.obstacle_updated[idx]:
                print("Obstacle erkannt:", idx+1)
                update = True
                self.obstacle_updated[idx] = True


        if update: # Falls irgendein Gate/Obstacle erkannt wurde, aber noch nicht geupdated uwrde -> Trajektorie neu planen

            with self._lock: # Lade alle Werte lokal um Lesen / Schreiben in compute_control möglich ist
                
                # Neuplanung mit Beginn der Werte aus observation; current_traj für initialwerte der Optimierung
                # current_traj = self.trajectory[:, :3] # MÖGLICHERWEISE BESSER MIT VOLLEM STATE-VEKTOR
                current_traj = self.trajectory

                rpy = R.from_quat(self.obs["quat"]).as_euler("xyz", degrees=False)
                start_state = np.concatenate([
                    self.obs["pos"],            # Position (px, py, pz)
                    self.obs["vel"],            # Velocity (vx, vy, vz)
                    rpy,                        # Roll, Pitch, Yaw
                    [self.last_f_collective],   # f_collective
                    [self.last_f_cmd],          # f_collective_cmd
                    self.last_rpy_cmd           # r_cmd, p_cmd, y_cmd
                ])

                current_tick = self.tick

                obstacles = self.obs["obstacles_pos"]
                gate_quat = self.obs["gates_quat"]
                gate_pos = self.obs["gates_pos"]

                N_list = self.section_ticks

                
            # Startposition / -geschw. aus den aktuellen Beob. oder aus der Trajektorie?
            start_time = time.time()
            traj_section, ab_tick = recompute_trajectory(current_traj, gate_pos, gate_quat, obstacles, N_list, current_tick, start_state)
            elapsed_time = time.time() - start_time
            print(f"⏱️ Recompute duration: {elapsed_time:.6f} seconds")


        with self._lock: # Speichere die Trajektorie im gemeinsamen Speicher
            self.trajectory[ab_tick:ab_tick + len(traj_section)] = traj_section





    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """
        Aufruf bei jedem Tick.
        Berechnung der Attitude Command basierend auf der geplanten Trajektorie.
        """
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        # Speichere Observations für Hintergrundprozess und lade Trajektorie und current_tick lokal
        with self._lock:
            self.obs = obs
            trajectory_local = self.trajectory
            tick_local = self.tick

        # Index for aktuellen Durchlauf
        idx = min(tick_local, trajectory_local.shape[0] - 1)

        # Keine Trajektorienwerte mehr ?
        if tick_local > idx:
            self.finished = True


        # Fix the start position for the MPC solver
        q = obs["quat"] 
        r = R.from_quat(q)
        rpy = r.as_euler("xyz", degrees=False) # current angle

        xcurrent = np.concatenate((obs["pos"], obs["vel"], rpy, [self.last_f_collective, self.last_f_cmd], self.last_rpy_cmd)) # sel.___ für Verzögerung des Models
        
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)


        # Set the desired trajectory for the MPC        # TRAJEKTORY MUSS UM N HORIZON VERLÄNGERT SEIN, DAMIT DIE MPC LÄUFT
        for j in range(self.N):
            if idx + j < trajectory_local.shape[0]:
                yref = trajectory_local[idx + j]
            else:
                yref = trajectory_local[-1]
            self.acados_ocp_solver.set(j, "yref", yref)

        yref_N = trajectory_local[idx + self.N]  # Letzter Wert der Trajektorie
        self.acados_ocp_solver.set(self.N, "yref", yref_N)
        

        self.acados_ocp_solver.solve()
        x1 = self.acados_ocp_solver.get(1, "x") # ertses Prädiktionsergebnis -> f_cmd, rpy_cmd
        
        '''
        # Low-Pass Filter for last_f_collective, x1[9] = aktueller Gesamtschub (Thrust); nötig falls frequ der Simulation ist ungleich der des MPCs
        w = 1 / self.freq / self.dt # falls dt = 1 / frequ --> w = 1 --> self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w = x1[9]
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w # nötig falls der MPC Prädiktionsschritt länger ist als der der Simulation -> dann ist der vorhergesegte Schub erst nach x Schritten erreicht
        '''
        # Annahme: MPC läuft mit der Frequenz der Simulation + Drone Modell wird bereits in MPC betrachtet und muss nicht hier noch: dot(f_collective) = 10 * (f_cmd - f) -> f_collective = last_f + 10 * (f_cmd - f) 
        self.last_f_collective = x1[9] # falls w = 1 -> aktueller Gesamtschub wird ganz übernommen

        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]


        return x1[10:14] # Rückgabe der Befehle: [f_collective_cmd, roll_cmd, pitch_cmd, yaw_cmd]





    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        with self._lock:
            self.tick += 1

        return self.finished

    def episode_callback(self):
        """Reset the integral error."""
        self.tick = 0