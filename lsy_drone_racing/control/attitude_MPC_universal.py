"""This module implements an MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired positions.

The desired positions are generated using cubic spline interpolation from a set of predefined waypoints.
The initial trajectory defined with the nominal gate and obstacle positions is updated dynamically based on the drone's observations.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.attitude_MPC_universal_functions import create_ocp_solver

# FINE TUNING PARAMETERS
# # Setting Tunnel-Constraints ? - And there respective parameters
SET_TUNNEL               = True
TUNNEL_WIDTH_GENERAL     = 0.4
TUNNEL_WIDTH_AT_GATE     = 0.2
TUNNEL_TRANSITION_LENGTH = 0.6

# # MPC Parameters
COMPLETION_TIME   = 7
MPC_HORIZON_STEPS = 30
MPC_HORIZON_TIME  = 1

# # Nominal Trajectory Parameters
WAYPOINTS = np.array([
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

GATE_MAP = {
    0 : 3,
    1 : 7,
    2 : 10,
    3 : 13
}

# # Check for Update of Gates 
GATE_UPDATE_THRESHOLD    = 0.05 # Threshold when to update
GATE_UPDATE_SHIFT_FACTOR = 1.2 # Factor to shift the waypoints when a gate is updated

# # Initial Estimation of Drone Mass - 20.9 for m = 0.027 kg
ACCELERATION_ESTIMATION = 20.907574256269616  # params_acc ≈ k_thrust / m

# # MPC Cost Function Parameters
Q_CONTROL       =  0.05
Q_ANGLE         =  0.05
Q_OBSTACLE      = 50.0
Q_POSITION      = 10.0
Q_POSITION_END  = 10.0
Q_ALL = np.array([Q_CONTROL, Q_ANGLE, Q_OBSTACLE, Q_POSITION, Q_POSITION_END], dtype=float)

# # any debug prints ? Self-written and MPC-debug messages spressed if "False"
PRINT_AUSGABE = False






class MPController(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self.freq = config.env.freq
        self._tick = 0
        self.y=[]
        self.y_mpc=[]


        # Waypoints as in the trajectory controller but tuned and extended by trial and error.
        self.waypoints = WAYPOINTS.copy().astype(float)
        self.gate_map = GATE_MAP


        self.init_gates=[ [0.45, -0.5, 0.56], [1.0, -1.05, 1.11], [0.0, 1.0, 0.56], [-0.5, 0.0, 1.11], ]

        self.prev_obstacle = np.array([ [1, 0, 1.4], [0.5, -1, 1.4], [0, 1.5, 1.4], [-0.5, 0.5, 1.4], ])
        self.prev_gates_quat = [ [0.0, 0.0, 0.92268986, 0.38554308], [0.0, 0.0, -0.38018841, 0.92490906], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], ]
        self.prev_gates=[ [0.45, -0.5, 0.56], [1.0, -1.05, 1.11], [0.0, 1.0, 0.56], [-0.5, 0.0, 1.11], ]


        # Scale trajectory between 0 and 1
        ts = np.linspace(0, 1, np.shape(self.waypoints)[0])
        cs_x = CubicSpline(ts, self.waypoints[:, 0])
        cs_y = CubicSpline(ts, self.waypoints[:, 1])
        cs_z = CubicSpline(ts, self.waypoints[:, 2])

        #visualising traj. Needed for visualisiing draw line##
        tvisual = np.linspace(0, 1, 50)
        x = cs_x(tvisual)
        y = cs_y(tvisual)
        z = cs_z(tvisual)
        self.traj_vis=np.array([x,y,z])
        self.update_traj_vis=np.array([x,y,z])
        


        self.des_completion_time = config.controller.get("COMPLETION_TIME", COMPLETION_TIME)
        self.N = config.controller.get("MPC_HORIZON_STEPS", MPC_HORIZON_STEPS)
        self.T_HORIZON = config.controller.get("MPC_HORIZON_TIME", MPC_HORIZON_TIME)
        self.dt = self.T_HORIZON / self.N

        ts = np.linspace(0, 1, int(self.freq * self.des_completion_time))

        ticks_per_segment = int(self.freq * self.des_completion_time) / (len(self.waypoints) - 1)
        self.ticks = np.round(np.arange(0, len(self.waypoints)) * ticks_per_segment).astype(int)




        self.x_des = cs_x(ts)
        self.y_des = cs_y(ts)
        self.z_des = cs_z(ts)

        # Append points after trajectory for MPC
        pad = 10 * self.N
        self.x_des = np.concatenate((self.x_des, [self.x_des[-1]] * pad))
        self.y_des = np.concatenate((self.y_des, [self.y_des[-1]] * pad))
        self.z_des = np.concatenate((self.z_des, [self.z_des[-1]] * pad))
    
        # Tunnel Constrain Settings
        self.set_tunnel = config.controller.get("SET_TUNNEL", SET_TUNNEL)
        self.tunnel_width = TUNNEL_WIDTH_GENERAL
        self.tunnel_w_gate = TUNNEL_WIDTH_AT_GATE
        self.tunnel_trans = TUNNEL_TRANSITION_LENGTH

        windows_workaround = config.controller.get("windows_workaround", True)
        Q_all_local = config.controller.get("Q_ALL", Q_ALL)
        self.acados_ocp_solver, self.ocp = create_ocp_solver(self.T_HORIZON, self.N, Q_all_local, self.set_tunnel, windows_workaround)

        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.config = config
        self.finished = False
        self.params_acc_0_hat = ACCELERATION_ESTIMATION # params_acc[0] ≈ k_thrust / m_nominal ; nominal value given for nominal_mass = 0.027
        self.vz_prev = 0.0 # estimated velocity at start = 0

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        self.mass_estimator(obs)

        # Check if "Update Gates" is nessessary
        updated_gate = self.check_for_update(obs)
        if updated_gate:
            self.update_traj(obs,updated_gate)
            
        if not np.array_equal(self.prev_obstacle,obs["obstacles_pos"]):
            if PRINT_AUSGABE:
                print('Obstacle has changed')  
            self.prev_obstacle=obs["obstacles_pos"]


        # Check if the desired trajectory is long enough for the MPC-Horizon
        # If not terminate porcess (because with the padding already added this should not happen)
        remaining = len(self.x_des) - 1 - self._tick
        if remaining < self.N + 1:
            self.finished = True
            hover_cmd = np.array([0.3, 0.0, 0.0, 0.0])
            return hover_cmd
        i = min(self._tick, len(self.x_des) - 1)
        

        q = obs["quat"]
        r = R.from_quat(q)
        rpy = r.as_euler("xyz", degrees=False)

        xcurrent = np.concatenate(
            (
                obs["pos"],
                obs["vel"],
                rpy,
                np.array([self.last_f_collective, self.last_f_cmd]),
                self.last_rpy_cmd,
            )
        )
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)

        
        # Set parameters for the MPC
        self.y=[] # self.y for debug visulization
        for j in range(self.N):

            y_ref = np.array([ self.x_des[i + j], self.y_des[i + j], self.z_des[i + j] ])
            if self.set_tunnel:
                # # Help-Parameters for the Tunnel Constraints
                y_ref_p1 = np.array([ self.x_des[i + j + 1], self.y_des[i + j + 1], self.z_des[i + j + 1] ])
                delta = np.array(y_ref_p1 - y_ref)
                tangent_norm = delta / ( np.linalg.norm(delta) + 1e-6 )
                tunnel_width = self._tunnel_radius(y_ref)
                # # Set parameters for the MPC
                yref = np.hstack([ # params = vertcat(p_obs1, p_obs2, p_obs3, p_obs4, p_ref, params_acc_0, p_tun_tan, p_tun_r)
                    self.prev_obstacle[:, :2].flatten(),
                    y_ref, 
                    self.params_acc_0_hat,
                    tangent_norm,
                    tunnel_width,
                    ])
                self.acados_ocp_solver.set(j, "p", yref)
                self.y.append(yref) # self.y for debug visulization
            else:
                yref = np.hstack([
                    self.prev_obstacle[:, :2].flatten(),
                    y_ref, 
                    self.params_acc_0_hat,
                ])
                self.acados_ocp_solver.set(j, "p", yref)
                self.y.append(yref) # self.y for debug visulization
            

        y_ref = np.array([ self.x_des[i + self.N], self.y_des[i + self.N], self.z_des[i + self.N] ])
        if self.set_tunnel:
            # # Help-Parameters for the Tunnel Constraints
            y_ref_p1 = np.array([ self.x_des[i + self.N + 1], self.y_des[i + self.N + 1], self.z_des[i + self.N + 1] ])
            delta = np.array(y_ref_p1 - y_ref)
            tangent_norm = delta / ( np.linalg.norm(delta) + 1e-6 )
            tunnel_width = self._tunnel_radius(y_ref)
            # # Set parameters for the MPC
            yref_N = np.hstack([
                self.prev_obstacle[:, :2].flatten(),
                y_ref,
                self.params_acc_0_hat,
                tangent_norm,
                tunnel_width,
            ])
            self.acados_ocp_solver.set(self.N, "p", yref_N)
        else:
            # # Set parameters for the MPC
            yref_N = np.hstack([
                self.prev_obstacle[:, :2].flatten(),
                y_ref, 
                self.params_acc_0_hat,
            ])
            self.acados_ocp_solver.set(self.N, "p", yref_N)
        

        self.acados_ocp_solver.solve()


        self.y_mpc = [] # for Debug: Plot the prediction of the MPC
        for j in range(self.N + 1):
            x_pred = self.acados_ocp_solver.get(j, "x")
            y_mpc = x_pred[:len(yref_N)] # [:len(yref_N)] # only take the nessessary states
            self.y_mpc.append(y_mpc)


        x1 = self.acados_ocp_solver.get(1, "x")
        w = 1 / self.config.env.freq / self.dt
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]

        cmd = x1[10:14]

        return cmd

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
        self._tick += 1

        return self.finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0
    
    def check_for_update(self, obs: dict[str, NDArray[np.floating]]) -> int | None:
        """Check if any gate's position has changed significantly.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.

        Returns:
            - `None` if no gate moved beyond threshold
            - The **index (int)** of the first changed gate (row-wise comparison).
        """
        threshold = GATE_UPDATE_THRESHOLD

        current_gates = np.asarray(obs["gates_pos"])
        for gate_idx in range(len(self.prev_gates)):  # Compare each gate (row) individually
            prev_gate = np.asarray(self.prev_gates[gate_idx])
            current_gate = np.asarray(current_gates[gate_idx])
            
            if np.linalg.norm(prev_gate - current_gate) > threshold:
                self.prev_gates = current_gates.copy()  # Update stored positions
                if PRINT_AUSGABE:
                    print(f"Gate {gate_idx} moved significantly.")
                    print(self.prev_gates[gate_idx])
                return gate_idx+1  # Add one, so that we can check update for gate 0 with if statement. 
        
        return None

    def update_traj(self, obs: dict[str, NDArray[np.floating]], updated_gate: int):
        """Set the cubic splines new from the current position.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            updated_gate: The (index+1) of the gate that has moved beyond the threshold.

        Returns:
            None
        """
        if self._tick == 0:
            if PRINT_AUSGABE:
                print("Kein Update, Tick == 0")
            return
        
        # update the waypoints that correspond to a specific gate
        for i, idx in self.gate_map.items(): # update the waypoints that correspond to a specific gate
            diff=self.prev_gates[i]-self.init_gates[i]
            self.waypoints[idx] += diff * GATE_UPDATE_SHIFT_FACTOR

        gate_idx = updated_gate-1 # Subtract the one we added in check_for_update because of if statement
        center_idx = self.gate_map[int(gate_idx)]

        # 1. Neue Sub-Waypoints auswählen
        rel_indices = [-1, 0, 1]
        abs_indices = [
            center_idx + i for i in rel_indices
            if 0 <= center_idx + i < len(self.waypoints)
        ]
        if len(abs_indices) < 2:
            print("⚠️ Nicht genug gültige Punkte für Splines.")
            return


        wp_section = self.waypoints[abs_indices]
        tick_section = [self.ticks[i] for i in abs_indices]
        tick_times = np.array(tick_section) / self.freq
        dt_segments = np.diff(tick_section)
        

        # 2. Preparation for new segment
        ts = []
        for i in range(len(dt_segments)):
            t_start = tick_times[i]
            t_end = tick_times[i + 1]
            n_points = max(2, dt_segments[i])  # mind. 2 Punkte pro Segment
            ts_seg = np.linspace(t_start, t_end, n_points, endpoint=False)
            ts.extend(ts_seg)
        ts.append(tick_times[-1])  # letzten Zeitpunkt ergänzen
        ts = np.array(ts)


        # 3. Neue Splines erstellen
        cs_x = CubicSpline(tick_times, wp_section[:, 0])
        cs_y = CubicSpline(tick_times, wp_section[:, 1])
        cs_z = CubicSpline(tick_times, wp_section[:, 2])
        x_new = cs_x(ts)
        y_new = cs_y(ts)
        z_new = cs_z(ts)
        # For Visulization of the updated Trajectory
        self.update_traj_vis=np.array([x_new,y_new,z_new])

        # 4. Aktuelle Trajektorie ersetzen
        tick_min = tick_section[0]
        tick_max = tick_section[-1]
       
        self.x_des[tick_min:tick_max + 1]  = x_new
        self.y_des[tick_min:tick_max + 1]  = y_new
        self.z_des[tick_min:tick_max + 1]  = z_new

        if PRINT_AUSGABE:
            print(f"✅ Neue Teiltrajektorie um Gate {gate_idx} aktualisiert.")
        

    def mass_estimator(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Updates the Acceleration Parameter in the MPC-Solver corresponding to the drone mass.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.

        Returns:
            None
        """
        max_angle = max_angle=np.deg2rad(20)


        params_acc = [20.907574256269616, 3.653687545690674] # params_acc[0] ≈ k_thrust / m_nominal
        GRAVITY = 9.806


        # Messgrößen
        vz_dot   = (obs["vel"][2] - self.vz_prev) / self.dt
        self.vz_prev = obs["vel"][2] # update für nächsten Durchlauf

        roll, pitch, _ = R.from_quat(obs["quat"]).as_euler("xyz", degrees=False)
        cos_roll_pitch   = np.cos(roll) * np.cos(pitch)
        
        # Only update, when Drone is upright
        if abs(roll) > max_angle or abs(pitch) > max_angle or cos_roll_pitch < 0.3:
            return


        denominator = self.last_f_collective * cos_roll_pitch + 1e-6    # safety against numerial errors

        params_acc_0   = (vz_dot + GRAVITY) / denominator - params_acc[1]/(self.last_f_collective + 1e-6)
        if params_acc_0 <= 0:                         # safety against numerial errors
            return

        alpha    = 0.02                               # Glättung
        self.params_acc_0_hat = (1 - alpha) * self.params_acc_0_hat + alpha * params_acc_0


    def _tunnel_radius(self, p_ref: np.ndarray) -> float:
        """ref_pt: np.array([x,y,z]) eines MPC-Knotens.
        
        Args:
            p_ref: 3D-position of the reference point in the MPC trajectory.

        Returns:
            The radius of the tunnel at the given reference point.
        """
        # Entfernung zum nächsten Gate-Zentrum
        d_gate = np.min(np.linalg.norm(self.prev_gates - p_ref, axis=1))

        # lineare Interpolation zwischen R_gate und R_far 
        alpha = np.clip(d_gate / self.tunnel_trans, 0.0, 1.0)

        return self.tunnel_w_gate + (self.tunnel_width - self.tunnel_w_gate) * alpha