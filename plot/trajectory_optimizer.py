"""This module optimizes the trajectory and plots it."""

import numpy as np
from optimizer_help_func import optimize_from_given_N_list_random, plot_waypoints_and_environment

obstacles_positions = [
    [1.0, 0.0, 1.4],
    [0.5, -1.0, 1.4],
    [0., 1.5, 1.4],
    [-0.5, 0.75, 1.4],
]

gates_positions = [
    [0.45, -0.5, 0.56],
    [1.0, -1.05, 1.11],
    [0.0, 1.0, 0.56],
    [-0.5, 0.0, 1.11],
]

gates_quat = [
    [0.0, 0.0, 0.92388, 0.38268],
    [0.0, 0.0, -0.38268, 0.92388],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
]




start_vel = [0, 0, 0.1]
velocity_gate4 = [0, -2, 0] # wird nicht verwendet - velocity in gate richtung wird betrachtet in UGB

start_pos = [1.0, 1.5, 0.0]

extended_gates = [start_pos] + gates_positions
extended_gates_quat = [[0, 0, 0, 0]] + gates_quat # der erste Eintrag wird nicht verwendet, da kein richtiges Gate




N_opt = [69, 95, 72, 78] # good starting distribution for: how many steps are in each segement between Gates - can be altered randomly within the optimization method.


X_opt, N_opt, cost = optimize_from_given_N_list_random(extended_gates, extended_gates_quat, obstacles_positions, start_vel, velocity_gate4, N_opt, iterations=1, max_shift=0.01, shorten=[], lengthen=[])




print("N_opt:", N_opt)
print("optimale Zeiten:", np.array(N_opt) * (1/50) )
print("optimale Gesamtzeit:", sum(N_opt) * (1/50))
print("optimale Kosten: ", cost)


plot_waypoints_and_environment(X_opt, obstacles_positions, gates_positions, gates_quat)
