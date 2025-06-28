import matplotlib.pyplot as plt
import numpy as np


# Definition aller drei Straf-Funktionen
distances = np.linspace(0, 0.275, 50)


# Variante 1:
multiply_1 = 1.0
falloff_1 = 1
faktor_1 = 100
penalty_lin = multiply_1 * np.exp(-faktor_1 * distances / falloff_1)

# Variante 2:
multiply_2 = 1.0
falloff_2 = 0.5
faktor_2 = 25
exp_2 = 2
penalty_quad = multiply_2 * np.exp( - faktor_2 * ((distances / falloff_2) ** exp_2))

# Variante 3:
multiply_3 = 1.0
falloff_3 = 1.0
faktor_3 = 100
exp_3 = 2
penalty_3 = multiply_3 * np.exp(- faktor_3 * ((distances / falloff_3) ** exp_3))

# Variante 4:
multiply_4 = 1.0
falloff_4 = 2.0
faktor_4 = 150
exp_4 = 2
penalty_4 = multiply_4 * np.exp(- faktor_4 * ((distances / falloff_4) ** exp_4))

# Variante 5:
multiply_5 = 1.0
falloff_5 = 2.5
faktor_5 = 200
exp_5 = 2
penalty_5 = multiply_5 * np.exp(- faktor_5 * ((distances / falloff_5) ** exp_5))



# Plot
plt.figure(figsize=(10, 6))

# plt.plot(distances, penalty_lin, label=f"exp( -{faktor_1} * dist / {falloff_1} )", linestyle="-")
plt.plot(distances, penalty_quad, label=f"exp( -{faktor_2} * ( (d/{falloff_2})^{exp_2} ) )", linestyle=":")
plt.plot(distances, penalty_3, label=f"exp( -{faktor_3} * ( (d/{falloff_3})^{exp_3} ) )", linestyle="--")
plt.plot(distances, penalty_4, label=f"exp( -{faktor_4} * ( (d/{falloff_4})^{exp_4} ) )", linestyle="-")
plt.plot(distances, penalty_5, label=f"exp( -{faktor_5} * ( (d/{falloff_5})^{exp_5} ) )", linestyle="-")

plt.axvline(0.05, color="black", linestyle="--", alpha=0.5)
plt.axvline(0.15, color="grey", linestyle="--", alpha=0.5)
plt.axvline(0.275, color="grey", linestyle="--", alpha=0.5)

plt.title("Vergleich von Straf-Funktionen nach Distanz")
plt.xlabel("Distanz (m)")
plt.ylabel("Penalty")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
