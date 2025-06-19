import matplotlib.pyplot as plt
import numpy as np

# Definition aller drei Straf-Funktionen
distances = np.linspace(0, 0.225, 50)


# Variante 1: linear
multiply_1 = 1.0
falloff_1 = 1
faktor_1 = 100
penalty_lin = multiply_1 * np.exp(-faktor_1 * distances / falloff_1)

# Variante 2: quadratisch
multiply_2 = 1.0
falloff_2 = 0.7
faktor_2 = 50
penalty_quad = multiply_2 * np.exp(-((distances / falloff_2) ** 2) * faktor_2)

# Variante 3: ^4
multiply_3 = 1.0
falloff_3 = 0.7
faktor_3 = 3000
penalty_3 = multiply_3 * np.exp(-((distances / falloff_3) ** 4) * faktor_3)

# Variante 4: ^6
multiply_4 = 1.0
falloff_4 = 0.4
faktor_4 = 1500
penalty_4 = multiply_4 * np.exp(-((distances / falloff_4) ** 6) * faktor_4)

# Variante 5: ^8
multiply_5 = 1.0
falloff_5 = 0.5
faktor_5 = 500
penalty_5 = multiply_5 * np.exp(-((distances / falloff_5) ** 8) * faktor_5)



# Plot
plt.figure(figsize=(10, 6))

# plt.plot(distances, penalty_lin, label=f"exp( -{faktor_1} * dist / {falloff_1} )", linestyle="-")
plt.plot(distances, penalty_quad, label=f"exp( -{faktor_2} * ( (d/{falloff_2})^2 ) )", linestyle=":")
plt.plot(distances, penalty_3, label=f"exp( -{faktor_3} * ( (d/{falloff_3})^4 ) )", linestyle="--")
plt.plot(distances, penalty_4, label=f"exp( -{faktor_4} * ( (d/{falloff_4})^6 ) )", linestyle="-")
plt.plot(distances, penalty_5, label=f"exp( -{faktor_5} * ( (d/{falloff_5})^8 ) )", linestyle="-")

plt.axvline(0.025, color="black", linestyle="--", alpha=0.5)
plt.axvline(0.075, color="grey", linestyle="--", alpha=0.5)
plt.axvline(0.125, color="grey", linestyle="--", alpha=0.5)
plt.axvline(0.175, color="grey", linestyle="--", alpha=0.5)

plt.title("Vergleich von Straf-Funktionen nach Distanz")
plt.xlabel("Distanz (m)")
plt.ylabel("Penalty")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
