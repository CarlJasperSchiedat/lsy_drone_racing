import matplotlib.pyplot as plt
import numpy as np

# Definition aller drei Straf-Funktionen
distances = np.linspace(0, 0.15, 50)


# Variante 1: linear
multiply_1 = 10
falloff_1 = 1
faktor_1 = 100
penalty_lin = multiply_1 * np.exp(-faktor_1 * distances / falloff_1)

# Variante 2: quadratisch
multiply_2 = 1.0
falloff_2 = 0.5
faktor_2 = 100
'''
penalty_bell = np.where(
    distances < falloff_3,
    1 * np.exp(-((distances / falloff_3) ** 2) * faktor_3),
    0.0
)
'''
penalty_quad = multiply_2 * np.exp(-((distances / falloff_2) ** 2) * faktor_2)

# Variante 3: ^4
multiply_3 = 5.0
falloff_3 = 0.4
faktor_3 = 200
penalty_3 = multiply_3 * np.exp(-((distances / falloff_3) ** 4) * faktor_3)



# Plot
plt.figure(figsize=(10, 6))

plt.plot(distances, penalty_lin, label=f"exp( -{faktor_1} * dist / {falloff_1} )", linestyle="-")
plt.plot(distances, penalty_quad, label=f"exp( -{faktor_2} * ( (d/{falloff_2})^2 ) )", linestyle=":")
plt.plot(distances, penalty_3, label=f"exp( -{faktor_3} * ( (d/{falloff_3})^4 ) )", linestyle="--")
plt.axvline(0.05/2, color="black", linestyle="--", alpha=0.5)
plt.axvline(0.1, color="grey", linestyle="--", alpha=0.5)
plt.title("Vergleich von Straf-Funktionen nach Distanz")
plt.xlabel("Distanz (m)")
plt.ylabel("Penalty")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
