import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Datos del problema
# X = Número de clientes (cientos), Y = Distancia (km)
X = np.array([8, 7, 6, 4, 2, 1])
Y = np.array([15, 19, 25, 23, 34, 40])

df = pd.DataFrame({'Nº Clientes (X)': X, 'Distancia km (Y)': Y})
print(df.to_string(index=False))

# Cálculo manual del coeficiente de correlación de Pearson
n = len(X)
mean_X = np.mean(X)
mean_Y = np.mean(Y)

sum_xy = np.sum((X - mean_X) * (Y - mean_Y))
sum_x2 = np.sum((X - mean_X)**2)
sum_y2 = np.sum((Y - mean_Y)**2)

r = sum_xy / np.sqrt(sum_x2 * sum_y2)

print(f"Media de X (clientes): {mean_X:.4f}")
print(f"Media de Y (distancia): {mean_Y:.4f}")
print(f"Σ(X-X̄)(Y-Ȳ) = {sum_xy:.4f}")
print(f"Σ(X-X̄)²    = {sum_x2:.4f}")
print(f"Σ(Y-Ȳ)²    = {sum_y2:.4f}")
print(f"\nCoeficiente de correlación r = {r:.4f}")

if abs(r) > 0.9:
    print("→ Correlación lineal MUY FUERTE y negativa (a mayor distancia, menos clientes)")
elif abs(r) > 0.7:
    print("→ Correlación lineal FUERTE")
else:
    print("→ Correlación lineal moderada o débil")

# Cálculo de pendiente (b) e intercepto (a)
b = sum_xy / sum_x2
a = mean_Y - b * mean_X

print(f"Pendiente  b = {b:.4f}")
print(f"Intercepto a = {a:.4f}")
print(f"\nEcuación de regresión: Y = {a:.4f} + ({b:.4f})·X")
print(f"Simplificado:          Y = {a:.2f} - {abs(b):.2f}·X")

# Recta de regresión X sobre Y: X = a' + b'·Y
b_prime = sum_xy / sum_y2
a_prime = mean_X - b_prime * mean_Y

print(f"Recta de regresión X sobre Y: X = {a_prime:.4f} + ({b_prime:.4f})·Y")

# Si Y = 2 km
Y_pred = 2
X_pred = a_prime + b_prime * Y_pred
print(f"\nPara Y = {Y_pred} km:")
print(f"X = {a_prime:.4f} + ({b_prime:.4f}) × {Y_pred} = {X_pred:.2f}")
print(f"\n→ Se esperan aproximadamente {X_pred:.0f} cientos de clientes ({X_pred*100:.0f} clientes)")

# Usando la recta Y sobre X: Y = a + b·X
X_objetivo = 5
Y_pred_obj = a + b * X_objetivo

print(f"Usando Y = {a:.4f} + ({b:.4f})·X")
print(f"Para X = {X_objetivo} (cientos de clientes):")
print(f"Y = {a:.4f} + ({b:.4f}) × {X_objetivo} = {Y_pred_obj:.2f} km")
print(f"\n→ Debe situarse aproximadamente a {Y_pred_obj:.2f} km del núcleo de población")

fig, ax = plt.subplots(figsize=(8, 5))

# Nube de puntos
ax.scatter(X, Y, color='steelblue', s=80, zorder=5, label='Datos observados')

# Recta de regresión Y sobre X
X_line = np.linspace(0, 10, 100)
Y_line = a + b * X_line
ax.plot(X_line, Y_line, color='crimson', linewidth=2, label=f'Recta: Y = {a:.2f} + ({b:.2f})·X')

# Punto predicho (pregunta 3)
ax.scatter([5], [Y_pred_obj], color='green', s=120, zorder=6, marker='*', label=f'Predicción X=5 → Y={Y_pred_obj:.2f} km')

ax.set_xlabel('Nº de Clientes X (cientos)', fontsize=12)
ax.set_ylabel('Distancia Y (km)', fontsize=12)
ax.set_title('Regresión Lineal: Clientes vs Distancia', fontsize=14)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print(f"\nResumen final:")
print(f"  r  = {r:.4f}")
print(f"  Ecuación: Y = {a:.2f} + ({b:.2f})·X")
print(f"  Pregunta 2 (Y=2 km) → X ≈ {X_pred:.0f} cientos de clientes")
print(f"  Pregunta 3 (X=5)    → Y ≈ {Y_pred_obj:.2f} km")