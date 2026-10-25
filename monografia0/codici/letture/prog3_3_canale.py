"""
prog3_3_canale.py -- PROGRAMMA 3.3
Corrente laminare in un canale inclinato a pelo libero di sezione quadrata.

Velocita' adimensionale U = u / (L^2 rho g sin(theta) / mu) nella sezione 0 <= Y, Z <= 1:
    U_YY + U_ZZ = -1,   U = 0 su fondo e pareti laterali,  dU/dY = 0 sul pelo libero (Y = 1).
Sovrarilassamento successivo con omega ottimo; la condizione di derivata si
impone con una riga fittizia sopra il pelo libero (U[i, n] = U[i, n-2]).
Portata calcolata pesando i nodi con le aree h^2 (interni), h^2/2 (contorno),
h^2/4 (spigoli), cioe' con la regola dei trapezi bidimensionale.
"""
import numpy as np
import matplotlib.pyplot as plt
from metodi_numerici import omega_ottimo
from stile_grafici import salva, BLU, ROSSO

H = 0.05
M = int(round(1.0 / H)) + 1          # linee verticali (indice i, coordinata Z)
N = int(round(1.0 / H)) + 1          # linee orizzontali (indice j, coordinata Y)
omega = omega_ottimo(M, N)           # come nel programma originale (a = cos(pi/M) + cos(pi/N))

U = np.zeros((M, N + 1))             # colonna N = riga fittizia sopra il pelo libero
U[1:M - 1, 1:N] = 0.04               # valori di primo tentativo

iterazioni = 0
errore = 1.0
while errore > 0.001:
    iterazioni += 1
    errore = 0.0
    for i in range(1, M - 1):
        for j in range(1, N):        # j = N-1 e' il pelo libero
            vecchio = U[i, j]
            if j == N - 1:
                U[i, j + 1] = U[i, j - 1]          # nodo fittizio: dU/dY = 0
            U[i, j] = (1 - omega) * U[i, j] + omega / 4.0 * (
                H * H + U[i - 1, j] + U[i + 1, j] + U[i, j - 1] + U[i, j + 1])
            errore += abs(U[i, j] - vecchio)
U = U[:, :N]

# portata: regola dei trapezi in due dimensioni
pesi = np.ones((M, N))
pesi[0, :] *= 0.5
pesi[-1, :] *= 0.5
pesi[:, 0] *= 0.5
pesi[:, -1] *= 0.5
portata = np.sum(pesi * U) * H * H

print(f"omega = {omega:.4f};  {iterazioni} iterazioni")
print(f"portata adimensionale Q / (L^4 rho g sin(theta)/mu) = {portata:.6f}")
print(f"velocita' massima (centro del pelo libero) = {U[M // 2, N - 1]:.5f}")
print("\nVelocita' (righe dal pelo libero al fondo, una colonna ogni due):")
for j in range(N - 1, -1, -2):
    print("  " + " ".join(f"{U[i, j]:.4f}" for i in range(0, M, 2)))

# confronto con il canale indefinitamente largo (senza pareti laterali)
print(f"\nCanale senza pareti laterali: U_max = 0,5,  Q = 1/3")
print(f"   rapporti: U_max(largo)/U_max = {0.5 / U[M//2, N-1]:.2f},  Q(largo)/Q = {(1/3)/portata:.2f}")

# ------------------------------------------------------------------ figura
Z = np.linspace(0, 1, M)
Y = np.linspace(0, 1, N)
fig, ax = plt.subplots(figsize=(4.0, 3.8))
livelli = [0.0, 0.018, 0.036, 0.054, 0.072, 0.090, 0.108]
cs = ax.contour(Z, Y, U.T, levels=livelli[1:], colors=BLU, linewidths=1.0)
ax.clabel(cs, fmt=lambda v: f"{v:.3f}".replace(".", ","), fontsize=6.5)
cf = ax.contourf(Z, Y, U.T, levels=40, cmap="Blues", alpha=0.35)
ax.plot([0, 0, 1, 1], [1, 0, 0, 1], color="k", lw=2.0)        # pareti e fondo
ax.plot([0, 1], [1, 1], color=ROSSO, lw=1.0, ls="--")         # pelo libero
ax.text(0.5, 1.02, "pelo libero", ha="center", fontsize=7, color=ROSSO)
ax.set_aspect("equal")
ax.set_xlabel("$Z=z/L$")
ax.set_ylabel("$Y=y/L$")
ax.set_ylim(0, 1.07)
salva(fig, "prog3_3_canale")
