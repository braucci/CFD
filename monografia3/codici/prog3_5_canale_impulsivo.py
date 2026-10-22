"""
prog3_5_canale_impulsivo.py -- PROGRAMMA 3.5
Avviamento di una corrente in un canale piano per effetto di un gradiente di
pressione costante applicato bruscamente.

Con Y = y/L, T = t/(L^2/nu), U = u / [-(L^2/(2 mu)) dp/dx] e W = (1 - Y^2) - U:
    W_T = W_YY,   W(1, T) = 0,   W_Y(0, T) = 0 (simmetria),   W(Y, 0) = 1 - Y^2.
Metodo implicito (indietro nel tempo, centrato nello spazio):
    R W_(i-1)^j - (1 + 2R) W_i^j + R W_(i+1)^j = -W_i^(j-1),
sistema tridiagonale risolto con TRID a ogni passo; nodo fittizio W_0 = W_2.
Confronto con la soluzione esatta
    W = sum_m 4 (-1)^(m+1) / lam_m^3 cos(lam_m Y) exp(-lam_m^2 T),  lam_m = (2m - 1) pi/2.
"""
import numpy as np
import matplotlib.pyplot as plt
from metodi_numerici import trid
from stile_grafici import salva, PALETTE, GRIGIO

N = 20
R = 0.4
TMAX = 1.0
NPRINT, NPLOT = 50, 200
H = 1.0 / N
TAU = R * H**2

Y = np.linspace(0.0, 1.0, N + 1)
US = 1.0 - Y**2                       # profilo di regime (Poiseuille)
W = US.copy()


def W_esatta(Y, T, mmax=200):
    W = np.zeros_like(Y)
    for m in range(1, mmax + 1):
        lam = (2 * m - 1) * np.pi / 2
        W += 4 * (-1) ** (m + 1) / lam**3 * np.cos(lam * Y) * np.exp(-lam**2 * T)
    return W


C = np.zeros((N, 4))
t, j = 0.0, 0
profili = [(0.0, US - W)]
print(f"tau = R h^2 = {TAU}")
print("     T       U a Y = 0; 0,1; ...; 1")
print(f"  {t:6.3f}  " + " ".join(f"{v:.4f}" for v in (US - W)[::2]))
while t < TMAX - 1e-12:
    C[:, 0] = R
    C[:, 1] = -1.0 - 2.0 * R
    C[:, 2] = R
    C[:, 3] = -W[:N]
    C[0, 2] = C[0, 0] + C[0, 2]       # simmetria in Y = 0 (nodo fittizio)
    C[0, 0] = 0.0
    C[N - 1, 3] -= C[N - 1, 2] * 0.0  # W = 0 alla parete Y = 1
    C[N - 1, 2] = 0.0
    W[:N] = trid(C)
    W[N] = 0.0
    j += 1
    t += TAU
    U = US - W
    if j % NPRINT == 0:
        print(f"  {t:6.3f}  " + " ".join(f"{v:.4f}" for v in U[::2]))
    if j % NPLOT == 0:
        profili.append((t, U.copy()))

print(f"\n{j} passi temporali senza instabilita'.")
print(f"Velocita' al centro a T = 1: U = {U[0]:.4f}  (esatta {1 - W_esatta(np.array([0.0]), 1.0)[0]:.4f})")
for tp, up in profili[1:]:
    err = np.max(np.abs(up - (US - W_esatta(Y, tp))))
    print(f"   T = {tp:.1f}   max|U - U_es| = {err:.2e}")
L, nu = 0.01, 1.0e-6
print(f"\nAcqua in un canale alto 2 cm: T = 1 corrisponde a t = L^2/nu = {L**2/nu:.0f} s")
for frac in [0.9, 0.99]:
    # tempo per cui la velocita' al centro raggiunge frac del valore di regime
    Ts = np.linspace(0.01, 3, 3000)
    Uc = 1 - np.array([W_esatta(np.array([0.0]), T)[0] for T in Ts])
    print(f"   U_centro = {frac:.2f} per T = {Ts[np.argmax(Uc >= frac)]:.3f}")

# ------------------------------------------------------------------ figura
fig, ax = plt.subplots(figsize=(4.6, 3.4))
yy = np.linspace(0, 1, 201)
simboli = ["x", "o", "s", "d", "v"]
for k, (tp, up) in enumerate(profili):
    col = PALETTE[k % len(PALETTE)]
    ax.plot(1 - yy**2 - W_esatta(yy, tp), yy, color=col, lw=0.9)
    ax.plot(up, Y, simboli[k % 5], ms=3.5, mfc="none", color=col, label=f"$T={tp:.1f}$")
ax.plot(1 - yy**2, yy, "--", color=GRIGIO, label="regime")
ax.set_xlabel("velocita' adimensionale $U$")
ax.set_ylabel("quota adimensionale $Y$")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1)
ax.legend(fontsize=7)
salva(fig, "prog3_5_canale_impulsivo")
