"""
prog2_2b_flussi_elementari.py -- i quattro flussi elementari e il cilindro
con circolazione.

1) Linee di corrente di corrente uniforme, sorgente, vortice e doppietta.
2) Cilindro circolare: corrente uniforme + doppietta + vortice. Per diversi
   valori della circolazione si traccia il campo, si calcola C_p sulla
   superficie e si integra la pressione: la resistenza e' nulla (paradosso di
   d'Alembert) e la portanza coincide con quella di Kutta-Joukowski, L = rho U Gamma.
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, ROSSO, GRIGIO, VERDE

U = 1.0

# ----------------------------------------------------------------- 1) galleria
x = np.linspace(-2, 2, 401)
X, Y = np.meshgrid(x, x, indexing="ij")
R2 = X**2 + Y**2
with np.errstate(divide="ignore", invalid="ignore"):
    flussi = {
        r"(a) corrente uniforme, $\alpha=30^\circ$":
            (U * (Y * np.cos(np.pi / 6) - X * np.sin(np.pi / 6)), np.linspace(-3, 3, 13)),
        r"(b) sorgente, $\sigma>0$":
            (np.arctan2(Y, X) / np.pi, np.linspace(-1, 1, 17)),
        r"(c) vortice, $\Gamma>0$ (orario)":
            (np.log(np.sqrt(R2)), np.linspace(-2.0, 0.6, 14)),
        r"(d) doppietta, $\kappa>0$":
            (-Y / R2, np.array([-8, -4, -2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2, 4, 8])),
    }
fig, axs = plt.subplots(2, 2, figsize=(6.0, 6.0))
for ax, (titolo, (campo, livelli)) in zip(axs.flat, flussi.items()):
    ax.contour(X, Y, campo, levels=livelli, colors=BLU, linewidths=0.8,
               linestyles="solid")
    ax.plot(0, 0, "o", color=ROSSO, ms=3) if "uniforme" not in titolo else None
    ax.set_aspect("equal"); ax.set_title(titolo)
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
    ax.grid(False)
# frecce indicative del verso del moto
axs[0, 0].annotate("", (1.2, 0.69), (0.2, 0.115), arrowprops=dict(arrowstyle="->", color=ROSSO))
for ang in np.radians([0, 90, 180, 270]):
    axs[0, 1].annotate("", (1.6 * np.cos(ang), 1.6 * np.sin(ang)),
                       (0.9 * np.cos(ang), 0.9 * np.sin(ang)),
                       arrowprops=dict(arrowstyle="->", color=ROSSO))
axs[1, 0].annotate("", (0.0, -1.2), (-0.05, 1.2), arrowprops=dict(arrowstyle="-", color="none"))
t = np.linspace(np.pi * 0.6, np.pi * 0.1, 30)
axs[1, 0].plot(1.35 * np.cos(t), 1.35 * np.sin(t), color=ROSSO, lw=1)
axs[1, 0].annotate("", (1.35 * np.cos(t[-1]) + 0.05, 1.35 * np.sin(t[-1]) - 0.12),
                   (1.35 * np.cos(t[-2]), 1.35 * np.sin(t[-2])),
                   arrowprops=dict(arrowstyle="->", color=ROSSO))
salva(fig, "prog2_2b_elementari")


# ------------------------------------------------- 2) cilindro con circolazione
A = 1.0                                   # raggio del cilindro
GAMMA = [0.0, 2.0 * np.pi, 4.0 * np.pi, 5.0 * np.pi]   # circolazioni (orarie)


def psi_cilindro(X, Y, gam):
    r = np.hypot(X, Y)
    return U * Y * (1 - A**2 / r**2) + gam / (2 * np.pi) * np.log(r / A)


xg = np.linspace(-3, 3, 481)
XG, YG = np.meshgrid(xg, xg, indexing="ij")
fig, axs = plt.subplots(2, 2, figsize=(6.0, 6.0))
theta = np.linspace(0, 2 * np.pi, 2001)
print(" Gamma/(U a)  sin(theta_s)    C_D (integr.)   C_L (integr.)   C_L = Gamma/(U a)")
for ax, gam in zip(axs.flat, GAMMA):
    with np.errstate(divide="ignore", invalid="ignore"):
        P = psi_cilindro(XG, YG, gam)
    P[np.hypot(XG, YG) < A] = np.nan
    ax.contour(XG, YG, P, levels=np.linspace(-3, 3, 25), colors=BLU,
               linewidths=0.7, linestyles="solid")
    ax.fill(A * np.cos(theta), A * np.sin(theta), color=GRIGIO, alpha=0.35)
    # velocita' tangenziale sulla superficie (theta antiorario da x positivo):
    # u_theta = -2 U sin(theta) - Gamma / (2 pi a)   (circolazione oraria > 0)
    ut = -2 * U * np.sin(theta) - gam / (2 * np.pi * A)
    cp = 1 - (ut / U)**2
    # forza per unita' di apertura: F = -int p n ds, n = (cos, sin)
    dth = theta[1] - theta[0]
    cx = -np.trapezoid(cp * np.cos(theta), dx=dth) * A / (2 * A)   # riferita al diametro
    cy = -np.trapezoid(cp * np.sin(theta), dx=dth) * A / (2 * A)
    s = gam / (4 * np.pi * U * A)
    print(f"   {gam/(U*A):7.4f}      {s:6.3f}        {cx:+.2e}       {cy:8.5f}        {gam/(U*A):8.5f}")
    if s <= 1:
        th_s = np.arcsin(-s)
        ax.plot(A * np.cos([th_s, np.pi - th_s]), A * np.sin([th_s, np.pi - th_s]),
                "o", color=ROSSO, ms=4)
    else:
        # punto di ristagno libero sull'asse y negativo: u_theta = 0 fuori dal cilindro
        rr = (gam / (4 * np.pi * U) + np.sqrt((gam / (4 * np.pi * U))**2 - A**2))
        ax.plot(0, -rr, "o", color=ROSSO, ms=4)
    ax.set_aspect("equal"); ax.set_xticks([-3, 0, 3]); ax.set_yticks([-3, 0, 3])
    ax.set_title(fr"$\Gamma/(4\pi U a) = {s:.2f}$"); ax.grid(False)
salva(fig, "prog2_2b_cilindro")
