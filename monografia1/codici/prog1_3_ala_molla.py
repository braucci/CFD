"""
prog1_3_ala_molla.py  --  Programma 1.3
Moto verticale di un'ala sostenuta da una molla in galleria del vento
(sistema aeroelastico a un grado di liberta', forma adimensionale).

    dZ/dT = V
    dV/dT = -(2 pi)^2 Z + beta c_l(alpha) U sqrt(U^2 + V^2)

    alpha = alpha0 - atan(V/U);   c_l = 2 pi alpha  per |alpha| <= 18 deg, 0 altrimenti
"""
import numpy as np
import matplotlib.pyplot as plt
from ode_solvers import integra
from stile_grafici import salva, PALETTE, BLU

# dati dimensionali e gruppi adimensionali
rhof, g, m, S, k = 1.22, 9.8, 3.0, 0.3, 980.0
alpha0 = np.radians(10.0)
beta = rhof * g * S / (2 * k)
t_rif = 2 * np.pi * np.sqrt(m / k)
l_rif = m * g / k
v_rif = l_rif / t_rif
print(f"beta = {beta:.5f}, t_rif = {t_rif:.3f} s, l_rif = {l_rif:.3f} m, v_rif = {v_rif:.4f} m/s")
print(f"U = 100  <->  u = {100*v_rif:.2f} m/s")

ALPHA_STALLO = np.radians(18.0)


def cl(alpha):
    return 2 * np.pi * alpha if abs(alpha) <= ALPHA_STALLO else 0.0


def sistema(U_di_T, V_raffica=lambda T: 0.0):
    """Restituisce f(T, s) per un vento orizzontale U(T) e un'eventuale raffica verticale."""
    def f(T, s):
        Z, V = s
        U = U_di_T(T)
        Vr = V + V_raffica(T)            # velocita' verticale relativa all'aria
        alpha = alpha0 - np.arctan2(Vr, U)
        return np.array([V, -(2 * np.pi) ** 2 * Z + beta * cl(alpha) * U * np.hypot(U, Vr)])
    return f


if __name__ == "__main__":
    h, Tmax = 0.01, 20.0
    n = int(Tmax / h)

    # ---------------- Fig. 1.3.4: cinque velocita' del vento
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for U, c in zip((25, 50, 75, 100, 125), PALETTE):
        T, S = integra(sistema(lambda T, U=U: U), 0.0, [0.0, 0.0], h, n)
        ax.plot(T, S[:, 0], color=c, label=f"$U={U}$")
        # posizione di equilibrio: (2pi)^2 Z = beta cl(alpha0) U^2
        Zeq = beta * cl(alpha0) * U ** 2 / (2 * np.pi) ** 2
        print(f"U = {U:3d}: Z(20) = {S[-1,0]:.4f}, Z_eq = {Zeq:.4f}, Z_max = {S[:,0].max():.4f}")
    ax.set_xlabel("$T$"); ax.set_ylabel("$Z$"); ax.legend(ncol=5, fontsize=7, loc="upper right")
    ax.set_ylim(0, 1.5)
    salva(fig, "prog1_3_vento")

    # ---------------- Fig. 1.3.5: raffica orizzontale U(1 + a sin(omega T))
    U0, a = 100.0, 0.2
    fig, axs = plt.subplots(3, 1, figsize=(5.2, 4.6), sharex=True)
    for ax, om in zip(axs, (2.0, 1.0, 0.5)):
        T, S = integra(sistema(lambda T, om=om: U0 * (1 + a * np.sin(om * T))), 0.0, [0, 0], h, n)
        ax.plot(T, S[:, 0], color=BLU)
        ax.set_ylabel("$Z$"); ax.set_title(f"$\\omega = {om}$", fontsize=8, loc="right")
        ax.set_ylim(0, 1.6)
    axs[-1].set_xlabel("$T$")
    salva(fig, "prog1_3_raffica_h")

    # ---------------- Fig. 1.3.6: raffica verticale -a U sin(omega T)
    fig, axs = plt.subplots(3, 1, figsize=(5.2, 4.6), sharex=True)
    for ax, om in zip(axs, (2.0, 1.0, 0.5)):
        T, S = integra(sistema(lambda T: U0, lambda T, om=om: -a * U0 * np.sin(om * T)),
                       0.0, [0, 0], h, n)
        ax.plot(T, S[:, 0], color=BLU)
        ax.set_ylabel("$Z$"); ax.set_title(f"$\\omega = {om}$", fontsize=8, loc="right")
        ax.set_ylim(-5, 5)
    axs[-1].set_xlabel("$T$")
    salva(fig, "prog1_3_raffica_v")
