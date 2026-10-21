"""
prog1_2_pendolo.py  --  Programma 1.2
Pendolo semplice con sfera immersa in un fluido (vuoto, aria, acqua).

    dy/dt = v                      (y = l*theta, spostamento lungo l'arco)
    dv/dt = [ -B sin(y/l) - C v|v| c_d(Re) ] / A
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk
from ode_solvers import integra, cd_sfera
from stile_grafici import salva, BLU, ARANCIO, VERDE

G = 9.8
d, rho, l = 0.01, 2500.0, 2.0          # sfera di vetro, filo di 2 m
theta0 = 45.0                           # gradi
tmax, h = 20.0, 0.1


def pendolo(rhof, nu):
    """Restituisce t e theta(t) [gradi] per un fluido di densita' rhof e viscosita' nu."""
    rb = rhof / rho
    A, B, C = 1 + rb / 2, (1 - rb) * G, 3 * rb / (4 * d)

    def f(t, s):
        y, v = s
        cd = cd_sfera(v * d / nu) if nu > 0 else 0.0
        return np.array([v, (-B * np.sin(y / l) - C * v * abs(v) * cd) / A])

    y0 = l * np.radians(theta0)
    T, S = integra(f, 0.0, [y0, 0.0], h, int(tmax / h))
    return T, np.degrees(S[:, 0] / l), S[:, 1]


if __name__ == "__main__":
    casi = (("nel vuoto", 0.0, 0.0, BLU), ("in aria", 1.22, 1.49e-5, ARANCIO),
            ("in acqua", 1000.0, 1.0e-6, VERDE))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for nome, rhof, nu, c in casi:
        T, TH, V = pendolo(rhof, nu)
        ax.plot(T, TH, color=c, label=nome)
        # periodo stimato dagli attraversamenti di theta = 0 con v<0 -> v>0 ... usiamo i massimi
        idx = [i for i in range(1, len(TH) - 1) if TH[i] > TH[i - 1] and TH[i] >= TH[i + 1]]
        if len(idx) >= 2:
            print(f"{nome:10s}: primo periodo ~ {T[idx[1]] - T[idx[0]]:.2f} s, "
                  f"ampiezza 2o massimo {TH[idx[1]]:.2f} deg")
        # istante del primo passaggio per la verticale
        i0 = np.where(TH <= 0)[0][0]
        print(f"{'':10s}  primo passaggio per theta=0 a t ~ {T[i0]:.2f} s")
    ax.set_xlabel("$t$ [s]"); ax.set_ylabel("$\\theta$ [gradi]")
    ax.set_ylim(-50, 50); ax.legend(ncol=3, loc="upper right")
    salva(fig, "prog1_2_pendolo")

    # periodo esatto nel vuoto (integrale ellittico completo di prima specie)
    k = np.sin(np.radians(theta0) / 2)
    T_lin = 2 * np.pi * np.sqrt(l / G)
    T_ex = 4 * np.sqrt(l / G) * ellipk(k ** 2)
    print(f"\nPeriodo linearizzato 2 pi sqrt(l/g) = {T_lin:.3f} s;  "
          f"periodo esatto a {theta0} deg = {T_ex:.3f} s  (+{(T_ex/T_lin-1)*100:.1f} %)")

    # Problema 3.1: periodo in funzione dell'ampiezza
    amp = np.linspace(5, 170, 34)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(amp, [4 * np.sqrt(l / G) * ellipk(np.sin(np.radians(a) / 2) ** 2) / T_lin
                  for a in amp], color=BLU)
    ax.set_xlabel("ampiezza $\\theta_0$ [gradi]")
    ax.set_ylabel("$T/T_{\\rm lin}$")
    salva(fig, "prog1_2_periodo")
