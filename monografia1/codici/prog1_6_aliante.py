"""
prog1_6_aliante.py  --  Programma 1.6
Traiettoria di un aliante a portanza e resistenza costanti (forma adimensionale).

    dX/dT = U,  dU/dT = -A (U^2+V^2)^{1/2} (B U + V)
    dY/dT = V,  dV/dT = -1 + A (U^2+V^2)^{1/2} (U - B V)

A = c_l (1/2 rho w0^2 S)/(m g)  (portanza iniziale / peso),   B = c_d / c_l.
"""
import numpy as np
import matplotlib.pyplot as plt
from ode_solvers import integra
from stile_grafici import salva, BLU, ARANCIO, VERDE, VIOLA


def aliante(A, B):
    def f(T, s):
        X, Y, U, V = s
        W = np.hypot(U, V)
        return np.array([U, V, -A * W * (B * U + V), -1.0 + A * W * (U - B * V)])
    return f


if __name__ == "__main__":
    A, B, h, W0 = 1.5, 0.06, 0.05, 1.0
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for th0, c, ls, lab in ((-90.0, BLU, "-", "$\\theta_0 = -90^\\circ$"),
                            (180.0, ARANCIO, "-.", "$\\theta_0 = 180^\\circ$")):
        th = np.radians(th0)
        T, S = integra(aliante(A, B), 0.0, [0, 0, W0 * np.cos(th), W0 * np.sin(th)], h, 200)
        ax.plot(S[:, 0], S[:, 1], ls, color=c, label=lab)
        W = np.hypot(S[:, 2], S[:, 3])
        print(f"theta0 = {th0:5.0f}: X(T=10) = {S[-1,0]:.3f}, Y = {S[-1,1]:.3f}; "
              f"W min = {W.min():.3f}, W max = {W.max():.3f}")
    ax.set_xlabel("$X$"); ax.set_ylabel("$Y$"); ax.set_aspect("equal")
    ax.set_xlim(-1, 5); ax.set_ylim(-2, 1.2)
    ax.legend(loc="lower right")
    salva(fig, "prog1_6_aliante")

    # Caso "fisico" di riferimento: volo planato rettilineo di equilibrio
    # a velocita' costante: la traiettoria tende all'angolo di planata gamma con tan(gamma) = B
    gamma = np.degrees(np.arctan(B))
    print(f"Angolo di planata di equilibrio: tan(gamma) = B -> gamma = {gamma:.2f} deg; "
          f"efficienza E = 1/B = {1/B:.1f}")

    # effetto di B: senza resistenza (B = 0) il moto non si smorza
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for Bv, c in ((0.0, VERDE), (0.03, VIOLA), (0.06, BLU), (0.12, ARANCIO)):
        th = np.radians(-90.0)
        T, S = integra(aliante(A, Bv), 0.0, [0, 0, 0.0, -W0], h, 400)
        ax.plot(S[:, 0], S[:, 1], color=c, label=f"$B = {Bv}$")
    ax.set_xlabel("$X$"); ax.set_ylabel("$Y$"); ax.set_aspect("equal")
    ax.set_xlim(-0.5, 10.5)
    ax.legend(fontsize=7, loc="lower left")
    salva(fig, "prog1_6_effettoB")
