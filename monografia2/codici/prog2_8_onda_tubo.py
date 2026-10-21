"""
prog2_8_onda_tubo.py -- PROGRAMMA 2.8
Propagazione e riflessione di un'ondina sinusoidale di piccola ampiezza in un
tubo lungo 1 m, chiuso a sinistra (u = 0) e aperto a destra (du/dx = 0).
Equazione delle onde  u_tt = a^2 u_xx  risolta con lo schema "leapfrog"

   u_i^{j+1} = 2 u_i^j + Co^2 (u_{i-1}^j - 2 u_i^j + u_{i+1}^j) - u_i^{j-1},
   Co = a tau / h  (numero di Courant).

Con Co = 1 lo schema diventa u_i^{j+1} = u_{i-1}^j + u_{i+1}^j - u_i^{j-1} ed e'
esatto. La funzione 'leapfrog' accetta un Co qualsiasi (usata da prog2_8b).
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, ROSSO, GRIGIO

A_SUONO = 340.0     # velocita' del suono [m/s]
L = 1.0             # lunghezza del tubo [m]


def f_iniziale(x):
    """Ondina di lunghezza d'onda 0,4 m e ampiezza 1 m/s fra x = 0,2 e 0,6 m."""
    return np.where((x >= 0.2) & (x <= 0.6), np.sin(5 * np.pi * (x - 0.2)), 0.0)


def g_iniziale(x):
    return np.zeros_like(x)


def leapfrog(h=0.02, Co=1.0, n_passi=115, f=f_iniziale, g=g_iniziale):
    """Restituisce x, i tempi e la matrice U[j, i] = u(x_i, t_j)."""
    m = int(round(L / h)) + 1
    x = np.linspace(0.0, L, m)
    tau = Co * h / A_SUONO
    C2 = Co**2
    U = np.zeros((n_passi + 1, m))
    U[0] = f(x)
    # formula di partenza (nodo fittizio u^{-1} = u^{1} - 2 tau g)
    U[1, 1:-1] = U[0, 1:-1] + 0.5 * C2 * (U[0, :-2] - 2 * U[0, 1:-1] + U[0, 2:]) \
        + tau * g(x)[1:-1]
    U[1, -1] = U[0, -1] + C2 * (U[0, -2] - U[0, -1]) + tau * g(x)[-1]   # u_{m+1} = u_{m-1}
    U[1, 0] = 0.0
    for j in range(1, n_passi):
        U[j + 1, 1:-1] = 2 * U[j, 1:-1] + C2 * (U[j, :-2] - 2 * U[j, 1:-1] + U[j, 2:]) \
            - U[j - 1, 1:-1]
        U[j + 1, -1] = 2 * U[j, -1] + 2 * C2 * (U[j, -2] - U[j, -1]) - U[j - 1, -1]
        U[j + 1, 0] = 0.0                        # estremo chiuso
    t = tau * np.arange(n_passi + 1)
    return x, t, U


def esatta(x, t, f=f_iniziale):
    """d'Alembert con estensione di f dispari rispetto a x=0 (estremo chiuso),
    pari rispetto a x=1 (estremo aperto), periodica di periodo 4L."""
    def F(s):
        s = np.mod(s + 2 * L, 4 * L) - 2 * L     # riporta in [-2L, 2L)
        segno = np.sign(s); segno[segno == 0] = 1
        sa = np.abs(s)
        val = np.where(sa <= L, f(sa), f(2 * L - sa))
        return segno * val
    return 0.5 * (F(x - A_SUONO * t) + F(x + A_SUONO * t))


if __name__ == "__main__":
    h = 0.02
    x, t, U = leapfrog(h, 1.0, 115)
    tau = t[1]
    print(f"m = {len(x)} nodi, tau = h/a = {tau:.4e} s")
    err = max(np.max(np.abs(U[j] - esatta(x, t[j]))) for j in range(len(t)))
    print(f"Co = 1: massimo scarto dalla soluzione di d'Alembert = {err:.2e}")

    # 24 fotogrammi, uno ogni 5 passi (serie temporale)
    fig, axs = plt.subplots(8, 3, figsize=(6.6, 9.2), sharex=True, sharey=True)
    for k, ax in enumerate(axs.T.flat):
        j = 5 * k
        ax.plot(x, U[j], color=BLU, lw=1.1)
        ax.axhline(0, color=GRIGIO, lw=0.4)
        ax.set_ylim(-1.1, 1.1); ax.set_yticks([-1, 0, 1]); ax.set_xticks([0, 0.5, 1])
        ax.text(0.98, 0.78, fr"$t={j}\tau$", transform=ax.transAxes, ha="right", fontsize=7)
        ax.axvline(0, color="k", lw=2.0)
    for ax in axs[-1]:
        ax.set_xlabel("$x$ [m]")
    fig.subplots_adjust(hspace=0.08, wspace=0.08)
    salva(fig, "prog2_8_serie")

    # diagramma x-t: le riflessioni seguono le caratteristiche x -+ a t = cost
    x2, t2, U2 = leapfrog(h, 1.0, 200)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    im = ax.pcolormesh(x2, t2 / tau, U2, cmap="RdBu_r", vmin=-0.6, vmax=0.6, shading="auto")
    fig.colorbar(im, ax=ax, label="$u$ [m/s]")
    ax.set_xlabel("$x$ [m]"); ax.set_ylabel(r"$t/\tau$")
    ax.set_title("estremo chiuso (sinistra) -- estremo aperto (destra)", fontsize=8)
    ax.grid(False)
    salva(fig, "prog2_8_xt")
