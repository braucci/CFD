"""
prog2_10_maccormack.py -- PROGRAMMA 2.10
Equazione di convezione non lineare  u_t + (u^2/2)_x = 0  risolta con lo
schema esplicito di MacCormack (predittore in avanti, correttore all'indietro).
Condizione iniziale a gradino: u = 10 per x < 20, u = 0 per 20 <= x <= 40;
condizioni al contorno di Dirichlet u(0) = 10, u(40) = 0.
La soluzione esatta e' un urto che si muove con la velocita' di
Rankine-Hugoniot  s = (u_s + u_d)/2 = 5.
Si confrontano Co = 1, 0,75 e 1,1, con Co = u_max dt/dx.
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, ROSSO, GRIGIO, VERDE, VIOLA

UMAX, L, NX = 10.0, 40.0, 200


def maccormack(Co, t_fin, nx=NX, uL=UMAX, uR=0.0, istanti=()):
    """Integra fino a t_fin; restituisce x, la soluzione finale e un dizionario
    con le soluzioni agli istanti richiesti (primo passo che li supera)."""
    x = np.linspace(0.0, L, nx)
    dx = x[1] - x[0]
    u = np.where(x < 20.0, uL, uR).astype(float)
    t, salvate = 0.0, {}
    richiesti = sorted(istanti)
    while t < t_fin:
        umax = np.max(np.abs(u))
        dt = min(Co * dx / umax, t_fin - t)
        r = dt / dx
        F = 0.5 * u**2
        us = u.copy()
        us[:-1] = u[:-1] - r * (F[1:] - F[:-1])          # predittore (in avanti)
        Fs = 0.5 * us**2
        un = u.copy()
        un[1:-1] = 0.5 * (u[1:-1] + us[1:-1] - r * (Fs[1:-1] - Fs[:-2]))   # correttore
        un[0], un[-1] = uL, uR                            # Dirichlet
        u, t = un, t + dt
        while richiesti and t >= richiesti[0] - 1e-12:
            salvate[richiesti.pop(0)] = (t, u.copy())
        if not np.all(np.isfinite(u)) or np.max(np.abs(u)) > 1e3:
            break
    return x, u, t, salvate


if __name__ == "__main__":
    u0 = lambda x: np.where(x < 20.0, UMAX, 0.0)
    fig, axs = plt.subplots(3, 1, figsize=(5.6, 6.6), sharex=True)
    print("  Co    t      posizione dell'urto (u = 5)   esatta 20 + 5t   max u")
    casi = ((1.0, (1.0, 2.0, 3.0)), (0.75, (1.0, 2.0, 3.0)), (1.5, (0.10, 0.16)))
    for ax, (Co, istanti) in zip(axs, casi):
        x, u, t, sal = maccormack(Co, max(istanti), istanti=istanti)
        ax.plot(x, u0(x), color=GRIGIO, lw=1.0, label="$t=0$")
        for (tr, (ts, us)), col in zip(sorted(sal.items()), (BLU, ARANCIO, VERDE)):
            ax.plot(x, us, color=col, lw=1.1, label=f"$t={ts:.2f}$ s")
            k = np.argmin(np.abs(us - 5.0) + 100 * (x < 15))
            print(f"  {Co:4.2f}  {ts:4.2f}         {x[k]:6.2f}                {20+5*ts:6.2f}"
                  f"       {us.max():6.2f}")
        ax.set_ylim(-2, 14) if Co <= 1 else ax.set_ylim(-6, 30)
        ax.set_ylabel("$u$")
        ax.set_title(f"Co = {Co:g}", fontsize=8); ax.legend(loc="upper right", fontsize=7)
    axs[-1].set_xlabel("$x$")
    salva(fig, "prog2_10_maccormack")

    # Co = 1,1 e 1,2: l'instabilita' lineare non esplode perche' le perturbazioni
    # vengono trasportate dentro l'urto
    for Co in (1.1, 1.2):
        x, u, t, _ = maccormack(Co, 4.0)
        print(f"  Co = {Co}: dopo 4 s max u = {u.max():.2f}, min u = {u.min():.2f}")

    # analisi di von Neumann (equazione lineare u_t + c u_x = 0): G di Lax-Wendroff
    kh = np.linspace(0, np.pi, 300)
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for Co, col in ((0.5, VERDE), (0.75, ARANCIO), (1.0, BLU), (1.1, ROSSO)):
        G = 1 - 1j * Co * np.sin(kh) - Co**2 * (1 - np.cos(kh))
        ax[0].plot(kh, np.abs(G), color=col, label=f"Co = {Co:g}")
        fase = -np.angle(G)
        with np.errstate(invalid="ignore", divide="ignore"):
            ax[1].plot(kh[1:], fase[1:] / (Co * kh[1:]), color=col)
    for a in ax:
        a.set_xticks([0, np.pi / 2, np.pi]); a.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
        a.set_xlabel("$kh$")
    ax[0].set_ylabel("$|G|$"); ax[0].legend(fontsize=7)
    ax[1].set_ylabel(r"$c_{num}/c$")
    salva(fig, "prog2_10_vonneumann")
