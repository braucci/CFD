"""
prog2_11_beam_warming.py -- PROGRAMMA 2.11
Equazione di convezione non lineare u_t + (u^2/2)_x = 0 risolta con lo schema
implicito di Beam e Warming (Crank-Nicolson + linearizzazione del flusso):

  -r u_{i-1}^n u_{i-1}^{n+1} + u_i^{n+1} + r u_{i+1}^n u_{i+1}^{n+1} = u_i^n + D_i,
  r = dt/(4 dx),
  D_i = -(mu_e/8)(u_{i+2} - 4u_{i+1} + 6u_i - 4u_{i-1} + u_{i-2})^n
        (dissipazione artificiale del quarto ordine).
Il sistema tridiagonale si risolve a ogni passo con l'algoritmo di Thomas.
Stessi dati del Programma 2.10, 101 nodi, Co = 1; mu_e = 0,8 e mu_e = 0.
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, ROSSO, GRIGIO, VERDE
from metodi_numerici import thomas

UMAX, L = 10.0, 40.0


def beam_warming(Co, mu_e, t_fin, nx=101, istanti=(), uL=UMAX, uR=0.0):
    x = np.linspace(0.0, L, nx)
    dx = x[1] - x[0]
    u = np.where(x < 20.0, uL, uR).astype(float)
    t, salvate, richiesti = 0.0, {}, sorted(istanti)
    while t < t_fin - 1e-12:
        dt = min(Co * dx / np.max(np.abs(u)), t_fin - t)
        r = dt / (4.0 * dx)
        n = nx - 2                                   # incognite: nodi 1..nx-2
        a = np.zeros(n); b = np.ones(n); c = np.zeros(n)
        a[1:] = -r * u[1:-2]                         # coefficiente di u_{i-1}^{n+1}
        c[:-1] = r * u[2:-1]                         # coefficiente di u_{i+1}^{n+1}
        d = u[1:-1].copy()
        D = np.zeros(nx)
        D[2:-2] = -mu_e / 8.0 * (u[4:] - 4 * u[3:-1] + 6 * u[2:-2] - 4 * u[1:-3] + u[:-4])
        d += D[1:-1]
        d[0] += r * u[0] * uL                        # termini noti dei nodi di contorno
        d[-1] -= r * u[-1] * uR
        un = np.empty(nx)
        un[1:-1] = thomas(a, b, c, d)
        un[0], un[-1] = uL, uR
        u, t = un, t + dt
        while richiesti and t >= richiesti[0] - 1e-9:
            salvate[richiesti.pop(0)] = (t, u.copy())
        if not np.all(np.isfinite(u)) or np.max(np.abs(u)) > 1e4:
            break
    return x, u, t, salvate


if __name__ == "__main__":
    fig, axs = plt.subplots(2, 1, figsize=(5.6, 4.8), sharex=True)
    print("  mu_e    t     posizione dell'urto   esatta    max u    min u")
    for ax, mu in zip(axs, (0.8, 0.0)):
        x, u, t, sal = beam_warming(1.0, mu, 3.6, istanti=(2.0, 3.6))
        ax.plot(x, np.where(x < 20, UMAX, 0), color=GRIGIO, lw=1.0, label="$t=0$")
        for (tr, (ts, us)), col in zip(sorted(sal.items()), (BLU, ARANCIO)):
            ax.plot(x, us, color=col, lw=1.1, label=f"$t={ts:.2f}$ s")
            k = np.argmin(np.abs(us - 5.0) + 100 * (x < 15))
            print(f"  {mu:4.2f}  {ts:4.2f}       {x[k]:6.2f}          {20+5*ts:6.2f}   "
                  f"{us.max():6.2f}   {us.min():6.2f}")
        ax.set_ylabel("$u$"); ax.legend(loc="upper right", fontsize=7)
        ax.set_title(fr"$\mu_e = {mu:g}$", fontsize=8)
    axs[1].set_xlabel("$x$")
    salva(fig, "prog2_11_beam_warming")
