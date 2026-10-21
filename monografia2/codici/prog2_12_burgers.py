"""
prog2_12_burgers.py -- equazione di Burgers viscosa adimensionale
        u_t + u u_x = u_xx ,     -6 <= x <= 6
con la soluzione esatta di Benton e Platzman
        u(x, t) = -2 sinh x / (cosh x - e^{-t})
(discontinua in x = 0 per t = 0; per t -> infinito tende a -2 tanh x).

Funzioni:
  esatta(x, t)
  maccormack_burgers(...)   schema esplicito (condizioni di Neumann)
  beam_warming_burgers(...) schema implicito (condizioni di Dirichlet)
Il programma principale disegna la soluzione esatta; gli schemi sono messi
alla prova negli esercizi svolti (esercizi_svolti.py, es_13 ed es_14).
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, ROSSO, GRIGIO, VERDE, VIOLA
from metodi_numerici import thomas

XL, XR, NX = -6.0, 6.0, 100


def esatta(x, t):
    return -2.0 * np.sinh(x) / (np.cosh(x) - np.exp(-t))


def griglia(nx=NX):
    return np.linspace(XL, XR, nx)


def maccormack_burgers(t_fin, nx=NX, dt=None, nu=1.0, istanti=()):
    """MacCormack: predittore in avanti, correttore all'indietro, diffusione
    centrata in entrambe le fasi; du/dx = 0 agli estremi (primo ordine)."""
    x = griglia(nx); dx = x[1] - x[0]
    u = esatta(x, 0.0)
    if dt is None:     # entrambe le condizioni: convettiva e viscosa
        dt = 0.9 * min(dx / np.max(np.abs(u)), 0.5 * dx**2 / nu)
    r, d = dt / dx, nu * dt / dx**2
    t, sal, rich = 0.0, {}, sorted(istanti)
    while t < t_fin - 1e-12:
        F = 0.5 * u**2
        us = u.copy()
        us[1:-1] = u[1:-1] - r * (F[2:] - F[1:-1]) + d * (u[2:] - 2 * u[1:-1] + u[:-2])
        us[0], us[-1] = us[1], us[-2]
        Fs = 0.5 * us**2
        un = u.copy()
        un[1:-1] = 0.5 * (u[1:-1] + us[1:-1] - r * (Fs[1:-1] - Fs[:-2])
                          + d * (us[2:] - 2 * us[1:-1] + us[:-2]))
        un[0], un[-1] = un[1], un[-2]          # Neumann: du/dx = 0
        u, t = un, t + dt
        while rich and t >= rich[0] - 1e-9:
            sal[rich.pop(0)] = (t, u.copy())
    return x, u, dt, sal


def beam_warming_burgers(t_fin, dt, nx=NX, nu=1.0, mu_e=0.0, istanti=()):
    """Beam-Warming (Crank-Nicolson + linearizzazione del flusso) con
    condizioni di Dirichlet u(-6) = +2, u(+6) = -2; dissipazione opzionale."""
    x = griglia(nx); dx = x[1] - x[0]
    u = esatta(x, 0.0)
    uL, uR = 2.0, -2.0
    u[0], u[-1] = uL, uR
    r, d = dt / (4 * dx), nu * dt / dx**2
    t, sal, rich = 0.0, {}, sorted(istanti)
    while t < t_fin - 1e-12:
        n = nx - 2
        a = -r * u[:-2] - 0.5 * d            # coefficiente di u_{i-1}^{n+1}
        b = np.full(n, 1.0 + d)
        c = r * u[2:] - 0.5 * d              # coefficiente di u_{i+1}^{n+1}
        rhs = u[1:-1] + 0.5 * d * (u[2:] - 2 * u[1:-1] + u[:-2])
        if mu_e > 0:
            D = np.zeros(nx)
            D[2:-2] = -mu_e / 8 * (u[4:] - 4 * u[3:-1] + 6 * u[2:-2] - 4 * u[1:-3] + u[:-4])
            rhs += D[1:-1]
        rhs[0] -= a[0] * uL
        rhs[-1] -= c[-1] * uR
        un = u.copy()
        un[1:-1] = thomas(a, b, c, rhs)
        u, t = un, t + dt
        while rich and t >= rich[0] - 1e-9:
            sal[rich.pop(0)] = (t, u.copy())
    return x, u, sal


if __name__ == "__main__":
    x = np.linspace(-4, 4, 801)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    with np.errstate(divide="ignore", invalid="ignore"):
        u0 = esatta(x, 0.0)
    u0[np.abs(u0) > 4.5] = np.nan
    ax.plot(x, u0, "--", color=GRIGIO, label="$t=0$")
    for t, col in ((0.2, VIOLA), (0.5, VERDE), (1.0, ARANCIO), (2.0, BLU)):
        ax.plot(x, esatta(x, t), color=col, label=f"$t={t:g}$")
    ax.plot(x, -2 * np.tanh(x), ":", color="k", lw=0.8, label=r"$-2\tanh x$")
    ax.set_ylim(-4, 4); ax.set_xlabel("$x$"); ax.set_ylabel("$u$"); ax.legend(fontsize=7)
    salva(fig, "prog2_12_burgers_esatta")
    for t in (0.2, 0.5, 1.0, 2.0):
        print(f"t = {t}: pendenza in x = 0  du/dx = {-2/(1-np.exp(-t)):.4f}")
