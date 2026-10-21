"""
prog2_9_onda_finita.py -- PROGRAMMA 2.9
Propagazione di un'onda di ampiezza finita in un tubo di 2 m, chiuso a
sinistra e aperto a destra: formazione di un'onda d'urto.
Equazioni per u (velocita') e a (velocita' del suono) in forma caratteristica:
   [d/dt + (u+a) d/dx] (u + 2a/(gamma-1)) = 0,
   [d/dt + (u-a) d/dx] (u - 2a/(gamma-1)) = 0,
risolte con lo schema di Courant, Isaacson e Rees (caratteristiche rettilinee
all'indietro + interpolazione lineare sulla riga di tempo precedente).
Confronto con la soluzione esatta dell'onda semplice prima dell'urto.
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, ROSSO, GRIGIO, VERDE

A0, H, GAMMA = 340.0, 0.02, 1.4
M = 101                                  # nodi fra x = 0 e x = 2 m
TAU = 0.5 * H / A0
R = TAU / H                               # rapporto tau/h
K = (GAMMA - 1.0) / 2.0                   # COEFF del programma originale
AMP = A0 / 2.0


def u_iniziale():
    i = np.arange(1, M + 1)               # numerazione del testo (1..M)
    u = np.zeros(M)
    u[(i >= 1) & (i <= 13)] = AMP * (i[(i >= 1) & (i <= 13)] - 1) / 12
    s = (i > 13) & (i <= 39); u[s] = AMP * (26 - i[s]) / 13
    s = (i > 39) & (i <= 51); u[s] = AMP * (i[s] - 51) / 12
    return u


def cir(n_passi, u0=None, verso=+1):
    """Schema di Courant-Isaacson-Rees. verso = +1: onda verso destra
    (a = a0 + K u), verso = -1: onda verso sinistra (a = a0 - K u)."""
    u = u_iniziale() if u0 is None else u0.copy()
    a = A0 + verso * K * u
    U = [u.copy()]; Asuono = [a.copy()]
    for n in range(n_passi):
        upa, uma = R * (u + a), R * (u - a)
        if np.any(np.abs(upa) > 1) or np.any(np.abs(uma) > 1):
            print(f"condizione di stabilita' violata al passo {n}")
            break
        un, an = np.empty(M), np.empty(M)
        # nodi interni: punti A (a sinistra) e B (a destra) sulla riga precedente
        uA = u[1:-1] + upa[1:-1] * (u[:-2] - u[1:-1])
        aA = a[1:-1] + upa[1:-1] * (a[:-2] - a[1:-1])
        uB = u[1:-1] - uma[1:-1] * (u[2:] - u[1:-1])
        aB = a[1:-1] - uma[1:-1] * (a[2:] - a[1:-1])
        un[1:-1] = 0.5 * ((uA + uB) + (aA - aB) / K)
        an[1:-1] = 0.5 * (K * (uA - uB) + (aA + aB))
        # estremo chiuso (x = 0): u = 0 e invariante J- dal punto B
        uB0 = u[0] - uma[0] * (u[1] - u[0]); aB0 = a[0] - uma[0] * (a[1] - a[0])
        un[0] = 0.0
        an[0] = aB0 - K * uB0
        # estremo aperto (x = L): a = a0 e invariante J+ dal punto A
        uAm = u[-1] + upa[-1] * (u[-2] - u[-1]); aAm = a[-1] + upa[-1] * (a[-2] - a[-1])
        an[-1] = A0
        un[-1] = uAm + (aAm - A0) / K
        u, a = un, an
        U.append(u.copy()); Asuono.append(a.copy())
    return np.array(U), np.array(Asuono)


def onda_semplice(x0, u0, t):
    """Soluzione esatta prima dell'urto: ogni valore u0(xi) viaggia lungo la
    caratteristica rettilinea x = xi + (a0 + (gamma+1)/2 u0) t."""
    return x0 + (A0 + 0.5 * (GAMMA + 1) * u0) * t, u0


if __name__ == "__main__":
    x = H * np.arange(M)
    U, Asu = cir(55)
    print(f"tau = {TAU:.4e} s;  max Co = max(|u|+a) tau/h = {np.max((np.abs(U)+Asu))*R:.3f}")
    dudx0 = -AMP / (13 * H)
    t_urto = 1.0 / (0.5 * (GAMMA + 1) * abs(dudx0))
    print(f"pendenza iniziale nella zona di compressione = {dudx0:.1f} 1/s")
    print(f"istante di formazione dell'urto (teoria)     = {t_urto:.4e} s = {t_urto/TAU:.1f} tau")

    pend = np.max(-np.diff(U, axis=1) / H, axis=1)
    print("\n  passo   max(-du/dx) [1/s]")
    for n in (0, 10, 20, 30, 35, 40, 45, 50, 55):
        print(f"   {n:3d}     {pend[n]:9.1f}")

    # fotogrammi, con la soluzione esatta dell'onda semplice prima dell'urto
    u0 = u_iniziale()
    fig, axs = plt.subplots(6, 2, figsize=(6.6, 8.0), sharex=True, sharey=True)
    for k, ax in enumerate(axs.T.flat):
        n = 5 * k
        ax.plot(x, U[n], color=BLU, lw=1.1)
        if n * TAU < t_urto:
            xe, ue = onda_semplice(x[:51], u0[:51], n * TAU)
            ax.plot(xe, ue, "--", color=ARANCIO, lw=0.9)
        ax.axhline(0, color=GRIGIO, lw=0.4)
        ax.set_ylim(-190, 190); ax.set_yticks([-150, 0, 150]); ax.set_xticks([0, 1, 2])
        ax.text(0.98, 0.78, fr"$t={n}\tau$", transform=ax.transAxes, ha="right", fontsize=7)
    for ax in axs[-1]:
        ax.set_xlabel("$x$ [m]")
    axs[0, 0].set_ylabel("$u$ [m/s]")
    fig.subplots_adjust(hspace=0.08, wspace=0.08)
    salva(fig, "prog2_9_serie")

    # diagramma delle caratteristiche C+ (onda semplice) e pendenza massima
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.0))
    tt = np.linspace(0, 1.3 * t_urto, 2)
    for i in range(0, 51, 2):
        xi, _ = onda_semplice(x[i], u0[i], tt)
        col = ROSSO if 13 <= i <= 39 else BLU
        ax[0].plot(xi, tt / TAU, color=col, lw=0.6)
    ax[0].axhline(t_urto / TAU, color=GRIGIO, ls="--", lw=0.8)
    ax[0].text(0.05, t_urto / TAU + 1, r"$t_u$", color=GRIGIO)
    ax[0].set_xlabel("$x$ [m]"); ax[0].set_ylabel(r"$t/\tau$"); ax[0].set_xlim(0, 2)
    ax[1].plot(np.arange(len(pend)), pend, color=BLU)
    ax[1].axvline(t_urto / TAU, color=GRIGIO, ls="--", lw=0.8)
    ax[1].set_xlabel(r"$t/\tau$"); ax[1].set_ylabel(r"$\max(-\partial u/\partial x)$ [1/s]")
    salva(fig, "prog2_9_caratteristiche")
