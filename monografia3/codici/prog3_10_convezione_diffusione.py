"""
prog3_10_convezione_diffusione.py -- PROGRAMMA 3.10
Convezione-diffusione bidimensionale di una temperatura in un campo di
velocita' assegnato, periodico in y (periodo 2):
    T_t + u T_x + v T_y = (T_xx + T_yy) / (Pr Re),
    u = -sin(pi x) cos(2 pi y),   v = cos(pi x) cos(pi y),
    T = 0 per x = 0 (parete fredda),  T = 1 per x = 1 (parete calda).
Metodo misto: differenze centrate in x (direzione non periodica), derivate in y
con la trasformata di Fourier (direzione periodica), Eulero esplicito nel tempo.
Si integra fino allo stato stazionario per Pr = 0,71 (aria) e Pr = 25 e si
confronta il risultato con la soluzione stazionaria calcolata direttamente
(sistema lineare con la matrice di differenziazione di Fourier).
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from metodi_numerici import numeri_onda
from stile_grafici import salva, BLU, ROSSO

RE = 1.0
NX, NY = 41, 32                        # nodi in x (pareti incluse) e in y (un periodo)
LY = 2.0
x = np.linspace(0.0, 1.0, NX)
y = LY * np.arange(NY) / NY
dx = x[1] - x[0]
X, Y = np.meshgrid(x, y, indexing="ij")
U = -np.sin(np.pi * X) * np.cos(2 * np.pi * Y)
V = np.cos(np.pi * X) * np.cos(np.pi * Y)
k = numeri_onda(NY, LY)
k1 = k.copy()
k1[NY // 2] = 0.0                      # modo di Nyquist nullo per la derivata prima


def dy(T):
    return np.fft.ifft(1j * k1 * np.fft.fft(T, axis=1), axis=1).real


def dyy(T):
    return np.fft.ifft(-k**2 * np.fft.fft(T, axis=1), axis=1).real


def evolvi(Pr, dt, toll=1e-4, nmax=400000):
    """Eulero esplicito fino a max|dT/dt| < toll. Restituisce T, passi, tempo."""
    a = 1.0 / (Pr * RE)
    T = np.tile(x[:, None], (1, NY))   # condizione iniziale: profilo conduttivo
    for n in range(1, nmax + 1):
        Tx = np.zeros_like(T)
        Txx = np.zeros_like(T)
        Tx[1:-1] = (T[2:] - T[:-2]) / (2 * dx)
        Txx[1:-1] = (T[2:] - 2 * T[1:-1] + T[:-2]) / dx**2
        dTdt = -U * Tx - V * dy(T) + a * (Txx + dyy(T))
        dTdt[0] = dTdt[-1] = 0.0       # pareti a temperatura imposta
        T = T + dt * dTdt
        if np.max(np.abs(dTdt)) < toll:
            return T, n, n * dt
    raise RuntimeError("stato stazionario non raggiunto")


def stazionario(Pr):
    """Soluzione diretta di u T_x + v T_y = a lap T con le stesse discretizzazioni."""
    a = 1.0 / (Pr * RE)
    # matrice di differenziazione di Fourier (prima e seconda) applicata a vettori in y
    I = np.eye(NY)
    Dy = np.fft.ifft(1j * k1[:, None] * np.fft.fft(I, axis=0), axis=0).real
    Dyy = np.fft.ifft(-(k**2)[:, None] * np.fft.fft(I, axis=0), axis=0).real
    n = (NX - 2) * NY
    A = np.zeros((n, n))
    b = np.zeros(n)
    idx = lambda i, jj: (i - 1) * NY + jj
    for i in range(1, NX - 1):
        for jj in range(NY):
            r = idx(i, jj)
            # u T_x - a T_xx
            coef = {i - 1: -U[i, jj] / (2 * dx) - a / dx**2,
                    i: 2 * a / dx**2,
                    i + 1: U[i, jj] / (2 * dx) - a / dx**2}
            for ii, c in coef.items():
                if ii == 0:
                    b[r] -= c * 0.0
                elif ii == NX - 1:
                    b[r] -= c * 1.0
                else:
                    A[r, idx(ii, jj)] += c
            # v T_y - a T_yy
            A[r, idx(i, 0):idx(i, 0) + NY] += V[i, jj] * Dy[jj] - a * Dyy[jj]
    Tin = np.linalg.solve(A, b)
    T = np.zeros((NX, NY))
    T[-1] = 1.0
    T[1:-1] = Tin.reshape(NX - 2, NY)
    return T


if __name__ == "__main__":
    risultati = {}
    for Pr, dt in ((0.71, 1.0e-4), (25.0, 1.0e-3)):
        a = 1 / (Pr * RE)
        lim_diff = 2.0 / (a * (4 / dx**2 + np.max(k**2)))
        t0 = time.perf_counter()
        T, passi, tfin = evolvi(Pr, dt)
        durata = time.perf_counter() - t0
        Ts = stazionario(Pr)
        risultati[Pr] = T
        print(f"Pr = {Pr:5.2f}: dt = {dt:.0e} (limite diffusivo ~ {lim_diff:.1e}), {passi} passi,"
              f" t = {tfin:.2f}, {durata:.1f} s")
        print(f"           scarto dalla soluzione stazionaria diretta: {np.max(np.abs(T - Ts)):.1e};"
              f"  T(x = 0,5) fra {T[NX//2].min():.3f} e {T[NX//2].max():.3f}")
    print(f"\nNumero di Peclet Pe = Re Pr: {RE*0.71:.2f} e {RE*25:.0f}")
    div = -np.pi * np.cos(np.pi * X) * np.cos(2 * np.pi * Y) - np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y)
    print(f"Divergenza del campo assegnato: max |div V| = {np.max(np.abs(div)):.2f} (il campo non e' solenoidale)")

    # ------------------------------------------------------------------ figure
    xe = np.concatenate([y, [LY]])
    fig, ax = plt.subplots(1, 3, figsize=(7.8, 3.6))
    s = 2
    ax[0].quiver(X[::s, ::2], Y[::s, ::2], U[::s, ::2], V[::s, ::2], color=BLU, scale=25, width=0.006)
    ax[0].set_title("campo di velocita'")
    for a_, Pr in zip(ax[1:], (0.71, 25.0)):
        T = risultati[Pr]
        Te = np.concatenate([T, T[:, :1]], axis=1)
        cs = a_.contour(x, xe, Te.T, levels=np.linspace(0.05, 0.95, 10), cmap="coolwarm", linewidths=1.0)
        a_.clabel(cs, fmt=lambda v: f"{v:.2f}".replace(".", ","), fontsize=5.5)
        a_.set_title(f"isoterme, $Pr={Pr:g}$".replace(".", "{,}"))
    for a_ in ax:
        a_.set_xlim(0, 1)
        a_.set_ylim(0, 2)
        a_.set_aspect("equal")
        a_.set_xlabel("$x$")
    ax[0].set_ylabel("$y$")
    fig.tight_layout()
    salva(fig, "prog3_10_convezione_diffusione")
