"""
prog2_6_camera.py -- PROGRAMMA 2.6
Flusso potenziale in una camera rettangolare (6 x 4) con l'angolo superiore
destro chiuso da una parete a 45 gradi; ingresso e uscita sono due aperture
sulla parete inferiore. Equazione di Laplace per psi risolta con il metodo
del sovrarilassamento successivo (SOR) e, per confronto, con Liebmann.

Condizioni al contorno: psi = 0 sul tratto AB della parete inferiore compreso
fra le due aperture, psi = 1 sul resto della parete (portata unitaria).
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, ROSSO, GRIGIO, VERDE
from metodi_numerici import sor_passo, cerca_xy

H = 0.25
M, N = 25, 17                      # nodi lungo x (0..6) e lungo y (0..4)


def prepara():
    """Valori al contorno, valori di tentativo e lista dei nodi interni."""
    jlast = np.full(M, N - 1)                     # indice (da 0) del nodo di parete superiore
    for i in range(17, M):                        # parete inclinata: da (4,4) a (6,2)
        jlast[i] = (N - 1) - (i - 16)
    psi = np.full((M, N), np.nan)                 # NaN = fuori dal dominio
    for i in range(M):
        psi[i, :jlast[i] + 1] = 0.5               # tentativo nei nodi interni
        psi[i, jlast[i]] = 1.0                    # parete superiore / inclinata
    psi[0, :] = 1.0                               # parete sinistra
    psi[M - 1, :jlast[M - 1] + 1] = 1.0           # parete destra (y <= 2)
    psi[:, 0] = 1.0                               # parete inferiore ...
    psi[7:21, 0] = 0.0                            # ... tratto AB fra le aperture
    interni = [(i, j) for i in range(1, M - 1) for j in range(1, jlast[i])]
    return psi, interni, jlast


def risolvi(omega, tol=0.001, kmax=5000):
    psi, interni, jlast = prepara()
    zeta = np.zeros((M, N))
    it, err = 0, 1.0
    while err > tol and it < kmax:
        it += 1
        err = sor_passo(psi, zeta, H, interni, omega)
    return psi, it, jlast


if __name__ == "__main__":
    # omega ottimo con la formula del rettangolo (come nel programma originale
    # si usano i numeri di nodi M, N; con il numero di intervalli M-1, N-1 il
    # valore cambia di poco)
    alfa = np.cos(np.pi / M) + np.cos(np.pi / N)
    omega = (8 - 4 * np.sqrt(4 - alfa**2)) / alfa**2
    psi, it, jlast = risolvi(omega)
    _, it_gs, _ = risolvi(1.0)
    print(f"SOR  (omega = {omega:.4f}): {it} iterazioni")
    print(f"Liebmann (omega = 1)    : {it_gs} iterazioni")
    print("\n omega   iterazioni")
    for w in (1.0, 1.2, 1.4, 1.6, 1.65, 1.7, 1.75, 1.8, 1.9):
        print(f"  {w:.2f}    {risolvi(w)[1]:4d}")

    x = np.arange(M) * H
    y = np.arange(N) * H
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    livelli = np.round(np.arange(0.1, 1.0, 0.1), 2)
    ax.contour(x, y, psi.T, levels=livelli, colors=BLU, linewidths=0.9, linestyles="solid")
    for c in livelli:
        xx, yy = cerca_xy(x, y, psi, c)
        ax.plot(xx, yy, ".", ms=2.5, color=ARANCIO)
    # pareti
    ax.plot([1.5, 0, 0, 4, 6, 6, 5.25], [0, 0, 4, 4, 2, 0, 0], "k", lw=2)
    ax.plot([1.75, 5.0], [0, 0], color=ROSSO, lw=2)
    ax.annotate("ingresso", (1.62, -0.05), (1.0, -0.55), ha="center",
                arrowprops=dict(arrowstyle="->", color=GRIGIO), color=GRIGIO)
    ax.annotate("uscita", (5.12, -0.05), (5.6, -0.55), ha="center",
                arrowprops=dict(arrowstyle="<-", color=GRIGIO), color=GRIGIO)
    ax.text(3.3, 0.12, r"$\psi=0$ (A--B)", color=ROSSO, ha="center")
    ax.text(0.3, 3.6, r"$\psi=1$", color="k")
    ax.set_aspect("equal"); ax.set_xlim(-0.2, 6.2); ax.set_ylim(-0.75, 4.2)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.grid(False)
    salva(fig, "prog2_6_camera")
