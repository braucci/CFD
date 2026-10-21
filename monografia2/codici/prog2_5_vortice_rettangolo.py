"""
prog2_5_vortice_rettangolo.py -- PROGRAMMA 2.5
Equazione di Poisson  psi_xx + psi_yy = -zeta  per il flusso indotto da un
vortice concentrato in (x0, y0) all'interno di una parete rettangolare
(-3 <= x <= 3, -2 <= y <= 2), risolta con la formula iterativa di Liebmann
(Gauss-Seidel). Condizione al contorno: psi = 0 sulla parete.

Verifica: la soluzione discreta si confronta con la soluzione esatta del
problema continuo (funzione di Green del rettangolo in serie doppia di Fourier)
per un vortice puntiforme di circolazione Gamma = zeta0 h^2.
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, VERDE, VIOLA, ROSSO, GRIGIO
from metodi_numerici import gauss_seidel_passo, nodi_interni_rettangolo, cerca_xy

XMIN, XMAX, YMIN, YMAX, H = -3.0, 3.0, -2.0, 2.0, 0.25
ZETA0, X0, Y0 = 100.0, 1.0, 1.0
ITMAX, ERRMAX = 200, 0.001


def griglia(h=H):
    x = np.arange(XMIN, XMAX + 1e-9, h)
    y = np.arange(YMIN, YMAX + 1e-9, h)
    return x, y


def psi_serie(xp, yp, gamma, mmax=200):
    """Soluzione esatta: vortice puntiforme di circolazione gamma in (X0, Y0)
    dentro il rettangolo con psi = 0 sul contorno (serie doppia di Fourier)."""
    Lx, Ly = XMAX - XMIN, YMAX - YMIN
    m = np.arange(1, mmax + 1)[:, None]
    n = np.arange(1, mmax + 1)[None, :]
    km, kn = m * np.pi / Lx, n * np.pi / Ly
    coef = 4.0 * gamma / (Lx * Ly) / (km**2 + kn**2) \
        * np.sin(km * (X0 - XMIN)) * np.sin(kn * (Y0 - YMIN))
    return np.sum(coef * np.sin(km * (xp - XMIN)) * np.sin(kn * (yp - YMIN)))


if __name__ == "__main__":
    x, y = griglia()
    M, N = len(x), len(y)
    psi = np.zeros((M, N))
    psi[1:-1, 1:-1] = 0.5                     # valori di primo tentativo
    zeta = np.zeros((M, N))
    i0, j0 = int(round((X0 - XMIN) / H)), int(round((Y0 - YMIN) / H))
    zeta[i0, j0] = ZETA0
    interni = nodi_interni_rettangolo(M, N)

    it, err = 0, 1.0
    storia = []
    while err > ERRMAX and it < ITMAX:
        it += 1
        err = gauss_seidel_passo(psi, zeta, H, interni)
        storia.append(err)
    print(f"M = {M}, N = {N}, iterazioni = {it}, somma degli errori = {err:.3e}")

    # tabella ridotta: una colonna ogni 3 e una riga ogni 2, dall'alto in basso
    print("\npsi(i,j) (righe dall'alto, una ogni 2; colonne una ogni 3):")
    for j in range(N - 1, -1, -2):
        print("  " + " ".join(f"{psi[i, j]:7.4f}" for i in range(0, M, 3)))

    # confronto con la soluzione esatta del problema continuo
    gamma = ZETA0 * H**2
    print(f"\ncircolazione equivalente Gamma = zeta0 h^2 = {gamma:.4f}")
    print("   punto (x, y)      psi numerico   psi esatto (serie)")
    for xp, yp in ((-2.0, 0.0), (0.0, 0.0), (1.0, -1.0), (2.0, 1.0), (1.0, 0.0)):
        i, j = int(round((xp - XMIN) / H)), int(round((yp - YMIN) / H))
        print(f"   ({xp:+.1f}, {yp:+.1f})       {psi[i, j]:8.5f}      {psi_serie(xp, yp, gamma):8.5f}")

    # linee di corrente
    COST = [0.05, 0.2, 0.5, 1.0, 1.5]
    SIMB = ["o", "s", "D", "v", "^"]
    COL = [BLU, ARANCIO, VERDE, VIOLA, ROSSO]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for c, s, col in zip(COST, SIMB, COL):
        xx, yy = cerca_xy(x, y, psi, c)
        ax.plot(xx, yy, s, ms=3, color=col, mfc="none", label=fr"$\psi={c:g}$")
    ax.contour(x, y, psi.T, levels=COST, colors=GRIGIO, linewidths=0.5, linestyles="solid")
    ax.plot(X0, Y0, "+", color="k", ms=8)
    ax.plot([XMIN, XMAX, XMAX, XMIN, XMIN], [YMIN, YMIN, YMAX, YMAX, YMIN], "k", lw=1.5)
    ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.legend(loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.12), handletextpad=0.2)
    salva(fig, "prog2_5_vortice")
