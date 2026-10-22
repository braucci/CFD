"""
prog3_8_orr_sommerfeld.py -- PROGRAMMA 3.8
Stabilita' lineare della corrente di Poiseuille piana U = 1 - y^2:
equazione di Orr-Sommerfeld risolta con la collocazione di Chebyshev.

Con lam = -i c (c velocita' complessa dell'onda) l'equazione diventa il problema
agli autovalori generalizzato  A v = lam B v  con
    A = (D^4 - 2 alfa^2 D^2 + alfa^4 I)/(alfa Re) - 2 i I - i diag(1 - y^2)(D^2 - alfa^2 I),
    B = D^2 - alfa^2 I.
Condizioni v = v' = 0 in y = +-1: si sostituiscono la prima e l'ultima riga di A
con le righe dell'identita' e la seconda e la penultima con la prima e l'ultima
riga di D (le righe corrispondenti di B si annullano: gli autovalori
"infiniti" che ne derivano vengono scartati).
Poiche' lam = -i c = c_I - i c_R, la parte reale lam_R coincide con c_I: se e'
positiva la perturbazione cresce come exp(alfa c_I t) (onda di Tollmien-Schlichting).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from metodi_numerici import cheb
from stile_grafici import salva, BLU, ROSSO, GRIGIO


def orr_sommerfeld(Re, alfa, N=100):
    D, y = cheb(N)
    D2 = D @ D
    D4 = D2 @ D2
    I = np.eye(N + 1)
    A = (D4 - 2 * alfa**2 * D2 + alfa**4 * I) / (Re * alfa) - 2j * I \
        - 1j * np.diag(1 - y**2) @ (D2 - alfa**2 * I)
    B = (D2 - alfa**2 * I).astype(complex)
    A[0, :] = 0.0
    A[0, 0] = 1.0                    # v(+1) = 0
    A[1, :] = D[0, :]                # v'(+1) = 0
    A[N, :] = 0.0
    A[N, N] = 1.0                    # v(-1) = 0
    A[N - 1, :] = D[N, :]            # v'(-1) = 0
    B[[0, 1, N - 1, N], :] = 0.0
    lam, V = eig(A, B)
    ok = np.isfinite(lam) & (np.abs(lam) < 1e3)
    return lam[ok], V[:, ok], y


if __name__ == "__main__":
    risultati = {}
    for Re in (5772.0, 7500.0):
        lam, V, y = orr_sommerfeld(Re, 1.02)
        j = np.argmax(lam.real)
        risultati[Re] = (lam, V[:, j], y)
        c = 1j * lam[j]                              # c = i lam
        print(f"Re = {Re:6.0f}, alfa = 1,02:  (lam_R)max = {lam[j].real:+.10f}")
        print(f"     modo meno stabile: c = c_R + i c_I = {c.real:.6f} {c.imag:+.6f} i "
              f"(velocita' di fase {c.real:.4f} U_c)")
    # sensibilita' alla risoluzione
    print("\nConvergenza in N (Re = 5772, alfa = 1,02):")
    for N in (40, 60, 80, 100, 140):
        lam, _, _ = orr_sommerfeld(5772.0, 1.02, N)
        print(f"   N = {N:4d}   (lam_R)max = {lam.real.max():+.3e}   lam = {lam[np.argmax(lam.real)]:.8f}")

    # ------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for a, Re in zip(ax, (5772.0, 7500.0)):
        lam = risultati[Re][0]
        a.plot(lam.real, lam.imag, ".", ms=4, color=BLU)
        jm = np.argmax(lam.real)
        a.plot(lam.real[jm], lam.imag[jm], "o", ms=7, mfc="none", color=ROSSO)
        a.axvline(0, color=GRIGIO, lw=0.8)
        a.set_xlim(-1, 0.2)
        a.set_ylim(-1, 0)
        a.set_aspect("equal")
        a.set_xlabel(r"$\lambda_R$")
        a.set_ylabel(r"$\lambda_I$")
        a.set_title(f"$Re={Re:.0f}$, $\\alpha=1{{,}}02$:  $(\\lambda_R)_{{\\max}}={lam.real.max():+.2e}$",
                    fontsize=8)
    fig.tight_layout()
    salva(fig, "prog3_8_spettro")

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    lam, v, y = risultati[7500.0]
    D, _ = cheb(len(y) - 1)
    alfa = 1.02
    u = 1j * (D @ v) / alfa                      # continuita': i alfa u + v' = 0
    scala = np.max(np.abs(v))
    c_R = (1j * lam[np.argmax(lam.real)]).real
    y_c = np.sqrt(1 - c_R)                       # strato critico: U(y_c) = c_R
    ax.plot(y, np.abs(v) / scala, color=BLU, label=r"$|\tilde v|$")
    ax.plot(y, np.abs(u) / scala, "--", color=ROSSO, label=r"$|\tilde u|$")
    for s in (-1, 1):
        ax.axvline(s * y_c, color=GRIGIO, ls=":", lw=0.9)
    ax.text(y_c - 0.02, 0.05, "strato\ncritico", fontsize=6.5, color=GRIGIO, ha="right")
    ax.set_xlabel("$y$")
    ax.set_ylabel("ampiezza (scala di $|\\tilde v|_{\\max}$)")
    ax.set_title(r"onda di Tollmien-Schlichting, $Re=7500$, $\alpha=1{,}02$")
    ax.legend(fontsize=7)
    salva(fig, "prog3_8_autofunzione")
    print(f"strato critico: U(y_c) = c_R per y_c = +-{y_c:.4f};  |u|max/|v|max = {np.abs(u).max()/scala:.2f}")
