"""
prog3_7a_benard.py -- stabilita' marginale della convezione di Rayleigh-Benard
con le differenze finite: problema agli autovalori generalizzato  A x = Ra B x.

Incognite: theta_1..theta_n e w_1..w_n (ampiezze delle perturbazioni di
temperatura e di velocita' verticale sui nodi z_i = (i-1) h, h = 1/(n-1)).
Equazioni (neutralita', sigma = 0):
    (D^2 - k^2) theta + w = 0                       (energia)
    (D^2 - k^2)^2 w - k^2 Ra theta = 0              (quantita' di moto)
Pareti rigide (z = 0, 1):  theta = 0,  w = 0,  Dw = 0 (formula unilatera del
secondo ordine, che sostituisce l'equazione di w nel primo nodo interno).
Per validare il codice si risolve anche il caso con pareti libere
(theta = w = D^2 w = 0, nodo fittizio w_(-1) = -w_1), la cui soluzione esatta e'
    Ra = (pi^2 + k^2)^3 / k^2,   minimo 27 pi^4 / 4 per k = pi / sqrt(2).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from stile_grafici import salva, BLU, ARANCIO, ROSSO, GRIGIO


def matrici(n, k, pareti="rigide"):
    """Restituisce A, B (2n x 2n) con l'ordine x = (theta_1..theta_n, w_1..w_n)."""
    h = 1.0 / (n - 1)
    a1, a2 = 1 / h**2, -(2 / h**2 + k**2)
    b1 = 1 / h**4
    b2 = -(4 / h**4 + 2 * k**2 / h**2)
    b3 = 6 / h**4 + 4 * k**2 / h**2 + k**4
    A = np.zeros((2 * n, 2 * n))
    B = np.zeros((2 * n, 2 * n))
    T, W = 0, n                                     # offset dei due blocchi
    # --- equazione dell'energia (righe 0..n-1)
    A[T + 0, T + 0] = 1.0                           # theta = 0 in z = 0
    A[T + n - 1, T + n - 1] = 1.0                   # theta = 0 in z = 1
    for i in range(1, n - 1):
        A[T + i, T + i - 1] = a1
        A[T + i, T + i] = a2
        A[T + i, T + i + 1] = a1
        A[T + i, W + i] = 1.0
    # --- equazione della quantita' di moto (righe n..2n-1)
    A[W + 0, W + 0] = 1.0                           # w = 0 in z = 0
    A[W + n - 1, W + n - 1] = 1.0                   # w = 0 in z = 1
    if pareti == "rigide":                          # Dw = 0 (unilatera, O(h^2))
        A[W + 1, W + 0:W + 3] = [-3 / (2 * h), 2 / h, -1 / (2 * h)]
        A[W + n - 2, W + n - 3:W + n] = [1 / (2 * h), -2 / h, 3 / (2 * h)]
        interni = range(2, n - 2)
    else:                                           # D^2 w = 0: nodo fittizio w_(-1) = -w_1
        interni = range(1, n - 1)
    for i in interni:
        for d, c in zip(range(-2, 3), [b1, b2, b3, b2, b1]):
            jj = i + d
            if jj < 0:                              # fittizio sotto la parete inferiore
                A[W + i, W - jj] -= c
            elif jj > n - 1:                        # fittizio sopra la parete superiore
                A[W + i, W + 2 * (n - 1) - jj] -= c
            else:
                A[W + i, W + jj] += c
        B[W + i, T + i] = k**2
    return A, B


def Ra_minimo(n, k, pareti="rigide", vettori=False):
    A, B = matrici(n, k, pareti)
    lam, X = eig(A, B)
    ok = np.isfinite(lam) & (np.abs(lam.imag) < 1e-6 * np.abs(lam)) & (lam.real > 0)
    idx = np.nonzero(ok)[0]
    j = idx[np.argmin(lam[idx].real)]
    if vettori:
        return lam[j].real, X[:, j].real
    return lam[j].real


if __name__ == "__main__":
    print("1) Validazione con pareti libere, k = pi/sqrt(2) (esatto 27 pi^4/4 = %.4f):" % (27 * np.pi**4 / 4))
    kc_l = np.pi / np.sqrt(2)
    prec = None
    for n in [11, 21, 41, 81, 161]:
        r = Ra_minimo(n, kc_l, "libere")
        err = r - 27 * np.pi**4 / 4
        ordine = "" if prec is None else f"  ordine {np.log2(abs(prec)/abs(err)):.2f}"
        print(f"   n = {n:4d}   Ra = {r:10.4f}   errore = {err:+.3e}{ordine}")
        prec = err

    print("\n2) Pareti rigide, k = 3,117 (valore di riferimento Ra_c = 1707,76):")
    prec = None
    ris = []
    for n in [7, 13, 25, 49, 97, 151, 193]:
        r = Ra_minimo(n, 3.117)
        ris.append((n, r))
        print(f"   n = {n:4d}   Ra = {r:10.3f}")
    # estrapolazione di Richardson sugli ultimi due (ordine 2 in h)
    (n1, r1), (n2, r2) = ris[-2], ris[-1]
    h1, h2 = 1 / (n1 - 1), 1 / (n2 - 1)
    r_inf = r2 + (r2 - r1) * h2**2 / (h1**2 - h2**2)
    print(f"   Richardson (n = {n1}, {n2}): Ra = {r_inf:.2f}")

    # ------------------------------------------------------------------ figura
    A, B = matrici(7, 3.117)
    fig, ax = plt.subplots(1, 3, figsize=(7.8, 2.8))
    ax[0].spy(A, markersize=4, color=BLU)
    ax[0].set_title(r"$\mathbf{A}$ ($n=7$)")
    ax[1].spy(B, markersize=4, color=ROSSO)
    ax[1].set_title(r"$\mathbf{B}$ ($n=7$)")
    for a in ax[:2]:
        a.axhline(6.5, color=GRIGIO, lw=0.6)
        a.axvline(6.5, color=GRIGIO, lw=0.6)
        a.tick_params(labelsize=6)
    kk = np.linspace(1.0, 8.0, 60)
    ax[2].plot(kk, (np.pi**2 + kk**2) ** 3 / kk**2, color=GRIGIO, lw=2.2, alpha=0.6, label="esatta")
    ax[2].plot(kk[::4], [Ra_minimo(41, k, "libere") for k in kk[::4]], "o", ms=3.5, color=ARANCIO,
               label="diff. finite, $n=41$")
    ax[2].set_ylim(0, 6000)
    ax[2].set_xlabel("$k$")
    ax[2].set_ylabel("$Ra$")
    ax[2].set_title("pareti libere: validazione")
    ax[2].legend(fontsize=6.5)
    fig.tight_layout()
    salva(fig, "prog3_7a_benard")
