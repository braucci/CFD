"""
prog3_3b_condotti.py -- validazione del Programma 3.3 con la soluzione esatta
e influenza delle pareti laterali in un canale a pelo libero.

Per simmetria, il canale largo W e profondo D con pelo libero equivale a meta'
di un condotto rettangolare chiuso W x 2D. La soluzione esatta di
U_yy + U_zz = -1 (U = 0 sulle pareti, U_y = 0 per y = D) e' la serie
    U = sum_{n dispari} 4 W^2/(pi^3 n^3) sin(n pi z/W) [1 - cosh(k (y - D))/cosh(k D)],
con k = n pi / W, e la portata
    Q = sum_{n dispari} 4 W^2/(pi^3 n^3) (2 W/(n pi)) [D - tanh(k D)/k].
1) Errore della soluzione alle differenze finite (risolta qui con un metodo
   diretto per matrici sparse) al raffinare della griglia; estrapolazione di
   Richardson.  2) Portata in funzione del rapporto W/D.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from stile_grafici import salva, BLU, ARANCIO, VERDE, GRIGIO


def serie_U(y, z, W=1.0, D=1.0, nmax=401):
    U = np.zeros(np.broadcast(y, z).shape)
    for n in range(1, nmax + 1, 2):
        k = n * np.pi / W
        # cosh(k(y-D))/cosh(kD) scritto in forma stabile per k D grande
        rapp = np.exp(-k * y) * (1 + np.exp(-2 * k * (D - y))) / (1 + np.exp(-2 * k * D))
        U += 4 * W**2 / (np.pi**3 * n**3) * np.sin(k * z) * (1 - rapp)
    return U


def serie_Q(W=1.0, D=1.0, nmax=2001):
    Q = 0.0
    for n in range(1, nmax + 1, 2):
        k = n * np.pi / W
        Q += 4 * W**2 / (np.pi**3 * n**3) * (2 * W / (n * np.pi)) * (D - np.tanh(k * D) / k)
    return Q


def differenze_finite(h):
    """Canale quadrato 1 x 1: incognite nei nodi i = 1..M-2 (z), j = 1..N-1 (y);
    la riga j = N-1 e' il pelo libero, con nodo fittizio U[i,N] = U[i,N-2]."""
    n = int(round(1.0 / h))
    nz, ny = n - 1, n                  # incognite per riga e numero di righe
    idx = lambda i, j: (j - 1) * nz + (i - 1)
    A = sp.lil_matrix((nz * ny, nz * ny))
    b = -h * h * np.ones(nz * ny)
    for j in range(1, ny + 1):
        for i in range(1, nz + 1):
            r = idx(i, j)
            A[r, r] = -4.0
            if i > 1:
                A[r, idx(i - 1, j)] = 1.0
            if i < nz:
                A[r, idx(i + 1, j)] = 1.0
            if j > 1:
                A[r, idx(i, j - 1)] += 1.0
            if j < ny:
                A[r, idx(i, j + 1)] = 1.0
            else:                          # pelo libero: il fittizio vale U[i, j-1]
                A[r, idx(i, j - 1)] += 1.0
    u = spla.spsolve(A.tocsr(), b)
    U = np.zeros((n + 1, n + 1))
    U[1:n, 1:n + 1] = u.reshape(ny, nz).T
    pesi = np.ones_like(U)
    pesi[0, :] *= 0.5
    pesi[-1, :] *= 0.5
    pesi[:, 0] *= 0.5
    pesi[:, -1] *= 0.5
    return U[n // 2, n], np.sum(pesi * U) * h * h


Umax_es = serie_U(1.0, 0.5)
Q_es = serie_Q()
print(f"Soluzione esatta (canale quadrato): U_max = {Umax_es:.6f},  Q = {Q_es:.6f}")
print("\n   h        U_max       errore      Q           errore     ordine(Q)")
passi = [1 / 10, 1 / 20, 1 / 40, 1 / 80, 1 / 160]
ris = []
for k, h in enumerate(passi):
    um, Q = differenze_finite(h)
    ris.append((um, Q))
    ordine = "" if k == 0 else f"{np.log2(abs(ris[k-1][1]-Q_es)/abs(Q-Q_es)):6.2f}"
    print(f"  1/{int(round(1/h)):<4d}  {um:.6f}  {um-Umax_es:+.2e}  {Q:.6f}  {Q-Q_es:+.2e}  {ordine}")
Q_rich = (4 * ris[2][1] - ris[1][1]) / 3
print(f"\nRichardson con h = 1/20 e 1/40: Q = {Q_rich:.6f}  (errore {Q_rich-Q_es:+.1e})")

# influenza del rapporto larghezza/profondita'
WD = np.logspace(-0.6, 1.5, 60)
Qrel = np.array([serie_Q(W=w, D=1.0) / (w / 3.0) for w in WD])   # rispetto al canale indefinito
print("\nPortata per unita' di larghezza rispetto al canale senza pareti (Q/(W D^3/3)):")
for w in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    print(f"   W/D = {w:5.1f}   {serie_Q(W=w, D=1.0)/(w/3.0):.4f}   "
          f"(stima per W >> D: 1 - 1,2605 D/W = {1 - 1.2605/w:.4f})")

# ------------------------------------------------------------------ figura
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
hh = np.array(passi)
err = np.array([abs(r[1] - Q_es) for r in ris])
err_u = np.array([abs(r[0] - Umax_es) for r in ris])
ax[0].loglog(hh, err, "o-", color=BLU, label="portata $Q$")
ax[0].loglog(hh, err_u, "s-", color=ARANCIO, label=r"$U_{\max}$")
ax[0].loglog(hh, err[0] * (hh / hh[0]) ** 2, ":", color="k", label=r"$\propto h^2$")
ax[0].set_xlabel("passo $h$")
ax[0].set_ylabel("errore rispetto alla serie")
ax[0].legend()
ax[1].semilogx(WD, Qrel, color=VERDE)
ax[1].axhline(1.0, color=GRIGIO, ls=":")
ax[1].axvline(1.0, color=GRIGIO, ls=":")
ax[1].set_xlabel("$W/D$")
ax[1].set_ylabel(r"$Q\,/\,(W D^3/3)$")
ax[1].set_title("effetto delle pareti laterali")
fig.tight_layout()
salva(fig, "prog3_3b_condotti")
