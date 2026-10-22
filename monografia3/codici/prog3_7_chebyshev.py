"""
prog3_7_chebyshev.py -- PROGRAMMA 3.7
Il metodo di collocazione di Chebyshev per un problema ai limiti:
    u'' + pi u = pi,   u(-1) = 0,   u(1) = 1,
la cui soluzione esatta e'
    u = 1 - cos(sqrt(pi) y) / (2 cos sqrt(pi)) + sin(sqrt(pi) y) / (2 sin sqrt(pi)).
Nei nodi y_j = cos(j pi/N) la derivata seconda e' D^2 u; le righe della
matrice relative ai nodi di contorno sono sostituite dalle condizioni al
contorno. Si confronta la convergenza (esponenziale) con quella delle
differenze finite centrate del secondo ordine.
"""
import numpy as np
import matplotlib.pyplot as plt
from metodi_numerici import cheb
from stile_grafici import salva, BLU, ARANCIO, ROSSO, GRIGIO

s = np.sqrt(np.pi)


def esatta(y):
    return 1 - np.cos(s * y) / (2 * np.cos(s)) + np.sin(s * y) / (2 * np.sin(s))


def chebyshev(N):
    D, y = cheb(N)
    A = D @ D + np.pi * np.eye(N + 1)
    c = np.pi * np.ones(N + 1)
    A[0, :] = 0.0                    # y_0 = +1 :  u = 1
    A[0, 0] = 1.0
    c[0] = 1.0
    A[N, :] = 0.0                    # y_N = -1 :  u = 0
    A[N, N] = 1.0
    c[N] = 0.0
    return y, np.linalg.solve(A, c)


def differenze_finite(N):
    y = np.linspace(-1, 1, N + 1)
    h = y[1] - y[0]
    A = np.zeros((N + 1, N + 1))
    c = np.pi * np.ones(N + 1)
    for i in range(1, N):
        A[i, i - 1:i + 2] = [1 / h**2, -2 / h**2 + np.pi, 1 / h**2]
    A[0, 0] = A[N, N] = 1.0
    c[0], c[N] = 0.0, 1.0
    return y, np.linalg.solve(A, c)


# verifica della matrice di differenziazione su u = y^2 (coefficienti a0 = a2 = 1/2)
D, y = cheb(8)
print(f"Verifica: max |D y^2 - 2y| = {np.max(np.abs(D @ y**2 - 2 * y)):.1e},"
      f"  max |D^2 y^2 - 2| = {np.max(np.abs(D @ D @ y**2 - 2)):.1e}")
print(f"Elementi d'angolo: D[0,0] = {D[0,0]:.4f} = (2N^2+1)/6 = {(2*64+1)/6:.4f}")

print("\n   N    errore Chebyshev    errore diff. finite")
Nlista = [4, 6, 8, 10, 12, 16, 20, 24, 32]
e_ch, e_df = [], []
for N in Nlista:
    yc, uc = chebyshev(N)
    yd, ud = differenze_finite(N)
    e_ch.append(np.max(np.abs(uc - esatta(yc))))
    e_df.append(np.max(np.abs(ud - esatta(yd))))
    print(f"  {N:3d}    {e_ch[-1]:.3e}           {e_df[-1]:.3e}")

# ------------------------------------------------------------------ figura
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
yy = np.linspace(-1, 1, 300)
y8, u8 = chebyshev(8)
ax[0].plot(yy, esatta(yy), color=GRIGIO, lw=1.8, label="esatta")
ax[0].plot(y8, u8, "o", color=BLU, ms=5, label="Chebyshev, 9 nodi")
ax[0].set_xlabel("$y$")
ax[0].set_ylabel("$u$")
ax[0].set_title(r"$u''+\pi u=\pi$")
ax[0].legend()
ax[1].semilogy(Nlista, e_ch, "o-", color=BLU, label="Chebyshev")
ax[1].semilogy(Nlista, e_df, "s-", color=ARANCIO, label="differenze finite")
ax[1].set_xlabel("$N$ (numero di intervalli)")
ax[1].set_ylabel("errore massimo")
ax[1].legend()
fig.tight_layout()
salva(fig, "prog3_7_chebyshev")
