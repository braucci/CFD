"""
prog3_7b_runge.py -- perche' i nodi di Chebyshev: il fenomeno di Runge,
la costante di Lebesgue e la velocita' di convergenza spettrale.

1) Interpolazione polinomiale della funzione di Runge f(y) = 1/(1 + 25 y^2)
   su nodi equispaziati e su nodi di Chebyshev-Gauss-Lobatto: con i primi
   l'errore cresce con N (vicino agli estremi), con i secondi decresce come
   rho^(-N), dove rho = 1/5 + sqrt(1 + 1/25) = 1,2198 e' il parametro
   dell'ellisse di Bernstein che passa per i poli y = +-i/5.
2) Costante di Lebesgue Lambda_N (norma dell'operatore di interpolazione):
   cresce come 2^N/(e N log N) per i nodi equispaziati e come (2/pi) log N
   per quelli di Chebyshev.
3) Numero di condizionamento della matrice di differenziazione: cond(D^2) ~ N^4.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator
from metodi_numerici import cheb
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

f = lambda y: 1.0 / (1 + 25 * y**2)
yy = np.linspace(-1, 1, 2001)
rho = 0.2 + np.sqrt(1 + 0.04)


def interpola(nodi, valori, y):
    return BarycentricInterpolator(nodi, valori)(y)


def lebesgue(nodi, y):
    """Lambda = max_y sum_j |l_j(y)| con i polinomi di Lagrange l_j."""
    n = len(nodi)
    somma = np.zeros_like(y)
    for j in range(n):
        e = np.zeros(n); e[j] = 1.0
        somma += np.abs(interpola(nodi, e, y))
    return somma.max()


print("Funzione di Runge 1/(1 + 25 y^2): errore massimo dell'interpolazione")
print("      N   equispaziati    Chebyshev     rho^-N      Lambda_eq    Lambda_cheb")
Ns = [4, 8, 12, 16, 20, 24, 32, 40, 48, 64]
err_eq, err_ch, leb_eq, leb_ch = [], [], [], []
for N in Ns:
    ye = np.linspace(-1, 1, N + 1)
    _, yc = cheb(N)
    ee = np.max(np.abs(interpola(ye, f(ye), yy) - f(yy)))
    ec = np.max(np.abs(interpola(yc, f(yc), yy) - f(yy)))
    le, lc = lebesgue(ye, yy), lebesgue(yc, yy)
    err_eq.append(ee); err_ch.append(ec); leb_eq.append(le); leb_ch.append(lc)
    print(f"   {N:4d}   {ee:11.3e}   {ec:11.3e}   {rho**-N:9.2e}   {le:10.3e}   {lc:8.3f}")
# tasso di convergenza osservato con Chebyshev
pend = -np.polyfit(Ns[2:7], np.log(err_ch[2:7]), 1)[0]
print(f"   tasso osservato (Chebyshev, 12 <= N <= 32): errore ~ exp(-{pend:.4f} N) = {np.exp(pend):.4f}^-N;"
      f" teoria: rho = {rho:.4f}")
print("\nCondizionamento della matrice di differenziazione (righe di contorno sostituite con l'identita'):")
for N in (8, 16, 32, 64):
    D, y = cheb(N)
    D2 = D @ D
    D2[0, :] = 0; D2[0, 0] = 1; D2[-1, :] = 0; D2[-1, -1] = 1
    print(f"   N = {N:3d}: cond(D^2) = {np.linalg.cond(D2):.2e}   (N^4 = {N**4:.1e})")

fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.8))
for N, col in ((8, VERDE), (16, ARANCIO)):
    ye = np.linspace(-1, 1, N + 1)
    ax[0].plot(yy, interpola(ye, f(ye), yy), color=col, label=f"equispaziati, $N={N}$")
_, yc = cheb(16)
ax[0].plot(yy, interpola(yc, f(yc), yy), "--", color=BLU, label="Chebyshev, $N=16$")
ax[0].plot(yy, f(yy), "k:", lw=1.2, label="$1/(1+25y^2)$")
ax[0].set_ylim(-0.8, 1.6)
ax[0].set_xlabel("$y$")
ax[0].legend(fontsize=6.5, loc="upper center")
ax[0].set_title("fenomeno di Runge")
ax[1].semilogy(Ns, err_eq, "s-", color=ROSSO, label="equispaziati")
ax[1].semilogy(Ns, err_ch, "o-", color=BLU, label="Chebyshev")
ax[1].semilogy(Ns, 2.0 * rho**(-np.array(Ns)), ":", color=GRIGIO, label="$\\propto\\rho^{-N}$")
ax[1].set_xlabel("$N$")
ax[1].set_ylabel("errore massimo")
ax[1].legend(fontsize=7)
ax[1].set_title("convergenza")
ax[2].semilogy(Ns, leb_eq, "s-", color=ROSSO, label="equispaziati")
ax[2].semilogy(Ns, leb_ch, "o-", color=BLU, label="Chebyshev")
ax[2].semilogy(Ns, 2 / np.pi * np.log(np.array(Ns)) + 1, ":", color=GRIGIO, label="$(2/\\pi)\\ln N+1$")
ax[2].set_xlabel("$N$")
ax[2].set_ylabel("$\\Lambda_N$")
ax[2].legend(fontsize=7)
ax[2].set_title("costante di Lebesgue")
fig.tight_layout()
salva(fig, "prog3_7b_runge")
