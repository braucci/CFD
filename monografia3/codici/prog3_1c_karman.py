"""
prog3_1c_karman.py -- metodo integrale di von Karman-Pohlhausen e serie di Blasius.

1) Profili polinomiali u/U = F(Y), Y = y/delta, che soddisfano le condizioni
   F(0) = 0, F(1) = 1, F'(1) = 0 (e, per il quartico, F''(0) = 0, F''(1) = 0):
   l'equazione integrale di von Karman  d(theta)/dx = tau_w/(rho U^2)  fornisce
   delta(x) in forma chiusa e, con essa, C_f, delta*, theta e H, da confrontare
   con i valori esatti di Blasius.
2) Serie di potenze di Blasius  f = sum_k (-1/2)^k A_k alfa^(k+1) eta^(3k+2)/(3k+2)!
   con alfa = f''(0), A_0 = A_1 = 1 e la ricorrenza
   A_k = sum_{j=0}^{k-1} C(3k-1, 3j) A_j A_{k-1-j}:
   stima del raggio di convergenza e confronto con l'integrazione numerica.
"""
import numpy as np
from math import comb, factorial
import matplotlib.pyplot as plt
from metodi_numerici import rk4_integra
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

ALFA = 0.332057336215          # f''(0) di Blasius (riferimento)


def blasius(eta, z):
    f, p, q = z
    return np.array([p, q, -0.5 * f * q])


# ---------------------------------------------------------------- 1) Karman-Pohlhausen
# ciascun profilo: nome, F(Y), F'(0), e gli integrali I1 = int(1-F), I2 = int F(1-F)
profili = {
    "lineare  F = Y":            (lambda Y: Y,                      1.0),
    "cubico   F = 3Y/2 - Y^3/2": (lambda Y: 1.5 * Y - 0.5 * Y**3,   1.5),
    "quartico F = 2Y-2Y^3+Y^4":  (lambda Y: 2 * Y - 2 * Y**3 + Y**4, 2.0),
    "seno     F = sin(pi Y/2)":  (lambda Y: np.sin(np.pi * Y / 2),  np.pi / 2),
}
Y = np.linspace(0, 1, 2001)
print("Metodo integrale di von Karman: delta/sqrt(nu x/U), C_f sqrt(Re_x), delta*, theta, H")
print(f"   {'profilo':28s} {'delta':>7s} {'Cf*sqrt(Re)':>12s} {'delta*':>8s} {'theta':>8s} {'H':>6s}")
risultati = {}
for nome, (F, Fp0) in profili.items():
    Fv = F(Y)
    I1 = np.trapezoid(1 - Fv, Y)            # delta*/delta
    I2 = np.trapezoid(Fv * (1 - Fv), Y)     # theta/delta
    # equazione integrale: U^2 d(I2 delta)/dx = nu U F'(0)/delta  ->  delta^2 = 2 nu x F'(0)/(U I2)
    d = np.sqrt(2 * Fp0 / I2)               # delta / sqrt(nu x / U)
    Cf = 2 * Fp0 / d                        # C_f sqrt(Re_x) = 2 F'(0) sqrt(nu x/U)/delta
    risultati[nome] = (d, Cf, I1 * d, I2 * d, I1 / I2)
    print(f"   {nome:28s} {d:7.3f} {Cf:12.4f} {I1*d:8.4f} {I2*d:8.4f} {I1/I2:6.3f}")
print(f"   {'Blasius (esatto)':28s} {'4.910':>7s} {2*ALFA:12.4f} {1.7208:8.4f} {2*ALFA:8.4f} {1.7208/(2*ALFA):6.3f}")

# ---------------------------------------------------------------- 2) serie di Blasius
K = 40
A = [1.0, 1.0]
for k in range(2, K + 1):
    A.append(sum(comb(3 * k - 1, 3 * j) * A[j] * A[k - 1 - j] for j in range(k)))
A = np.array(A, dtype=float)


def serie(eta, nterm):
    s = 0.0
    for k in range(nterm):
        s += (-0.5)**k * A[k] * ALFA**(k + 1) * eta**(3 * k + 2) / factorial(3 * k + 2)
    return s


# raggio di convergenza: rapporto fra termini successivi (test del rapporto)
k = np.arange(5, K)
c_k = np.array([0.5**kk * A[kk] * ALFA**(kk + 1) / factorial(3 * kk + 2) for kk in range(K + 1)])
rho = (c_k[k] / c_k[k + 1])**(1 / 3)         # |c_k eta^(3k)| ~ costante -> eta ~ (c_k/c_(k+1))^(1/3)
print("\nSerie di Blasius: A_k =", [int(a) for a in A[:6]], "...")
print(f"   raggio di convergenza stimato dal test del rapporto: {rho[-1]:.3f} (k = {k[-1]})")
eta_rk, Z = rk4_integra(blasius, np.array([0, 0, ALFA]), 8.0, 0.01)
for eta_t in (2.0, 4.0, 5.0, 5.5, 6.0):
    i = int(round(eta_t / 0.01))
    err = [abs(serie(eta_t, n) - Z[i, 0]) for n in (10, 20, 40)]
    print(f"   eta = {eta_t:3.1f}: |serie - RK4| con 10, 20, 40 termini = "
          + ", ".join(f"{e:.1e}" for e in err))

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.9))
eta = np.linspace(0, 6, 300)
_, Zb = rk4_integra(blasius, np.array([0, 0, ALFA]), 6.0, 0.02)
ax[0].plot(Zb[:, 1], np.linspace(0, 6, Zb.shape[0]), "k", lw=1.8, label="Blasius")
for (nome, (F, _)), col in zip(profili.items(), [GRIGIO, VERDE, ARANCIO, ROSSO]):
    d = risultati[nome][0]
    ax[0].plot(F(np.minimum(eta / d, 1.0)), eta, "--", color=col, lw=1.1, label=nome.split()[0])
ax[0].set_xlabel("$u/U$")
ax[0].set_ylabel("$\\eta$")
ax[0].legend(fontsize=7)
ax[0].set_title("profili di prova (scala $\\delta$ di Karman)")
nomi = list(risultati)
ax[1].bar(range(4), [risultati[n][1] for n in nomi], color=[GRIGIO, VERDE, ARANCIO, ROSSO], width=0.6)
ax[1].axhline(2 * ALFA, color="k", ls="--", lw=1.0, label="Blasius $0{,}664$")
ax[1].set_xticks(range(4))
ax[1].set_xticklabels([n.split()[0] for n in nomi], fontsize=7)
ax[1].set_ylabel("$C_f\\sqrt{Re_x}$")
ax[1].set_ylim(0.5, 0.8)
ax[1].legend(fontsize=7)
ax[1].set_title("sforzo alla parete")
for n, col in zip((10, 20, 40), (GRIGIO, ARANCIO, BLU)):
    e = np.array([abs(serie(x, n) - Z[int(round(x / 0.01)), 0]) for x in eta])
    ax[2].semilogy(eta, np.maximum(e, 1e-16), color=col, label=f"{n} termini")
ax[2].axvline(rho[-1], color=ROSSO, ls=":", lw=1.0)
ax[2].text(rho[-1] - 0.1, 1e-8, "raggio di\nconvergenza", ha="right", fontsize=7, color=ROSSO)
ax[2].set_xlabel("$\\eta$")
ax[2].set_ylabel("$|f_{serie}-f|$")
ax[2].set_ylim(1e-16, 1e2)
ax[2].legend(fontsize=7, loc="upper left")
ax[2].set_title("serie di Blasius")
fig.tight_layout()
salva(fig, "prog3_1c_karman")
