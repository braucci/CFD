"""
prog3_5b_crank_nicolson.py -- confronto fra metodo implicito (Eulero all'indietro)
e metodo di Crank-Nicolson per l'equazione della diffusione.

Schema theta (theta = 1: implicito; theta = 1/2: Crank-Nicolson):
  -theta R W_(i-1)^(n+1) + (1 + 2 theta R) W_i^(n+1) - theta R W_(i+1)^(n+1)
     = (1-theta) R W_(i-1)^n + [1 - 2 (1-theta) R] W_i^n + (1-theta) R W_(i+1)^n
1) Ordine nel tempo sul problema del Programma 3.5 (h piccolo, tau variabile).
2) Fattori di amplificazione: G_I = 1/(1 + 4R s^2), G_CN = (1 - 2R s^2)/(1 + 2R s^2),
   s = sin(kh/2): per R grande G_CN -> -1 (le componenti brevi non si smorzano).
3) Dato iniziale discontinuo (problema di Rayleigh, R = 10): Crank-Nicolson
   produce oscillazioni spurie, il metodo implicito no.
"""
import numpy as np
import matplotlib.pyplot as plt
from metodi_numerici import thomas
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO


def passo_theta(W, R, theta, simm=True, w_sup=0.0):
    """Un passo dello schema theta su 0 <= Y <= 1, nodi 0..N; W[N] = w_sup.
    Se simm=True in Y = 0 vale W_Y = 0 (nodo fittizio), altrimenti W[0] e' fisso."""
    N = len(W) - 1
    d = W.copy()
    lap = np.zeros_like(W)
    lap[1:-1] = W[:-2] - 2 * W[1:-1] + W[2:]
    if simm:
        lap[0] = 2 * (W[1] - W[0])
    rhs = W + (1 - theta) * R * lap
    a = -theta * R * np.ones(N + 1)
    b = (1 + 2 * theta * R) * np.ones(N + 1)
    c = -theta * R * np.ones(N + 1)
    if simm:
        c[0] = -2 * theta * R                 # W_(-1) = W_1
        i0 = 0
    else:
        i0 = 1
    # incognite: i0..N-1; condizione di Dirichlet in N (e in 0 se non simmetrico)
    rhs[N - 1] += theta * R * w_sup
    if not simm:
        rhs[1] += theta * R * W[0]
    sol = thomas(a[i0:N], b[i0:N], c[i0:N], rhs[i0:N])
    nuovo = W.copy()
    nuovo[i0:N] = sol
    nuovo[N] = w_sup
    return nuovo


def W_esatta(Y, T, mmax=400):
    W = np.zeros_like(Y)
    for m in range(1, mmax + 1):
        lam = (2 * m - 1) * np.pi / 2
        W += 4 * (-1) ** (m + 1) / lam**3 * np.cos(lam * Y) * np.exp(-lam**2 * T)
    return W


# ------------------------------------------------- 1) ordine nel tempo
N = 400
Y = np.linspace(0, 1, N + 1)
h = 1.0 / N
T_fin = 0.5
Wes = W_esatta(Y, T_fin)
print("1) Errore a T = 0,5 (h = 1/400) al variare del passo temporale:")
print("     tau        implicito    ordine    Crank-Nicolson   ordine")
taus = [0.05, 0.025, 0.0125, 0.00625, 0.003125]
err = {1.0: [], 0.5: []}
for tau in taus:
    for th in (1.0, 0.5):
        W = 1 - Y**2
        R = tau / h**2
        for _ in range(int(round(T_fin / tau))):
            W = passo_theta(W, R, th)
        err[th].append(np.max(np.abs(W - Wes)))
for k, tau in enumerate(taus):
    o1 = "" if k == 0 else f"{np.log2(err[1.0][k-1]/err[1.0][k]):6.2f}"
    o2 = "" if k == 0 else f"{np.log2(err[0.5][k-1]/err[0.5][k]):6.2f}"
    print(f"  {tau:8.5f}    {err[1.0][k]:.3e}  {o1:>7s}     {err[0.5][k]:.3e}   {o2:>7s}")

# ------------------------------------------------- 3) dato discontinuo
N3 = 40
Y3 = np.linspace(0, 1, N3 + 1)
R3 = 10.0
tau3 = R3 / N3**2
n3 = 5
ris = {}
for th in (1.0, 0.5):
    u = np.zeros(N3 + 1)
    u[-1] = 1.0                                # lastra superiore in moto
    for _ in range(n3):
        u = passo_theta(u, R3, th, simm=False, w_sup=1.0)
    ris[th] = u
from scipy.special import erfc
T3 = n3 * tau3
u_es = np.zeros_like(Y3)
for n in range(1, 400):
    u_es += 2 * (-1) ** n / (n * np.pi) * np.sin(n * np.pi * Y3) * np.exp(-(n * np.pi) ** 2 * T3)
u_es += Y3
print(f"\n3) Lastra messa in moto, R = {R3}, {n3} passi (T = {T3:.4f}):")
print(f"   implicito:       min u (interni) = {ris[1.0][1:-1].min():+.4f}, max u (interni) = {ris[1.0][:-1].max():.4f}")
print(f"   Crank-Nicolson:  min u (interni) = {ris[0.5][1:-1].min():+.4f}, max u (interni) = {ris[0.5][:-1].max():.4f}")
# avvio di Rannacher: i primi due mezzi passi con il metodo implicito, poi Crank-Nicolson
u = np.zeros(N3 + 1)
u[-1] = 1.0
for _ in range(2):
    u = passo_theta(u, R3 / 2, 1.0, simm=False, w_sup=1.0)
for _ in range(n3 - 1):
    u = passo_theta(u, R3, 0.5, simm=False, w_sup=1.0)
print(f"   CN con avvio di Rannacher (2 mezzi passi impliciti): max u (interni) = {u[:-1].max():.4f}")
print(f"   errore massimo rispetto alla serie esatta: implicito {np.max(np.abs(ris[1.0]-u_es)):.2e},"
      f" Crank-Nicolson {np.max(np.abs(ris[0.5]-u_es)):.2e}, Rannacher {np.max(np.abs(u-u_es)):.2e}")
print(f"   (la soluzione esatta e' compresa fra 0 e 1)")

# ------------------------------------------------- figura
fig, ax = plt.subplots(1, 3, figsize=(7.8, 2.7))
tt = np.array(taus)
ax[0].loglog(tt, err[1.0], "o-", color=BLU, label="implicito")
ax[0].loglog(tt, err[0.5], "s-", color=VERDE, label="Crank-Nicolson")
ax[0].loglog(tt, err[1.0][0] * (tt / tt[0]), ":", color="k", label=r"$\tau$")
ax[0].loglog(tt, err[0.5][0] * (tt / tt[0]) ** 2, "--", color="k", label=r"$\tau^2$")
ax[0].set_xlabel(r"$\tau$")
ax[0].set_ylabel("errore massimo")
ax[0].legend(fontsize=6.5)
kh = np.linspace(0, np.pi, 300)
s2 = np.sin(kh / 2) ** 2
for R, ls in [(0.4, "-"), (10.0, "--")]:
    ax[1].plot(kh, 1 / (1 + 4 * R * s2), ls, color=BLU, label=f"impl., $R={R:g}$")
    ax[1].plot(kh, (1 - 2 * R * s2) / (1 + 2 * R * s2), ls, color=VERDE, label=f"C-N, $R={R:g}$")
ax[1].axhline(0, color=GRIGIO, lw=0.6)
ax[1].set_xlabel("$kh$")
ax[1].set_ylabel("$G$")
ax[1].legend(fontsize=6)
ax[2].plot(u_es, Y3, color=GRIGIO, lw=2.0, alpha=0.6, label="esatta")
ax[2].plot(ris[1.0], Y3, "o-", ms=2.5, color=BLU, label="implicito")
ax[2].plot(ris[0.5], Y3, "s-", ms=2.5, color=VERDE, label="Crank-Nicolson")
ax[2].set_xlabel("$u/u_0$")
ax[2].set_ylabel("$y/d$")
ax[2].set_ylim(0.5, 1.0)
ax[2].legend(fontsize=6.5)
ax[2].set_title(f"$R={R3:g}$, {n3} passi")
fig.tight_layout()
salva(fig, "prog3_5b_crank_nicolson")
