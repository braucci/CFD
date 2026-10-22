"""
prog3_1b_tiro.py -- metodi di tiro piu' efficienti per il problema di Blasius
e studio dell'accuratezza.

1) Metodo di Newton con le equazioni variazionali: insieme a (f, p, q) si
   integrano le sensibilita' (F, P, Q) = d(f, p, q)/dq0, che soddisfano
       F' = P,  P' = Q,  Q' = -(F q + f Q)/2,   F(0) = P(0) = 0, Q(0) = 1,
   e si corregge  q0 <- q0 - (p(eta_max) - 1) / P(eta_max).
2) Metodo della secante (non richiede le sensibilita').
3) Confronto con il risolutore di problemi ai limiti scipy.integrate.solve_bvp.
4) Errore su f''(0) al variare del passo h (ordine 4 di RK4) e di eta_max.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, solve_ivp
from metodi_numerici import rk4_integra
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO


def blasius(eta, z):
    f, p, q = z
    return np.array([p, q, -0.5 * f * q])


def blasius_var(eta, z):
    """Sistema esteso: (f, p, q) e sensibilita' (F, P, Q) rispetto a q0."""
    f, p, q, F, P, Q = z
    return np.array([p, q, -0.5 * f * q, P, Q, -0.5 * (F * q + f * Q)])


def newton(q0, h=0.1, etamax=10.0, tol=1e-12, stampa=False):
    storia = []
    for it in range(1, 30):
        _, Z = rk4_integra(blasius_var, np.array([0, 0, q0, 0, 0, 1.0]), etamax, h)
        res = Z[-1, 1] - 1.0
        storia.append((it, q0, res))
        if stampa:
            print(f"   it {it}:  q0 = {q0:.12f}   p(eta_max) - 1 = {res:+.3e}")
        if abs(res) < tol:
            return q0, it, storia
        q0 -= res / Z[-1, 4]
    raise RuntimeError("Newton non converge")


def secante(qa, qb, h=0.1, etamax=10.0, tol=1e-12):
    def r(q):
        _, Z = rk4_integra(blasius, np.array([0, 0, q]), etamax, h)
        return Z[-1, 1] - 1.0
    ra, rb = r(qa), r(qb)
    for it in range(1, 50):
        qc = qb - rb * (qb - qa) / (rb - ra)
        qa, ra, qb, rb = qb, rb, qc, r(qc)
        if abs(rb) < tol:
            return qb, it
    raise RuntimeError("secante non converge")


# ------------------------------------------------------------ 1) e 2)
print("1) Newton con equazioni variazionali (h = 0,1, eta_max = 10), partenza q0 = 1:")
t0 = time.perf_counter()
q_n, it_n, storia = newton(1.0, stampa=True)
t_n = time.perf_counter() - t0
print(f"   f''(0) = {q_n:.10f}  in {it_n} iterazioni ({t_n*1e3:.0f} ms)")
t0 = time.perf_counter()
q_s, it_s = secante(0.2, 0.5)
t_s = time.perf_counter() - t0
print(f"2) Secante (partenze 0,2 e 0,5): f''(0) = {q_s:.10f} in {it_s} iterazioni ({t_s*1e3:.0f} ms)")

# ------------------------------------------------------------ valore di riferimento
# integrazione adattiva molto accurata su un dominio ampio, con Newton
def residuo_ivp(q0, etamax=20.0):
    s = solve_ivp(blasius_var, (0, etamax), [0, 0, q0, 0, 0, 1], rtol=1e-13, atol=1e-14,
                  method="DOP853")
    return s.y[1, -1] - 1.0, s.y[4, -1]


q_rif = 0.33
for _ in range(12):
    r, dr = residuo_ivp(q_rif)
    q_rif -= r / dr
print(f"   valore di riferimento (DOP853, eta_max = 20): f''(0) = {q_rif:.12f}")

# ------------------------------------------------------------ 3) solve_bvp
def bc(za, zb):
    return np.array([za[0], za[1], zb[1] - 1.0])

eta0 = np.linspace(0, 10, 51)
z0 = np.vstack([eta0 - 1.7 * (1 - np.exp(-eta0)), 1 - np.exp(-eta0), 0.33 * np.exp(-eta0)])
t0 = time.perf_counter()
sol = solve_bvp(blasius, bc, eta0, z0, tol=1e-8, max_nodes=20000)
t_b = time.perf_counter() - t0
print(f"3) solve_bvp (collocazione): f''(0) = {sol.sol(0.0)[2]:.10f}, "
      f"{sol.x.size} nodi finali, stato = {sol.status} ({t_b*1e3:.0f} ms)")

# ------------------------------------------------------------ 4) accuratezza
passi = np.array([0.4, 0.2, 0.1, 0.05, 0.025])
err_h = []
print("\n4a) Errore su f''(0) al variare del passo (eta_max = 10):")
q_10, _, _ = newton(0.33, h=0.0025, etamax=10.0)   # riferimento con lo stesso eta_max
for h in passi:
    q, _, _ = newton(0.33, h=h, etamax=10.0)
    err_h.append(abs(q - q_10))
for k, h in enumerate(passi):
    ordine = "" if k == 0 else f"   ordine osservato {np.log2(err_h[k-1]/err_h[k]):.2f}"
    print(f"   h = {h:6.3f}   |errore| = {err_h[k]:.3e}{ordine}")

emax = np.array([4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0])
err_e = []
print("\n4b) Errore su f''(0) al variare di eta_max (h = 0,01):")
for e in emax:
    q, _, _ = newton(0.33, h=0.01, etamax=e)
    err_e.append(abs(q - q_rif))
    print(f"   eta_max = {e:5.1f}   f''(0) = {q:.10f}   |errore| = {err_e[-1]:.2e}")

# ------------------------------------------------------------ figura
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.6))
it = [s[0] for s in storia]
ax[0].semilogy(it, [abs(s[2]) + 1e-17 for s in storia], "o-", color=BLU)
ax[0].set_xlabel("iterazione di Newton")
ax[0].set_ylabel(r"$|f'(\eta_{\max})-1|$")
ax[0].set_title("convergenza quadratica")
ax[1].loglog(passi, err_h, "s-", color=ARANCIO, label="RK4")
ax[1].loglog(passi, err_h[2] * (passi / passi[2]) ** 4, ":", color="k", label=r"$\propto h^4$")
ax[1].set_xlabel("passo $h$")
ax[1].set_ylabel(r"errore su $f''(0)$")
ax[1].legend()
ax[1].set_title("errore di discretizzazione")
ax[2].semilogy(emax, err_e, "d-", color=VERDE)
ax[2].set_xlabel(r"$\eta_{\max}$")
ax[2].set_ylabel(r"errore su $f''(0)$")
ax[2].set_title("troncamento del dominio")
fig.tight_layout()
salva(fig, "prog3_1b_tiro")
