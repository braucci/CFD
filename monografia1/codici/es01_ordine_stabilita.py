"""
es01_ordine_stabilita.py
Esperimenti numerici del Capitolo 1 (metodi a un passo):
  (a) verifica dell'ordine di accuratezza di Eulero, RK2, RK4
      sul problema test  dx/dt = x - t^2 + 1,  x(0) = 0.5
      la cui soluzione esatta e'  x(t) = (t+1)^2 - 0.5 e^t
  (b) stabilita' assoluta di Eulero sul problema modello dx/dt = -lambda x
"""
import numpy as np
import matplotlib.pyplot as plt
from ode_solvers import integra
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO


# ---------------------------------------------------------------- (a)
def f_test(t, x):
    return x - t ** 2 + 1.0


def x_esatta(t):
    return (t + 1.0) ** 2 - 0.5 * np.exp(t)


print("=== (a) Verifica dell'ordine: errore globale in t = 2 ===")
passi = [0.4, 0.2, 0.1, 0.05, 0.025]
errori = {m: [] for m in ("eulero", "rk2", "rk4")}
for h in passi:
    n = int(round(2.0 / h))
    for m in errori:
        t, X = integra(f_test, 0.0, 0.5, h, n, metodo=m)
        errori[m].append(abs(X[-1, 0] - x_esatta(2.0)))

print(f"{'h':>7} {'Eulero':>12} {'RK2':>12} {'RK4':>12}")
for i, h in enumerate(passi):
    print(f"{h:7.3f} {errori['eulero'][i]:12.3e} {errori['rk2'][i]:12.3e} "
          f"{errori['rk4'][i]:12.3e}")

print("\nOrdine osservato p = log2(E(h)/E(h/2)):")
for m in errori:
    e = np.array(errori[m])
    p = np.log2(e[:-1] / e[1:])
    print(f"  {m:7s}: " + "  ".join(f"{v:5.2f}" for v in p))

fig, ax = plt.subplots(figsize=(4.2, 3.0))
for m, lab, c in (("eulero", "Eulero (ordine 1)", BLU),
                  ("rk2", "RK2 (ordine 2)", ARANCIO),
                  ("rk4", "RK4 (ordine 4)", VERDE)):
    ax.loglog(passi, errori[m], "o-", label=lab, color=c)
hh = np.array(passi)
ax.loglog(hh, 0.6 * hh ** 4, ":", color=VERDE, lw=0.8)
ax.loglog(hh, 0.5 * hh ** 2, ":", color=ARANCIO, lw=0.8)
ax.loglog(hh, 1.0 * hh, ":", color=BLU, lw=0.8)
ax.set_xlabel("passo $h$")
ax.set_ylabel("errore globale $|x_N - x(2)|$")
ax.legend()
salva(fig, "es01_ordine")

# ---------------------------------------------------------------- (b)
print("\n=== (b) Stabilita' di Eulero su dx/dt = -lambda x ===")
lam = 1.0
fig, ax = plt.subplots(figsize=(4.6, 3.0))
tt = np.linspace(0, 12, 400)
ax.plot(tt, np.exp(-lam * tt), "k-", lw=1.0, label="esatta $e^{-\\lambda t}$")
for hl, c in ((0.5, VERDE), (1.5, ARANCIO), (2.1, ROSSO)):
    h = hl / lam
    n = int(12 / h)
    t, X = integra(lambda t, x: -lam * x, 0.0, 1.0, h, n, metodo="eulero")
    ax.plot(t, X[:, 0], "o-", ms=3, color=c, label=f"Eulero, $h\\lambda={hl}$")
    print(f"  h*lambda = {hl}: x_N = {X[-1,0]: .3e}  (fattore 1-h*lambda = {1-hl:+.2f})")
ax.set_ylim(-3, 3)
ax.set_xlabel("$t$")
ax.set_ylabel("$x$")
ax.legend(ncol=2)
salva(fig, "es01_stabilita")

# ---------------------------------------------------------------- (c)
# Regioni di stabilita' assoluta nel piano complesso z = h*lambda
print("\n=== (c) Regioni di stabilita' assoluta ===")
re = np.linspace(-4.0, 1.0, 400)
im = np.linspace(-3.5, 3.5, 400)
RE, IM = np.meshgrid(re, im)
z = RE + 1j * IM
R = {
    "Eulero": 1 + z,
    "RK2": 1 + z + z ** 2 / 2,
    "RK4": 1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24,
}
fig, ax = plt.subplots(figsize=(4.2, 3.6))
for (nome, Rz), c in zip(R.items(), (BLU, ARANCIO, VERDE)):
    ax.contour(RE, IM, np.abs(Rz), levels=[1.0], colors=[c], linewidths=1.4)
    ax.plot([], [], color=c, label=nome)
ax.axhline(0, color="k", lw=0.5)
ax.axvline(0, color="k", lw=0.5)
ax.set_aspect("equal")
ax.set_xlabel("Re$(h\\lambda)$")
ax.set_ylabel("Im$(h\\lambda)$")
ax.legend(loc="upper left")
salva(fig, "es01_regioni")
for nome in R:
    # intersezione con l'asse reale negativo
    rr = np.linspace(-3.0, 0, 30001)
    Rr = {"Eulero": 1 + rr, "RK2": 1 + rr + rr**2/2,
          "RK4": 1 + rr + rr**2/2 + rr**3/6 + rr**4/24}[nome]
    lim = rr[np.where(np.abs(Rr) <= 1.0)[0][0]]
    print(f"  {nome:7s}: stabile sull'asse reale per  -{abs(lim):.3f} <= h*lambda <= 0")
