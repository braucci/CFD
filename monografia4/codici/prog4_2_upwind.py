"""
prog4_2_upwind.py -- differenze a monte (upwind) e viscosita' artificiale per
l'equazione modello della convezione  zeta_t + u zeta_x = 0,  u > 0 costante.

1) Analisi della perturbazione discreta: un disturbo unitario in x_m a t_j si
   propaga nei passi successivi con i pesi binomiali (1-C)^k C^(n-k): si
   verifica che per 0 < C <= 1 nessun valore supera 1 e che la perturbazione
   viaggia solo verso valle (proprieta' di trasporto). Con le differenze
   centrate il disturbo si propaga anche a monte e cresce.
2) Schema upwind, downwind, centrato (FTCS) e Lax-Wendroff su un dato
   iniziale gaussiano: stabilita' e diffusione numerica.
3) Verifica dell'equazione modificata: la varianza della gaussiana cresce
   come 2 mu_e t con mu_e = u h (1 - C)/2 (viscosita' artificiale).
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

# ---------------------------------------------------------------- 1) perturbazione discreta
def propaga(schema, C, passi, N=15):
    z = np.zeros(N); z[N // 2] = 1.0            # disturbo unitario in x_m
    storia = [z.copy()]
    for _ in range(passi):
        zn = z.copy()
        if schema == "upwind":
            zn[1:] = (1 - C) * z[1:] + C * z[:-1]
        elif schema == "downwind":
            zn[:-1] = (1 + C) * z[:-1] - C * z[1:]
        elif schema == "centrato":
            zn[1:-1] = z[1:-1] - 0.5 * C * (z[2:] - z[:-2])
        z = zn
        storia.append(z.copy())
    return np.array(storia)


print("Perturbazione unitaria in x_m: valori ai passi successivi (nodi m-2 ... m+3)")
for schema, C in (("upwind", 0.5), ("upwind", 1.0), ("upwind", 1.5), ("centrato", 0.5), ("downwind", 0.5)):
    S = propaga(schema, C, 3)
    print(f"   {schema:9s} C = {C:3.1f}:")
    for k in range(4):
        print(f"      passo {k}: " + "  ".join(f"{v:+7.4f}" for v in S[k, 7 - 2:7 + 4]) + f"   max|zeta| = {np.max(np.abs(S[k])):.4f}")

# ---------------------------------------------------------------- 2) gaussiana
u, L, N = 1.0, 20.0, 400
h = L / N
x = np.arange(N) * h
x0, s0 = 4.0, 0.5
z0 = np.exp(-(x - x0)**2 / (2 * s0**2))


def integra(schema, C, t_fin):
    tau = C * h / u
    z = z0.copy()
    n = int(round(t_fin / tau))
    for _ in range(n):
        zm, zp = np.roll(z, 1), np.roll(z, -1)        # periodico (il dominio e' largo abbastanza)
        if schema == "upwind":
            z = z - C * (z - zm)
        elif schema == "centrato":
            z = z - 0.5 * C * (zp - zm)
        elif schema == "lax-wendroff":
            z = z - 0.5 * C * (zp - zm) + 0.5 * C * C * (zp - 2 * z + zm)
        elif schema == "downwind":
            z = z - C * (zp - z)
    return z, n * tau


def varianza(z):
    m = np.sum(z * x) / np.sum(z)
    return m, np.sum(z * (x - m)**2) / np.sum(z)


t_fin = 8.0
print(f"\nGaussiana trasportata per t = {t_fin} (u = 1, h = {h}):")
print("   schema          C     max      errore max    posizione    varianza   2 mu_e t + s0^2")
ris = {}
for schema, C in (("upwind", 0.25), ("upwind", 0.5), ("upwind", 0.9), ("upwind", 1.0),
                  ("lax-wendroff", 0.5), ("centrato", 0.5), ("centrato", 0.05), ("downwind", 0.5)):
    z, t = integra(schema, C, t_fin)
    ze = np.exp(-(x - x0 - u * t)**2 / (2 * s0**2))
    ris[(schema, C)] = z
    if np.all(np.isfinite(z)) and np.max(np.abs(z)) < 1e3:
        m, v = varianza(np.maximum(z, 0))
        mue = 0.5 * u * h * (1 - C)
        print(f"   {schema:13s} {C:5.2f}  {z.max():7.4f}   {np.max(np.abs(z - ze)):.3e}    {m:7.3f}     {v:7.4f}     {s0**2 + 2*mue*t:7.4f}")
    else:
        print(f"   {schema:13s} {C:5.2f}  instabile (max|zeta| = {np.nanmax(np.abs(z)):.1e})")

# ---------------------------------------------------------------- 3) verifica quantitativa della viscosita' artificiale
print("\nViscosita' artificiale misurata dalla crescita della varianza (upwind, t = 8):")
for C in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
    z, t = integra("upwind", C, t_fin)
    m, v = varianza(z)
    mu_mis = (v - s0**2) / (2 * t)
    print(f"   C = {C:4.2f}: mu_e misurata = {mu_mis:.5f}, prevista u h (1 - C)/2 = {0.5*u*h*(1-C):.5f}")

fig, ax = plt.subplots(1, 3, figsize=(8.6, 2.8))
S = propaga("upwind", 0.5, 4, N=11)
for k, col in zip(range(5), [GRIGIO, VERDE, ARANCIO, ROSSO, BLU]):
    ax[0].plot(np.arange(11) - 5, S[k], "o-", color=col, ms=3, lw=1, label=f"passo {k}")
ax[0].set_xlabel("$i-m$"); ax[0].set_ylabel("$\\zeta/\\epsilon$"); ax[0].legend(fontsize=6.5)
ax[0].set_title("upwind, $C=0{,}5$: perturbazione discreta")
ze = np.exp(-(x - x0 - u * t_fin)**2 / (2 * s0**2))
ax[1].plot(x, ze, "k", lw=1.6, label="esatta")
for (schema, C), col in ((("upwind", 0.5), BLU), (("upwind", 0.9), VERDE), (("lax-wendroff", 0.5), ARANCIO)):
    ax[1].plot(x, ris[(schema, C)], "--", color=col, lw=1.1, label=f"{schema}, $C={C}$".replace(".", "{,}"))
ax[1].set_xlim(8, 16); ax[1].set_xlabel("$x$"); ax[1].legend(fontsize=6.5)
ax[1].set_title("gaussiana dopo $t=8$")
Cs = np.linspace(0.05, 1.0, 20)
mu_m = []
for C in Cs:
    z, t = integra("upwind", C, t_fin)
    mu_m.append((varianza(z)[1] - s0**2) / (2 * t))
ax[2].plot(Cs, mu_m, "o", color=BLU, ms=3.5, label="misurata")
ax[2].plot(Cs, 0.5 * u * h * (1 - Cs), color=ROSSO, lw=1.2, label="$uh(1-C)/2$")
ax[2].set_xlabel("$C$"); ax[2].set_ylabel("$\\mu_e$"); ax[2].legend(fontsize=7)
ax[2].set_title("viscosità artificiale")
fig.tight_layout()
salva(fig, "prog4_2_upwind")
