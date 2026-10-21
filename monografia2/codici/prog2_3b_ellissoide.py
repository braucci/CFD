"""
prog2_3b_ellissoide.py -- il metodo di von Karman applicato a un corpo
"a dirigibile": l'ellissoide di rivoluzione allungato (semiassi a, b; finezza
a/b = 4), per il quale la soluzione esatta e' nota.

  velocita' sulla superficie:  V/U = (1+k) sqrt[(1 - z^2/a^2)/(1 - e^2 z^2/a^2)]
  con e = sqrt(1 - b^2/a^2),  1 + k = 2/(2 - alfa0),
  alfa0 = 2(1-e^2)/e^3 [ (1/2) ln((1+e)/(1-e)) - e ].
  La distribuzione esatta di sorgenti e' lineare fra i fuochi z = -ae, +ae:
  q(zeta) = -K zeta,  K = U / [2 e a^2/b^2 - ln((1+e)/(1-e))].
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, ROSSO, GRIGIO, VERDE, VIOLA
from prog2_3_karman_sfera import karman_risolvi, karman_velocita

U, A, B = 1.0, 1.0, 0.25
e = np.sqrt(1 - (B / A)**2)
alfa0 = 2 * (1 - e**2) / e**3 * (0.5 * np.log((1 + e) / (1 - e)) - e)
uno_piu_k = 2 / (2 - alfa0)
K = U / (2 * e * A**2 / B**2 - np.log((1 + e) / (1 - e)))
print(f"eccentricita' e = {e:.5f}, fuochi in z = +-{A*e:.4f}")
print(f"alfa0 = {alfa0:.5f}, 1 + k = {uno_piu_k:.5f}, K = {K:.5f}")


def V_esatta(z):
    return U * uno_piu_k * np.sqrt((1 - z**2 / A**2) / (1 - e**2 * z**2 / A**2))


def calcola(n, frazione=1.0):
    """n segmenti uguali fra -c e +c (c = frazione * a e); n-1 punti di
    controllo equispaziati in z sulla superficie."""
    c = frazione * A * e
    zz = np.linspace(-c, c, n + 1)
    zi = np.linspace(-A, A, n + 1)[1:-1]
    ri = B * np.sqrt(1 - zi**2 / A**2)
    q, M = karman_risolvi(zz, ri, zi)
    return zz, q, M, zi, ri


zt = np.linspace(-0.995, 0.995, 600) * A          # punti di verifica
rt = B * np.sqrt(1 - zt**2 / A**2)
print("\n frazione   n    cond(M)      max|V-Vex|/U   max|q|/U")
risultati = {}
for frazione in (1.0, 0.9):
    for n in (6, 10, 20, 40):
        zz, q, M, zi, ri = calcola(n, frazione)
        ur, uz = karman_velocita(q, zz, rt, zt, U)
        err = np.max(np.abs(np.hypot(ur, uz) - V_esatta(zt)))
        risultati[(frazione, n)] = (zz, q, ur, uz)
        print(f"   {frazione:.1f}    {n:3d}   {np.linalg.cond(M):.2e}     {err:.2e}      {np.max(np.abs(q)):.2e}")

# ----- figura 1: forma, V/U e C_p per n = 10 ---------------------------------
zz, q, ur, uz = risultati[(1.0, 10)]
V = np.hypot(ur, uz)
fig, ax = plt.subplots(2, 1, figsize=(5.8, 4.8), sharex=True,
                       gridspec_kw={"height_ratios": [1, 2.2]})
ax[0].fill_between(zt, -rt, rt, color=GRIGIO, alpha=0.3)
ax[0].plot(zt, rt, color=BLU); ax[0].plot(zt, -rt, color=BLU)
for j in range(len(q)):
    ax[0].plot(zz[j:j + 2], [0, 0], color=ROSSO if q[j] > 0 else BLU, lw=3,
               solid_capstyle="butt")
ax[0].plot([-A * e, A * e], [0, 0], "k|", ms=8)
ax[0].set_ylim(-0.4, 0.4); ax[0].set_ylabel("$r/a$")
ax[1].plot(zt, V_esatta(zt), color=GRIGIO, lw=2.2, label="$V/U$ esatta")
ax[1].plot(zt, V, "--", color=BLU, label="$V/U$, von Karman ($n=10$)")
ax[1].plot(zt, 1 - V_esatta(zt)**2, color=GRIGIO, lw=2.2)
ax[1].plot(zt, 1 - V**2, "--", color=ARANCIO, label="$C_p$, von Karman ($n=10$)")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_xlabel("$z/a$"); ax[1].legend(loc="center")
salva(fig, "prog2_3b_ellissoide")

# ----- figura 2: intensita' dei segmenti -------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.8))
for n, col, mk in ((10, BLU, "o"), (40, ARANCIO, "s")):
    zz, q, _, _ = risultati[(1.0, n)]
    ax[0].plot(0.5 * (zz[1:] + zz[:-1]), q, mk, color=col, ms=3.5, label=f"$n={n}$")
zl = np.linspace(-A * e, A * e, 10)
ax[0].plot(zl, -K * zl, color=GRIGIO, label=r"esatta $-K\zeta$")
ax[0].set_xlabel(r"$\zeta/a$"); ax[0].set_ylabel("$q/U$"); ax[0].legend()
ax[0].set_title("segmenti fra i fuochi")
zz, q, _, _ = risultati[(0.9, 20)]
ax[1].plot(0.5 * (zz[1:] + zz[:-1]), q, "o-", color=ROSSO, ms=3.5, lw=0.8)
ax[1].set_xlabel(r"$\zeta/a$"); ax[1].set_ylabel("$q/U$")
ax[1].set_title(r"segmenti troppo corti ($c=0{,}9\,ae$, $n=20$)")
salva(fig, "prog2_3b_intensita")
