"""
prog3_8b_curva_neutra.py -- curva di stabilita' neutra della corrente di
Poiseuille piana (luogo dei punti in cui (lam_R)max = c_I = 0) e punto critico.

1) Si calcola c_I,max = max lam_R su una griglia (log Re, alfa) e si traccia la
   linea di livello zero.
2) Il punto critico (Re minimo della curva) si ottiene risolvendo, per alfa
   fissato, c_I,max(Re, alfa) = 0 con il metodo di Brent e minimizzando poi
   Re(alfa) con la ricerca della sezione aurea.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar
from prog3_8_orr_sommerfeld import orr_sommerfeld
from stile_grafici import salva, BLU, ROSSO, GRIGIO

N = 80


def ci_max(Re, alfa):
    lam, _, _ = orr_sommerfeld(Re, alfa, N)
    return lam.real.max()


def Re_neutro(alfa):
    """Ramo inferiore della curva neutra per il numero d'onda alfa."""
    return brentq(lambda lr: ci_max(np.exp(lr), alfa), np.log(5000.0), np.log(12000.0), xtol=1e-7)


ris = minimize_scalar(lambda a: Re_neutro(a), bracket=(0.98, 1.02, 1.06), tol=1e-5)
alfa_c = ris.x
Re_c = np.exp(Re_neutro(alfa_c))
print(f"Punto critico: Re_c = {Re_c:.2f},  alfa_c = {alfa_c:.4f}")
lam, _, _ = orr_sommerfeld(Re_c, alfa_c, N)
c = 1j * lam[np.argmax(lam.real)]
print(f"   velocita' di fase dell'onda critica c_R = {c.real:.5f} U_c")

logRe = np.linspace(np.log10(3000), np.log10(1.0e5), 46)
alfas = np.linspace(0.3, 1.3, 41)
CI = np.zeros((len(alfas), len(logRe)))
for i, a in enumerate(alfas):
    for j, lr in enumerate(logRe):
        CI[i, j] = ci_max(10**lr, a)
print(f"c_I massimo sulla griglia: {CI.max():.4f} "
      f"(Re = {10**logRe[np.unravel_index(CI.argmax(), CI.shape)[1]]:.0f},"
      f" alfa = {alfas[np.unravel_index(CI.argmax(), CI.shape)[0]]:.2f})")
for Re in (1.0e4, 3.0e4, 1.0e5):
    j = np.argmin(np.abs(10**logRe - Re))
    inst = alfas[CI[:, j] > 0]
    if inst.size:
        print(f"   Re = {Re:8.0f}: onde instabili per {inst.min():.2f} < alfa < {inst.max():.2f}")

fig, ax = plt.subplots(figsize=(4.8, 3.4))
cs = ax.contourf(10**logRe, alfas, CI, levels=[0, 0.002, 0.004, 0.006, 0.008], cmap="Reds", alpha=0.6)
ax.contour(10**logRe, alfas, CI, levels=[0.0], colors=BLU, linewidths=1.5)
ax.plot(Re_c, alfa_c, "o", color="k", ms=4)
ax.annotate(f"({Re_c:.0f}; {alfa_c:.3f})".replace(".", ","), (Re_c, alfa_c), xytext=(7000, 1.18),
            fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6))
ax.set_xscale("log")
ax.set_xlabel("$Re=U_c h/\\nu$")
ax.set_ylabel(r"numero d'onda $\alpha$")
ax.text(3.0e4, 0.85, "instabile", fontsize=8, color=ROSSO)
ax.text(4.0e3, 0.45, "stabile", fontsize=8, color=GRIGIO)
fig.colorbar(cs, ax=ax, label="$c_I$")
salva(fig, "prog3_8b_curva_neutra")
