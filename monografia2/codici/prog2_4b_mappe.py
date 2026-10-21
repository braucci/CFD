"""
prog2_4b_mappe.py -- esempi di trasformazioni conformi.

(a) z = zeta^2 : la corrente uniforme del semipiano superiore diventa il
    flusso nell'angolo retto (linee di corrente iperboliche 2xy = cost).
(b) z = zeta + b^2/zeta con b < a: il cerchio |zeta| = a diventa un'ellisse e
    il flusso attorno al cilindro diventa il flusso attorno al cilindro ellittico.
(c) la stessa trasformazione applicata a cerchi passanti per zeta = +-b:
    centro sull'asse immaginario -> arco di cerchio (lamina curva);
    centro fuori dagli assi -> profilo di Joukowski.
Le linee di corrente si tracciano nel piano zeta (dove il flusso e' noto) e
se ne trasformano i punti nel piano z.
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, ROSSO, GRIGIO, VERDE, VIOLA

U = 1.0


def linee_mappate(ax, zeta_grid, campo, livelli, mappa, colore, stile="-", lw=0.8):
    """Traccia nel piano z le curve di livello di 'campo' calcolate nel piano zeta."""
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(zeta_grid.real, zeta_grid.imag, campo, levels=livelli)
    for segs in cs.allsegs:
        for seg in segs:
            if len(seg) < 2:
                continue
            zm = mappa(seg[:, 0] + 1j * seg[:, 1])
            ax.plot(zm.real, zm.imag, stile, color=colore, lw=lw)
    plt.close(fig_tmp)


fig, axs = plt.subplots(2, 2, figsize=(6.4, 6.0))

# ---------------- (a) angolo retto ----------------------------------------
s = np.linspace(0, 2.5, 301)
ZETA = s[:, None] + 1j * s[None, :]            # primo quadrante del piano zeta
W = U * ZETA**2                                # w = U zeta^2 = U(x^2-y^2) + i 2U xy
ax = axs[0, 0]
ax.contour(ZETA.real, ZETA.imag, W.imag, levels=np.arange(0.5, 6.1, 0.5),
           colors=BLU, linewidths=0.8, linestyles="solid")
ax.contour(ZETA.real, ZETA.imag, W.real, levels=np.arange(-6, 6.1, 0.5),
           colors=ARANCIO, linewidths=0.6, linestyles="--")
ax.plot([0, 0, 2.5], [2.5, 0, 0], color="k", lw=2)
ax.set_aspect("equal"); ax.set_title(r"(a) $w=U\zeta^2$: angolo di $90^\circ$")
ax.set_xlim(0, 2.5); ax.set_ylim(0, 2.5); ax.grid(False)

# ---------------- (b) ellisse -----------------------------------------------
a, b = 1.0, 0.8
r = np.linspace(a, 4.0, 300)
t = np.linspace(0, 2 * np.pi, 721)
ZC = r[:, None] * np.exp(1j * t[None, :])     # esterno del cerchio
PSI = (U * (ZC + a**2 / ZC)).imag
ax = axs[0, 1]
linee_mappate(ax, ZC, PSI, np.linspace(-3, 3, 13), lambda zz: zz + b**2 / zz, BLU)
ell = a * np.exp(1j * t) + b**2 / (a * np.exp(1j * t))
ax.fill(ell.real, ell.imag, color=GRIGIO, alpha=0.4); ax.plot(ell.real, ell.imag, "k", lw=0.8)
ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-2.5, 2.5); ax.grid(False)
ax.set_title(f"(b) cilindro ellittico ($b/a={b/a:.1f}$)")
print(f"ellisse: semiassi {a + b**2/a:.3f} e {a - b**2/a:.3f}")

# ---------------- (c) arco di cerchio e profilo ------------------------------
b = 0.8
ax = axs[1, 0]
for centro, col, nome in ((0.0 + 0.3j, VERDE, "arco"),
                          (-0.15 + 0.3j, BLU, "profilo")):
    rag = abs(b - centro)                      # il cerchio passa per zeta = b
    cerchio = centro + rag * np.exp(1j * t)
    ax.plot(cerchio.real, cerchio.imag, color=col, lw=1.0)
    ax.plot(centro.real, centro.imag, "+", color=col)
ax.plot([-b, b], [0, 0], "o", color=ROSSO, ms=3)
ax.set_aspect("equal"); ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.2, 1.6); ax.grid(False)
ax.set_title(r"(c) piano $\zeta$: cerchi per $\zeta=\pm b$")
ax = axs[1, 1]
for centro, col in ((0.0 + 0.3j, VERDE), (-0.15 + 0.3j, BLU)):
    rag = abs(b - centro)
    cerchio = centro + rag * np.exp(1j * t)
    zz = cerchio + b**2 / cerchio
    ax.plot(zz.real, zz.imag, color=col, lw=1.2)
ax.plot([-2 * b, 2 * b], [0, 0], "o", color=ROSSO, ms=3)
ax.set_aspect("equal"); ax.set_xlim(-2.2, 2.2); ax.set_ylim(-0.6, 1.0); ax.grid(False)
ax.set_title(r"(d) piano $z$: arco e profilo di Joukowski")
salva(fig, "prog2_4b_mappe")
