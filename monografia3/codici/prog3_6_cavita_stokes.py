"""
prog3_6_cavita_stokes.py -- PROGRAMMA 3.6
Flusso di Stokes in una cavita' quadrata 1 x 1 mossa dalla parete superiore
(velocita' 1 verso x positivo). Equazione biarmonica scritta come coppia
    lap(psi) = -zeta,    lap(zeta) = 0,
con psi = 0 su tutto il contorno e vorticita' di parete calcolata da psi con
le formule a cinque punti (accuratezza O(h^2)), per esempio sulla parete mobile
    zeta_(i,m) = [ -psi_(i-1,m-1) + 8/3 psi_(i,m-1) - psi_(i+1,m-1) - 2/3 psi_(i,m-2) ]/h^2
                 - 2/(3h) (dpsi/dy)_(i,m),     (dpsi/dy) = 1.
Iterazione esterna: (1) vorticita' di parete da psi; (2) Liebmann per zeta;
(3) Liebmann per psi; finche' le somme delle variazioni ERZETA ed ERPSI non
scendono sotto EPSLON. Come nella subroutine originale, ogni chiamata di
Liebmann riparte da valori interni nulli.
"""
import numpy as np
import matplotlib.pyplot as plt
from metodi_numerici import liebmann
from stile_grafici import salva, BLU, ROSSO, GRIGIO

M = 11
H = 1.0 / (M - 1)
EPSLON = 1.0e-3
RF = 1.0                                     # fattore di rilassamento (1: nessuno)

psi = np.zeros((M, M))                       # psi[i, j]: i -> x, j -> y
zeta = np.zeros((M, M))


def vorticita_parete(psi, zeta):
    i = np.arange(1, M - 1)
    j = np.arange(1, M - 1)
    zeta[0, j] = (-psi[1, j + 1] + 8 / 3 * psi[1, j] - psi[1, j - 1] - 2 / 3 * psi[2, j]) / H**2
    zeta[M - 1, j] = (-psi[M - 2, j + 1] + 8 / 3 * psi[M - 2, j] - psi[M - 2, j - 1]
                      - 2 / 3 * psi[M - 3, j]) / H**2
    zeta[i, 0] = (-psi[i + 1, 1] + 8 / 3 * psi[i, 1] - psi[i - 1, 1] - 2 / 3 * psi[i, 2]) / H**2
    zeta[i, M - 1] = ((-psi[i + 1, M - 2] + 8 / 3 * psi[i, M - 2] - psi[i - 1, M - 2]
                       - 2 / 3 * psi[i, M - 3]) / H**2 - 2 / (3 * H) * 1.0)


iterazioni = 0
erzeta = erpsi = 1.0
storia = []
while erzeta > EPSLON or erpsi > EPSLON:
    iterazioni += 1
    vorticita_parete(psi, zeta)
    zeta_vecchia = zeta.copy()
    zeta, _ = liebmann(zeta, np.zeros((M, M)), H, 0.001)
    zeta = RF * zeta + (1 - RF) * zeta_vecchia
    erzeta = np.sum(np.abs(zeta - zeta_vecchia))
    psi_vecchia = psi.copy()
    psi, _ = liebmann(psi, -zeta, H, 0.001)
    psi = RF * psi + (1 - RF) * psi_vecchia
    erpsi = np.sum(np.abs(psi - psi_vecchia))
    storia.append((iterazioni, erzeta, erpsi))

# vorticita' negli spigoli: media dei due valori adiacenti
zeta[0, 0] = 0.5 * (zeta[0, 1] + zeta[1, 0])
zeta[M - 1, 0] = 0.5 * (zeta[M - 1, 1] + zeta[M - 2, 0])
zeta[0, M - 1] = 0.5 * (zeta[1, M - 1] + zeta[0, M - 2])
zeta[M - 1, M - 1] = 0.5 * (zeta[M - 1, M - 2] + zeta[M - 2, M - 1])

print(f"Funzione di corrente dopo {iterazioni} iterazioni, ERPSI = {erpsi:.4e}")
for j in range(M - 1, -1, -1):
    print("  " + " ".join(f"{psi[i, j]:+.4f}" for i in range(M)))
print(f"\nVorticita', ERZETA = {erzeta:.4e}")
for j in range(M - 1, -1, -1):
    print("  " + " ".join(f"{zeta[i, j]:+7.3f}" for i in range(M)))
i0, j0 = np.unravel_index(np.argmin(psi), psi.shape)
print(f"\npsi minimo = {psi.min():.5f} nel nodo x = {i0*H:.1f}, y = {j0*H:.1f}")
print(f"asimmetria massima |psi(x) - psi(1-x)| = {np.max(np.abs(psi - psi[::-1, :])):.1e}")
print(f"asimmetria massima |zeta(x) - zeta(1-x)| = {np.max(np.abs(zeta - zeta[::-1, :])):.1e}")

# ------------------------------------------------------------------ figura
x = np.linspace(0, 1, M)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4))
livelli = [-0.090, -0.075, -0.060, -0.045, -0.030, -0.015, 0.0]
cs = ax[0].contour(x, x, psi.T, levels=livelli, colors=BLU, linewidths=1.0, linestyles="solid")
ax[0].clabel(cs, fmt=lambda v: f"{v:.3f}".replace(".", ","), fontsize=6)
ax[0].set_title(r"funzione di corrente $\psi$ (griglia $11\times 11$)")
cz = ax[1].contour(x, x, zeta.T, levels=np.arange(-7, 2, 1), colors=ROSSO, linewidths=0.9,
                   linestyles="solid")
ax[1].clabel(cz, fmt="%d", fontsize=6)
ax[1].set_title(r"vorticita' $\zeta$")
for a in ax:
    a.set_aspect("equal")
    a.set_xlabel("$x$")
    a.set_ylabel("$y$")
    a.annotate("", xy=(0.7, 1.0), xytext=(0.3, 1.0),
               arrowprops=dict(arrowstyle="->", color="k", lw=1.2), annotation_clip=False)
    a.set_title(a.get_title(), pad=10)
fig.tight_layout()
salva(fig, "prog3_6_cavita_stokes")

fig, ax = plt.subplots(figsize=(4.0, 2.6))
it = [s[0] for s in storia]
ax.semilogy(it, [s[1] for s in storia], color=ROSSO, label="ERZETA")
ax.semilogy(it, [s[2] for s in storia], color=BLU, label="ERPSI")
ax.axhline(EPSLON, color=GRIGIO, ls=":")
ax.set_xlabel("iterazione esterna")
ax.set_ylabel("somma delle variazioni")
ax.legend()
salva(fig, "prog3_6_convergenza")
