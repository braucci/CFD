"""
prog2_2_doppiette.py -- PROGRAMMA 2.2
Metodo inverso di sovrapposizione: corrente uniforme + cinque doppiette
sull'asse x. Si calcola la funzione di corrente adimensionale sui nodi di una
griglia rettangolare, si cercano i punti di cinque linee di corrente lungo le
verticali della griglia (ricerca con interpolazione lineare) e si disegnano.
Si calcola poi il coefficiente di pressione sul corpo (Bernoulli).

  Psi = Y - sum_i c_i Y / ((X - d_i)^2 + Y^2)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from stile_grafici import salva, BLU, ARANCIO, VERDE, VIOLA, ROSSO, GRIGIO
from metodi_numerici import cerca_x

D = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])     # posizioni d_i delle doppiette
C = np.array([0.15, 0.30, 0.20, 0.10, 0.05])  # intensita' adimensionali c_i
EPS = 1e-6       # spostamento delle doppiette: i loro centri non cadono sui nodi


def Psi(X, Y, alfa=0.0, d=D, c=C):
    """Funzione di corrente: corrente uniforme inclinata di alfa + doppiette."""
    p = Y * np.cos(alfa) - X * np.sin(alfa)
    for di, ci in zip(d, c):
        p = p - ci * Y / ((X - di - EPS)**2 + Y**2)
    return p


def velocita(X, Y, d=D, c=C):
    """u = dPsi/dY, v = -dPsi/dX (derivate analitiche, corrente con alfa = 0)."""
    u = np.ones_like(X)
    v = np.zeros_like(X)
    for di, ci in zip(d, c):
        xr = X - di - EPS
        r2 = xr**2 + Y**2
        u -= ci * (xr**2 - Y**2) / r2**2
        v -= ci * 2.0 * xr * Y / r2**2
    return u, v


# ----- griglia e funzione di corrente nei nodi ------------------------------
DX = DY = 0.2
x = np.arange(-3.0, 3.0 + 1e-9, DX)
y = np.arange(-2.0, 2.0 + 1e-9, DY)
X, Y = np.meshgrid(x, y, indexing="ij")
PSI = Psi(X, Y)

COSTANTI = [0.0, -0.5, -1.0, 0.5, 1.0]
SIMBOLI = ["s", "o", "D", "v", "^"]
COLORI = [ROSSO, BLU, VIOLA, ARANCIO, VERDE]

fig, ax = plt.subplots(figsize=(6.0, 4.0))
for psia, simb, col in zip(COSTANTI, SIMBOLI, COLORI):
    xx, yy = cerca_x(x, y, PSI, psia)
    ax.plot(xx, yy, simb, ms=3.2, color=col, mfc="none", label=fr"$\Psi={psia:g}$")
    if psia == 0.0:
        print(f"Psi = 0: {len(xx)} punti trovati (compresi quelli sull'asse x)")
# confronto: contorno 'esatto' tracciato su una griglia molto fine
xf = np.linspace(-3, 3, 601); yf = np.linspace(-2, 2, 401)
XF, YF = np.meshgrid(xf, yf, indexing="ij")
ax.contour(XF, YF, Psi(XF, YF), levels=sorted(COSTANTI), colors=GRIGIO, linewidths=0.5,
           linestyles="solid")
ax.plot(D, 0 * D, "k+", ms=6)
ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-2, 2)
ax.set_xlabel("$X = x/l$"); ax.set_ylabel("$Y = y/l$")
ax.legend(loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.13), handletextpad=0.2)
salva(fig, "prog2_2_doppiette")

# ----- forma del corpo: estremi e spessore -----------------------------------
# sull'asse il corpo inizia e finisce nei punti di ristagno u(X,0) = 0
u_asse = lambda X: velocita(np.array([X]), np.array([0.0]))[0][0]
x_ant = brentq(u_asse, -3.0, -1.01)
x_post = brentq(u_asse, 1.01, 3.0)
# semispessore: per ogni X la radice Y>0 di Psi(X,Y)=0
Xc = np.linspace(x_ant + 1e-4, x_post - 1e-4, 400)
Yc = np.array([brentq(lambda yy: Psi(xc, yy), 1e-6, 2.0) for xc in Xc])
k = np.argmax(Yc)
print(f"punti di ristagno: X = {x_ant:.4f} e X = {x_post:.4f}; "
      f"lunghezza = {x_post - x_ant:.4f} l")
print(f"spessore massimo = {2*Yc[k]:.4f} l in X = {Xc[k]:.3f}; "
      f"rapporto di finezza = {(x_post-x_ant)/(2*Yc[k]):.3f}")

# ----- coefficiente di pressione sul corpo (Bernoulli) -----------------------
u, v = velocita(Xc, Yc)
cp = 1.0 - (u**2 + v**2)
print(f"C_p minimo = {cp.min():.4f} in X = {Xc[np.argmin(cp)]:.3f}")
fig, ax = plt.subplots(2, 1, figsize=(5.6, 5.4),
                       gridspec_kw={"height_ratios": [1.0, 1.0]})
ax[0].fill_between(Xc, -Yc, Yc, color=GRIGIO, alpha=0.3)
ax[0].plot(Xc, Yc, color=BLU); ax[0].plot(Xc, -Yc, color=BLU)
ax[0].plot(D, 0 * D, "k+", ms=5)
ax[0].set_xlim(-1.6, 1.4); ax[0].set_ylim(-0.85, 0.85)
ax[0].set_aspect("equal"); ax[0].set_ylabel("$Y$")
ax[1].set_xlim(-1.6, 1.4)
ax[1].plot(Xc, cp, color=ARANCIO)
ax[1].invert_yaxis()
ax[1].set_xlabel("$X$"); ax[1].set_ylabel("$C_p$")
salva(fig, "prog2_2_cp_corpo")

# ----- una sola doppietta e corrente inclinata di 30 gradi -------------------
alfa = np.radians(30.0)
PSI30 = Psi(X, Y, alfa, d=[0.0], c=[1.0])
fig, ax = plt.subplots(figsize=(6.0, 4.0))
for psia, simb, col in zip(COSTANTI, SIMBOLI, COLORI):
    xx, yy = cerca_x(x, y, PSI30, psia)
    ax.plot(xx, yy, simb, ms=3.2, color=col, mfc="none", label=fr"$\Psi={psia:g}$")
ax.contour(XF, YF, Psi(XF, YF, alfa, d=[0.0], c=[1.0]), levels=sorted(COSTANTI),
           colors=GRIGIO, linewidths=0.5, linestyles="solid")
ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-2, 2)
ax.set_xlabel("$X$"); ax.set_ylabel("$Y$")
ax.legend(loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.13), handletextpad=0.2)
salva(fig, "prog2_2_doppietta30")
