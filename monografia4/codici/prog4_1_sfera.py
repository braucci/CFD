"""
prog4_1_sfera.py -- Programma 4.1: corrente attorno a una sfera a numeri di
Reynolds finiti con il metodo di Galerkin (funzione di prova di Kawaguti).

Funzione di corrente di prova (adimensionale, Re = 2 U a / nu):
    psi = (r^2/2 + A1/r + A2/r^2 + A3/r^3 + A4/r^4)(1 - z^2)
          + (B1/r + B2/r^2 + B3/r^3 + B4/r^4) z (1 - z^2),     z = cos(theta).
Le condizioni di aderenza e l'annullamento del residuo sulla sfera forniscono sei
relazioni lineari; le due condizioni di ortogonalita' di Galerkin danno il sistema
non lineare (vedi prog4_1b_galerkin_simbolico.py per la derivazione)
    g(A1, B1) = 9.931035 A1/Re + 0.011713 B1 - 0.002546 A1 B1 + 44.689656/Re = 0
    h(A1, B1) = 0.050148 A1 + 2.201058 B1/Re + 0.002421 A1^2 - 0.000376 B1^2 + 0.379018 = 0
risolto con il metodo di Newton-Raphson (NEWTN2). Il programma calcola per
20 numeri di Reynolds i coefficienti di resistenza di pressione, di attrito e
totale, la lunghezza r_s della bolla di ricircolo (radice di u_r = 0 sull'asse,
trovata con NEWTN1) e traccia le linee di corrente per Re = 10, 100, 300.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

# coefficienti del sistema ridotto (4 cifre del testo; la derivazione simbolica
# fornisce 9.931034, 0.01171243, 0.002545953, 44.68966; 0.05015221, 2.201058,
# 0.00242145, 0.0003756646, 0.3790197)
C = dict(a1=9.931035, a2=0.011713, a3=0.002546, a4=44.689656,
         b1=0.050148, b2=2.201058, b3=0.002421, b4=0.000376, b5=0.379018)


def g(x, y, Re):
    return C["a1"] * x / Re + C["a2"] * y - C["a3"] * x * y + C["a4"] / Re


def h(x, y, Re):
    return C["b1"] * x + C["b2"] * y / Re + C["b3"] * x * x - C["b4"] * y * y + C["b5"]


def gx(x, y, Re):
    return C["a1"] / Re - C["a3"] * y


def hy(x, y, Re):
    return C["b2"] / Re - 2 * C["b4"] * y


def newtn1(f, fp, x, eps=1e-6, itmax=100):
    """Newton-Raphson per una sola equazione f(x) = 0."""
    for it in range(1, itmax + 1):
        dx = -f(x) / fp(x)
        x += dx
        if abs(dx) < eps:
            return x, it
    raise RuntimeError("NEWTN1: convergenza non raggiunta")


def newtn2(x, y, Re, eps=1e-6, itmax=200):
    """Newton-Raphson 'disaccoppiato' del testo per g(x,y) = 0, h(x,y) = 0:
    x aggiornato con dg/dx, poi y con dh/dy usando il nuovo x."""
    for it in range(1, itmax + 1):
        xn = x - g(x, y, Re) / gx(x, y, Re)
        yn = y - h(xn, y, Re) / hy(xn, y, Re)
        dx, dy = abs(xn - x), abs(yn - y)
        x, y = xn, yn
        if dx < eps and dy < eps:
            return x, y, it
    raise RuntimeError("NEWTN2: convergenza non raggiunta")


def coefficienti(A1, B1):
    A2 = -15 / 29 * (8 + 5 * A1)
    A3 = 9 / 29 * (17 + 7 * A1)
    A4 = -(95 + 34 * A1) / 58
    B2 = -23 / 9 * B1
    B3 = 19 / 9 * B1
    B4 = -5 / 9 * B1
    return np.array([A1, A2, A3, A4]), np.array([B1, B2, B3, B4])


def psi(r, z, A, B):
    return (r**2 / 2 + A[0] / r + A[1] / r**2 + A[2] / r**3 + A[3] / r**4) * (1 - z**2) \
        + (B[0] / r + B[1] / r**2 + B[2] / r**3 + B[3] / r**4) * z * (1 - z**2)


def ur_asse(r, A, B):
    """u_r/U sull'asse a valle (theta = 0): (4.1.44) a meno del fattore 2."""
    return 0.5 + (A[0] + B[0]) / r**3 + (A[1] + B[1]) / r**4 + (A[2] + B[2]) / r**5 + (A[3] + B[3]) / r**6


def dur_asse(r, A, B):
    return -3 * (A[0] + B[0]) / r**4 - 4 * (A[1] + B[1]) / r**5 - 5 * (A[2] + B[2]) / r**6 - 6 * (A[3] + B[3]) / r**7


Res = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
print("    Re        A1          B1         CDP        CDS         CD         RS    iter")
x, y = -5.0, -0.5                       # valori di partenza (poi si usa la soluzione precedente)
risultati = []
for Re in Res:
    x, y, it = newtn2(x, y, Re)
    # verifica con un risolutore di libreria (Newton completo con jacobiano numerico)
    xs, ys = fsolve(lambda v: [g(v[0], v[1], Re), h(v[0], v[1], Re)], [x, y], xtol=1e-10)
    A, B = coefficienti(x, y)
    cdp = 32 / (3 * Re) * (8 * A[1] + 25 * A[2] + 54 * A[3])
    cds = 64 / (3 * Re) * (2 * A[1] + 5 * A[2] + 9 * A[3])
    # lunghezza della bolla: radice di u_r = 0 con r > 1 (se esiste)
    rs = 1.0
    rr = np.linspace(1.002, 6.0, 2000)      # si parte poco oltre la superficie (u_r = 0 in r = 1)
    u = ur_asse(rr, A, B)
    seg = np.nonzero((u[:-1] < 0) & (u[1:] >= 0))[0]
    if len(seg):
        rs, _ = newtn1(lambda r: ur_asse(r, A, B), lambda r: dur_asse(r, A, B), rr[seg[-1]])
    risultati.append((Re, x, y, cdp, cds, cdp + cds, rs, A, B))
    print(f"{Re:6.0f} {x:11.4e} {y:11.4e} {cdp:10.4e} {cds:10.4e} {cdp+cds:10.4e} {rs:8.4f} {it:5d}"
          f"   (fsolve: {abs(xs-x)+abs(ys-y):.1e})")

# confronto con la correlazione sperimentale di Schiller-Naumann (Re < 800)
print("\nConfronto con la correlazione sperimentale C_D = 24/Re (1 + 0,15 Re^0,687):")
for Re, x, y, cdp, cds, cd, rs, A, B in risultati:
    if Re in (5, 10, 20, 50, 100, 200, 500, 1000):
        cde = 24 / Re * (1 + 0.15 * Re**0.687)
        print(f"   Re = {Re:5.0f}: C_D Galerkin = {cd:.3f}, sperimentale = {cde:.3f}, scarto {100*(cd/cde-1):+.1f} %;"
              f" quota di pressione {100*cdp/cd:.0f} %")
# separazione: sulla sfera u_r = 0 e du_r/dr = 0 (aderenza); il segno di u_r subito
# dietro la sfera e' quello di d^2 u_r/dr^2 in r = 1, che cambia segno alla separazione
def curvatura(Re, x0, y0):
    xx, yy, _ = newtn2(x0, y0, Re)
    A, B = coefficienti(xx, yy)
    return 12 * (A[0] + B[0]) + 20 * (A[1] + B[1]) + 30 * (A[2] + B[2]) + 42 * (A[3] + B[3])
for k in range(1, len(risultati)):
    if risultati[k][6] > 1.0 + 1e-9:
        Re_a, Re_b = risultati[k - 1][0], risultati[k][0]
        x0, y0 = risultati[k - 1][1], risultati[k - 1][2]
        lo, hi = Re_a, Re_b
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if curvatura(mid, x0, y0) * curvatura(lo, x0, y0) > 0:
                lo = mid
            else:
                hi = mid
        print(f"\nSeparazione (r_s > 1) fra Re = {Re_a:.0f} e {Re_b:.0f}: Re_sep = {0.5*(lo+hi):.2f}"
              f" (esperimenti di Taneda: circa 24)")
        break

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(8.6, 2.6))
xg = np.linspace(-3, 3, 301)
yg = np.linspace(0, 3, 151)
X, Y = np.meshgrid(xg, yg)
R = np.sqrt(X**2 + Y**2)
Z = np.where(R > 0, X / np.maximum(R, 1e-9), 0)
for a_, Re in zip(ax, (10, 100, 300)):
    ris = [t for t in risultati if t[0] == Re][0]
    A, B = ris[7], ris[8]
    PSI = psi(np.maximum(R, 1.0), Z, A, B)
    PSI = np.where(R < 1, np.nan, PSI)
    lv = [-0.01, -0.003, -0.0005, 0.0, 0.05, 0.15, 0.3, 0.6, 1.0, 1.5, 2.2, 3.0, 4.0]
    a_.contour(X, Y, PSI, levels=lv, colors=[ROSSO if v < 0 else BLU for v in lv], linewidths=0.8)
    a_.add_patch(plt.Circle((0, 0), 1, color=GRIGIO))
    a_.set_aspect("equal")
    a_.set_xlim(-3, 3); a_.set_ylim(0, 3)
    a_.set_title(f"$Re={Re}$, $r_s={ris[6]:.2f}$".replace(".", "{,}"))
    a_.set_xlabel("$x/a$")
ax[0].set_ylabel("$y/a$")
fig.tight_layout()
salva(fig, "prog4_1_sfera_linee")

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
Rv = np.array([t[0] for t in risultati])
ax[0].loglog(Rv, [t[5] for t in risultati], "o-", color=BLU, ms=3, label="Galerkin, $C_D$")
ax[0].loglog(Rv, [t[3] for t in risultati], "s--", color=ARANCIO, ms=3, label="pressione")
ax[0].loglog(Rv, [t[4] for t in risultati], "^--", color=VERDE, ms=3, label="attrito")
Rf = np.logspace(0.5, 3, 100)
ax[0].loglog(Rf, 24 / Rf * (1 + 0.15 * Rf**0.687), "k:", lw=1.2, label="Schiller--Naumann")
ax[0].loglog(Rf, 24 / Rf, color=GRIGIO, lw=0.8, label="Stokes $24/Re$")
ax[0].set_xlabel("$Re$"); ax[0].set_ylabel("$C_D$")
ax[0].legend(fontsize=6.5)
ax[1].semilogx(Rv, [t[6] for t in risultati], "o-", color=BLU, ms=3)
ax[1].set_xlabel("$Re$"); ax[1].set_ylabel("$r_s/a$")
ax[1].set_title("lunghezza della bolla di ricircolo")
fig.tight_layout()
salva(fig, "prog4_1_sfera_cd")
