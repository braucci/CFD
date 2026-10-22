"""
prog3_6b_cavita_fine.py -- la cavita' di Stokes su griglie fini.

Poiche' il problema e' lineare, le equazioni discrete
    lap_h(psi) + zeta = 0,   lap_h(zeta) = 0   (nodi interni)
    zeta_parete = formula di parete (lineare in psi)
formano UN sistema lineare sparso, risolto qui con un metodo diretto invece
che con le iterazioni annidate di Liebmann. Si confrontano due formule di parete:
  * formula a cinque punti del Programma 3.6 (errore O(h^2));
  * formula di Thom, zeta_w = -2 psi_1/h^2 - 2 U_w/h (ottenuta da Taylor lungo la normale).
Si studiano la convergenza di psi_min, l'estrapolazione di Richardson e i
vortici secondari (di Moffatt) negli spigoli inferiori.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from stile_grafici import salva, BLU, ARANCIO, ROSSO, VERDE, GRIGIO


def risolvi(n, formula="cinque"):
    """Griglia di (n+1) x (n+1) nodi, h = 1/n. Incognite: psi nei nodi interni,
    zeta in tutti i nodi tranne gli spigoli. Restituisce psi, zeta (n+1 x n+1)."""
    h = 1.0 / n
    ip = -np.ones((n + 1, n + 1), dtype=int)        # indici delle incognite psi
    iz = -np.ones((n + 1, n + 1), dtype=int)        # indici delle incognite zeta
    k = 0
    for i in range(1, n):
        for j in range(1, n):
            ip[i, j] = k
            k += 1
    for i in range(n + 1):
        for j in range(n + 1):
            spigolo = (i in (0, n)) and (j in (0, n))
            if not spigolo:
                iz[i, j] = k
                k += 1
    righe, colonne, valori = [], [], []
    b = np.zeros(k)
    r = 0

    def aggiungi(riga, col, val):
        righe.append(riga)
        colonne.append(col)
        valori.append(val)

    def P(i, j):           # contributo di psi[i,j] (zero se sul contorno)
        return ip[i, j]

    # equazioni nei nodi interni
    for i in range(1, n):
        for j in range(1, n):
            # (1) lap psi + zeta = 0
            aggiungi(r, ip[i, j], -4.0 / h**2)
            for (a, c) in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if P(a, c) >= 0:
                    aggiungi(r, P(a, c), 1.0 / h**2)
            aggiungi(r, iz[i, j], 1.0)
            r += 1
            # (2) lap zeta = 0
            aggiungi(r, iz[i, j], -4.0)
            for (a, c) in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                aggiungi(r, iz[a, c], 1.0)
            r += 1
    # equazioni di parete per la vorticita'
    pareti = []
    for t in range(1, n):
        pareti.append(((0, t), (1, 0), 0.0))       # parete sinistra, normale verso +x
        pareti.append(((n, t), (-1, 0), 0.0))      # parete destra
        pareti.append(((t, 0), (0, 1), 0.0))       # fondo
        pareti.append(((t, n), (0, -1), 1.0))      # coperchio mobile, u = 1
    for (i, j), (ni, nj), uw in pareti:
        aggiungi(r, iz[i, j], 1.0)
        ti, tj = nj, ni                           # direzione tangente
        i1, j1 = i + ni, j + nj                    # primo nodo interno
        i2, j2 = i + 2 * ni, j + 2 * nj            # secondo nodo interno
        if formula == "cinque":
            # zeta = [-psi(t-) + 8/3 psi_1 - psi(t+) - 2/3 psi_2]/h^2 - (2/3h) (dpsi/dn)*segno
            for (a, c, w) in ((i1 - ti, j1 - tj, -1.0), (i1, j1, 8 / 3), (i1 + ti, j1 + tj, -1.0),
                              (i2, j2, -2 / 3)):
                if P(a, c) >= 0:
                    aggiungi(r, P(a, c), -w / h**2)
            b[r] = -2.0 / (3.0 * h) * uw
        else:                                      # Thom
            aggiungi(r, P(i1, j1), 2.0 / h**2)
            b[r] = -2.0 * uw / h
        r += 1
    A = sp.csr_matrix((valori, (righe, colonne)), shape=(k, k))
    x = spla.spsolve(A, b)
    psi = np.zeros((n + 1, n + 1))
    zeta = np.zeros((n + 1, n + 1))
    m = ip >= 0
    psi[m] = x[ip[m]]
    m = iz >= 0
    zeta[m] = x[iz[m]]
    return psi, zeta


def minimo_interpolato(psi):
    """Minimo di psi con un paraboloide sui 3x3 nodi attorno al minimo discreto."""
    n = psi.shape[0] - 1
    i, j = np.unravel_index(np.argmin(psi), psi.shape)
    f = psi[i - 1:i + 2, j - 1:j + 2]
    fx = (f[2, 1] - f[0, 1]) / 2
    fy = (f[1, 2] - f[1, 0]) / 2
    fxx = f[2, 1] - 2 * f[1, 1] + f[0, 1]
    fyy = f[1, 2] - 2 * f[1, 1] + f[1, 0]
    fxy = (f[2, 2] - f[2, 0] - f[0, 2] + f[0, 0]) / 4
    Hm = np.array([[fxx, fxy], [fxy, fyy]])
    d = -np.linalg.solve(Hm, [fx, fy])
    val = f[1, 1] + fx * d[0] + fy * d[1] + 0.5 * d @ Hm @ d
    return val, (i + d[0]) / n, (j + d[1]) / n


print("  n     formula     psi_min       x_c      y_c      zeta_c     tempo")
risultati = {"cinque": [], "Thom": []}
griglie = [10, 20, 40, 80, 160]
for formula in ("cinque", "Thom"):
    for n in griglie:
        t0 = time.perf_counter()
        psi, zeta = risolvi(n, formula)
        dt = time.perf_counter() - t0
        val, xc, yc = minimo_interpolato(psi)
        zc = zeta[int(round(xc * n)), int(round(yc * n))]
        risultati[formula].append(val)
        print(f"  {n:3d}   {formula:7s}   {val:.6f}   {xc:.4f}   {yc:.4f}   {zc:8.4f}   {dt:5.2f} s")

for formula in ("cinque", "Thom"):
    v = risultati[formula]
    ordini = [np.log2(abs(v[k] - v[k - 1]) / abs(v[k + 1] - v[k])) for k in range(1, len(v) - 1)]
    rich = v[-1] + (v[-1] - v[-2]) / 3
    print(f"\n{formula}: ordini osservati " + ", ".join(f"{o:.2f}" for o in ordini) +
          f";  Richardson (80, 160): psi_min = {rich:.6f}")

# vortici di Moffatt sulla griglia piu' fine
n = 320
psi, zeta = risolvi(n, "cinque")
x = np.linspace(0, 1, n + 1)
basso = psi[: n // 4, : n // 4]
i, j = np.unravel_index(np.argmax(basso), basso.shape)
print(f"\nGriglia {n}x{n}: psi_min = {minimo_interpolato(psi)[0]:.6f}")
print(f"vortice secondario nello spigolo in basso a sinistra: psi_max = {basso.max():.3e} "
      f"in x = {x[i]:.4f}, y = {x[j]:.4f}  (rapporto con |psi_min|: {basso.max()/abs(psi.min()):.2e})")
# estensione del vortice secondario lungo il fondo (dove psi cambia segno sulla riga j = 1)
riga = psi[:, 1]
xs = x[np.nonzero(np.diff(np.sign(riga[1:n // 2])))[0] + 1]
print(f"cambio di segno di psi accanto al fondo: x = {xs[:2]}")

# teoria di Moffatt per uno spigolo retto: psi ~ r^lam f(theta) con f simmetrica
# rispetto alla bisettrice; lam - 1 = z risolve sin(z pi/2) = -z (radice complessa)
from scipy.optimize import fsolve
def _eq(v):
    z = v[0] + 1j * v[1]
    r = np.sin(z * np.pi / 2) + z
    return [r.real, r.imag]
p, q = fsolve(_eq, [2.7, 1.1])
print(f"\nMoffatt (spigolo di 90 gradi): lam = {p+1:.4f} + {q:.4f} i;  rapporto fra le dimensioni"
      f" di vortici successivi = exp(pi/q) = {np.exp(np.pi/q):.2f};")
print(f"   rapporto fra le intensita' di psi = exp(pi Re(lam)/q) = {np.exp(np.pi*(p+1)/q):.3g},"
      f"  fra le velocita' = {np.exp(np.pi*p/q):.3g}")

# ------------------------------------------------------------------ figura
fig, ax = plt.subplots(1, 3, figsize=(8.0, 2.9))
nn = np.array(griglie)
for formula, col, mk in (("cinque", BLU, "o"), ("Thom", ARANCIO, "s")):
    v = np.array(risultati[formula])
    rif = risultati["cinque"][-1] + (risultati["cinque"][-1] - risultati["cinque"][-2]) / 3
    ax[0].loglog(1 / nn, np.abs(v - rif), mk + "-", color=col, label=f"formula {formula}")
ax[0].loglog(1 / nn, 0.5 * (1 / nn) ** 2, ":", color="k", label="$h^2$")
ax[0].set_xlabel("$h$")
ax[0].set_ylabel(r"$|\psi_{\min}-\psi_{\min,\infty}|$")
ax[0].legend(fontsize=6.5)
lv = np.linspace(psi.min(), 0, 11)[:-1]
ax[1].contour(x, x, psi.T, levels=lv, colors=BLU, linewidths=0.8, linestyles="solid")
ax[1].contour(x, x, psi.T, levels=[0.0], colors=GRIGIO, linewidths=0.6)
pos = basso.max() * np.array([0.01, 0.1, 0.3, 0.6, 0.9])
ax[1].contour(x, x, psi.T, levels=pos, colors=ROSSO, linewidths=0.7, linestyles="solid")
ax[1].set_aspect("equal")
ax[1].set_title(f"$\\psi$, griglia ${n}\\times{n}$")
ax[2].contour(x, x, psi.T, levels=[0.0], colors=GRIGIO, linewidths=0.6)
ax[2].contour(x, x, psi.T, levels=pos, colors=ROSSO, linewidths=0.8, linestyles="solid")
ax[2].contour(x, x, psi.T, levels=-np.logspace(-7, -3, 5)[::-1], colors=BLU, linewidths=0.6,
              linestyles="solid")
ax[2].set_xlim(0, 0.25)
ax[2].set_ylim(0, 0.25)
ax[2].set_aspect("equal")
ax[2].set_title("spigolo in basso a sinistra")
for a in ax[1:]:
    a.set_xlabel("$x$")
    a.set_ylabel("$y$")
fig.tight_layout()
salva(fig, "prog3_6b_cavita_fine")
