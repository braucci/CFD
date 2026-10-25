"""
prog0_6_scipy.py -- Programma 0.6: la cassetta degli attrezzi di SciPy.

Le funzioni di SciPy usate nelle monografie, ciascuna su un problema della
fluidodinamica con soluzione nota:
1) scipy.optimize: brentq (radice scalare: velocita' limite), fsolve (sistema non
   lineare: punto di ristagno di una corrente), minimize_scalar (gittata massima).
2) scipy.integrate: quad (integrale: portata di Poiseuille), solve_ivp (problema ai
   valori iniziali con passo adattivo ed evento: caduta della sfera fino al suolo),
   solve_bvp (problema ai limiti: Blasius).
3) scipy.linalg: solve, solve_banded (sistema tridiagonale di Crank-Nicolson), eigh
   (autovalori dell'operatore delle differenze seconde), lu_factor/lu_solve.
4) scipy.sparse e sparse.linalg: matrice del laplaciano 2D in formato lil/csr,
   spsolve e splu (fattorizza una volta, risolve molte volte); confronto dei tempi
   con la matrice piena.
5) scipy.interpolate: interp1d e CubicSpline (polare di un profilo), scipy.special:
   erf, funzioni di Bessel.
"""
import time
import numpy as np
from scipy import optimize, integrate, linalg, sparse, interpolate, special
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

# ------------------------------------------------------------------ 1) optimize
print("1) scipy.optimize")
RHO_A, NU, G = 1.225, 1.46e-5, 9.80665
d, m = 0.01, 7800.0 * np.pi * 0.01**3 / 6
cd = lambda Re: 24 / Re * (1 + 0.15 * Re**0.687) + 0.42 / (1 + 42500 * Re**-1.16)
f = lambda v: G - 0.5 * RHO_A * v**2 * np.pi * d**2 / 4 * cd(max(v * d / NU, 1e-9)) / m
vt = optimize.brentq(f, 1.0, 100.0, xtol=1e-12)
print(f"   brentq: velocita' limite v_t = {vt:.6f} m/s (f(v_t) = {f(vt):.1e})")
# punto di ristagno della corrente intorno al cilindro con circolazione: u = v = 0 (sistema 2x2)
gamma = 2 * np.pi * 0.8
def velocita(p):
    x, y = p; r2 = x * x + y * y
    return [1 - (x * x - y * y) / r2**2 + gamma / (2 * np.pi) * y / r2, -2 * x * y / r2**2 - gamma / (2 * np.pi) * x / r2]
sol, info, ier, msg = optimize.fsolve(velocita, [-0.9, -0.4], full_output=True)
print(f"   fsolve: punto di ristagno in ({sol[0]:.6f}, {sol[1]:.6f}); esatto: sin(theta) = -Gamma/(4 pi U a) = -0,4 ->"
      f" ({-np.sqrt(1 - 0.16):.6f}, {-0.4:.6f}); residuo {np.max(np.abs(velocita(sol))):.1e}, {info['nfev']} valutazioni")
gittata = lambda alfa: -(20.0**2 / G) * np.sin(2 * alfa)         # da minimizzare (segno meno)
ris = optimize.minimize_scalar(gittata, bounds=(0, np.pi / 2), method="bounded")
print(f"   minimize_scalar: angolo di gittata massima {np.degrees(ris.x):.4f} gradi (esatto 45), gittata {-ris.fun:.4f} m")

# ------------------------------------------------------------------ 2) integrate
print("\n2) scipy.integrate")
R, mu, dpdx = 0.01, 1e-3, -100.0
portata, err = integrate.quad(lambda r: 2 * np.pi * r * (-dpdx / (4 * mu)) * (R**2 - r**2), 0, R)
print(f"   quad: portata di Poiseuille {portata:.6e} m^3/s (esatta pi R^4 (-dp/dx)/(8 mu) = {np.pi * R**4 * (-dpdx) / (8 * mu):.6e}), errore stimato {err:.1e}")


def caduta(t, y):                                  # y = [z, v], z verso il basso
    z, v = y
    return [v, f(v)]


suolo = lambda t, y: 100.0 - y[0]                  # evento: z = 100 m
suolo.terminal = True
s = integrate.solve_ivp(caduta, (0, 60), [0.0, 0.0], method="RK45", rtol=1e-10, atol=1e-12, events=suolo, dense_output=True)
t_suolo = s.t_events[0][0]
print(f"   solve_ivp: la sfera tocca il suolo (100 m) a t = {t_suolo:.6f} s con v = {s.y_events[0][0][1]:.6f} m/s;"
      f" {len(s.t)} passi adattivi (passo medio {s.t[-1] / len(s.t):.3f} s)")
print(f"   soluzione densa: z(2 s) = {s.sol(2.0)[0]:.6f} m; nel vuoto sarebbe {0.5 * G * 4:.4f} m")


def blasius(eta, y):                               # y = [f, f', f'']
    return np.vstack([y[1], y[2], -0.5 * y[0] * y[2]])


def cc(ya, yb):
    return np.array([ya[0], ya[1], yb[1] - 1.0])


eta = np.linspace(0, 10, 50)
y0 = np.zeros((3, eta.size)); y0[1] = 1 - np.exp(-eta)
sb = integrate.solve_bvp(blasius, cc, eta, y0, tol=1e-10)
print(f"   solve_bvp: Blasius f''(0) = {sb.sol(0)[2]:.8f} (riferimento 0,33205734), {sb.x.size} nodi dopo il raffinamento")

# ------------------------------------------------------------------ 3) linalg
print("\n3) scipy.linalg")
n = 1000
dd = 0.8                                           # Crank-Nicolson: -d/2 u_(i-1) + (1+d) u_i - d/2 u_(i+1)
ab = np.zeros((3, n)); ab[0, 1:] = -dd / 2; ab[1] = 1 + dd; ab[2, :-1] = -dd / 2
rhs = np.sin(np.linspace(0, np.pi, n))
A = np.diag(np.full(n, 1 + dd)) + np.diag(np.full(n - 1, -dd / 2), 1) + np.diag(np.full(n - 1, -dd / 2), -1)
linalg.solve_banded((1, 1), ab, rhs); linalg.solve(A, rhs)        # prima chiamata "a freddo": non si cronometra
t0 = time.perf_counter(); x_b = linalg.solve_banded((1, 1), ab, rhs); tb = time.perf_counter() - t0
t0 = time.perf_counter(); x_f = linalg.solve(A, rhs); tf = time.perf_counter() - t0
print(f"   solve_banded contro solve (n = {n}): {tb*1e3:.2f} ms contro {tf*1e3:.2f} ms, differenza {np.max(np.abs(x_b - x_f)):.1e}")
lu, piv = linalg.lu_factor(A)
t0 = time.perf_counter()
for _ in range(100):
    linalg.lu_solve((lu, piv), rhs)
print(f"   lu_factor una volta + 100 lu_solve: {(time.perf_counter() - t0)*1e3:.1f} ms (100 solve: ~{100*tf*1e3:.0f} ms)")
m = 50; h = 1.0 / (m + 1)
D2 = (np.diag(np.full(m, -2.0)) + np.diag(np.ones(m - 1), 1) + np.diag(np.ones(m - 1), -1)) / h**2
lam = linalg.eigh(D2, eigvals_only=True)
print(f"   eigh: primi autovalori di d^2/dx^2 discreto {lam[-3:][::-1]}; esatti -(k pi)^2 = {-(np.arange(1, 4) * np.pi)**2}")

# ------------------------------------------------------------------ 4) sparse
print("\n4) scipy.sparse: equazione di Poisson lap u = f su griglia N x N con u = 0 al contorno")


def matrice_poisson(N, h):
    """Laplaciano a cinque punti sui nodi interni, formato lil (comodo da riempire) -> csc (comodo da fattorizzare)."""
    n = N * N
    idx = lambda i, j: i * N + j
    A = sparse.lil_matrix((n, n))
    for i in range(N):
        for j in range(N):
            p = idx(i, j)
            A[p, p] = -4.0 / h**2
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < N and 0 <= b < N:
                    A[p, idx(a, b)] = 1.0 / h**2
    return A.tocsc()


N = 60; h = 1.0 / (N + 1)
xg = np.linspace(h, 1 - h, N)
X, Y = np.meshgrid(xg, xg, indexing="ij")
u_es = np.sin(np.pi * X) * np.sin(np.pi * Y)
fsrc = (-2 * np.pi**2 * u_es).ravel()
t0 = time.perf_counter(); A = matrice_poisson(N, h); t_costr = time.perf_counter() - t0
print(f"   matrice {A.shape} con {A.nnz} elementi non nulli su {A.shape[0]**2} ({A.nnz / A.shape[0]**2:.2%}), costruita in {t_costr*1e3:.0f} ms")
t0 = time.perf_counter(); u1 = spla.spsolve(A, fsrc); t1 = time.perf_counter() - t0
t0 = time.perf_counter(); LU = spla.splu(A); t2 = time.perf_counter() - t0
t0 = time.perf_counter(); u2 = LU.solve(fsrc); t3 = time.perf_counter() - t0
t0 = time.perf_counter(); u3 = linalg.solve(A.toarray(), fsrc); t4 = time.perf_counter() - t0
print(f"   spsolve {t1*1e3:.1f} ms; splu {t2*1e3:.1f} ms + solve {t3*1e3:.2f} ms; matrice piena {t4*1e3:.0f} ms")
print(f"   errore rispetto alla soluzione esatta: {np.max(np.abs(u2 - u_es.ravel())):.2e} (O(h^2) = {h**2:.1e});"
      f" spsolve e splu coincidono: {np.allclose(u1, u2)}")

# ------------------------------------------------------------------ 5) interpolate e special
print("\n5) scipy.interpolate e scipy.special")
alfa = np.array([-4, -2, 0, 2, 4, 6, 8, 10, 12, 14])                 # polare (dati tipici di un profilo)
cl = np.array([-0.2, 0.0, 0.25, 0.45, 0.67, 0.88, 1.08, 1.25, 1.35, 1.30])
lin = interpolate.interp1d(alfa, cl)
spl = interpolate.CubicSpline(alfa, cl)
print(f"   C_L(5 gradi): lineare {lin(5.0):.4f}, spline cubica {spl(5.0):.4f}; pendenza della spline in 0: {spl(0.0, 1):.4f} per grado "
      f"({np.degrees(spl(0.0, 1)):.3f} per radiante; teoria sottile: 2 pi = {2*np.pi:.3f})")
print(f"   np.interp e' l'equivalente NumPy per l'interpolazione lineare: {np.interp(5.0, alfa, cl):.4f}")
print(f"   erf(1) = {special.erf(1.0):.8f}, erfc(1) = {special.erfc(1.0):.8f} (soluzione di Rayleigh); J_0(2,4048) = {special.j0(2.4048):.2e} (zero di Bessel)")

# ------------------------------------------------------------------ figura
fig, ax = plt.subplots(1, 3, figsize=(9.0, 2.8))
tt = np.linspace(0, t_suolo, 200)
ax[0].plot(tt, s.sol(tt)[1], color=BLU, label="solve_ivp")
ax[0].plot(s.t, s.y[1], "o", color=ARANCIO, ms=2.5, label="passi adattivi")
ax[0].axhline(vt, color=GRIGIO, ls=":", label="$v_t$ (brentq)")
ax[0].set_xlabel("$t$ [s]"); ax[0].set_ylabel("$v$ [m/s]"); ax[0].legend(fontsize=7)
ax[1].plot(sb.x, sb.sol(sb.x)[1], color=BLU); ax[1].set_xlabel("$\\eta$"); ax[1].set_ylabel("$f'(\\eta)$"); ax[1].set_title("Blasius con solve\\_bvp", fontsize=8)
aa = np.linspace(-4, 14, 200)
ax[2].plot(aa, spl(aa), color=BLU, label="spline cubica"); ax[2].plot(aa, lin(aa), "--", color=ARANCIO, label="lineare")
ax[2].plot(alfa, cl, "o", color=GRIGIO, ms=3, label="dati")
ax[2].set_xlabel("$\\alpha$ [gradi]"); ax[2].set_ylabel("$C_L$"); ax[2].legend(fontsize=7)
fig.tight_layout()
salva(fig, "prog0_6_scipy")
