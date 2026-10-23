"""
prog4_1c_jeffery_hamel.py -- metodo di Galerkin per la corrente radiale fra due
pareti piane inclinate (corrente di Jeffery-Hamel), confrontato con la soluzione
esatta del problema ai limiti.

Corrente puramente radiale u_r = (Q/(rho r)) f(theta) fra le pareti theta = +-alfa,
portata Q per unita' di larghezza, Re = Q/mu. L'equazione del moto si riduce a
    f'' + 4 f + Re f^2 + 2 C0 = 0,   f(+-alfa) = 0,   2 int_0^alfa f dtheta = 1,
con C0 costante della pressione, p - p0 = (Q mu/(rho r^2)) [C0 + 2 f].
Galerkin (Snyder, Spriggs e Stewart): f = sum_{j=1}^n C_j (alfa^2 - theta^2)^j, residuo
E(theta) ortogonale a (alfa^2 - theta^2)^k, k = 1..n, piu' il vincolo di portata:
n + 1 equazioni (quadratiche) nelle n + 1 incognite C0, C_1..C_n, risolte con
Newton-Raphson (fsolve). Gli integrali sono calcolati in forma chiusa con SymPy.
"""
import numpy as np
import sympy as sp
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

RE, ALFA = 14.164, 0.36


def galerkin(n, Re=RE, alfa=ALFA):
    th = sp.symbols("theta")
    C = sp.symbols(f"C0:{n+1}")
    base = [(alfa**2 - th**2)**j for j in range(1, n + 1)]
    f = sum(C[j] * base[j - 1] for j in range(1, n + 1))
    E = sp.diff(f, th, 2) + 4 * f + Re * f**2 + 2 * C[0]
    eqs = [2 * sp.integrate(f, (th, 0, alfa)) - 1]
    eqs += [sp.integrate(sp.expand(E * base[k]), (th, 0, alfa)) for k in range(n)]
    F = sp.lambdify([C], eqs, "numpy")
    J = sp.lambdify([C], sp.Matrix(eqs).jacobian(C), "numpy")
    c0 = np.zeros(n + 1); c0[1] = 1.0 / alfa**3
    c, info, ier, msg = fsolve(lambda v: np.array(F(v), float), c0, fprime=lambda v: np.array(J(v), float),
                               full_output=True, xtol=1e-13)
    assert ier == 1, msg
    fnum = sp.lambdify(th, f.subs(dict(zip(C, c))), "numpy")
    return c, fnum


def esatta(Re=RE, alfa=ALFA, guida=(2.4, -11.3)):
    """Soluzione 'esatta' per tiro: incognite f(0) e C0, condizioni f(alfa) = 0 e
    2 int_0^alfa f dtheta = 1; 'guida' e' il punto di partenza (continuazione in Re)."""
    def integra(f0, C0):
        return solve_ivp(lambda t, y: [y[1], -4 * y[0] - Re * y[0]**2 - 2 * C0, y[0]], (0, alfa),
                         [f0, 0.0, 0.0], rtol=1e-11, atol=1e-13, dense_output=True)
    def residuo(v):
        s = integra(*v)
        return [s.y[0, -1], s.y[2, -1] - 0.5]
    v, info, ier, msg = fsolve(residuo, guida, xtol=1e-12, full_output=True)
    assert ier == 1, msg
    s = integra(*v)
    s.p = [v[1]]
    return s


sol = esatta()
print(f"Corrente di Jeffery-Hamel, Re = {RE}, alfa = {ALFA} rad ({np.degrees(ALFA):.1f} gradi):")
print(f"   soluzione esatta: C0 = {sol.p[0]:.6f}, f(0) = {sol.sol(0)[0]:.6f}, f'(alfa) = {sol.sol(ALFA)[1]:.5f}")
th = np.linspace(0, ALFA, 181)
fe = sol.sol(th)[0]
risultati = {}
for n in (1, 2, 3, 4):
    c, fnum = galerkin(n)
    err = np.max(np.abs(fnum(th) - fe))
    risultati[n] = (c, fnum)
    print(f"   Galerkin n = {n}: C0 = {c[0]:.6f}, C = {np.array2string(c[1:], precision=4)},"
          f" f(0) = {fnum(0.0):.6f}, errore max {err:.2e}")

# continuazione in Re: comparsa della corrente inversa e punto di svolta
print("\nAl crescere di Re (pareti divergenti) il profilo si gonfia al centro e si appiattisce vicino")
print("alle pareti; oltre un valore critico compare una corrente inversa (f'(alfa) > 0):")
soluzioni = {}
guida = (2.4, -11.3)
Re = RE
while Re < 60:
    try:
        s = esatta(Re, guida=guida)
    except AssertionError:
        print(f"   Re = {Re:7.3f}: il tiro non converge piu': la famiglia di soluzioni simmetriche con sola"
              f" corrente uscente termina (punto di svolta) fra Re = {Re-0.5:.1f} e {Re:.1f}")
        break
    guida = (s.sol(0)[0], s.p[0])
    soluzioni[round(Re, 3)] = s
    if abs((Re - RE) % 5.0) < 1e-9:
        print(f"   Re = {Re:7.3f}: f(0) = {s.sol(0)[0]:.4f}, C0 = {s.p[0]:8.3f}, f'(alfa) = {s.sol(ALFA)[1]:+.4f}"
              + ("  <- corrente inversa" if s.sol(ALFA)[1] > 0 else ""))
    Re += 0.5
Re_s = [Re for Re, s in soluzioni.items() if s.sol(ALFA)[1] > 0]
if Re_s:
    print(f"   prima corrente inversa alla parete fra Re = {max(Re for Re in soluzioni if Re not in Re_s):.1f}"
          f" e {min(Re_s):.1f}")

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
ax[0].plot(th, fe, "k", lw=1.8, label="esatta")
for n, col in zip((1, 2, 3), (GRIGIO, ARANCIO, BLU)):
    ax[0].plot(th, risultati[n][1](th), "--", color=col, lw=1.1, label=f"Galerkin $n={n}$")
ax[0].set_xlabel("$\\theta$ (rad)")
ax[0].set_ylabel("$u_r\\,\\rho r/Q$")
ax[0].legend(fontsize=7)
ax[0].set_title(f"$Re={RE}$, $\\alpha={ALFA}$".replace(".", "{,}"))
for Re, col in zip((14.164, 19.164, 24.164, 29.164), (BLU, VERDE, ARANCIO, ROSSO)):
    s = soluzioni[Re]
    ax[1].plot(th, s.sol(th)[0], color=col, label=f"$Re={Re:.1f}$".replace(".", "{,}"))
ax[1].axhline(0, color="k", lw=0.6)
ax[1].set_xlabel("$\\theta$ (rad)")
ax[1].set_ylabel("$u_r\\,\\rho r/Q$")
ax[1].legend(fontsize=7)
ax[1].set_title("effetto del numero di Reynolds")
fig.tight_layout()
salva(fig, "prog4_1c_jeffery_hamel")
