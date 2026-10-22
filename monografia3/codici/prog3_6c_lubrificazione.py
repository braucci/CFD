"""
prog3_6c_lubrificazione.py -- teoria della lubrificazione: il pattino piano
(cuscinetto di Michell) e la legge di Stokes per la sfera.

1) Pattino inclinato di lunghezza L che scorre con velocita' U su un piano,
   meato di altezza h(x) = h2 [k - (k - 1) x/L], k = h1/h2 > 1 (ingresso in x = 0).
   Equazione di Reynolds (film sottile, flusso di Stokes locale):
       d/dx [ h^3 dp/dx ] = 6 mu U dh/dx,   p(0) = p(L) = 0.
   Con X = x/L, H = h/h2, P = p h2^2/(mu U L):
       P(X) = 6 (k - 1) X (1 - X) / [ (k + 1) H(X)^2 ].
   Portanza per unita' di larghezza W = int p dx e attrito F = int tau dx:
       W h2^2/(mu U L^2) = 6/(k-1)^2 [ ln k - 2 (k-1)/(k+1) ],
       F h2/(mu U L)     = 4 ln k/(k-1) - 6/(k+1).
   Il programma verifica le formule risolvendo l'equazione di Reynolds alle
   differenze finite, traccia P(X) e W(k), e trova il k ottimo.
2) Sfera di raggio a in moto lento: verifica che la funzione di corrente di
   Stokes soddisfa E^4 psi = 0 e calcolo della resistenza 6 pi mu a U; velocita'
   di sedimentazione di una particella di polvere.
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from metodi_numerici import thomas
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

# ---------------------------------------------------------------- 1) pattino
def pressione_esatta(X, k):
    H = k - (k - 1) * X
    return 6 * (k - 1) * X * (1 - X) / ((k + 1) * H**2)


def portanza(k):
    return 6 / (k - 1)**2 * (np.log(k) - 2 * (k - 1) / (k + 1))


def attrito(k):
    return 4 * np.log(k) / (k - 1) - 6 / (k + 1)


def pressione_fd(k, n=200):
    """Risolve d/dX (H^3 dP/dX) = 6 dH/dX = -6 (k-1) con P(0) = P(1) = 0 (forma conservativa)."""
    X = np.linspace(0, 1, n + 1)
    hX = 1.0 / n
    H = k - (k - 1) * X
    Hm = (0.5 * (H[:-1] + H[1:]))**3          # H^3 nei punti intermedi
    a = Hm[:-1] / hX**2
    c = Hm[1:] / hX**2
    b = -(a + c)
    d = np.full(n - 1, -6 * (k - 1))
    P = np.zeros(n + 1)
    P[1:-1] = thomas(a, b, c, d)
    return X, P


print("Pattino piano: k = h1/h2")
print("   k      W (FD)     W (esatta)   F (esatta)   (W/F) h2/L")
for k in (1.5, 2.0, 2.2, 3.0, 4.0):
    X, P = pressione_fd(k)
    Wfd = np.trapezoid(P, X)
    print(f"   {k:3.1f}   {Wfd:.5f}    {portanza(k):.5f}      {attrito(k):.4f}      {portanza(k)/attrito(k):.4f}")
kopt = minimize_scalar(lambda k: -portanza(k), bounds=(1.2, 5), method="bounded").x
print(f"   portanza massima per k = {kopt:.4f}: W h2^2/(mu U L^2) = {portanza(kopt):.5f}")
# esempio numerico: pattino di una ralla, olio SAE 30
mu, U, L, h2, rho = 0.05, 5.0, 0.05, 50e-6, 900.0
k = kopt
W = portanza(k) * mu * U * L**2 / h2**2
F = attrito(k) * mu * U * L / h2
print(f"   esempio: olio a caldo mu = {mu} Pa s, U = {U} m/s, L = {L*1e3:.0f} mm, h2 = {h2*1e6:.0f} um:")
print(f"      portanza {W/1e3:.1f} kN per metro di larghezza, attrito {F:.0f} N/m, W/F = {W/F:.0f},"
      f" p_max = {pressione_esatta(np.linspace(0,1,1001),k).max()*mu*U*L/h2**2/1e6:.2f} MPa,"
      f" Re ridotto rho U h2^2/(mu L) = {rho*U*h2**2/(mu*L):.1e}")

# ---------------------------------------------------------------- 2) sfera di Stokes
r, th, a, Us = sp.symbols("r theta a U", positive=True)
psi = sp.Rational(1, 4) * Us * r**2 * sp.sin(th)**2 * (2 - 3 * a / r + a**3 / r**3)
E2 = lambda f: sp.diff(f, r, 2) + sp.sin(th) / r**2 * sp.diff(sp.diff(f, th) / sp.sin(th), th)
print("\nSfera in moto lento (sfera ferma, corrente U all'infinito):")
print("   E^4 psi =", sp.simplify(E2(E2(psi))))
ur = sp.simplify(sp.diff(psi, th) / (r**2 * sp.sin(th)))
ut = sp.simplify(-sp.diff(psi, r) / (r * sp.sin(th)))
print("   u_r =", ur, ";  u_theta =", ut)
print("   sulla sfera (r = a): u_r =", sp.simplify(ur.subs(r, a)), ", u_theta =", sp.simplify(ut.subs(r, a)))
mu_s = sp.symbols("mu", positive=True)
# vorticita' omega_phi = (1/r) d(r u_theta)/dr - (1/r) d u_r/d theta ; pressione da mu lap V = grad p
om = sp.simplify(sp.diff(r * ut, r) / r - sp.diff(ur, th) / r)
print("   vorticita' omega_phi =", om)
# pressione: p = -3 mu a U cos(theta)/(2 r^2) (da integrare mu rot(omega) = -grad p)
p = -sp.Rational(3, 2) * mu_s * a * Us * sp.cos(th) / r**2
# sforzo tangenziale sulla sfera: tau = mu * (d u_theta/dr - u_theta/r + (1/r) d u_r/d theta) in r = a
tau = sp.simplify((mu_s * (sp.diff(ut, r) - ut / r + sp.diff(ur, th) / r)).subs(r, a))
print("   sforzo tangenziale sulla sfera tau_rtheta =", tau)
# resistenza = int (-p cos + tau_rtheta sin ... ) dA ; con segno: forza lungo la corrente
Fp = sp.integrate(-p.subs(r, a) * sp.cos(th) * 2 * sp.pi * a**2 * sp.sin(th), (th, 0, sp.pi))
Ft = sp.integrate(-tau * sp.sin(th) * 2 * sp.pi * a**2 * sp.sin(th), (th, 0, sp.pi))
print("   resistenza di pressione =", sp.simplify(Fp), ";  di attrito =", sp.simplify(Ft), ";  totale =", sp.simplify(Fp + Ft))
# sedimentazione di una particella di polvere in aria
rho_p, rho_a, mu_a, nu_a, g = 2500.0, 1.2, 1.8e-5, 1.5e-5, 9.81
for d in (1e-6, 10e-6, 50e-6, 100e-6):
    ap = d / 2
    v = 2 * ap**2 * (rho_p - rho_a) * g / (9 * mu_a)
    Re = d * v / nu_a
    print(f"   particella di {d*1e6:5.0f} um: v = {v*1e3:8.3f} mm/s, Re = {Re:.2e}"
          + ("  (legge di Stokes non piu' valida)" if Re > 1 else ""))

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.8))
X = np.linspace(0, 1, 201)
for k, col in zip((1.5, 2.2, 4.0), (VERDE, BLU, ARANCIO)):
    ax[0].plot(X, pressione_esatta(X, k), color=col, label=f"$k={k:g}$".replace(".", "{,}"))
    Xf, Pf = pressione_fd(k, 20)
    ax[0].plot(Xf, Pf, "o", color=col, ms=2.5)
ax[0].set_xlabel("$X=x/L$")
ax[0].set_ylabel("$P=p\\,h_2^2/(\\mu UL)$")
ax[0].legend(fontsize=7)
ax[0].set_title("pressione nel meato")
ks = np.linspace(1.05, 6, 300)
ax[1].plot(ks, portanza(ks), color=BLU, label="portanza $W$")
ax[1].plot(ks, attrito(ks) / 10, "--", color=ARANCIO, label="attrito $F/10$")
ax[1].axvline(kopt, color=GRIGIO, ls=":", lw=1)
ax[1].set_xlabel("$k=h_1/h_2$")
ax[1].set_ylabel("$W h_2^2/(\\mu UL^2)$, $F h_2/(\\mu UL)$")
ax[1].legend(fontsize=7)
ax[1].set_title("carico e attrito")
# linee di corrente attorno alla sfera (soluzione di Stokes nel piano meridiano)
R, TH = np.meshgrid(np.linspace(1, 4, 200), np.linspace(0, 2 * np.pi, 361), indexing="ij")
PSI = 0.25 * R**2 * np.sin(TH)**2 * (2 - 3 / R + 1 / R**3)
Xs, Ys = R * np.cos(TH), R * np.sin(TH)
ax[2].contour(Xs, Ys, PSI, levels=np.linspace(0.1, 4.0, 14), colors=BLU, linewidths=0.7)
ax[2].add_patch(plt.Circle((0, 0), 1, color=GRIGIO))
ax[2].set_aspect("equal")
ax[2].set_xlim(-3, 3); ax[2].set_ylim(-3, 3)
ax[2].set_xticks([]); ax[2].set_yticks([])
ax[2].set_title("sfera di Stokes: linee di corrente")
fig.tight_layout()
salva(fig, "prog3_6c_lubrificazione")
