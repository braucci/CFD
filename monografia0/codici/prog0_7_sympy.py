"""
prog0_7_sympy.py -- Programma 0.7: calcolo simbolico con SymPy.

1) Simboli, espressioni, semplificazione, sostituzione, valutazione numerica.
2) Derivate e integrali: la portata di Poiseuille, la funzione di corrente del
   dipolo, verifica che psi = U y (1 - a^2/r^2) e' armonica.
3) Serie di Taylor e derivazione delle formule alle differenze finite: la
   differenza centrata, la formula unilatera a tre punti e il suo errore di
   troncamento (il metodo dei coefficienti indeterminati, risolto con solve).
4) Equazioni: soluzione simbolica (velocita' limite con c_d costante), sistema
   lineare (coefficienti dello spessore NACA imposti come vincoli).
5) lambdify: da espressione simbolica a funzione NumPy vettorializzata; lo
   jacobiano esatto per il metodo di Newton.
6) Verifica di un risultato delle monografie: la funzione di corrente di Stokes
   per la sfera soddisfa D^4 psi = 0 e le condizioni di aderenza.
"""
import numpy as np
import sympy as sp

# ------------------------------------------------------------------ 1) simboli ed espressioni
print("1) Simboli ed espressioni")
x, y, r, theta, a, U = sp.symbols("x y r theta a U", positive=True)
espr = (x**2 - y**2) / (x**2 + y**2)**2
print(f"   espressione: {espr};  in (1, 0): {espr.subs({x: 1, y: 0})};  numerica in (1,2, 0,3): {espr.subs({x: 1.2, y: 0.3}).evalf(6)}")
print(f"   semplificazione: {sp.simplify((x**2 - 1) / (x - 1))};  espansione: {sp.expand((x + y)**3)};  fattorizzazione: {sp.factor(x**3 - y**3)}")
print(f"   trigonometria: {sp.trigsimp(sp.sin(theta)**2 + sp.cos(theta)**2)};  pi con 30 cifre: {sp.N(sp.pi, 30)};  razionali esatti: {sp.Rational(1, 3) + sp.Rational(1, 6)}")

# ------------------------------------------------------------------ 2) derivate e integrali
print("\n2) Derivate e integrali")
R, mu, G = sp.symbols("R mu G", positive=True)      # G = -dp/dx
u_pois = G / (4 * mu) * (R**2 - r**2)
Q = sp.integrate(2 * sp.pi * r * u_pois, (r, 0, R))
print(f"   portata di Poiseuille: Q = {sp.simplify(Q)};  sforzo alla parete tau = mu du/dr|_R = {sp.simplify(mu * sp.diff(u_pois, r).subs(r, R))}")
psi = U * y * (1 - a**2 / (x**2 + y**2))
lap = sp.simplify(sp.diff(psi, x, 2) + sp.diff(psi, y, 2))
u_c, v_c = sp.diff(psi, y), -sp.diff(psi, x)
print(f"   psi del cilindro: laplaciano = {lap};  velocita' in (a, 0): u = {sp.simplify(u_c.subs({x: a, y: 0}))}, v = {sp.simplify(v_c.subs({x: a, y: 0}))} (ristagno)")
print(f"   velocita' in (0, a): u = {sp.simplify(u_c.subs({x: 0, y: a}))} (il doppio di U: C_p = -3)")
print(f"   integrale indefinito: {sp.integrate(sp.exp(-x**2) * x, x)};  definito: {sp.integrate(sp.exp(-x**2), (x, -sp.oo, sp.oo))};  limite: {sp.limit(sp.sin(x) / x, x, 0)}")

# ------------------------------------------------------------------ 3) serie di Taylor e differenze finite
print("\n3) Serie di Taylor e formule alle differenze finite")
h = sp.symbols("h", positive=True)
print(f"   serie di exp(x) in x = 0: {sp.series(sp.exp(x), x, 0, 5)};  di sin(x)/x: {sp.series(sp.sin(x) / x, x, 0, 6)}")
# Per ricavare le formule alle differenze si rappresentano i valori di f, f', f'', ... nel nodo x con
# simboli f0, f1, f2, ... e si scrive lo sviluppo di Taylor di f(x + k h) "a mano"
f0, f1, f2, f3, f4 = sp.symbols("f0 f1 f2 f3 f4")
taylor = lambda k: f0 + k * h * f1 + (k * h)**2 / 2 * f2 + (k * h)**3 / 6 * f3 + (k * h)**4 / 24 * f4
centrata = sp.expand((taylor(1) - taylor(-1)) / (2 * h))
print(f"   differenza centrata: {centrata}  -> f1 + h^2 f3/6: errore di troncamento del secondo ordine")
seconda = sp.expand((taylor(1) - 2 * taylor(0) + taylor(-1)) / h**2)
print(f"   derivata seconda centrata: {seconda}  -> f2 + h^2 f4/12")
# metodo dei coefficienti indeterminati: f' ~ [c0 f(x) + c1 f(x+h) + c2 f(x+2h)]/h esatta per i polinomi di grado <= 2
c0, c1, c2 = sp.symbols("c0 c1 c2")
approx = sp.expand((c0 * taylor(0) + c1 * taylor(1) + c2 * taylor(2)) / h)
eqs = [approx.coeff(f0), approx.coeff(f1) - 1, approx.coeff(f2)]   # coefficiente di f0 nullo, di f1 unitario, di f2 nullo
sol = sp.solve(eqs, [c0, c1, c2])
print(f"   formula unilatera a tre punti: c0 = {sol[c0]}, c1 = {sol[c1]}, c2 = {sol[c2]}  ->  f' = (-3f0 + 4f1 - f2)/(2h)")
errore = sp.expand(approx.subs(sol) - f1)
print(f"   errore di troncamento: {errore.coeff(f3) * f3}  (secondo ordine; e' la formula della vorticita' di parete del Modulo 4)")
# la formula di parete di Thom: psi(h) = psi(0) + h psi'(0) + h^2/2 psi''(0) con psi(0) = psi'(0) = 0
psi1, om = sp.symbols("psi_1 omega")
print(f"   vorticita' di parete (Thom): da psi_1 = h^2/2 psi'' e omega = -psi'' segue omega = {sp.solve(sp.Eq(psi1, h**2 / 2 * (-om)), om)[0]}")

# ------------------------------------------------------------------ 4) equazioni e sistemi
print("\n4) Equazioni e sistemi")
v, m, g, rho, A_, cd = sp.symbols("v m g rho A c_d", positive=True)
print(f"   velocita' limite (c_d costante): v_t = {sp.solve(sp.Eq(m * g, sp.Rational(1, 2) * rho * v**2 * A_ * cd), v)[0]}")
print(f"   radici di x^3 - 6x^2 + 11x - 6: {sp.solve(x**3 - 6 * x**2 + 11 * x - 6, x)};  numeriche di cos x = x: {sp.nsolve(sp.cos(x) - x, x, 0.7)}")
# spessore NACA: il profilo passa per (0,3; 0,1) con tangente nulla ed e' chiuso al bordo d'uscita (t = 0,2):
# si impongono tre vincoli sui coefficienti e si ritrovano (quasi) quelli classici
a0, a1, a2, a3, a4 = sp.symbols("a0:5")
yt = a0 * sp.sqrt(x) - a1 * x - a2 * x**2 + a3 * x**3 - a4 * x**4
vincoli = [sp.Eq(yt.subs(x, sp.Rational(3, 10)), sp.Rational(1, 10)), sp.Eq(sp.diff(yt, x).subs(x, sp.Rational(3, 10)), 0),
           sp.Eq(yt.subs(x, 1), 0)]
sol_naca = sp.solve(vincoli, [a2, a3, a4])          # a0 = 0,2969 e a1 = 0,1260 dati
sol_num = {k: float(val.subs({a0: 0.2969, a1: 0.1260})) for k, val in sol_naca.items()}
print(f"   coefficienti NACA dai vincoli (a0, a1 dati): {sol_num}  (classici: 0,3516, 0,2843, 0,1036)")

# ------------------------------------------------------------------ 5) lambdify e jacobiano
print("\n5) lambdify e jacobiano esatto per Newton")
g1 = x**2 + y**2 - 4
g2 = sp.exp(x) + y - 1
F = sp.Matrix([g1, g2]); J = F.jacobian([x, y])
F_num = sp.lambdify([x, y], F, "numpy"); J_num = sp.lambdify([x, y], J, "numpy")
p = np.array([-1.5, 1.0])
for k in range(6):
    p = p - np.linalg.solve(np.array(J_num(*p), float), np.array(F_num(*p), float).ravel())
print(f"   J = {J.tolist()}; Newton da (-1,5, 1): {p} con residuo {np.max(np.abs(F_num(*p))):.1e}")
cp = sp.lambdify(theta, 1 - (2 * sp.sin(theta))**2, "numpy")
th = np.linspace(0, np.pi, 5)
print(f"   lambdify di C_p(theta) valutata su un array: {cp(th)}")

# ------------------------------------------------------------------ 6) verifica: funzione di corrente di Stokes
print("\n6) La funzione di corrente di Stokes per la sfera (Modulo 3 e 4)")
z = sp.symbols("zeta", real=True)
psi_s = (r**2 / 2 - sp.Rational(3, 4) * r + 1 / (4 * r)) * (1 - z**2)
D2 = lambda F: sp.diff(F, r, 2) + (1 - z**2) / r**2 * sp.diff(F, z, 2)
print(f"   D^4 psi = {sp.simplify(D2(D2(psi_s)))};  psi(1) = {sp.simplify(psi_s.subs(r, 1))};  psi_r(1) = {sp.simplify(sp.diff(psi_s, r).subs(r, 1))}")
print(f"   per r -> oo: psi/(r^2/2 (1 - zeta^2)) -> {sp.limit(psi_s / (r**2 / 2 * (1 - z**2)), r, sp.oo)} (corrente uniforme)")
print(f"   LaTeX pronto per il documento: {sp.latex(psi_s)}")
