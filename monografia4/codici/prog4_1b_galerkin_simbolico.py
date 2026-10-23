"""
prog4_1b_galerkin_simbolico.py -- derivazione simbolica (SymPy) delle equazioni
di Galerkin per la sfera a Reynolds finito.

Partendo dalla funzione di corrente di prova di Kawaguti e dall'equazione di
Navier-Stokes in forma di funzione di corrente (coordinate r, zeta = cos theta),
    (2/Re) D^4 psi + (1/r^2) [ psi_z d/dr - psi_r d/dz - 2 z/(1-z^2) psi_r - (2/r) psi_z ] D^2 psi = 0,
    D^2 = d^2/dr^2 + (1 - z^2)/r^2 d^2/dz^2,
il programma:
 1) calcola il residuo (NS) e ne impone l'annullamento sulla sfera (r = 1),
    ottenendo le due condizioni 9 A2 + 35 A3 + 90 A4 = 0 e B1 - 6 B3 - 21 B4 = 0;
 2) impone le due condizioni di ortogonalita' di Galerkin
       int_1^oo int_-1^1 (NS) (1/r) [P0 - P2] dz dr = 0,
       int_1^oo int_-1^1 (NS) (1/r^2) [P1 - P3] dz dr = 0,
    con P_n polinomi di Legendre, e verifica i coefficienti razionali del testo;
 3) risolve le sei relazioni lineari e riduce il sistema a due equazioni in
    (A1, B1), i cui coefficienti numerici sono quelli usati in prog4_1_sfera.py.
Tempo di esecuzione: circa 15-20 s.
"""
import sympy as sp

r, z, Re = sp.symbols("r zeta Re", positive=True)
A1, A2, A3, A4, B1, B2, B3, B4 = sp.symbols("A1:5 B1:5")
psi = (r**2 / 2 + A1 / r + A2 / r**2 + A3 / r**3 + A4 / r**4) * (1 - z**2) \
    + (B1 / r + B2 / r**2 + B3 / r**3 + B4 / r**4) * z * (1 - z**2)


def D2(f):
    return sp.diff(f, r, 2) + (1 - z**2) / r**2 * sp.diff(f, z, 2)


D2psi = D2(psi)
pr, pz = sp.diff(psi, r), sp.diff(psi, z)
NS = 2 / Re * D2(D2psi) + (pz * sp.diff(D2psi, r) - pr * sp.diff(D2psi, z)
                           - 2 * z / (1 - z**2) * pr * D2psi - 2 / r * pz * D2psi) / r**2
NS = sp.expand(sp.simplify(NS))

# 1) condizioni di aderenza e residuo nullo sulla sfera
aderenza = [sp.Eq(A1 + A2 + A3 + A4, -sp.Rational(1, 2)), sp.Eq(A1 + 2 * A2 + 3 * A3 + 4 * A4, 1),
            sp.Eq(B1 + B2 + B3 + B4, 0), sp.Eq(B1 + 2 * B2 + 3 * B3 + 4 * B4, 0)]
visc = sp.factor(sp.expand(2 * D2(D2psi).subs(r, 1)))
conv = sp.expand((NS - 2 / Re * D2(D2psi)).subs(r, 1))
sol_ad = sp.solve(aderenza, [A3, A4, B3, B4])
print("Parte viscosa del residuo sulla sfera, 2 D^4 psi (r = 1):")
print("  ", visc)
print("Parte convettiva sulla sfera dopo aver imposto l'aderenza:", sp.simplify(conv.subs(sol_ad)))

# 2) ortogonalita' di Galerkin
P = lambda n: sp.legendre(n, z)
pesi = [(P(0) - P(2)) / r, (P(1) - P(3)) / r**2]
G = []
for w in pesi:
    Iz = sp.expand(sp.integrate(sp.expand(NS * w), (z, -1, 1)))
    G.append(sp.expand(sp.integrate(Iz, (r, 1, sp.oo))))
print("\nPrima equazione di Galerkin (4.1.24):")
print("  ", G[0])
print("Seconda equazione di Galerkin (4.1.25):")
print("  ", G[1])
# alcuni coefficienti da confrontare con il testo
print("\nVerifica di alcuni coefficienti: A1*B1 ->", G[0].coeff(A1 * B1), "(testo -512/245);",
      "A2/Re ->", sp.simplify(G[0].coeff(A2) * Re), "(testo 32/Re * 6/5 = 192/5);",
      "B1^2 ->", G[1].coeff(B1**2), "(testo -8/63)")

# 3) riduzione a due incognite
lineari = aderenza + [sp.Eq(9 * A2 + 35 * A3 + 90 * A4, 0), sp.Eq(B1 - 6 * B3 - 21 * B4, 0)]
sol = sp.solve(lineari, [A2, A3, A4, B2, B3, B4])
print("\nRelazioni lineari (4.1.26):")
for k, v in sol.items():
    print(f"   {k} = {v}")
E1 = sp.expand(G[0].subs(sol))
E2 = sp.expand(G[1].subs(sol))
print("\nSistema ridotto in (A1, B1):")
print("   g =", sp.N(E1, 7))
print("   h =", sp.N(E2, 7))
print("   (testo: g = 9.931035 A1/Re + 0.011713 B1 - 0.002546 A1 B1 + 44.689656/Re,")
print("           h = 0.050148 A1 + 2.201058 B1/Re + 0.002421 A1^2 - 0.000376 B1^2 + 0.379018)")

# la funzione di prova per Re piccolo (4.1.52) contiene la soluzione di Stokes
A1s, A2s, A3s, A4s = sp.symbols("a1:5")
psi_stokes = (r**2 / 2 - 3 * r / 4 + 1 / (4 * r)) * (1 - z**2)
print("\nSoluzione di Stokes: D^4 psi =", sp.simplify(D2(D2(psi_stokes))),
      "; psi(1) =", sp.simplify(psi_stokes.subs(r, 1)), "; psi_r(1) =", sp.simplify(sp.diff(psi_stokes, r).subs(r, 1)))
