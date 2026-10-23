"""
esercizi_svolti.py -- verifiche numeriche e simboliche degli esercizi del Modulo 4.

Es. 1  Stabilita' di von Neumann dello schema upwind (fattore di amplificazione).
Es. 2  Schema FTCS per la convezione: instabile per ogni C > 0 e privo della
       proprieta' trasportiva (analisi della perturbazione discreta).
Es. 3  Analisi di Hirt (equazione modificata) per gli schemi FTCS e BTCS
       dell'equazione della diffusione: derivazione simbolica dei coefficienti e
       verifica sperimentale dei limiti di stabilita'.
Es. 4  Analisi della perturbazione discreta dello schema FTCS per la diffusione.
Es. 5  Coefficienti dello schema upwind di Torrance e criterio di stabilita' (4.3.33).
Es. 6  Fattorizzazione LU contro algoritmo di Thomas (equazione di Burgers AB-CN).
Es. 7  Coordinate trasformate del cilindro: verifica simbolica del laplaciano e
       del termine convettivo (4.6.44)-(4.6.46).
Es. 8  Funzione di prova di Kawaguti per piccoli Re: riduzione alla soluzione di Stokes.
Es. 9  Problema di Neumann per l'equazione di Poisson: matrice singolare, fissaggio
       di un nodo, autovalori e condizionamento.
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO


def titolo(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


# ============================================================ Es. 1: von Neumann per lo schema upwind
titolo("Es. 1 -- schema upwind: fattore di amplificazione G = 1 - C (1 - e^{-i beta})")
C, beta = sp.symbols("C beta", real=True)
G = 1 - C * (1 - sp.exp(-sp.I * beta))
G2 = sp.simplify(sp.expand(G * sp.conjugate(G), complex=True))
G2 = sp.simplify(G2.rewrite(sp.cos))
print("   |G|^2 =", G2, "  =  1 - 2 C (1 - C)(1 - cos beta)")
print("   verifica:", sp.simplify(G2 - (1 - 2 * C * (1 - C) * (1 - sp.cos(beta)))) == 0)
Cs = np.array([0.25, 0.5, 0.75, 1.0, 1.1, 1.5])
bs = np.linspace(0, np.pi, 721)
print("   massimo di |G| sul cerchio unitario: " + ", ".join(
    f"C={c:g}: {np.max(np.abs(1 - c*(1 - np.exp(-1j*bs)))):.3f}" for c in Cs))

# ============================================================ Es. 2: FTCS per la convezione
titolo("Es. 2 -- schema FTCS per zeta_t + u zeta_x = 0: G = 1 - i C sin beta")
print("   |G|^2 = 1 + C^2 sin^2 beta >= 1 per ogni C != 0: stabile solo per C -> 0")
print("   massimo di |G|: " + ", ".join(f"C={c:g}: {np.sqrt(1 + c*c):.3f}" for c in (0.1, 0.5, 1.0)))
print("   perturbazione discreta: zeta = 0 ovunque, zeta_m = eps all'istante j (C = 0,5):")
z = np.zeros(11); z[5] = 1.0
for passo in range(3):
    zn = z.copy(); zn[1:-1] = z[1:-1] - 0.25 * (z[2:] - z[:-2])
    z = zn
    print(f"      j+{passo+1}: " + " ".join(f"{v:+6.3f}" for v in z[2:9]) + "   (nodi m-3 .. m+3)")
print("   la perturbazione raggiunge anche i nodi a monte (m-1, m-2, ...): nessuna proprieta' trasportiva.")
print("   con lo schema upwind (C = 0,5):")
z = np.zeros(11); z[5] = 1.0
for passo in range(3):
    zn = z.copy(); zn[1:] = z[1:] - 0.5 * (z[1:] - z[:-1])
    z = zn
    print(f"      j+{passo+1}: " + " ".join(f"{v:+6.3f}" for v in z[2:9]) + "   (nodi m-3 .. m+3)")

# ============================================================ Es. 3: analisi di Hirt per la diffusione
titolo("Es. 3 -- analisi di Hirt: equazione modificata degli schemi FTCS e BTCS per u_t = nu u_yy")
y, t, nu, tau, h = sp.symbols("y t nu tau h", positive=True)
u = sp.Function("u")(y, t)


def serie(f, dy, dtt, ordine=5):
    """Sviluppo di Taylor di u(y + dy, t + dtt) fino all'ordine indicato (in dy e dtt)."""
    s = 0
    for n in range(ordine + 1):
        for k in range(n + 1):
            s += sp.diff(u, y, k, t, n - k) * dy**k * dtt**(n - k) / (sp.factorial(k) * sp.factorial(n - k))
    return s


def equazione_modificata(implicito):
    liv = tau if implicito else 0
    lhs = (serie(u, 0, tau) - u) / tau
    rhs = nu * (serie(u, h, liv) - 2 * serie(u, 0, liv) + serie(u, -h, liv)) / h**2
    E = sp.expand(lhs - rhs)                              # = 0 e' l'equazione modificata
    # eliminazione delle derivate temporali di ordine > 1 con l'equazione stessa (iterativamente)
    utt = sp.diff(u, t, 2); uyyt = sp.diff(u, y, 2, t)
    # al primo ordine in tau: u_t = nu u_yy + O(tau, h^2) => u_tt = nu u_yyt = nu^2 u_yyyy, u_yyt = nu u_yyyy
    E = E.subs({utt: nu**2 * sp.diff(u, y, 4), uyyt: nu * sp.diff(u, y, 4)})
    E = E.subs(sp.diff(u, t, 3), 0).subs(sp.diff(u, y, 2, t, 2), 0).subs(sp.diff(u, y, 4, t), 0)
    E = sp.expand(E)
    # si trattengono i termini fino a O(tau, h^2)
    coeff4 = -E.coeff(sp.diff(u, y, 4))
    return sp.simplify(coeff4)


R = sp.symbols("R", positive=True)
for nome, impl in (("FTCS (esplicito)", False), ("BTCS (implicito)", True)):
    c4 = equazione_modificata(impl)
    c4R = sp.simplify(c4.subs(tau, R * h**2 / nu))
    print(f"   {nome}: u_t = nu u_yy + [{sp.factor(c4R)}] u_yyyy + ...   "
          f"(= nu h^2/2 ({'1/6 - R' if not impl else '1/6 + R'}))")
print("   Per la (4.2.23) il coefficiente del termine del quarto ordine e' nu h^2 (1/6 - R)/2.")
print("   Verifica sperimentale (FTCS, onda sin(pi y) su [0,1], 41 nodi, t = 0,1):")
Ny = 41; yy = np.linspace(0, 1, Ny); hh = yy[1] - yy[0]
for Rr in (0.1, 1 / 6, 0.3, 0.45, 0.5, 0.55):
    dt = Rr * hh**2
    uu = np.sin(np.pi * yy)
    n = int(round(0.1 / dt))
    for _ in range(n):
        uu[1:-1] = uu[1:-1] + Rr * (uu[2:] - 2 * uu[1:-1] + uu[:-2])
        if np.max(np.abs(uu)) > 1e3:
            break
    err = np.max(np.abs(uu - np.sin(np.pi * yy) * np.exp(-np.pi**2 * n * dt)))
    print(f"      R = {Rr:.3f}: " + ("instabile" if not np.isfinite(err) or err > 1 else f"errore massimo {err:.2e}"))
print("   -> lo schema e' stabile fino a R = 1/2 (von Neumann), non fino a R = 1/6.")

# ============================================================ Es. 4: perturbazione discreta FTCS diffusione
titolo("Es. 4 -- perturbazione discreta per u_(i,j+1) = u_(i,j) + R (u_(i+1,j) - 2 u_(i,j) + u_(i-1,j))")
print("   una perturbazione eps in i = m da' u_m = (1 - 2R) eps, u_(m+-1) = R eps al passo successivo;")
print("   la risposta nel nodo perturbato non cambia segno se 1 - 2R >= 0, cioe' R <= 1/2;")
print("   |1 - 2R| <= 1 (sola ampiezza) darebbe R <= 1, condizione insufficiente:")
for Rr in (0.4, 0.5, 0.6, 0.8):
    z = np.zeros(41); z[20] = 1.0
    for _ in range(200):
        z[1:-1] = z[1:-1] + Rr * (z[2:] - 2 * z[1:-1] + z[:-2])
    print(f"      R = {Rr}: max|u| dopo 200 passi = {np.max(np.abs(z)):.3e}")

# ============================================================ Es. 5: coefficienti dello schema di Torrance
titolo("Es. 5 -- schema upwind di Torrance: coefficienti a1..a5 e criterio (4.3.33)")
Uf, Ub, Vf, Vb, dT, hh_, Pr = sp.symbols("U_f U_b V_f V_b Delta_T h Pr", real=True)
B = 1 / Pr
a1 = -dT / (2 * hh_) * (Uf - sp.Abs(Uf)) + dT * B / hh_**2
a2 = dT / (2 * hh_) * (Ub + sp.Abs(Ub)) + dT * B / hh_**2
a4 = -dT / (2 * hh_) * (Vf - sp.Abs(Vf)) + dT * B / hh_**2
a5 = dT / (2 * hh_) * (Vb + sp.Abs(Vb)) + dT * B / hh_**2
a3 = 1 - dT / (2 * hh_) * (Uf + sp.Abs(Uf) - Ub + sp.Abs(Ub) + Vf + sp.Abs(Vf) - Vb + sp.Abs(Vb)) - 4 * dT * B / hh_**2
print("   a1 =", a1, "\n   a2 =", a2, "\n   a4 =", a4, "\n   a5 =", a5, "\n   a3 =", a3)
print("   a1, a2, a4, a5 >= 0 perche' |U| - U >= 0 e |U| + U >= 0;")
print("   a1+...+a5 = 1 - (Delta T/h)(Uf - Ub + Vf - Vb):", sp.simplify(a1 + a2 + a3 + a4 + a5 - (1 - dT / hh_ * (Uf - Ub + Vf - Vb))) == 0,
      "-> = 1 (consistenza) se il campo discreto e' solenoidale")
print("   a3 >= 0  <=>  Delta T <= [ (Uf+|Uf| - Ub+|Ub| + Vf+|Vf| - Vb+|Vb|)/(2h) + 4/(h^2 Pr) ]^(-1)   (4.3.33)")
from prog4_3_benard import simula as benard, DT as DT_B
print("   Verifica sul problema di Benard (Programma 4.3), 400 passi:")
with np.errstate(all="ignore"):
    sal, sto, dtmin = benard(maxstp=400, stampa=False)
    print(f"      passo massimo consentito dalla (4.3.33) nei primi 400 passi: {dtmin:.5f} (usato {DT_B})")
    for fatt in (0.9, 1.1, 1.5):
        try:
            s2, st2, _ = benard(maxstp=int(400 * DT_B / (fatt * dtmin)), dt=fatt * dtmin, stampa=False)
            m = max(v[1] for v in st2)
            print(f"      dT = {fatt:.1f} x limite = {fatt*dtmin:.5f}: " + ("instabile (overflow)" if not np.isfinite(m) or m > 1e3 else f"max|Psi| = {m:.4f}, regolare"))
        except RuntimeError as e:
            print(f"      dT = {fatt:.1f} x limite: instabile ({e})")

# ============================================================ Es. 6: LU contro Thomas
titolo("Es. 6 -- Burgers AB-CN: fattorizzazione LU contro algoritmo di Thomas")
import prog4_4_burgers_abcn as b4
x1, u1, _ = b4.integra("abcn", 201, 0.005, 1.0, risolutore="thomas")
x2, u2, _ = b4.integra("abcn", 201, 0.005, 1.0, risolutore="lu")
print(f"   differenza massima fra le due soluzioni: {np.max(np.abs(u1 - u2)):.2e} (precisione di macchina)")
n = 199
print(f"   operazioni per passo (n = {n} incognite): Thomas ~ 8n = {8*n} flop (rifa' ogni volta l'eliminazione);")
print(f"   LU: fattorizzazione una volta (~3n), poi ~5n = {5*n} flop per passo (sostituzioni in avanti e all'indietro).")

# ============================================================ Es. 7: coordinate del cilindro
titolo("Es. 7 -- cilindro: laplaciano e termine convettivo nelle coordinate xi, eta")
xi, eta = sp.symbols("xi eta", real=True)
r = sp.exp(sp.pi * xi); th = sp.pi * eta
f = sp.Function("f")(xi, eta)
# derivate rispetto a r e theta tramite la regola della catena: d/dr = (1/(pi r)) d/dxi, d/dtheta = (1/pi) d/deta
d_r = lambda F: sp.diff(F, xi) / (sp.pi * r)
d_th = lambda F: sp.diff(F, eta) / sp.pi
lap_polare = d_r(d_r(f)) + d_r(f) / r + d_th(d_th(f)) / r**2
g = sp.pi**2 * sp.exp(2 * sp.pi * xi)
print("   nabla^2 f - (f_xixi + f_etaeta)/g =", sp.simplify(lap_polare - (sp.diff(f, xi, 2) + sp.diff(f, eta, 2)) / g))
psi, om = sp.Function("psi")(xi, eta), sp.Function("omega")(xi, eta)
vr = d_th(psi) / r; vth = -d_r(psi)
conv = vr * d_r(om) + vth / r * d_th(om)
print("   v_r omega_r + (v_theta/r) omega_theta - (psi_eta omega_xi - psi_xi omega_eta)/g =",
      sp.simplify(conv - (sp.diff(psi, eta) * sp.diff(om, xi) - sp.diff(psi, xi) * sp.diff(om, eta)) / g))
print("   v_r = psi_eta/(pi e^{pi xi}):", sp.simplify(vr - sp.diff(psi, eta) / (sp.pi * r)) == 0,
      ";  v_theta = -psi_xi/(pi e^{pi xi}):", sp.simplify(vth + sp.diff(psi, xi) / (sp.pi * r)) == 0)
# condizione al contorno esterno (4.6.58): psi potenziale
print("   psi potenziale r sin(theta)(1 - 1/r^2) = 2 sin(pi eta) sinh(pi xi):",
      sp.simplify(sp.expand((r * sp.sin(th) * (1 - 1 / r**2) - 2 * sp.sin(sp.pi * eta) * sp.sinh(sp.pi * xi)).rewrite(sp.exp))) == 0)

# ============================================================ Es. 8: funzione di prova per piccoli Re
titolo("Es. 8 -- funzione di prova (4.1.52) per Re -> 0 e soluzione di Stokes")
rr, zz = sp.symbols("r zeta", positive=True)
A1, A2, A3, A4, B1, B2, B3, B4 = sp.symbols("A1:5 B1:5")
f_r = rr**2 / 2 + A1 * rr + A2 + A3 / rr + A4 / rr**2
g_r = B1 * rr + B2 + B3 / rr + B4 / rr**2
psi_t = f_r * (1 - zz**2) + g_r * zz * (1 - zz**2)
D2 = lambda F: sp.diff(F, rr, 2) + (1 - zz**2) / rr**2 * sp.diff(F, zz, 2)
D4 = sp.simplify(D2(D2(psi_t)))
print("   D^4 psi =", sp.factor(D4))
print("   D^4 psi = 0 per ogni r richiede A2 = A4 = 0 e B1 = B3 = 0; le condizioni di aderenza sulla parte")
print("   dispari danno allora B2 = B4 = 0 (nessuna asimmetria avanti-dietro nel limite di Stokes)")
sol = sp.solve([sp.Eq(f_r.subs(rr, 1), 0), sp.Eq(sp.diff(f_r, rr).subs(rr, 1), 0), A2, A4], [A1, A2, A3, A4], dict=True)[0]
print("   con le condizioni di aderenza psi(1) = psi_r(1) = 0:", sol, " -> psi = (r^2/2 - 3r/4 + 1/(4r)) sin^2 theta  (Stokes)")
cd_stokes = sp.Rational(24, 1)
print("   resistenza di Stokes: D = 6 pi mu a U, C_D = 24/Re (Re basato sul diametro)")

# ============================================================ Es. 9: problema di Neumann
titolo("Es. 9 -- equazione di Poisson con condizioni di Neumann su 4 nodi interni")
A = np.array([[-2, 1, 1, 0], [1, -2, 0, 1], [1, 0, -2, 1], [0, 1, 1, -2]], float)
w = np.sort(np.linalg.eigvalsh(A))
print("   autovalori di A:", np.round(w, 4), " -> uno nullo: matrice singolare (rango 3)")
print("   vettore nullo: (1,1,1,1)/2 (la pressione e' definita a meno di una costante); la compatibilita'")
print("   richiede che il termine noto sia ortogonale a (1,1,1,1): somma delle r_ij = 0")
A1m = A.copy(); A1m[0, :] = [1, 0, 0, 0]
w1 = np.linalg.eigvals(A1m)
print("   fissando phi_11 = 0: autovalori", np.round(np.sort(w1.real), 4), f", rcond = {1/np.linalg.cond(A1m, 1):.4f}")
# esempio su griglia N x N: confronto del condizionamento fra Dirichlet e Neumann fissato
for N in (8, 16, 32):
    n = N * N; idx = lambda i, j: i * N + j
    An = np.zeros((n, n)); Ad = np.zeros((n, n))
    for i in range(N):
        for j in range(N):
            p = idx(i, j); Ad[p, p] = -4
            for (a, b) in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < N and 0 <= b < N:
                    An[p, idx(a, b)] = 1; Ad[p, idx(a, b)] = 1; An[p, p] -= 1
    An[0, :] = 0; An[0, 0] = 1
    print(f"   N = {N:2d}: numero di condizionamento (norma 2): Neumann fissato {np.linalg.cond(An):9.1f}, Dirichlet {np.linalg.cond(Ad):9.1f}")

# ------------------------------------------------------------ figura riassuntiva (Es. 1-4)
fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.6))
for c, col in zip((0.25, 0.5, 0.75, 1.0, 1.25), (GRIGIO, VERDE, ARANCIO, BLU, ROSSO)):
    ax[0].plot(bs, np.abs(1 - c * (1 - np.exp(-1j * bs))), color=col, label=f"$C={c:g}$".replace(".", "{,}"))
ax[0].axhline(1, color="k", lw=0.5); ax[0].set_xlabel("$\\beta = kh$"); ax[0].set_ylabel("$|G|$ upwind"); ax[0].legend(fontsize=6)
for c, col in zip((0.1, 0.5, 1.0), (VERDE, ARANCIO, BLU)):
    ax[1].plot(bs, np.sqrt(1 + c * c * np.sin(bs)**2), color=col, label=f"$C={c:g}$".replace(".", "{,}"))
ax[1].axhline(1, color="k", lw=0.5); ax[1].set_xlabel("$\\beta$"); ax[1].set_ylabel("$|G|$ FTCS"); ax[1].legend(fontsize=6)
Rs = np.linspace(0, 1, 201)
ax[2].plot(Rs, np.abs(1 - 4 * Rs), color=BLU, label="von Neumann $|G|_{\\max}=|1-4R|$")
ax[2].plot(Rs, np.abs(1 - 2 * Rs), "--", color=ARANCIO, label="pert. discreta $|1-2R|$")
ax[2].plot(Rs, 6 * (1 / 6 - Rs), ":", color=GRIGIO, label="Hirt: coeff. di $u_{yyyy}$, $6(1/6-R)$")
ax[2].axhline(1, color="k", lw=0.5); ax[2].axhline(0, color="k", lw=0.5)
ax[2].set_xlabel("$R=\\nu\\tau/h^2$"); ax[2].set_ylim(-1.2, 3); ax[2].legend(fontsize=6)
fig.tight_layout()
salva(fig, "esercizi_stabilita")
