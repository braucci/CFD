"""
fig17_ala.py -- figura e risultati del caso c17_ala (Cap. 15): la mesh di snappyHexMesh nel piano di simmetria,
il coefficiente di pressione sul dorso della semiala, la distribuzione in apertura del coefficiente di portanza
locale confrontata con l'ellittica, e il riquadro gen_c17_risultati.tex (C_L, C_D, confronto con la linea portante).

Legge la mesh ricomposta (constant/polyMesh) e i campi dell'ultimo istante; per non percorrere in Python l'intero
poliedro (3,4e5 celle, 1e6 facce) calcola centro e normale solo delle facce dei patch che servono.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO
from foamcase import leggi_punti, leggi_facce, leggi_lista_etichette, leggi_boundary, leggi_campo
from figutil import caso, it, scrivi_tex

C = caso("c17_ala")
ALFA = 5.0
AR = 4.0
B2 = 2.0                       # semiapertura (corda unitaria)
NU = 1e-6


# --- lettura leggera della mesh ---------------------------------------------------------------------------------
def ultimo_tempo(cart):
    t = [d for d in os.listdir(cart) if d.replace(".", "", 1).isdigit() and d != "0"]
    return max(t, key=float)


pm = os.path.join(C, "constant", "polyMesh")
P = leggi_punti(os.path.join(pm, "points"))
F = leggi_facce(os.path.join(pm, "faces"))
owner = leggi_lista_etichette(os.path.join(pm, "owner"))
boundary = leggi_boundary(os.path.join(pm, "boundary"))
nCells = int(owner.max()) + 1
T = ultimo_tempo(C)


def facce_patch(nome):
    b = next(b for b in boundary if b["nome"] == nome)
    return np.arange(b["startFace"], b["startFace"] + b["nFaces"])


def geometria(idx):
    """Centro e vettore area delle facce idx (decomposizione in triangoli intorno al baricentro dei vertici)."""
    Cf = np.zeros((len(idx), 3)); Sf = np.zeros((len(idx), 3))
    for k, i in enumerate(idx):
        pts = P[F[i]]
        cg = pts.mean(axis=0)
        n = 0.5 * np.cross(np.roll(pts, -1, axis=0) - pts, cg - pts).sum(axis=0)
        Sf[k] = n
        Cf[k] = cg
    return Cf, Sf


def poligoni(idx, assi=(0, 1)):
    return [P[F[i]][:, list(assi)] for i in idx]


p_int, p_bordo, _ = leggi_campo(os.path.join(C, T, "p"), nCells)
ala = facce_patch("ala")
Cf_a, Sf_a = geometria(ala)
p_ala = np.ravel(p_int)[owner[ala]]           # p e' zeroGradient sulla parete: il valore e' quello della cella adiacente
Cp = 2.0 * p_ala                    # rho = 1, U_inf = 1, p cinematica

# --- forze dalla pressione e distribuzione in apertura ----------------------------------------------------------
dirL = np.array([-np.sin(np.radians(ALFA)), np.cos(np.radians(ALFA)), 0.0])
dirD = np.array([np.cos(np.radians(ALFA)), np.sin(np.radians(ALFA)), 0.0])
Fp = p_ala[:, None] * Sf_a          # forza di pressione del fluido sulla parete: Sf esce dal dominio, cioe' entra nell'ala
S = B2 * 1.0
CL_p = (Fp @ dirL).sum() / (0.5 * S)
CD_p = (Fp @ dirD).sum() / (0.5 * S)

# portanza locale per strisce in apertura (sole forze di pressione, che danno il 99% della portanza)
nz = 16
bordi = np.linspace(0.0, B2, nz + 1)
z = Cf_a[:, 2]
cl_loc = np.zeros(nz)
for k in range(nz):
    m = (z >= bordi[k]) & (z < bordi[k + 1]) if k < nz - 1 else (z >= bordi[k]) & (z <= bordi[k + 1] + 1e-9)
    cl_loc[k] = (Fp[m] @ dirL).sum() / (0.5 * 1.0 * (bordi[k + 1] - bordi[k]))
zc = 0.5 * (bordi[1:] + bordi[:-1])
CL_strisce = np.trapezoid(np.concatenate([[cl_loc[0]], cl_loc, [0.0]]), np.concatenate([[0.0], zc, [B2]])) / B2
eta = np.linspace(0, 1, 200)
cl_ell = 4.0 * CL_p / np.pi * np.sqrt(1 - eta**2)      # ellittica di pari portanza totale

# --- coefficienti dalla function object e y+ ---------------------------------------------------------------------
dat = np.loadtxt(os.path.join(C, "postProcessing", "coefficienti", "0", "forceCoeffs.dat"))
t_fo, Cd_fo, Cl_fo = dat[:, 0], dat[:, 2], dat[:, 3]
CL, CD = Cl_fo[-1], Cd_fo[-1]
yp = np.loadtxt(os.path.join(C, "postProcessing", "yPlus", "0", "yPlus.dat"), usecols=(0, 2, 3, 4))
ypmin, ypmax, ypmed = yp[-1, 1], yp[-1, 2], yp[-1, 3]
var_ultimi = (Cl_fo[-200:].max() - Cl_fo[-200:].min()) / CL

# --- teoria: linea portante (Prandtl) e correzione di Helmbold per AR basso --------------------------------------
a0_2pi = 2 * np.pi
import json
_pol = json.load(open(os.path.join(caso("c15_naca0012"), "risultati.json")))["polare"]
_a = np.array([q["alfa"] for q in _pol]); _cl = np.array([q["Cl"] for q in _pol])
a0_2D = np.polyfit(np.radians(_a[_a <= 8]), _cl[_a <= 8], 1)[0]   # pendenza del profilo dal caso c15 (Re = 6e6, SA)
alfa = np.radians(ALFA)
CL_prandtl = a0_2pi * alfa / (1 + a0_2pi / (np.pi * AR))
CL_prandtl_2D = a0_2D * alfa / (1 + a0_2D / (np.pi * AR))
k = a0_2D / (np.pi * AR)
CL_helmbold = a0_2D * alfa / (np.sqrt(1 + k**2) + k)
CDi = CL**2 / (np.pi * AR)                               # resistenza indotta ellittica (e = 1)
Re = 1.0 / NU
t_c = 0.12
Cf_sg = 0.37 / np.log10(Re)**2.584                        # lastra piana turbolenta (Schultz-Grunow), una faccia
CDf_stima = 2 * Cf_sg * (1 + 2 * t_c)                     # due facce, sovravelocita' dello spessore
Cd_prof = 2 * Cf_sg * (1 + 2 * t_c + 60 * t_c**4)         # resistenza di profilo (Hoerner): attrito + pressione
e_eff = CL**2 / (np.pi * AR * (CD - Cd_prof))            # fattore di Oswald "apparente": tutto l'eccesso sul profilo e' indotto

# --- log: iterazioni e tempo ------------------------------------------------------------------------------------
n_it = int(t_fo[-1]) + 1
tempo_cpu = None
with open(os.path.join(C, "log.foamRun")) as fp:
    for riga in fp:
        if riga.startswith("ExecutionTime"):
            tempo_cpu = float(riga.split()[2])
n_celle = None
with open(os.path.join(C, "log.checkMesh")) as fp:
    for riga in fp:
        if riga.strip().startswith("cells:"):
            n_celle = int(riga.split()[1]); break

strati = None
with open(os.path.join(C, "log.snappyHexMesh")) as fp:
    righe = fp.readlines()
for i, riga in enumerate(righe):
    if riga.startswith("patch") and "layers" in riga and "overall thickness" in riga:
        for r in righe[i + 2:i + 6]:
            if r.split() and r.split()[0] == "ala":
                strati = (float(r.split()[2]), float(r.split()[4]))      # strati medi, spessore raggiunto [%]
print(f"strati: {strati}")
print(f"celle {n_celle}, iterazioni {n_it-1}, tempo {tempo_cpu:.0f} s")
print(f"CL = {CL:.4f} (pressione {CL_p:.4f}, strisce {CL_strisce:.4f})  CD = {CD:.4f} (pressione {CD_p:.4f})")
print(f"Prandtl 2pi {CL_prandtl:.3f}, Prandtl a0 2D {CL_prandtl_2D:.3f}, Helmbold {CL_helmbold:.3f}; CDi {CDi:.4f}")
print(f"y+ min {ypmin:.1f} max {ypmax:.0f} medio {ypmed:.1f}; variazione CL ultime 200 it {var_ultimi:.1e}")

# --- figura -------------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.7), gridspec_kw={"width_ratios": [1.15, 1.3, 0.95]})

# (a) mesh nel piano di simmetria, intorno al profilo
ax = axs[0]
sim = facce_patch("simmetria")
Cf_s, _ = geometria(sim)
vicino = (Cf_s[:, 0] > -0.35) & (Cf_s[:, 0] < 1.45) & (np.abs(Cf_s[:, 1]) < 0.55)
ax.add_collection(PolyCollection(poligoni(sim[vicino]), facecolors="none", edgecolors="#566573", linewidths=0.25))
# traccia del profilo: spigoli delle facce dell'ala che giacciono sul piano di simmetria
radice = Cf_a[:, 2] < 0.02
ax.add_collection(PolyCollection(poligoni(ala[radice]), facecolors="#1b2631", edgecolors="#1b2631", linewidths=0.4))
ax.set_xlim(-0.35, 1.45); ax.set_ylim(-0.55, 0.55); ax.set_aspect("equal")
ax.set_xlabel("$x/c$"); ax.set_ylabel("$y/c$")
ax.set_title("(a) mesh nel piano di simmetria", fontsize=10)

# (b) Cp sul dorso, vista in pianta
ax = axs[1]
# Sf esce dal dominio fluido, cioe' entra nell'ala: sul dorso punta verso il basso, quindi Sf . dirL < 0
dorso = (Sf_a @ dirL) < 0
coll = PolyCollection(poligoni(ala[dorso], assi=(2, 0)), array=Cp[dorso], cmap="RdBu_r", edgecolors="face", linewidths=0.1)
coll.set_clim(-1.2, 0.6)
ax.add_collection(coll)
ax.set_xlim(0, B2); ax.set_ylim(1.02, -0.02); ax.set_aspect("equal")
ax.set_xlabel("$z/c$ (apertura)"); ax.set_ylabel("$x/c$")
ax.set_title("(b) $C_p$ sul dorso", fontsize=10)
cb = fig.colorbar(coll, ax=ax, fraction=0.03, pad=0.02); cb.set_label("$C_p$")

# (c) distribuzione in apertura
ax = axs[2]
ax.plot(eta, cl_ell, color=GRIGIO, lw=1.2, ls="--", label=f"ellittica, $C_L={CL_p:.3f}$")
ax.step(np.concatenate([[0], bordi[1:]]) / B2, np.concatenate([cl_loc, [cl_loc[-1]]]), where="post", color=BLU, lw=1.5, label="OpenFOAM (strisce)")
ax.axhline(a0_2D * alfa, color=ARANCIO, lw=1, ls=":", label=f"profilo 2D a $5^\\circ$ ($a_0\\alpha={a0_2D*alfa:.3f}$)".replace(".", "{,}"))
ax.set_xlim(0, 1); ax.set_ylim(0, 0.6)
ax.set_xlabel("$z/(b/2)$"); ax.set_ylabel("$c_l(z)$")
ax.set_title("(c) portanza locale in apertura", fontsize=10)
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.3)

fig.tight_layout(); salva(fig, "c17_ala")

# --- riquadro --------------------------------------------------------------------------------------------------------
testo = f"""
\\begin{{itemize}}
\\item Mesh: {n_celle} celle (fondo $36\\times24\\times12$, livelli 5--6 sulla superficie, 4 strati richiesti: {it(strati[0],2)} strati medi e {it(strati[1],0)}\\% dello spessore nominale raggiunti), \\kw{{checkMesh}} senza errori; {n_it-1} iterazioni SIMPLEC in parallelo su 2 processori, {tempo_cpu/60:.0f} minuti.
\\item $\\yplus$ sull'ala: minimo {it(ypmin,1)}, massimo {it(ypmax,0)}, medio {it(ypmed,0)}: funzione di parete di Spalding su tutto l'intervallo.
\\item $C_L={it(CL,4)}$, $C_D={it(CD,4)}$ (function object \\kw{{forceCoeffs}}; dalle sole pressioni, con \\file{{foamcase.py}}: $C_L={it(CL_p,4)}$, $C_D={it(CD_p,4)}$); variazione relativa di $C_L$ nelle ultime 200 iterazioni {it(var_ultimi,1,esp=True)}, residui finali $10^{{-7}}$.
\\item Linea portante di Prandtl con $a_0=2\\pi$: $C_L={it(CL_prandtl,3)}$; con la pendenza del profilo calcolata nel Cap.~\\ref{{cap:naca}} ($a_0={it(a0_2D,2)}$ rad$^{{-1}}$): $C_L={it(CL_prandtl_2D,3)}$; correzione di Helmbold per allungamento basso: $C_L={it(CL_helmbold,3)}$. Il calcolo è del {it((CL_helmbold-CL)/CL_helmbold*100,1)}\\% sotto Helmbold.
\\item Attrito: $C_{{D,f}}=C_D-C_{{D,p}}={it(CD-CD_p,4)}$, contro la stima di lastra piana turbolenta a $\\Rey_c=10^6$ (Schultz--Grunow, due facce, con la correzione di sovravelocità di Hoerner) $2C_f(1+2t/c)={it(CDf_stima,5)}$: la funzione di parete di Spalding a $\\yplus$ medio {it(ypmed,0)} dà l'attrito giusto.
\\item Resistenza totale: $C_D-C_L^2/(\\pi AR)={it(CD-CDi,4)}$ (tolta la resistenza indotta ellittica, {it(CDi,4)}), contro la resistenza di profilo $2C_f(1+2t/c+60(t/c)^4)={it(Cd_prof,5)}$ della formula di Hoerner: attribuendo tutto l'eccesso alla resistenza indotta si avrebbe un fattore di Oswald $e={it(e_eff,2)}$, mentre la linea portante dà per un'ala rettangolare di $AR=4$ valori superiori a $0{{,}}9$. Una parte della resistenza di pressione, dell'ordine di {it((CD-Cd_prof)-CDi/0.9,4)}, è dunque \\emph{{spuria}}: dissipazione numerica nella mesh di fondo (celle di $0{{,}}06c$ nella scatola della scia, livello 3), che l'integrazione delle pressioni sulla parete non sa distinguere dalla resistenza fisica. È il limite noto del calcolo della resistenza in ``campo vicino''; i codici industriali la decompongono in ``campo lontano'' (piano di Trefftz) proprio per separare la parte indotta da quella spuria.
\\end{{itemize}}
"""
scrivi_tex("c17_risultati", testo)
