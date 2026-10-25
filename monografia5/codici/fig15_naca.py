"""
fig15_naca.py -- figure e risultati del caso c15_naca0012 (Cap. 13): mesh a C, convergenza, polare C_L(alfa) e
C_D(alfa) contro i dati di Ladson e i risultati CFL3D/FUN3D (NASA TMR), C_p a tre incidenze, campo intorno al
profilo a 10 gradi, GCI su tre mesh.
Uso: python3 fig15_naca.py   (dopo python3 genera.py tutto nel caso)
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO, VIOLA
from foamcase import Caso
from foamgci import gci, h_da_celle
from figutil import caso, campo2d, it, scrivi_tex

C = caso("c15_naca0012")
R = json.load(open(os.path.join(C, "risultati.json")))
POL = R["polare"]; G = R["gci"]
NU = 1.6667e-7

# Ladson (1988), NASA TM 4074, 180 grit, Re = 6e6, M = 0,15 (valori tabulati sul sito NASA TMR)
LADSON = np.array([[-4.04, -0.4417, 0.00871], [-2.14, -0.2385, 0.00800], [-0.05, -0.0048, 0.00809], [2.05, 0.2125, 0.00816],
                   [4.04, 0.4316, 0.00823], [6.09, 0.6546, 0.00885], [8.30, 0.8873, 0.01050], [10.12, 1.0707, 0.01201],
                   [11.13, 1.1685, 0.01239], [12.12, 1.2605, 0.01332], [13.08, 1.3455, 0.01503], [14.22, 1.4365, 0.01625],
                   [15.26, 1.5129, 0.01900], [16.30, 1.5739, 0.02218], [17.13, 1.6116, 0.02560], [18.02, 1.5784, 0.03611]])
# CFL3D e FUN3D con Spalart-Allmaras sulla mesh piu' fine della NASA TMR (897x257)
CFL3D = {0: (0.0, 0.00819), 10: (1.0909, 0.01231), 15: (1.5461, 0.02124)}
FUN3D = {0: (0.0, 0.00812), 10: (1.0983, 0.01242), 15: (1.5547, 0.02159)}
# ripartizione attrito / pressione sulla griglia 897x257 (NASA TMR; a 10 gradi CFL3D, a 0 e 15 gradi Flow360)
RIPARTIZIONE = {0: (0.006811, 0.001267), 10: (0.006203, 0.006200), 15: (0.005112, 0.016473)}

alfa = np.array([p["alfa"] for p in POL]); Cl = np.array([p["Cl"] for p in POL]); Cd = np.array([p["Cd"] for p in POL])
Clpy = np.array([p["Cl_python"] for p in POL]); Cdpy = np.array([p["Cd_python"] for p in POL])
Cdp = np.array([p["Cd_pressione"] for p in POL]); Cdf = np.array([p["Cd_attrito"] for p in POL])

# --- mesh --------------------------------------------------------------------------------------------------
c = Caso(C); m = c.mesh
fig, ax = plt.subplots(1, 3, figsize=(9.8, 3.1))
m.disegna(ax[0], lw=0.12); ax[0].set_xlim(-21, 21); ax[0].set_ylim(-21, 21); ax[0].set_aspect("equal"); ax[0].set_title("dominio: $R=20c$, $L=20c$", fontsize=9)
m.disegna(ax[1], lw=0.2); ax[1].set_xlim(-0.3, 1.3); ax[1].set_ylim(-0.6, 0.6); ax[1].set_aspect("equal"); ax[1].set_title("il profilo e la scia", fontsize=9)
m.disegna(ax[2], lw=0.3); ax[2].set_xlim(-0.02, 0.06); ax[2].set_ylim(-0.04, 0.04); ax[2].set_aspect("equal"); ax[2].set_title("bordo d'attacco: strati a parete", fontsize=9)
for a in ax: a.set_xlabel("$x/c$")
ax[0].set_ylabel("$y/c$")
fig.tight_layout(); salva(fig, "c15_mesh")

# --- polare ------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(9.8, 3.2))
ax[0].plot(LADSON[:, 0], LADSON[:, 1], "o", color="k", ms=3.5, mfc="none", label="Ladson, 180 grit")
ax[0].plot(alfa, Cl, "s-", color=BLU, ms=4, lw=1.1, label="OpenFOAM, SA (forceCoeffs)")
ax[0].plot(list(CFL3D), [v[0] for v in CFL3D.values()], "^", color=ROSSO, ms=5, label="CFL3D, SA")
aa = np.linspace(0, 10, 2); ax[0].plot(aa, 2 * np.pi * np.radians(aa), ":", color=GRIGIO, lw=0.9, label="$2\\pi\\alpha$")
ax[0].set_xlabel("$\\alpha$ [gradi]"); ax[0].set_ylabel("$C_L$"); ax[0].legend(fontsize=6.5); ax[0].set_xlim(-0.5, 16)
ax[1].plot(LADSON[:, 0], LADSON[:, 2], "o", color="k", ms=3.5, mfc="none", label="Ladson, 180 grit")
ax[1].plot(alfa, Cd, "s-", color=BLU, ms=4, lw=1.1, label="OpenFOAM, SA")
ax[1].plot(alfa, Cdp, "--", color=ARANCIO, lw=0.9, label="$C_{D,p}$ (pressione)")
ax[1].plot(alfa, Cdf, "--", color=VERDE, lw=0.9, label="$C_{D,f}$ (attrito)")
ax[1].plot(list(CFL3D), [v[1] for v in CFL3D.values()], "^", color=ROSSO, ms=5, label="CFL3D, SA")
ax[1].plot(list(RIPARTIZIONE), [v[1] for v in RIPARTIZIONE.values()], "v", color=ARANCIO, ms=4, mfc="none", label="$C_{D,p}$ rif. (897$\\times$257)")
ax[1].plot(list(RIPARTIZIONE), [v[0] for v in RIPARTIZIONE.values()], "v", color=VERDE, ms=4, mfc="none", label="$C_{D,f}$ rif. (897$\\times$257)")
ax[1].set_xlabel("$\\alpha$ [gradi]"); ax[1].set_ylabel("$C_D$"); ax[1].legend(fontsize=6.5); ax[1].set_xlim(-0.5, 16); ax[1].set_ylim(0, 0.03)
for a_, col in ((0, GRIGIO), (10, BLU), (15, ARANCIO)):
    d = np.load(os.path.join(C, "dati", f"a{a_:02d}.npz"))
    x, cp = d["x"], d["Cp"]; o = np.argsort(x)
    # dorso e ventre separati con il segno di y
    sup = d["y"] >= 0
    for sel, ls in ((sup, "-"), (~sup, "--")):
        xx, cc = x[sel], cp[sel]; oo = np.argsort(xx)
        ax[2].plot(xx[oo], cc[oo], ls, color=col, lw=1.0, label=f"$\\alpha={a_}^\\circ$" if ls == "-" else None)
ax[2].invert_yaxis(); ax[2].set_xlabel("$x/c$"); ax[2].set_ylabel("$C_p$"); ax[2].legend(fontsize=7); ax[2].set_title("dorso (continua) e ventre (tratteggio)", fontsize=9)
fig.tight_layout(); salva(fig, "c15_polare")

# --- campo a 10 gradi e convergenza --------------------------------------------------------------------------
d10 = np.load(os.path.join(C, "dati", "a10.npz"))
fig, ax = plt.subplots(1, 3, figsize=(10.0, 3.1), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
# nel caso restano la mesh media e i campi dell'ultima esecuzione (alfa = 10 gradi)
tempi = c.tempi()
if len(tempi) > 1:
    U = c.campo("U"); p = c.campo("p")
    cf = campo2d(ax[0], m, np.linalg.norm(U, axis=1), livelli=np.linspace(0, 1.6, 17), cmap="viridis", maschera_lato=0.3, extend="max")
    fig.colorbar(cf, ax=ax[0], label="$|\\mathbf{U}|/U_\\infty$", shrink=0.85)
    from nacamesh import naca4
    dorso, ventre = naca4("0012", 200)
    ax[0].fill(np.concatenate([dorso[:, 0], ventre[::-1, 0]]), np.concatenate([dorso[:, 1], ventre[::-1, 1]]), color="#d5d8dc", ec="k", lw=0.5, zorder=5)
    ax[0].set_xlim(-0.5, 1.5); ax[0].set_ylim(-0.7, 0.7); ax[0].set_title("$|\\mathbf{U}|$ a $\\alpha=10^\\circ$", fontsize=9)
ax[0].set_xlabel("$x/c$"); ax[0].set_ylabel("$y/c$")
st = d10["storia"]; col = {n: i for i, n in enumerate(d10["colonne"])}
ax[1].plot(st[:, 0], st[:, col["Cl"]], color=BLU, lw=1.0, label="$C_L$"); ax[1].set_ylabel("$C_L$", color=BLU)
ax1b = ax[1].twinx(); ax1b.plot(st[:, 0], st[:, col["Cd"]], color=ARANCIO, lw=1.0); ax1b.set_ylabel("$C_D$", color=ARANCIO); ax1b.set_ylim(0, 0.05)
ax[1].set_xlabel("iterazione"); ax[1].set_title("$\\alpha=10^\\circ$: storia dei coefficienti", fontsize=9); ax[1].set_ylim(0, 1.5)
ax[2].semilogy(d10["res_t"], d10["res_p"], color=VERDE, lw=1.0, label="$p$"); ax[2].semilogy(d10["res_t"], d10["res_Ux"], color=BLU, lw=1.0, label="$U_x$")
ax[2].set_xlabel("iterazione"); ax[2].set_ylabel("residuo iniziale"); ax[2].legend(fontsize=7); ax[2].set_title("$\\alpha=10^\\circ$: residui", fontsize=9)
fig.tight_layout(); salva(fig, "c15_convergenza")

# --- GCI ---------------------------------------------------------------------------------------------------
liv = ("fine", "media", "grossa")
N = [G[l]["celle"] for l in liv]; h = h_da_celle(N)
gL = gci(h, [G[l]["Cl"] for l in liv]); gD = gci(h, [G[l]["Cd"] for l in liv])
DOM = R.get("dominio", {})
fig, ax = plt.subplots(1, 3, figsize=(10.0, 3.0))
for a_, chiave, g, col in ((ax[0], "Cl", gL, BLU), (ax[1], "Cd", gD, ARANCIO)):
    v = [G[l][chiave] for l in liv]
    a_.plot(h / h[0], v, "o-", color=col, ms=5, lw=1.1, label="calcolato")
    a_.plot(0, g["f_ext"], "*", color=col, ms=9, label=f"estrapolato: {g['f_ext']:.4f}")
    a_.errorbar(h[0] / h[0], v[0], yerr=g["GCI_fine"] * abs(v[0]), fmt="none", ecolor="k", capsize=4, lw=0.8, label=f"GCI fine {100*g['GCI_fine']:.2f}\\%")
    a_.set_xlabel("$h/h_{\\mathrm{fine}}$"); a_.set_ylabel("$C_L$" if chiave == "Cl" else "$C_D$"); a_.legend(fontsize=7); a_.set_xlim(-0.1, 2.2)
    a_.set_title(f"ordine apparente $p={g['p']:.2f}$", fontsize=9)
# effetto del dominio: R = 20 (la polare) e R = 40, contro CFL3D
i10 = int(np.argmin(abs(alfa - 10)))
Rs = [20.0] + [DOM[k]["R"] for k in sorted(DOM)]
ClR = [Cl[i10]] + [DOM[k]["Cl"] for k in sorted(DOM)]
CdR = [Cd[i10]] + [DOM[k]["Cd"] for k in sorted(DOM)]
# l'errore di dominio di un profilo portante decresce come c/R (velocita' indotta dal vortice al contorno):
# con due domini si estrapola linearmente in c/R fino a R infinito
invR = np.array([1.0 / r for r in Rs])
Cl_inf = ClR[-1] - (ClR[-2] - ClR[-1]) * invR[-1] / (invR[-2] - invR[-1]) if len(Rs) > 1 else np.nan
Cd_inf = CdR[-1] - (CdR[-2] - CdR[-1]) * invR[-1] / (invR[-2] - invR[-1]) if len(Rs) > 1 else np.nan
a3 = ax[2]
a3.plot(invR, ClR, "o-", color=BLU, ms=5, lw=1.1, label="$C_L$")
a3.plot([0, invR[-1]], [Cl_inf, ClR[-1]], ":", color=BLU, lw=1.0)
a3.plot(0, Cl_inf, "o", mfc="none", color=BLU, ms=6)
a3.plot(1 / 500, CFL3D[10][0], "*", color=BLU, ms=9, label="CFL3D ($500c$)")
a3.set_xlabel("$c/R$ (inverso della distanza del contorno)"); a3.set_ylabel("$C_L$", color=BLU); a3.set_ylim(1.0, 1.12)
a3b = a3.twinx()
a3b.plot(invR, CdR, "s-", color=ARANCIO, ms=5, lw=1.1, label="$C_D$")
a3b.plot([0, invR[-1]], [Cd_inf, CdR[-1]], ":", color=ARANCIO, lw=1.0)
a3b.plot(0, Cd_inf, "s", mfc="none", color=ARANCIO, ms=6)
a3b.plot(1 / 500, CFL3D[10][1], "*", color=ARANCIO, ms=9, label="CFL3D ($500c$)")
a3b.set_ylabel("$C_D$", color=ARANCIO); a3b.set_ylim(0.010, 0.022)
h1, l1 = a3.get_legend_handles_labels(); h2, l2 = a3b.get_legend_handles_labels()
a3.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="center right")
a3.set_xlim(-0.003, 0.06); a3.set_title("effetto del dominio a $10^\\circ$ (mesh media)", fontsize=9)
fig.tight_layout(); salva(fig, "c15_gci")

# --- riquadri ----------------------------------------------------------------------------------------------
def lad(a_, k):
    return np.interp(a_, LADSON[:, 0], LADSON[:, k])

json.dump({"gci_Cl": gL, "gci_Cd": gD}, open(os.path.join(C, "gci.json"), "w"), indent=1)
Cla = np.polyfit(np.radians(alfa[alfa <= 8]), Cl[alfa <= 8], 1)[0]
righe = ["\\begin{center}\\small\\begin{tabular}{rrrrrrrr}\\toprule $\\alpha$ & $C_L$ & $C_L$ Ladson & $C_D$ & $C_D$ Ladson & $C_{D,p}$ & $C_{D,f}$ & iter.\\\\\\midrule"]
for p_ in POL:
    righe.append(f"{p_['alfa']}$^\\circ$ & {it(p_['Cl'],4)} & {it(lad(p_['alfa'],1),4)} & {it(p_['Cd'],5)} & {it(lad(p_['alfa'],2),5)} & {it(p_['Cd_pressione'],5)} & {it(p_['Cd_attrito'],5)} & {p_['iterazioni']}\\\\")
righe.append("\\bottomrule\\end{tabular}\\end{center}")
i10 = int(np.argmin(abs(alfa - 10))); i0 = int(np.argmin(abs(alfa))); i15 = int(np.argmin(abs(alfa - 15)))
righe.append("\\begin{itemize}")
righe.append(f"\\item Mesh media: {R['celle_media']} celle, $y^+$ medio {it(POL[i10]['yplus_medio'],2)} (massimo {it(POL[i10]['yplus_max'],2)}) a $10^\\circ$: lo strato limite è risolto fino alla parete, "
             f"come richiede \\kw{{nutUSpaldingWallFunction}} usata in modalità ``low-Re''.")
righe.append(f"\\item Pendenza della retta di portanza (fit $0$--$8^\\circ$): $C_{{L\\alpha}}={it(Cla,3)}$ per radiante, cioè {it(Cla/(2*np.pi),3)}$\\cdot2\\pi$ "
             f"(la teoria del profilo sottile dà $2\\pi$, l'esperimento circa $0{{,}}97\\cdot2\\pi$ per lo spessore e lo strato limite).")
righe.append(f"\\item A $\\alpha=10^\\circ$: $C_L={it(Cl[i10],4)}$ contro $1{{,}}0909$ (CFL3D), $1{{,}}0983$ (FUN3D), $1{{,}}0707$ a $10{{,}}12^\\circ$ (Ladson); "
             f"$C_D={it(Cd[i10],5)}$ contro $0{{,}}01231$ (CFL3D) e $0{{,}}01201$ (Ladson): scarto {it(100*abs(Cl[i10]/1.0909-1),1)}\\% su $C_L$ e {it(100*abs(Cd[i10]/0.01231-1),1)}\\% su $C_D$ rispetto a CFL3D.")
righe.append(f"\\item Ripartizione della resistenza contro i valori di riferimento sulla griglia $897\\times257$ (CFL3D a $10^\\circ$~\\cite{{tmr}}, Flow360 a $0^\\circ$ e $15^\\circ$~\\cite{{flow360}}): a $0^\\circ$ attrito {it(Cdf[i0],5)} contro {it(RIPARTIZIONE[0][0],5)} "
             f"e pressione {it(Cdp[i0],5)} contro {it(RIPARTIZIONE[0][1],5)}; a $10^\\circ$ attrito {it(Cdf[i10],5)} contro {it(RIPARTIZIONE[10][0],5)} e pressione {it(Cdp[i10],5)} contro {it(RIPARTIZIONE[10][1],5)}; "
             f"a $15^\\circ$ attrito {it(Cdf[i15],5)} contro {it(RIPARTIZIONE[15][0],5)} e pressione {it(Cdp[i15],5)} contro {it(RIPARTIZIONE[15][1],5)}. "
             f"L'attrito è sovrastimato di un {it(100*np.mean([Cdf[i]/RIPARTIZIONE[a_][0]-1 for i, a_ in ((i0,0),(i10,10),(i15,15))]),0)}\\% circa a tutte le incidenze; la resistenza di pressione è sovrastimata di poco a portanza nulla e sempre di più al crescere di $C_L$.")
righe.append(f"\\item A $15^\\circ$ $C_L={it(Cl[i15],3)}$ contro $1{{,}}546$ (CFL3D) e {it(lad(15,1),3)} (Ladson): il dominio a $20c$ abbassa la portanza anche qui, e CFL3D sta sopra l'esperimento perché il modello SA, senza transizione, ritarda lo stallo.")
righe.append(f"\\item Le forze integrate da \\file{{foamcase.py}} coincidono con \\kw{{forceCoeffs}}: scarto massimo {it(100*np.max(np.abs(Clpy/np.where(Cl==0,1,Cl)-1)[alfa>0]),2)}\\% su $C_L$ e "
             f"{it(100*np.max(np.abs(Cdpy/Cd-1)),2)}\\% su $C_D$ (lo stesso gradiente di parete, la stessa somma $p\\,\\Sf$).")
righe.append(f"\\item GCI a $10^\\circ$ su {N[2]}, {N[1]} e {N[0]} celle ($r\\approx\\sqrt2$): $C_L$ = {it(G['grossa']['Cl'],4)}, {it(G['media']['Cl'],4)}, {it(G['fine']['Cl'],4)}, "
             f"ordine apparente {it(gL['p'],2)}, valore estrapolato {it(gL['f_ext'],4)}, $\\mathrm{{GCI}}_{{\\mathrm{{fine}}}}={it(100*gL['GCI_fine'],2)}$\\%; "
             f"$C_D$ = {it(G['grossa']['Cd'],5)}, {it(G['media']['Cd'],5)}, {it(G['fine']['Cd'],5)}, ordine {it(gD['p'],2)}, estrapolato {it(gD['f_ext'],5)}, "
             f"$\\mathrm{{GCI}}_{{\\mathrm{{fine}}}}={it(100*gD['GCI_fine'],2)}$\\% (convergenza {gD['convergenza']}).")
if DOM:
    k = sorted(DOM)[-1]; d = DOM[k]
    righe.append(f"\\item Dominio: con il contorno a $R=L={d['R']:.0f}c$ (stessa mesh media, stessa prima cella) $C_L={it(d['Cl'],4)}$ e $C_D={it(d['Cd'],5)}$ "
                 f"($C_{{D,p}}={it(d['Cd_pressione'],5)}$) contro {it(Cl[i10],4)} e {it(Cd[i10],5)} a $20c$: "
                 f"{it(100*(d['Cl']/Cl[i10]-1),1)}\\% su $C_L$ e {it(1e4*(d['Cd']-Cd[i10]),0)} conti su $C_D$, "
                 f"cioè {it(100*(Cd[i10]-d['Cd'])/(Cd[i10]-CFL3D[10][1]),0)}\\% dello scarto da CFL3D. L'errore di dominio decresce come $c/R$: "
                 f"l'estrapolazione lineare in $c/R$ dà $C_{{L,\\infty}}\\approx{it(Cl_inf,3)}$ (CFL3D {it(CFL3D[10][0],3)}, FUN3D {it(FUN3D[10][0],3)}) e $C_{{D,\\infty}}\\approx{it(Cd_inf,4)}$. "
                 f"A $60c$ e $100c$ la stessa famiglia di mesh (80 celle normali, scia alta $y_1$ fino all'uscita) diverge: per allontanare il contorno serve ridisegnare la mesh.")
righe.append("\\end{itemize}")
scrivi_tex("c15_risultati", "\n".join(righe))
