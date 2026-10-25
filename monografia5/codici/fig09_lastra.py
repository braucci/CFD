"""
fig09_lastra.py -- figura e risultati del caso c09_lastra (Cap. 9): C_f(Re_x) per il caso laminare (Blasius) e i
tre turbolenti (Karman-Schoenherr); profili u+(y+) a x = 0,5 m con legge lineare e logaritmica.
Uso: python3 fig09_lastra.py   (dopo casi/c09_lastra/genera.py)
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO
from figutil import caso, it, scrivi_tex

C = caso("c09_lastra")
R = json.load(open(os.path.join(C, "risultati.json")))
P = np.load(os.path.join(C, "profili.npz"))
COL = {"laminare": GRIGIO, "SST_risolta": BLU, "SST_wf": ARANCIO, "kEpsilon_wf": VERDE}
ETI = {"laminare": "laminare", "SST_risolta": "$k$–$\\omega$ SST, $y^+\\approx1$", "SST_wf": "$k$–$\\omega$ SST, funzioni di parete", "kEpsilon_wf": "$k$–$\\varepsilon$, funzioni di parete"}

fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.3))
def rif(nome, Rex):
    return 0.664 / np.sqrt(Rex) if nome == "laminare" else 0.370 * np.log10(Rex) ** (-2.584)

# errore medio e C_f di riferimento ricalcolati qui dai profili (Schultz-Grunow per il turbolento)
for nome in COL:
    Rex, cf = P[f"{nome}|Rex"], P[f"{nome}|cf"]
    x = Rex * (1e-5 if nome == "laminare" else 1e-7); sel = (x > 0.1) & (x < 0.95)
    R[nome]["errore_cf_medio"] = float(np.mean(np.abs(cf[sel] - rif(nome, Rex[sel])) / rif(nome, Rex[sel])))
    i5 = np.argmin(np.abs(x - 0.5)); R[nome]["cf_rif_x05"] = float(rif(nome, Rex[i5]))
    R[nome]["cf_white_x05"] = float(0.455 / np.log(0.06 * Rex[i5]) ** 2)

for nome in COL:
    Rex, cf = P[f"{nome}|Rex"], P[f"{nome}|cf"]
    sel = Rex > 0
    ax[0].loglog(Rex[sel], cf[sel], "-", color=COL[nome], lw=1.1, label=ETI[nome])
Rl = np.logspace(3.5, 5, 50); ax[0].loglog(Rl, 0.664 / np.sqrt(Rl), "k:", lw=0.9, label="Blasius $0{,}664\\,\\mathrm{Re}_x^{-1/2}$")
Rt = np.logspace(5.5, 7, 50); ax[0].loglog(Rt, 0.370 * np.log10(Rt) ** (-2.584), "k--", lw=0.9, label="Schultz–Grunow")
ax[0].loglog(Rt, 0.455 / np.log(0.06 * Rt) ** 2, "k-.", lw=0.9, label="White")
ax[0].set_xlabel("$\\mathrm{Re}_x$"); ax[0].set_ylabel("$C_f$"); ax[0].set_ylim(1e-3, 2e-2); ax[0].legend(fontsize=6.5)
yp = np.logspace(-0.5, 3.7, 200)
ax[1].semilogx(yp[yp < 12], yp[yp < 12], "k:", lw=0.9, label="$u^+=y^+$")
ax[1].semilogx(yp[yp > 8], np.log(yp[yp > 8]) / 0.41 + 5.2, "k--", lw=0.9, label="$u^+=\\ln y^+/0{,}41+5{,}2$")
for nome in ("SST_risolta", "SST_wf", "kEpsilon_wf"):
    y, u = P[f"{nome}|yplus"], P[f"{nome}|uplus"]
    sel = (y > 0) & (y < 4000)
    ax[1].semilogx(y[sel], u[sel], "o" if "wf" in nome else "-", color=COL[nome], ms=3, mfc="none", lw=1.1, label=ETI[nome])
ax[1].set_xlabel("$y^+$"); ax[1].set_ylabel("$u^+$"); ax[1].set_ylim(0, 30); ax[1].legend(fontsize=6.5, loc="upper left")
ax[1].set_title("profilo a $x=0{,}5$ m ($\\mathrm{Re}_x=5\\times10^6$)", fontsize=9)
fig.tight_layout(); salva(fig, "c09_lastra")

righe = ["\\begin{center}\\footnotesize\\begin{tabular}{lrrrrr}\\toprule variante & iter. & $y^+$ medio & $C_f(0{,}5)$ & riferimento & scarto medio\\\\\\midrule"]
for nome in COL:
    r = R[nome]
    righe.append(f"{ETI[nome]} & {r['iterazioni']} & {it(r['yplus_medio'],1)} & {it(r['cf_x05'],5)} & {it(r['cf_rif_x05'],5)} & {it(100*r['errore_cf_medio'],1)}\\%\\\\")
righe.append("\\bottomrule\\end{tabular}\\end{center}")
righe.append("\\begin{itemize}")
righe.append(f"\\item Il caso laminare ($\\mathrm{{Re}}_L=10^5$) segue Blasius con errore medio del {it(100*R['laminare']['errore_cf_medio'],1)}\\%: "
             f"è la verifica che la mesh e lo schema risolvono lo strato limite ($\\delta/x\\approx5/\\sqrt{{\\mathrm{{Re}}_x}}$, circa {int(5/np.sqrt(1e5)*1/2e-4):d} celle nello strato a $x=1$~m).")
righe.append(f"\\item Nel caso turbolento risolto ($y^+\\approx{it(R['SST_risolta']['yplus_medio'],1)}$) il profilo ricostruisce il sottostrato viscoso e la legge logaritmica senza alcun modello di parete; "
             f"nelle varianti con funzioni di parete ($y^+\\approx{it(R['SST_wf']['yplus_medio'],0)}$) il primo centro cella cade al limite inferiore della regione logaritmica (\\kw{{nutkWallFunction}} applica la legge logaritmica sopra $y^+\\approx11$ e quella lineare sotto) e il profilo è noto solo da lì in su.")
righe.append(f"\\item $C_f$ a $x=0{{,}}5$~m: SST risolta {it(R['SST_risolta']['cf_x05'],5)}, SST con funzioni di parete {it(R['SST_wf']['cf_x05'],5)}, "
             f"$k$--$\\varepsilon$ {it(R['kEpsilon_wf']['cf_x05'],5)} contro {it(R['SST_risolta']['cf_rif_x05'],5)} (Schultz--Grunow) e {it(R['SST_risolta']['cf_white_x05'],5)} (White): tutti entro il "
             f"{it(100*max(R[n]['errore_cf_medio'] for n in ('SST_risolta','SST_wf','kEpsilon_wf')),1)}\\%, che è anche la dispersione fra le correlazioni (e fra gli esperimenti) a questo numero di Reynolds; il modello SST risolto a parete è il più basso, come nei confronti della NASA TMR sulla lastra piana.")
righe.append(f"\\item Lo sforzo di parete calcolato da \\file{{foamcase.py}} e quello della function object \\kw{{wallShearStress}} differiscono al più di "
             f"{it(100*max(R[n]['scarto_FO_vs_python'] for n in COL if R[n]['scarto_FO_vs_python']==R[n]['scarto_FO_vs_python']),2)}\\% (stessa formula: $\\nu_{{\\mathrm{{eff}}}}\\,\\partial u/\\partial n$ con il gradiente della prima cella).")
righe.append("\\end{itemize}")
scrivi_tex("c09_risultati", "\n".join(righe))
