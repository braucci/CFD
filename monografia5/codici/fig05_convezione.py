"""
fig05_convezione.py -- figure e risultati del caso c05_convezione (Cap. 5): profili di s lungo x = 0,5 per gli
schemi convettivi (da profili.npz), tabella delle metriche (risultati.json) e campi s per upwind, linear e
vanLeer su 40 x 40 (calcolati qui in copie temporanee del caso).
Uso: python3 fig05_convezione.py   (dopo casi/c05_convezione/genera.py)
"""
import os
import re
import sys
import json
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, VIOLA, GIALLO, GRIGIO
from foamcase import Caso
from figutil import caso, campo2d, it, scrivi_tex

C = caso("c05_convezione")
R = {(r["schema"], r["N"]): r for r in json.load(open(os.path.join(C, "risultati.json")))}
P = np.load(os.path.join(C, "profili.npz"))

GRUPPI = [("primo ordine e centrato", ["upwind", "linear"]),
          ("ordine superiore non limitati", ["linearUpwind", "QUICK"]),
          ("TVD", ["vanLeer", "limitedLinear 1", "Minmod", "SuperBee", "MUSCL", "limitedLinear 0.2"])]
COL = [BLU, ARANCIO, VERDE, ROSSO, VIOLA, GIALLO]

fig, ax = plt.subplots(1, 3, figsize=(9.8, 3.2), sharey=True)
for k, (titolo, schemi) in enumerate(GRUPPI):
    ax[k].plot([0, 0, 1, 1], [0, 0.5, 0.5, 1], color=GRIGIO, lw=1.6, alpha=0.6, label="esatta")
    for j, s in enumerate(schemi):
        chiave = f"{s}|40"
        if chiave not in P:
            continue
        yy, ss = P[chiave].T
        ax[k].plot(ss, yy, "-", color=COL[j], lw=1.0, marker="o" if k < 2 else None, ms=2.5, label=s)
    ax[k].set_title(titolo, fontsize=9); ax[k].set_xlabel("$s$ su $x=0{,}5$"); ax[k].legend(fontsize=6.5, loc="lower left")
    ax[k].set_xlim(-0.25, 1.25); ax[k].axvline(0, color="k", lw=0.4, ls=":"); ax[k].axvline(1, color="k", lw=0.4, ls=":")
ax[0].set_ylabel("$y$")
fig.tight_layout(); salva(fig, "c05_profili")

# --- campi su 40 x 40 per tre schemi (copie temporanee) --------------------------------------------
def esegui_copia(schema, voce, N=40):
    dest = C + f"_{re.sub(r'[^A-Za-z0-9]', '', schema)}{N}"
    if not os.path.isdir(dest):
        shutil.copytree(C, dest, ignore=shutil.ignore_patterns("[1-9]*", "log.*", "postProcessing", "*.npz", "*.json"))
        for f, k, v in (("blockMeshDict", "N", str(N)), ("fvSchemes", "div(phi,s)", voce)):
            p = os.path.join(dest, "system", f); s = open(p).read()
            s = re.sub(rf"(\n\s*{re.escape(k)}\s+)[^;]*;", rf"\g<1>{v};", s, count=1); open(p, "w").write(s)
        subprocess.run("blockMesh > log.blockMesh && foamRun > log.foamRun", cwd=dest, shell=True, check=True)
    return Caso(dest)

fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.2))
for k, (s, voce) in enumerate((("upwind", "Gauss upwind"), ("linear", "Gauss linear"), ("vanLeer", "Gauss vanLeer"))):
    c = esegui_copia(s, voce)
    ss = c.campo("s")
    cf = campo2d(ax[k], c.mesh, ss, livelli=np.linspace(-0.2, 1.2, 15), cmap="RdYlBu_r", extend="both")
    ax[k].tricontour(c.mesh.C[:, 0], c.mesh.C[:, 1], ss, levels=[0.1, 0.5, 0.9], colors="k", linewidths=0.5)
    ax[k].set_title(f"{s}: min {ss.min():+.3f}, max {ss.max():.3f}", fontsize=9); ax[k].set_xlabel("$x$")
    ax[k].plot([0, 1], [0, 1], color="w", lw=0.6, ls="--")
ax[0].set_ylabel("$y$")
fig.colorbar(cf, ax=ax, label="$s$", shrink=0.8, pad=0.02)
salva(fig, "c05_campi")

# --- riquadro -------------------------------------------------------------------------------------
def r(s, N=40):
    return R[(s, N)]

righe = ["\\begin{center}\\small\\begin{tabular}{lrrrrr}\\toprule schema ($40\\times40$) & $s_{\\min}$ & $s_{\\max}$ & spessore & errore medio & iterazioni\\\\\\midrule"]
for _, schemi in GRUPPI:
    for s in schemi:
        d = r(s)
        if d.get("fallito"):
            righe.append(f"\\kw{{{s}}} & \\multicolumn{{5}}{{c}}{{non converge}}\\\\"); continue
        righe.append(f"\\kw{{{s}}} & {it(d['min'],3)} & {it(d['max'],3)} & {it(d['spessore'],3)} & {it(d['errore_medio'],4)} & {d['iterazioni']}\\\\")
righe.append("\\bottomrule\\end{tabular}\\end{center}")
u40, u80 = r("upwind", 40), r("upwind", 80); v40, v80 = r("vanLeer", 40), r("vanLeer", 80)
lin40 = r("linear", 40)
righe.append("\\begin{itemize}")
righe.append(f"\\item Raddoppiando $N$ lo spessore del gradino con \\kw{{upwind}} passa da {it(u40['spessore'],3)} a {it(u80['spessore'],3)} "
             f"(rapporto {it(u40['spessore']/u80['spessore'],2)}, cioè $\\propto\\sqrt{{h}}$ come prevede la diffusione numerica $\\Gamma_{{\\mathrm{{num}}}}\\propto h$ "
             f"su una lunghezza fissa), con \\kw{{vanLeer}} da {it(v40['spessore'],3)} a {it(v80['spessore'],3)} (rapporto {it(v40['spessore']/v80['spessore'],2)}).")
righe.append(f"\\item Lo schema \\kw{{linear}} converge lentissimamente (residuo {it(lin40['residuo_finale'],1,True)} dopo 1500 iterazioni: la matrice centrata non è diagonalmente dominante) e viola la limitatezza: $s\\in[{it(lin40['min'],3)},\\,{it(lin40['max'],3)}]$ contro $[0,1]$ "
             f"della soluzione esatta; tutti gli schemi TVD restano in $[0,1]$ a meno di {it(max(abs(r(s)['min']) + abs(r(s)['max']-1) for s in GRUPPI[2][1] if not r(s).get('fallito')),1,True)}.")
falliti = [s for _, ss in GRUPPI for s in ss if r(s).get("fallito")]
stallo = [s for s in GRUPPI[2][1] if not r(s).get("fallito") and r(s)["residuo_finale"] > 1e-6]
righe.append(f"\\item Errore medio su $40\\times40$: \\kw{{upwind}} {it(u40['errore_medio'],4)}, \\kw{{vanLeer}} {it(v40['errore_medio'],4)}, "
             f"\\kw{{limitedLinear 1}} {it(r('limitedLinear 1')['errore_medio'],4)}, \\kw{{limitedLinear 0.2}} {it(r('limitedLinear 0.2')['errore_medio'],4)}, \\kw{{Minmod}} {it(r('Minmod')['errore_medio'],4)}: "
             f"il limitatore più compressivo dà il gradino più netto, il più diffusivo (\\kw{{Minmod}}) sta fra \\kw{{upwind}} e \\kw{{vanLeer}}.")
if falliti:
    righe.append("\\item " + " e ".join(f"\\kw{{{s}}}" for s in falliti) + " non arrivano a una soluzione stazionaria: il limitatore vale 2 su molte facce, "
                 "il peso di faccia diventa interamente sottovento e il coefficiente diagonale di alcune celle si annulla (il precondizionatore DILU divide per zero). "
                 "Nei calcoli transitori il termine $\\partial/\\partial t$ rende la diagonale positiva e questi schemi funzionano; nei calcoli stazionari vanno evitati.")
if stallo:
    righe.append("\\item Con " + ", ".join(f"\\kw{{{s}}}" for s in stallo) + " il residuo si arresta fra $10^{-4}$ e $10^{-6}$ senza scendere oltre nelle 1500 iterazioni: "
                 "il limitatore cambia valore da un'iterazione all'altra sulle facce vicine al gradino (\\emph{limiter cycling}); il campo è però stazionario alla quarta cifra, "
                 "ed è il comportamento normale degli schemi TVD nei calcoli stazionari, da cui la raccomandazione di giudicare la convergenza anche sulle grandezze integrali.")
righe.append("\\end{itemize}")
scrivi_tex("c05_risultati", "\n".join(righe))
