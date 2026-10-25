"""
fig06_tempo.py -- figura e risultati del caso c06_tempo (Cap. 6): ordine di convergenza temporale (diffusione) e
gaussiana trasportata a Courant alto con i tre schemi temporali.
Uso: python3 fig06_tempo.py   (dopo casi/c06_tempo/genera.py)
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO
from figutil import caso, it, scrivi_tex

C = caso("c06_tempo")
R = json.load(open(os.path.join(C, "risultati.json")))
P = np.load(os.path.join(C, "profili.npz"))
COL = {"Euler": BLU, "backward": ARANCIO, "CrankNicolson 1": VERDE, "CrankNicolson 0.9": ROSSO}


def serie(schema):
    r = sorted([d for d in R["diffusione"] if d["schema"] == schema], key=lambda d: d["dt"])
    return np.array([d["dt"] for d in r]), np.array([d["errore"] for d in r])


def ordine(dt, e):
    return np.polyfit(np.log(dt), np.log(e), 1)[0]


fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.2), gridspec_kw=dict(width_ratios=[1, 1.4]))
ordini = {}
for s, col in COL.items():
    dt, e = serie(s)
    ordini[s] = ordine(dt, e)
    ax[0].loglog(dt, e, "o-", color=col, ms=4, lw=1.0, label=f"{s} ($p={ordini[s]:.2f}$)")
dd = np.array([dt.min(), dt.max()])
e1 = serie("Euler")[1]
ax[0].loglog(dd, e1[-1] * dd / dd[-1], color=GRIGIO, lw=0.7, alpha=0.7); ax[0].text(dd[0] * 1.05, e1[-1] * dd[0] / dd[-1] * 1.5, "$\\Delta t$", fontsize=8, color=GRIGIO)
ax[0].loglog(dd, e1[-1] * 0.3 * (dd / dd[-1])**2, color=GRIGIO, lw=0.7, alpha=0.7); ax[0].text(dd[0] * 1.05, e1[-1] * 0.3 * (dd[0] / dd[-1])**2 * 1.5, "$\\Delta t^2$", fontsize=8, color=GRIGIO)
ax[0].set_xlabel("$\\Delta t$ [s]"); ax[0].set_ylabel("errore massimo a $t=0{,}1$ s"); ax[0].legend(fontsize=7)
ax[0].set_title("diffusione: $T=\\sin\\pi x\\,e^{-\\pi^2 D t}$", fontsize=9)

x = P["x"]
ax[1].plot(x, P["esatta"], color=GRIGIO, lw=1.8, alpha=0.7, label="esatta")
for s in ("Euler", "backward", "CrankNicolson 1"):
    ax[1].plot(x, P[f"{s}|4"], "-", color=COL[s], lw=1.0, label=s)
ax[1].plot(x, P["Euler|0.5"], ":", color=BLU, lw=1.0, label="Euler, $\\mathrm{Co}=0{,}5$")
ax[1].set_xlim(4.5, 9.5); ax[1].set_xlabel("$x$"); ax[1].set_ylabel("$s$ a $t=5$ s"); ax[1].legend(fontsize=7)
ax[1].set_title("convezione a $\\mathrm{Co}=4$ (400 celle, schema spaziale linear)", fontsize=9)
fig.tight_layout(); salva(fig, "c06_tempo")


def conv(s, Co):
    return next(d for d in R["convezione"] if d["schema"] == s and d["Co"] == Co)

righe = ["\\begin{itemize}"]
righe.append(f"\\item Diffusione, ordine osservato nel tempo: \\kw{{Euler}} {it(ordini['Euler'],2)}, \\kw{{backward}} {it(ordini['backward'],2)}, "
             f"\\kw{{CrankNicolson 1}} {it(ordini['CrankNicolson 1'],2)}, \\kw{{CrankNicolson 0.9}} {it(ordini['CrankNicolson 0.9'],2)}. "
             f"Con $\\Delta t=0{{,}}0025$~s l'errore di Euler è {it(serie('Euler')[1][0],2,True)}, quello di Crank--Nicolson {it(serie('CrankNicolson 1')[1][0],2,True)}: "
             f"a parità di passo la differenza è di {serie('Euler')[1][0]/serie('CrankNicolson 1')[1][0]:.0f} volte.")
righe.append("\\item Convezione a $t=5$~s (la gaussiana ha percorso 5 unità, 200 celle): errore massimo\\\\[2pt]"
             "\\begin{tabular}{lcccc} $\\mathrm{Co}$ & 0{,}5 & 1 & 2 & 4\\\\\\hline "
             + " ".join(f"\\kw{{{s}}} & " + " & ".join(it(conv(s, Co)['errore'], 3) for Co in (0.5, 1.0, 2.0, 4.0)) + "\\\\" for s in ("Euler", "backward", "CrankNicolson 1"))
             + "\\end{tabular}")
e4, b4, c4 = conv("Euler", 4.0), conv("backward", 4.0), conv("CrankNicolson 1", 4.0)
righe.append(f"\\item A $\\mathrm{{Co}}=4$ nessuno schema diverge (sono tutti impliciti), ma \\kw{{Euler}} spiana il picco a {it(e4['max'],3)} "
             f"(diffusione numerica temporale $\\propto\\Delta t$), \\kw{{backward}} a {it(b4['max'],3)}, \\kw{{CrankNicolson 1}} a {it(c4['max'],3)} "
             f"con un minimo di {it(c4['min'],3)}: le oscillazioni dietro il picco sono l'errore di fase dello schema non dissipativo.")
righe.append(f"\\item \\kw{{Euler}} a $\\mathrm{{Co}}=0{{,}}5$ (passo otto volte più piccolo, otto volte il costo) ha ancora un errore di {it(conv('Euler',0.5)['errore'],3)}, "
             f"maggiore di quello di \\kw{{backward}} a $\\mathrm{{Co}}=2$ ({it(conv('backward',2.0)['errore'],3)}) e {conv('Euler',0.5)['errore']/conv('CrankNicolson 1',0.5)['errore']:.0f} volte quello di "
             f"\\kw{{CrankNicolson 1}} allo stesso passo: nei transitori il primo ordine nel tempo non si recupera riducendo il passo, si cambia schema.")
righe.append("\\end{itemize}")
scrivi_tex("c06_risultati", "\n".join(righe))
