"""
fig07_solutori.py -- figura e risultati per il Cap. 7 dal passo 'solutori' di casi/c08_cavita_re100/genera.py:
tempo di calcolo e iterazioni del solutore lineare della pressione per otto configurazioni, residuo iniziale di p
in funzione dell'iterazione SIMPLE per tre di esse.
Uso: python3 fig07_solutori.py   (dopo python3 genera.py solutori)
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stile_grafici import salva, BLU, ARANCIO, VERDE, GRIGIO, PALETTE
from foamlog import leggi_log
from figutil import caso, it, scrivi_tex

C = caso("c08_cavita_re100")
S = json.load(open(os.path.join(C, "risultati.json")))["solutori"]
nomi = [s["solutore"].replace(" relTol ", "\nrelTol ") for s in S]
tempi = np.array([s["tempo"] for s in S]); iters = np.array([s["iter_p_medie"] for s in S])

fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.4), gridspec_kw=dict(width_ratios=[1.5, 1]))
pos = np.arange(len(S))
b = ax[0].bar(pos - 0.2, tempi, 0.4, color=BLU, label="tempo per 100 iterazioni SIMPLE [s]")
ax2 = ax[0].twinx()
ax2.bar(pos + 0.2, iters, 0.4, color=ARANCIO, label="iterazioni medie del solutore di $p$")
ax[0].set_xticks(pos); ax[0].set_xticklabels(nomi, fontsize=6, rotation=35, ha="right")
ax[0].set_ylabel("tempo [s]", color=BLU); ax2.set_ylabel("iterazioni lineari per passo", color=ARANCIO)
ax[0].set_yscale("log"); ax2.set_yscale("log"); ax[0].set_ylim(1, 60); ax2.set_ylim(1, 2000)
for i, (t, n) in enumerate(zip(tempi, iters)):
    ax[0].text(i - 0.2, t, f"{t:.0f}", ha="center", va="bottom", fontsize=6, color=BLU)
    ax2.text(i + 0.2, n, f"{n:.0f}", ha="center", va="bottom", fontsize=6, color=ARANCIO)
h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax[0].legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")

scelti = [0, 3, 6]
for j, k in enumerate(scelti):
    d = leggi_log(os.path.join(C, f"log.foamRun_solutori_{k}"))
    ax[1].semilogy(d["campi"]["p"]["t"], d["campi"]["p"]["res0"], color=PALETTE[j], lw=1.0, label=S[k]["solutore"])
ax[1].set_xlabel("iterazione SIMPLE"); ax[1].set_ylabel("residuo iniziale di $p$"); ax[1].legend(fontsize=6.5)
fig.tight_layout(); salva(fig, "c07_solutori")

i_min = int(np.argmin(tempi)); i_max = int(np.argmax(tempi))
righe = ["\\begin{center}\\footnotesize\\begin{tabular}{lrrrr}\\toprule solutore di $p$ & tempo [s] & iter.\\ medie & iter.\\ totali & residuo $p$ (100)\\\\\\midrule"]
for s in S:
    righe.append(f"\\kw{{{s['solutore']}}} & {it(s['tempo'],1)} & {it(s['iter_p_medie'],1)} & {s['iter_p_totali']} & {it(s['res_p_100'],1,True)}\\\\")
righe.append("\\bottomrule\\end{tabular}\\end{center}")
righe.append("\\begin{itemize}")
righe.append(f"\\item Mesh $128\\times128$ (16\\,384 celle), 100 iterazioni SIMPLE identiche in tutto tranne il solutore di $p$ ($U$: \\kw{{smoothSolver}} con \\kw{{relTol 0.1}} ovunque). "
             f"La configurazione più rapida è \\kw{{{S[i_min]['solutore']}}} ({it(tempi[i_min],1)}~s), la più lenta \\kw{{{S[i_max]['solutore']}}} ({it(tempi[i_max],1)}~s): "
             f"un fattore {it(tempi[i_max]/tempi[i_min],1)} a parità di risultato.")
buoni = [s_ for s_ in S if s_["iter_p_medie"] < 999]
righe.append(f"\\item Il residuo iniziale di $p$ dopo 100 iterazioni è lo stesso per tutte le configurazioni che raggiungono \\kw{{relTol}}, a meno di un fattore "
             f"{it(max(s_['res_p_100'] for s_ in buoni)/min(s_['res_p_100'] for s_ in buoni),2)}: la velocità di convergenza esterna dipende dall'algoritmo SIMPLE e dal rilassamento, "
             f"non dal solutore lineare, finché \\kw{{relTol}} è abbastanza piccolo da ridurre l'errore di un ordine di grandezza per passo. "
             f"Il \\kw{{smoothSolver}} con Gauss--Seidel non ci arriva: esaurisce i 1000 \\kw{{maxIter}} a ogni passo senza raggiungere \\kw{{relTol 0.1}} (il residuo esterno resta a {it(S[6]['res_p_100'],1,True)}): "
             f"il rilassamento puntuale smorza solo gli errori ad alta frequenza, e l'equazione della pressione è dominata da quelli a bassa frequenza. È il motivo per cui esiste il multigriglia.")
righe.append(f"\\item Passare da \\kw{{relTol 0.1}} a \\kw{{relTol 0.01}} con GAMG costa {it(S[1]['tempo']/S[0]['tempo'],2)} volte il tempo "
             f"({S[1]['iter_p_medie']:.0f} contro {S[0]['iter_p_medie']:.0f} cicli per passo) e lascia il residuo esterno a {it(S[1]['res_p_100'],1,True)} contro {it(S[0]['res_p_100'],1,True)}: "
             f"il lavoro in più è sprecato, perché l'equazione della pressione cambia alla iterazione successiva.")
righe.append(f"\\item Il numero di iterazioni di \\kw{{PCG DIC}} ({S[3]['iter_p_medie']:.0f} per passo) cresce come $N$ (il numero di celle per lato), quello di GAMG ({S[0]['iter_p_medie']:.0f}) resta costante: "
             f"su mesh grandi il multigriglia vince sempre; qui, su $128^2$ celle, il vantaggio è già di {it(S[3]['tempo']/S[0]['tempo'],1)} volte (e di {it(S[4]['tempo']/S[1]['tempo'],1)} con \\kw{{relTol 0.01}}).")
righe.append("\\end{itemize}")
scrivi_tex("c07_risultati", "\n".join(righe))
