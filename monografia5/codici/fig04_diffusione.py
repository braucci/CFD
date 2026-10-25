"""
fig04_diffusione.py -- figure e risultati del caso c04_diffusione (Cap. 4): errore massimo in funzione di h per
le mesh (quadrato, parallelogramma a 45 gradi, quadrato con punti perturbati a caso) con i quattro schemi del
laplaciano; errore del gradiente con tre gradSchemes; campo T sulla mesh perturbata lasciata nel caso da genera.py.
Uso: python3 fig04_diffusione.py   (dopo casi/c04_diffusione/genera.py)
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO
from foamcase import Caso
from figutil import caso, campo2d, it, scrivi_tex

C = caso("c04_diffusione")
R = json.load(open(os.path.join(C, "risultati.json")))
D = R["diffusione"]; G = R["gradiente"]
COL = {"orthogonal": GRIGIO, "uncorrected": ROSSO, "corrected": BLU, "limited 0.5": VERDE}
MRK = {"orthogonal": "s", "uncorrected": "^", "corrected": "o", "limited 0.5": "d"}
TIT = {"quadrato": "mesh cartesiana", "S=0.5": "parallelogramma, $\\theta=26{,}6^\\circ$", "S=1": "parallelogramma, $\\theta=45^\\circ$", "perturbata": "mesh perturbata a caso"}


def serie(lista, **filtro):
    r = [d for d in lista if all(d[k] == v for k, v in filtro.items())]
    r.sort(key=lambda d: d["h"])
    return np.array([d["h"] for d in r]), np.array([d["errore"] for d in r])


def ordine(h, e):
    """Ordine di convergenza osservato (pendenza del fit in scala logaritmica)."""
    return np.polyfit(np.log(h), np.log(e), 1)[0]


def info(mesh_, N):
    return next(d for d in D if d["mesh"] == mesh_ and d["N"] == N)


# --- diffusione -----------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(9.8, 3.1), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
h0, e0 = serie(D, mesh="quadrato", schema="corrected")
for k, mesh_ in enumerate(("S=1", "perturbata")):
    for sch in COL:
        h, e = serie(D, mesh=mesh_, schema=sch)
        ax[k].loglog(h, e, MRK[sch] + "-", color=COL[sch], ms=4, lw=1.0, label=f"{sch} ($p={ordine(h, e):.2f}$)")
    ax[k].loglog(h0, e0, "k:", lw=0.9, label=f"cartesiana ($p={ordine(h0, e0):.2f}$)")
    hh = np.array([h0.min(), h0.max()])
    ax[k].loglog(hh, e0[-1] * (hh / hh[-1]) ** 2, color=GRIGIO, lw=0.7, alpha=0.6)
    ax[k].text(hh[0] * 1.1, e0[-1] * (hh[0] / hh[-1]) ** 2 * 1.4, "$h^2$", fontsize=8, color=GRIGIO)
    ax[k].set_title(TIT[mesh_] + (f" ($\\theta_{{\\max}}={info(mesh_, 40)['nonorto']:.0f}^\\circ$)" if mesh_ == "perturbata" else ""), fontsize=9)
    ax[k].set_xlabel("$h$"); ax[k].legend(fontsize=6.5, loc="lower right")
ax[0].set_ylabel("errore massimo $|T-T_{\\mathrm{es}}|$")

c = Caso(C); m = c.mesh; T = c.campo("T")
cf = campo2d(ax[2], m, T, livelli=np.linspace(0, 1, 11), cmap="viridis", linee=True, colore_linee="w", lw=0.4)
m.disegna(ax[2], lw=0.15, colore="0.6")
fig.colorbar(cf, ax=ax[2], label="$T$", shrink=0.85)
ax[2].set_xlabel("$x$"); ax[2].set_ylabel("$y$"); ax[2].set_title("$T$ sulla mesh perturbata, $N=40$", fontsize=9)
fig.tight_layout(); salva(fig, "c04_diffusione")

# --- gradiente -------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(9.8, 3.0), sharey=True)
colg = {"Gauss linear": BLU, "leastSquares": ARANCIO, "Gauss pointLinear": VERDE}
ord_g = {}
for k, mesh_ in enumerate(("quadrato", "S=1", "perturbata")):
    for gs in colg:
        h, e = serie(G, mesh=mesh_, schema=gs)
        ord_g[(mesh_, gs)] = ordine(h, e)
        ax[k].loglog(h, e, "o-", color=colg[gs], ms=4, lw=1.0, label=f"{gs} ($p={ord_g[(mesh_, gs)]:.2f}$)")
    ax[k].set_title(TIT[mesh_], fontsize=9); ax[k].set_xlabel("$h$"); ax[k].legend(fontsize=6.5)
ax[0].set_ylabel("errore massimo $|\\nabla T-\\nabla T_{\\mathrm{es}}|$")
fig.tight_layout(); salva(fig, "c04_gradiente")

# --- riquadro ---------------------------------------------------------------------------------------------
def err(mesh_, N, sch):
    return next(d["errore"] for d in D if d["mesh"] == mesh_ and d["N"] == N and d["schema"] == sch)

ip = info("perturbata", 40)
righe = ["\\begin{itemize}"]
righe.append(f"\\item Mesh cartesiana: con $N=10,20,40,80$ l'errore massimo è {it(err('quadrato',10,'corrected'),2,True)}, {it(err('quadrato',20,'corrected'),2,True)}, "
             f"{it(err('quadrato',40,'corrected'),2,True)}, {it(err('quadrato',80,'corrected'),2,True)}: ordine osservato ${it(ordine(h0, e0),2)}$; "
             f"i quattro schemi del laplaciano coincidono perché il termine di correzione è nullo.")
righe.append(f"\\item Parallelogramma a $45^\\circ$ ($\\theta=45^\\circ$ su tutte le facce interne, skewness interna nulla), $N=40$: "
             f"\\kw{{orthogonal}} {it(err('S=1',40,'orthogonal'),2,True)}, \\kw{{uncorrected}} {it(err('S=1',40,'uncorrected'),2,True)}, "
             f"\\kw{{corrected}} {it(err('S=1',40,'corrected'),2,True)}, \\kw{{limited 0.5}} {it(err('S=1',40,'limited 0.5'),2,True)}. "
             f"Senza correzione l'errore non diminuisce con $h$ (ordine ${it(ordine(*serie(D,mesh='S=1',schema='uncorrected')),2)}$): lo schema converge "
             f"alla soluzione di un'equazione diversa; con la correzione l'errore scende con ordine ${it(ordine(*serie(D,mesh='S=1',schema='corrected')),2)}$, "
             f"limitato dal trattamento del primo ordine delle facce di contorno, tutte oblique.")
righe.append(f"\\item Mesh perturbata ($\\theta_{{\\max}}={ip['nonorto']:.0f}^\\circ$, skewness massima {it(ip['skew'],2)}, $N=40$): \\kw{{uncorrected}} {it(err('perturbata',40,'uncorrected'),2,True)}, "
             f"\\kw{{corrected}} {it(err('perturbata',40,'corrected'),2,True)} (ordine ${it(ordine(*serie(D,mesh='perturbata',schema='corrected')),2)}$), "
             f"\\kw{{limited 0.5}} {it(err('perturbata',40,'limited 0.5'),2,True)}: la correzione dimezza l'errore e recupera la convergenza, il limitatore sta a metà.")
righe.append(f"\\item Gradiente di un campo esatto (celle interne). Mesh cartesiana: \\kw{{Gauss linear}} e \\kw{{leastSquares}} coincidono (ordine ${it(ord_g[('quadrato','Gauss linear')],2)}$), "
             f"\\kw{{Gauss pointLinear}} è anomalo e converge con ordine ${it(ord_g[('quadrato','Gauss pointLinear')],1)}$ (la sua correzione esplicita cancella il termine d'errore dominante sulle griglie uniformi). "
             f"Parallelogramma: ancora coincidenti, ordine ${it(ord_g[('S=1','Gauss linear')],2)}$, perché la retta $CF$ passa per il centro della faccia. "
             f"Mesh perturbata: \\kw{{Gauss linear}} \\emph{{non converge}} (errore {it(next(d['errore'] for d in G if d['mesh']=='perturbata' and d['N']==80 and d['schema']=='Gauss linear'),2)} a $N=80$, ordine ${it(ord_g[('perturbata','Gauss linear')],2)}$), "
             f"\\kw{{leastSquares}} converge (ordine ${it(ord_g[('perturbata','leastSquares')],2)}$, errore {it(next(d['errore'] for d in G if d['mesh']=='perturbata' and d['N']==80 and d['schema']=='leastSquares'),3)} a $N=80$): "
             f"è la ragione per cui sulle mesh non strutturate si preferiscono i minimi quadrati.")
righe.append("\\end{itemize}")
scrivi_tex("c04_risultati", "\n".join(righe))
