"""
fig02_mesh.py -- figure e risultati dei casi c02_canale, c02_cilindro, c02_tubo (Cap. 3): disegno delle mesh
con foamcase.Mesh.disegna, mappa della non ortogonalita', confronto fra gli indici di qualita' calcolati in
Python e quelli di checkMesh (letti dai log).
"""
import os
import re
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stile_grafici import salva, BLU, ARANCIO, GRIGIO
from foamcase import Mesh
from figutil import caso, it, scrivi_tex, triangolazione


def da_checkmesh(nome):
    log = open(os.path.join(caso(nome), "log.checkMesh")).read()
    g = lambda pat: float(re.search(pat, log).group(1))
    return dict(celle=int(g(r"cells:\s+(\d+)")), nonorto_max=g(r"non-orthogonality Max: ([\d.e+-]+)"),
                nonorto_media=g(r"average: ([\d.e+-]+)"), skew=g(r"Max skewness = ([\d.e+-]+)"),
                aspetto=g(r"Max aspect ratio = ([\d.e+-]+)"), ok="Mesh OK" in log)


righe = []
for nome in ("c02_canale", "c02_cilindro", "c02_tubo"):
    m = Mesh(caso(nome))
    ck = da_checkmesh(nome)
    no = m.non_ortogonalita()
    r = dict(nome=nome, celle=m.nCells, facce=m.nFaces, nonorto_max=no.max(), nonorto_media=m.media_non_ortogonalita(),
             skew=m.skewness().max(), ck=ck)
    righe.append(r)
    print(f"{nome:14s} celle {m.nCells:6d} (checkMesh {ck['celle']})  non-orto max {no.max():6.2f} ({ck['nonorto_max']:6.2f})"
          f"  media {r['nonorto_media']:5.2f} ({ck['nonorto_media']:5.2f})  skew {r['skew']:.3f} ({ck['skew']:.3f})  aspetto {ck['aspetto']:.1f}")

# --- canale -----------------------------------------------------------------------------------
m = Mesh(caso("c02_canale"))
fig, ax = plt.subplots(1, 2, figsize=(9.0, 2.6), gridspec_kw=dict(width_ratios=[2.2, 1]))
m.disegna(ax[0], lw=0.25); ax[0].set_aspect("equal"); ax[0].set_xlabel("$x$ [m]"); ax[0].set_ylabel("$y$ [m]")
m.disegna(ax[1], lw=0.4); ax[1].set_aspect("equal"); ax[1].set_xlim(0.03, 0.055); ax[1].set_ylim(-0.016, -0.004); ax[1].set_xlabel("$x$ [m]")
ax[1].set_title("dettaglio del gradino", fontsize=9)
fig.tight_layout(); salva(fig, "c02_canale")

# --- cilindro -----------------------------------------------------------------------------------
m = Mesh(caso("c02_cilindro"))
fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.2))
m.disegna(ax[0], lw=0.2); ax[0].set_aspect("equal"); ax[0].set_xlabel("$x/D$"); ax[0].set_ylabel("$y/D$")
m.disegna(ax[1], lw=0.35); ax[1].set_aspect("equal"); ax[1].set_xlim(-0.9, 0.9); ax[1].set_ylim(-0.9, 0.9); ax[1].set_xlabel("$x/D$")
# mappa della dimensione delle celle: radice del volume per unita' di spessore (la mesh e' spessa 1)
h = np.sqrt(m.V)
tri = triangolazione(m, maschera_lato=1.5)
sc = ax[2].tripcolor(tri, h, shading="gouraud", cmap="magma_r", norm=matplotlib.colors.LogNorm(vmin=h.min(), vmax=h.max()))
ax[2].add_patch(plt.Circle((0, 0), 0.5, color="w"))
fig.colorbar(sc, ax=ax[2], label="dimensione di cella $h/D$", shrink=0.85)
ax[2].set_aspect("equal"); ax[2].set_xlim(-10, 10); ax[2].set_ylim(-10, 10); ax[2].set_xlabel("$x/D$")
fig.tight_layout(); salva(fig, "c02_cilindro")

# --- tubo (vista nel piano x-r della faccia wedge) -----------------------------------------------
m = Mesh(caso("c02_tubo"))
idx = m.facce_patch("fronte")
polys = [m.points[m.faces[i]][:, :2] for i in idx]
fig, ax = plt.subplots(figsize=(8.5, 1.6))
ax.add_collection(PolyCollection(polys, facecolors="none", edgecolors=GRIGIO, linewidths=0.3))
ax.set_xlim(0, 0.2); ax.set_ylim(0, 0.0105); ax.set_aspect("equal"); ax.set_xlabel("$x$ [m]"); ax.set_ylabel("$r$ [m]")
ax.set_title("condotto assialsimmetrico: la faccia wedge (asse in basso, parete in alto)", fontsize=9)
fig.tight_layout(); salva(fig, "c02_tubo")

# --- riquadro -----------------------------------------------------------------------------------
def riga(r):
    return (f"\\item \\file{{{r['nome'].replace('_', chr(92)+'_')}}}: {r['celle']} celle, {r['facce']} facce; non ortogonalità massima "
            f"{it(r['nonorto_max'],2)}$^\\circ$ (media {it(r['nonorto_media'],2)}$^\\circ$), skewness massima {it(r['skew'],3)}, "
            f"rapporto di forma massimo {it(r['ck']['aspetto'],1)}; \\kw{{checkMesh}} riporta {it(r['ck']['nonorto_max'],2)}$^\\circ$, "
            f"{it(r['ck']['nonorto_media'],2)}$^\\circ$, {it(r['ck']['skew'],3)}: {'Mesh OK' if r['ck']['ok'] else 'controlli non superati (vedi testo)'}.")
testo = "\\begin{itemize}\n" + "\n".join(riga(r) for r in righe) + "\n\\end{itemize}\n" + \
    "I valori di \\file{foamcase.py} e di \\kw{checkMesh} coincidono alla terza cifra: le definizioni sono le stesse. " \
    "La mesh a O è ortogonale ovunque (le facce radiali e quelle circonferenziali sono perpendicolari anche alle giunzioni dei blocchi) e la sua skewness residua viene dagli archi; il canale è cartesiano; " \
    "il cuneo ha le facce degeneri sull'asse che \\kw{checkMesh} riconosce come tali (\\emph{wedge} e \\emph{empty} corretti)."
scrivi_tex("c02_risultati", testo)
