"""
fig10_gradino.py -- figura e risultati del caso c10_gradino (Cap. 9): C_f sul fondo per tre modelli RANS con il
riattacco sperimentale di Driver & Seegmiller (x_r/h = 6,26), linee di corrente e |U| con k-omega SST.
Uso: python3 fig10_gradino.py   (dopo casi/c10_gradino/genera.py)
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stile_grafici import salva, BLU, ARANCIO, VERDE, GRIGIO
from foamcase import Caso
from figutil import caso, campo2d, linee_corrente, it, scrivi_tex

C = caso("c10_gradino")
R = json.load(open(os.path.join(C, "risultati.json")))
P = np.load(os.path.join(C, "profili.npz"))
H = 0.0127; U0 = 44.2; XR_ESP = 6.26
COL = {"kEpsilon": VERDE, "kOmegaSST": BLU, "SpalartAllmaras": ARANCIO}
ETI = {"kEpsilon": "$k$–$\\varepsilon$", "kOmegaSST": "$k$–$\\omega$ SST", "SpalartAllmaras": "Spalart–Allmaras"}

fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.2), gridspec_kw=dict(width_ratios=[1, 1.5]))
for m in COL:
    ax[0].plot(P[f"{m}|x_h"], 1e3 * P[f"{m}|cf"], "-", color=COL[m], lw=1.1, label=f"{ETI[m]} ($x_r/h={R[m]['x_r_h']:.2f}$)")
ax[0].axhline(0, color="k", lw=0.5); ax[0].axvline(XR_ESP, color=GRIGIO, lw=1.0, ls="--")
ax[0].text(XR_ESP + 0.3, ax[0].get_ylim()[0] * 0.9 if ax[0].get_ylim()[0] < 0 else -2.5, "riattacco sperimentale\n$x_r/h=6{,}26$", fontsize=7, color=GRIGIO)
ax[0].set_xlim(0, 30); ax[0].set_xlabel("$x/h$"); ax[0].set_ylabel("$C_f\\times10^3$ sul fondo"); ax[0].legend(fontsize=6.5, loc="lower right")

c = Caso(C); m = c.mesh; U = c.campo("U")
magU = np.linalg.norm(U, axis=1) / U0
x_h = m.C[:, 0] / H; y_h = m.C[:, 1] / H
sel = (x_h > -4) & (x_h < 16)
import matplotlib.tri as mtri
tri = mtri.Triangulation(x_h[sel], y_h[sel])
# si eliminano i triangoli che attraversano il gradino (regione x<0, y<0)
cx = x_h[sel][tri.triangles].mean(axis=1); cy = y_h[sel][tri.triangles].mean(axis=1)
tri.set_mask((cx < 0) & (cy < 0))
u_U = U[:, 0] / U0
cf = ax[1].tricontourf(tri, u_U[sel], levels=np.linspace(-0.3, 1.2, 16), cmap="RdYlBu_r", extend="both")
ax[1].tricontour(tri, u_U[sel], levels=[0.0], colors="k", linewidths=0.9)          # u = 0: il confine della bolla di ricircolo
ax[1].tricontour(tri, u_U[sel], levels=[0.2, 0.4, 0.6, 0.8], colors="k", linewidths=0.3, alpha=0.6)
ax[1].fill([-4, 0, 0, -4], [-1, -1, 0, 0], color="0.3")
ax[1].set_xlim(-4, 16); ax[1].set_ylim(-1, 4); ax[1].set_aspect("equal"); ax[1].set_xlabel("$x/h$"); ax[1].set_ylabel("$y/h$")
ax[1].plot(R["kOmegaSST"]["x_r_h"], -1, "^", color=ARANCIO, ms=6, clip_on=False)
fig.colorbar(cf, ax=ax[1], label="$u/U_0$", shrink=0.8)
ax[1].set_title("$k$–$\\omega$ SST: $u/U_0$; linea nera: $u=0$ (bolla di ricircolo)", fontsize=9)
fig.tight_layout(); salva(fig, "c10_gradino")

righe = ["\\begin{center}\\small\\begin{tabular}{lrrrrr}\\toprule modello & $x_r/h$ & scarto & iterazioni & $y^+$ fondo & $C_{f,\\min}\\times10^3$\\\\\\midrule"]
for m_ in COL:
    r = R[m_]
    righe.append(f"{ETI[m_]} & {it(r['x_r_h'],2)} & {it(100*(r['x_r_h']-XR_ESP)/XR_ESP,0)}\\% & {r['iterazioni']} & {it(r['yplus_fondo_medio'],0)} & {it(1e3*r['cf_min'],2)}\\\\")
righe.append(f"esperimento & {it(XR_ESP,2)} & -- & -- & -- & --\\\\")
righe.append("\\bottomrule\\end{tabular}\\end{center}")
righe.append("\\begin{itemize}")
righe.append(f"\\item Il $k$--$\\varepsilon$ riattacca troppo presto ($x_r/h={it(R['kEpsilon']['x_r_h'],2)}$, scarto del {it(100*abs(R['kEpsilon']['x_r_h']-XR_ESP)/XR_ESP,0)}\\%): "
             f"la sua viscosità turbolenta, troppo alta dove il gradiente di pressione è avverso, fa diffondere lo strato di taglio e chiude la bolla prima. "
             f"È il difetto documentato del modello che ha motivato le correzioni di Menter.")
righe.append(f"\\item SST e Spalart--Allmaras danno $x_r/h={it(R['kOmegaSST']['x_r_h'],2)}$ e $x_r/h={it(R['SpalartAllmaras']['x_r_h'],2)}$: "
             f"entro il {it(100*max(abs(R[m]['x_r_h']-XR_ESP) for m in ('kOmegaSST','SpalartAllmaras'))/XR_ESP,0)}\\% dell'esperimento, che è quanto ci si può aspettare da un calcolo RANS 2D "
             f"con funzioni di parete ($y^+\\approx{it(R['kOmegaSST']['yplus_fondo_medio'],0)}$).")
righe.append(f"\\item Il minimo di $C_f$ nel ricircolo è di $-{it(1e3*abs(R['kOmegaSST']['cf_min']),2)}\\times10^{{-3}}$ per SST: lo sforzo è negativo perché il fluido a parete scorre verso il gradino; "
             f"il riattacco è il punto dove $C_f$ torna positivo, la definizione usata da \\file{{genera.py}}.")
# confronto fra lo sforzo di parete di foamcase e la function object wallShearStress (campi SST, ultimo istante)
xf, tau = c.sforzo_parete("fondo", 1.5e-5); wss = c.valore_patch("wallShearStress", "fondo")
scarto = np.max(np.abs(np.abs(wss[:, 0]) - np.abs(tau[:, 0]))) / np.max(np.abs(tau[:, 0]))
st = R["kOmegaSST"]["storia_xr"]
righe.append(f"\\item Convergenza: i residui si assestano intorno a $10^{{-4}}$ ($U$) e $2\\times10^{{-3}}$ ($p$) dopo circa 1000 iterazioni senza scendere oltre, ma il punto di riattacco "
             f"è fermo: per SST vale {it(st[0][1],3)}, {it(st[1][1],3)}, {it(st[2][1],3)}, {it(st[3][1],3)} alle iterazioni 1000, 2000, 3000, 4000. "
             f"È un caso tipico di \\emph{{stallo del residuo}} con soluzione integrale convergente (Cap.~\\ref{{cap:tempo}}): si giudica sulle grandezze d'interesse, non sulla sola riga del registro.")
righe.append(f"\\item Lo sforzo di parete sul fondo calcolato da \\file{{foamcase.py}} e dalla function object \\kw{{wallShearStress}} differiscono al più di {it(100*scarto,2)}\\%.")
righe.append("\\end{itemize}")
scrivi_tex("c10_risultati", "\n".join(righe))
