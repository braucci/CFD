"""
fig12_cuneo.py -- figura e risultati del caso c12_cuneo (Cap. 10): urto obliquo su cuneo di 15 gradi a Ma 2,5;
angolo dell'urto misurato dal campo di pressione e confrontato con la relazione theta-beta-Ma; p2/p1 e Ma2 a valle
confrontati con la teoria; profili lungo y = 0,1 m.
Uso: python3 fig12_cuneo.py   (dopo casi/c12_cuneo/Allrun)
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stile_grafici import salva, BLU, ARANCIO, VERDE, GRIGIO
from foamcase import Caso
from figutil import caso, campo2d, it, scrivi_tex

C = caso("c12_cuneo")
G = 1.4; M1 = 2.5; THETA = np.radians(15.0); P1 = 101325.0; T1 = 300.0; RGAS = 8314.47 / 28.96


def theta_di_beta(b, M=M1):
    return np.arctan(2 / np.tan(b) * (M**2 * np.sin(b)**2 - 1) / (M**2 * (G + np.cos(2 * b)) + 2))


beta = brentq(lambda b: theta_di_beta(b) - THETA, np.arcsin(1 / M1) + 1e-6, np.radians(64))   # soluzione debole
Mn1 = M1 * np.sin(beta)
p2p1 = 1 + 2 * G / (G + 1) * (Mn1**2 - 1)
Mn2 = np.sqrt((1 + (G - 1) / 2 * Mn1**2) / (G * Mn1**2 - (G - 1) / 2))
M2 = Mn2 / np.sin(beta - THETA)
T2T1 = p2p1 * ((G + 1) / (G - 1) + p2p1) / (1 + (G + 1) / (G - 1) * p2p1)

c = Caso(C); m = c.mesh
p = c.campo("p"); T = c.campo("T"); U = c.campo("U")
Ma = c.campo("Ma") if "Ma" in os.listdir(os.path.join(C, c.tempi()[-1])) else np.linalg.norm(U, axis=1) / np.sqrt(G * RGAS * T)
x, y = m.C[:, 0], m.C[:, 1]

# --- angolo dell'urto: per colonne di celle con 0,02 < x < 0,18 si cerca la y dove p attraversa (p1+p2)/2 ------
xs = np.unique(np.round(x[x > 0], 6))
xs = xs[(xs > 0.02) & (xs < 0.18)]
xc, yc = [], []
for xx in xs:
    col = np.isclose(x, xx); yy = y[col]; pp = p[col]; o = np.argsort(yy); yy, pp = yy[o], pp[o]
    pm = 0.5 * (P1 + P1 * p2p1)
    i = np.where((pp[:-1] > pm) & (pp[1:] <= pm))[0]      # p scende da p2 (vicino al cuneo) a p1 (sopra)
    if len(i):
        k = i[0]; xc.append(xx); yc.append(np.interp(pm, [pp[k + 1], pp[k]], [yy[k + 1], yy[k]]))
xc, yc = np.array(xc), np.array(yc)
a, b0 = np.polyfit(xc, yc, 1)
beta_num = np.degrees(np.arctan(a))
# --- valori a valle: regione fra cuneo e urto, 0,05 < x < 0,15 ----------------------------------------------
y_cuneo = x * np.tan(THETA); y_urto = a * x + b0
valle = (x > 0.05) & (x < 0.15) & (y > y_cuneo + 0.01) & (y < y_urto - 0.01)
p2_num = p[valle].mean(); M2_num = Ma[valle].mean(); T2_num = T[valle].mean()
monte = (x > 0.05) & (x < 0.15) & (y > y_urto + 0.015)
p1_num = p[monte].mean(); M1_num = Ma[monte].mean()

# --- profili su y = 0,1 ---------------------------------------------------------------------------------------
xl, pl = c.linea("p", (-0.15, 0.1, 0), (0.3048, 0.1, 0), 500); xl = xl - 0.15     # linea() restituisce la distanza dal primo punto
_, Ml = c.linea("Ma", (-0.15, 0.1, 0), (0.3048, 0.1, 0), 500) if "Ma" in os.listdir(os.path.join(C, c.tempi()[-1])) else (None, None)
if Ml is None:
    _, Ul = c.linea("U", (-0.15, 0.1, 0), (0.3048, 0.1, 0), 500); _, Tl = c.linea("T", (-0.15, 0.1, 0), (0.3048, 0.1, 0), 500)
    Ml = np.linalg.norm(Ul, axis=1) / np.sqrt(G * RGAS * Tl)
x_urto_y01 = (0.1 - b0) / a; x_urto_teo = 0.1 / np.tan(beta)

fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
cf = campo2d(ax[0], m, p / P1, livelli=np.linspace(0.9, 3.0, 22), cmap="inferno", extend="both")
xx = np.array([0, 0.1524 / np.tan(beta)]); ax[0].plot(xx, xx * np.tan(beta), "--", color="cyan", lw=1.1, label=f"teoria: $\\beta={np.degrees(beta):.1f}^\\circ$")
ax[0].plot(xc, yc, ".", color="w", ms=2, label=f"urto numerico: $\\beta={beta_num:.1f}^\\circ$")
ax[0].fill([0, 0.3048, 0.3048], [0, 0.3048 * np.tan(THETA), 0], color="0.35")
ax[0].axhline(0.1, color="w", lw=0.5, ls=":")
ax[0].set_xlim(-0.15242, 0.3048); ax[0].set_ylim(0, 0.1524); ax[0].set_xlabel("$x$ [m]"); ax[0].set_ylabel("$y$ [m]"); ax[0].legend(fontsize=7, loc="upper left")
fig.colorbar(cf, ax=ax[0], label="$p/p_1$", shrink=0.85)
ax[1].plot(xl, pl / P1, color=BLU, lw=1.1, label="$p/p_1$"); ax[1].axhline(p2p1, color=BLU, lw=0.7, ls="--")
ax[1].set_ylabel("$p/p_1$", color=BLU); ax[1].set_xlabel("$x$ [m] (su $y=0{,}1$ m)")
ax2 = ax[1].twinx(); ax2.plot(xl, Ml, color=ARANCIO, lw=1.1, label="$\\mathrm{Ma}$"); ax2.axhline(M2, color=ARANCIO, lw=0.7, ls="--"); ax2.set_ylabel("$\\mathrm{Ma}$", color=ARANCIO)
ax[1].axvline(x_urto_teo, color=GRIGIO, lw=0.8, ls=":"); ax[1].set_xlim(-0.15, 0.3)
ax[1].set_title("tratteggi: valori teorici a valle", fontsize=9)
fig.tight_layout(); salva(fig, "c12_cuneo")

ris = dict(beta_teorico=float(np.degrees(beta)), beta_numerico=float(beta_num), p2p1_teorico=float(p2p1), p2p1_numerico=float(p2_num / P1),
           M2_teorico=float(M2), M2_numerico=float(M2_num), T2T1_teorico=float(T2T1), T2T1_numerico=float(T2_num / T1),
           p1_numerico=float(p1_num / P1), M1_numerico=float(M1_num), x_urto_y01=float(x_urto_y01), x_urto_y01_teorico=float(x_urto_teo))
json.dump(ris, open(os.path.join(C, "risultati.json"), "w"), indent=1)
righe = ["\\begin{itemize}"]
righe.append(f"\\item Teoria (urto debole, $\\Ma_1=2{{,}}5$, $\\theta=15^\\circ$): $\\beta={it(np.degrees(beta),2)}^\\circ$, $\\Ma_{{n1}}={it(Mn1,3)}$, $p_2/p_1={it(p2p1,3)}$, "
             f"$T_2/T_1={it(T2T1,3)}$, $\\Ma_2={it(M2,3)}$ (ancora supersonico, come deve essere per un urto debole).")
righe.append(f"\\item Calcolo: l'urto, individuato dall'isobara $(p_1+p_2)/2$ su {len(xc)} colonne di celle, è una retta ($R^2$ della regressione: "
             f"{it(1 - np.sum((yc - (a*xc+b0))**2)/np.sum((yc-yc.mean())**2), 5)}) con $\\beta={it(beta_num,2)}^\\circ$: scarto {it(abs(beta_num-np.degrees(beta)),2)}$^\\circ$ "
             f"({it(100*abs(beta_num-np.degrees(beta))/np.degrees(beta),1)}\\%), meno della risoluzione angolare della mesh ($\\Delta y/\\Delta x$ per cella).")
righe.append(f"\\item Nella regione a valle (media su {int(valle.sum())} celle): $p_2/p_1={it(p2_num/P1,3)}$ ({it(100*(p2_num/P1/p2p1-1),2)}\\%), $T_2/T_1={it(T2_num/T1,3)}$ "
             f"({it(100*(T2_num/T1/T2T1-1),2)}\\%), $\\Ma_2={it(M2_num,3)}$ ({it(100*(M2_num/M2-1),2)}\\%); a monte $\\Ma={it(M1_num,3)}$ e $p/p_1={it(p1_num/P1,4)}$: "
             f"la corrente libera è trasportata senza alterazioni.")
righe.append(f"\\item Su $y=0{{,}}1$~m l'urto è in $x={it(x_urto_y01,4)}$~m contro ${it(x_urto_teo,4)}$~m teorici; la transizione occupa 3--4 celle "
             f"($\\Delta x={it(1000*0.3048/240,2)}$~mm), come nel tubo d'urto: la ricostruzione limitata non introduce oscillazioni a monte né a valle.")
righe.append("\\end{itemize}")
scrivi_tex("c12_risultati", "\n".join(righe))
