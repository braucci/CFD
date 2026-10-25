"""
fig08_cavita.py -- figure e risultati del caso c08_cavita_re100 (Cap. 8): confronto con Ghia et al., linee di
corrente e pressione, residui per i diversi rilassamenti e SIMPLEC, effetto degli schemi convettivi su 32 x 32.
Uso: python3 fig08_cavita.py   (dopo python3 genera.py tutto)
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, VIOLA, GRIGIO, PALETTE
from foamcase import Caso
from foamlog import leggi_log
from figutil import caso, campo2d, linee_corrente, it, scrivi_tex
sys.path.insert(0, caso("c08_cavita_re100"))
from genera import GHIA_X, GHIA_Y, GHIA_U, GHIA_V   # noqa: E402

C = caso("c08_cavita_re100")
R = json.load(open(os.path.join(C, "risultati.json")))
B = np.load(os.path.join(C, "profili_base.npz")); PI = np.load(os.path.join(C, "profili_piso.npz"))

# --- Ghia + campi -------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(9.8, 3.3), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
ax[0].plot(B["u"], B["y"], "-", color=BLU, lw=1.2, label="SIMPLE")
ax[0].plot(PI["u"], PI["y"], "--", color=ARANCIO, lw=1.2, label="PISO, $t=20$ s")
ax[0].plot(GHIA_U, GHIA_Y, "o", color="k", ms=3.5, mfc="none", label="Ghia et al.")
ax[0].set_xlabel("$u/U$ su $x=L/2$"); ax[0].set_ylabel("$y/L$"); ax[0].legend(fontsize=7)
ax[1].plot(B["x"], B["v"], "-", color=BLU, lw=1.2, label="SIMPLE")
ax[1].plot(PI["x"], PI["v"], "--", color=ARANCIO, lw=1.2, label="PISO, $t=20$ s")
ax[1].plot(GHIA_X, GHIA_V, "o", color="k", ms=3.5, mfc="none", label="Ghia et al.")
ax[1].set_xlabel("$x/L$"); ax[1].set_ylabel("$v/U$ su $y=L/2$"); ax[1].legend(fontsize=7)
c = Caso(C); m = c.mesh
U = c.campo("U"); p = c.campo("p")
cf = campo2d(ax[2], m, p, livelli=np.linspace(np.percentile(p, 0.5), np.percentile(p, 99.5), 21), cmap="RdBu_r", extend="both")
linee_corrente(ax[2], m, U, n=160, densita=1.4, colore="k", lw=0.45)
fig.colorbar(cf, ax=ax[2], label="$p$ [m$^2$/s$^2$]", shrink=0.85)
ax[2].set_xlim(0, 1); ax[2].set_ylim(0, 1); ax[2].set_xlabel("$x/L$"); ax[2].set_title("linee di corrente e pressione", fontsize=9)
fig.tight_layout(); salva(fig, "c08_ghia")

# --- convergenza -----------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.3), gridspec_kw=dict(width_ratios=[1.35, 1]))
for j, r in enumerate(R["rilassamento"]):
    cons = "yes" if r["SIMPLEC"] else "no"
    d = leggi_log(os.path.join(C, f"log.foamRun_rilassamento_{r['alpha_U']}_{r['alpha_p']}_{cons}"))
    et = f"SIMPLEC $\\alpha_U={r['alpha_U']}$" if r["SIMPLEC"] else f"$\\alpha_U={r['alpha_U']}$, $\\alpha_p={r['alpha_p']}$"
    ax[0].semilogy(d["campi"]["p"]["t"], d["campi"]["p"]["res0"], color=PALETTE[j % len(PALETTE)], lw=1.0,
                   ls="-" if not r["SIMPLEC"] else "--", label=et + (" (non converge)" if not r["convergenza"] else ""))
ax[0].axhline(1e-6, color="k", lw=0.5, ls=":"); ax[0].set_xlabel("iterazione SIMPLE"); ax[0].set_ylabel("residuo iniziale di $p$"); ax[0].legend(fontsize=6)
for nome, col in (("upwind", ROSSO), ("linear", VERDE), ("linearUpwind", BLU)):
    d = np.load(os.path.join(C, f"profili_32_{nome}.npz"))
    ax[1].plot(d["u"], d["y"], "-", color=col, lw=1.1, label=f"{nome}, $32\\times32$")
ax[1].plot(B["u"], B["y"], ":", color=GRIGIO, lw=1.0, label="linearUpwind, $128\\times128$")
ax[1].plot(GHIA_U, GHIA_Y, "o", color="k", ms=3.5, mfc="none", label="Ghia et al.")
ax[1].set_xlabel("$u/U$ su $x=L/2$"); ax[1].set_ylabel("$y/L$"); ax[1].legend(fontsize=6.5)
fig.tight_layout(); salva(fig, "c08_convergenza")

# --- riquadro -------------------------------------------------------------------------------------------
b = R["base"]; ps = R["piso"]; s32 = R["schemi_32"]
ril = R["rilassamento"]
def trova(aU, ap, c_): return next(r for r in ril if r["alpha_U"] == aU and r["alpha_p"] == ap and r["SIMPLEC"] == c_)
righe = ["\\begin{itemize}"]
righe.append(f"\\item Caso base ($\\alpha_U=0{{,}}7$, $\\alpha_p=0{{,}}3$, \\kw{{linearUpwind}}, $128\\times128$): \\kw{{residualControl}} raggiunto dopo {b['iterazioni']} iterazioni in {it(b['tempo'],1)}~s; "
             f"scarto massimo dai dati di Ghia et al. {it(b['err_u'],4)} su $u$ e {it(b['err_v'],4)} su $v$ (i dati di riferimento sono su $129\\times129$ nodi).")
righe.append("\\item Rilassamento (iterazioni fino a $10^{-6}$): " + "; ".join(
    (f"SIMPLEC $\\alpha_U={it(r['alpha_U'],2)}$" if r["SIMPLEC"] else f"$({it(r['alpha_U'],1)},\\,{it(r['alpha_p'],1)})$") + f" $\\to$ " + (f"{r['iterazioni']}" if r["convergenza"] else f"non converge in {r['iterazioni']}")
    for r in ril) + ". " + f"Il numero di iterazioni è governato da $\\alpha_U$: con $\\alpha_U=0{{,}}7$ le coppie $(0{{,}}7,\\,0{{,}}3)$ e $(0{{,}}7,\\,0{{,}}5)$ convergono nello stesso numero di iterazioni "
    f"({trova(0.7,0.3,False)['iterazioni']} e {trova(0.7,0.5,False)['iterazioni']}), perché la velocità asintotica di convergenza è quella del modo più lento, smorzato dal rilassamento della quantità di moto; "
    f"$\\alpha_p$ decide invece la stabilità (con $\\alpha_p=0{{,}}9$ il calcolo diverge nelle prime iterazioni). Portare $\\alpha_U$ a $0{{,}}9$ riduce le iterazioni a {trova(0.9,0.1,False)['iterazioni']}; "
    f"SIMPLEC con $\\alpha_U=0{{,}}95$ e $\\alpha_p=1$ ne richiede {trova(0.95,1.0,True)['iterazioni']}, {it(b['iterazioni']/max(trova(0.95,1.0,True)['iterazioni'],1),1)} volte meno del caso base: "
    "su questo problema, laminare e su mesh ortogonale, è la scelta migliore.")
righe.append(f"\\item PISO (Euler, $\\Delta t=0{{,}}004$~s, $\\mathrm{{Co}}_{{\\max}}={it(ps['Co_max'],2)}$): {ps['passi']} passi in {it(ps['tempo_calcolo'],0)}~s, cioè {it(ps['tempo_calcolo']/b['tempo'],1)} volte il tempo di SIMPLE; "
             f"a $t=20$~s lo scarto dal profilo stazionario di SIMPLE è {it(ps['storia'][-1]['scarto_da_simple'],1,True)} (a $t=5$~s era {it(ps['storia'][0]['scarto_da_simple'],1,True)}): lo stato stazionario è lo stesso, raggiunto per via transitoria.")
righe.append(f"\\item Schemi su $32\\times32$ (errore massimo su $u$ contro Ghia): \\kw{{upwind}} {it(s32['upwind']['err_u'],4)}, \\kw{{linear}} {it(s32['linear']['err_u'],4)}, \\kw{{linearUpwind}} {it(s32['linearUpwind']['err_u'],4)}; "
             f"la mesh grossa con \\kw{{upwind}} sottostima la velocità di ritorno al fondo del {100*(1-abs(min(np.load(os.path.join(C,'profili_32_upwind.npz'))['u']))/abs(min(B['u']))):.0f}\\%.")
righe.append("\\end{itemize}")
scrivi_tex("c08_risultati", "\n".join(righe))
