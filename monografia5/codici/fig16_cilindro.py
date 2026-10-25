"""
fig16_cilindro.py -- figure e risultati del caso c16_cilindro (Cap. 14): storia di C_L e C_D, spettro (FFT) e numero
di Strouhal, confronto con le correlazioni sperimentali (Williamson; Roshko), campo di vorticita' all'ultimo istante,
sonde di velocita' nella scia.
Uso: python3 fig16_cilindro.py   (dopo casi/c16_cilindro/Allrun)
"""
import os
import sys
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO
from foamcase import Caso
from foamlog import leggi_log
from figutil import caso, campo2d, it, scrivi_tex

C = caso("c16_cilindro")
RE = 100.0; D = 1.0; U0 = 1.0


def leggi_coefficienti():
    f = sorted(glob.glob(os.path.join(C, "postProcessing", "coefficienti", "*", "*.dat")))
    d = np.vstack([np.loadtxt(x, comments="#") for x in f])
    with open(f[0]) as fp:
        intest = [l for l in fp if l.startswith("#")][-1].strip("# \n").split()
    col = {n: i for i, n in enumerate(intest)}
    return d[:, 0], d[:, col["Cl"]], d[:, col["Cd"]]


def leggi_sonde():
    f = sorted(glob.glob(os.path.join(C, "postProcessing", "sonda", "*", "U")))
    righe = []
    for x in f:
        for l in open(x):
            if l.startswith("#"):
                continue
            righe.append([float(v) for v in l.replace("(", " ").replace(")", " ").split()])
    d = np.array(righe)
    return d[:, 0], d[:, 1:4], d[:, 4:7]


def strouhal(t, s, t0):
    """Frequenza dominante (FFT con finestra di Hann, parabola sui tre punti del picco) del segnale s per t > t0."""
    sel = t > t0; tt, ss = t[sel], s[sel] - s[sel].mean()
    dt = np.mean(np.diff(tt)); n = len(tt)
    w = np.hanning(n); S = np.abs(np.fft.rfft(ss * w)); f = np.fft.rfftfreq(n, dt)
    k = np.argmax(S[1:]) + 1
    # interpolazione parabolica del picco
    if 1 <= k < len(S) - 1:
        a, b, c_ = np.log(S[k - 1]), np.log(S[k]), np.log(S[k + 1]); delta = 0.5 * (a - c_) / (a - 2 * b + c_)
    else:
        delta = 0.0
    return (k + delta) * (f[1] - f[0]), f, S


t, Cl, Cd = leggi_coefficienti()
ts, U1, U2 = leggi_sonde()
T0 = 0.6 * t[-1]            # si scarta il transitorio (i primi 60% del tempo)
St_Cl, f, S = strouhal(t, Cl, T0)
St_v, fs, Ss = strouhal(ts, U2[:, 1], T0)
sel = t > T0
Cl_amp = 0.5 * (Cl[sel].max() - Cl[sel].min()); Cl_rms = np.sqrt(np.mean((Cl[sel] - Cl[sel].mean())**2))
Cd_med = Cd[sel].mean(); Cd_amp = 0.5 * (Cd[sel].max() - Cd[sel].min())
# Williamson (1988): St = 0,2175 - 5,1064/Re (regime laminare parallelo, 49 < Re < 180); Roshko (1954): St = 0,212 (1 - 21,2/Re)
St_W = 0.2175 - 5.1064 / RE; St_R = 0.212 * (1 - 21.2 / RE)
f_Cd = strouhal(t, Cd, T0)[0]

c = Caso(C); m = c.mesh
om = c.campo("vorticity"); U = c.campo("U"); p = c.campo("p")
log = leggi_log(os.path.join(C, "log.foamRun"))

fig, ax = plt.subplots(2, 2, figsize=(9.6, 5.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
ax[0, 0].plot(t, Cl, color=BLU, lw=0.8, label="$C_L$"); ax[0, 0].set_ylabel("$C_L$", color=BLU)
a2 = ax[0, 0].twinx(); a2.plot(t, Cd, color=ARANCIO, lw=0.8); a2.set_ylabel("$C_D$", color=ARANCIO); a2.set_ylim(1.0, 1.8)
ax[0, 0].axvspan(T0, t[-1], color=GRIGIO, alpha=0.12); ax[0, 0].set_xlabel("$tU/D$"); ax[0, 0].set_title("coefficienti di portanza e resistenza (ombreggiato: finestra dell'analisi)", fontsize=9)
ax[0, 1].semilogy(f[1:], S[1:] / S.max(), color=BLU, lw=1.0, label="$C_L$")
ax[0, 1].semilogy(fs[1:], Ss[1:] / Ss.max(), color=VERDE, lw=1.0, label="$v$ alla sonda $(6,\\,0{,}5)$")
ax[0, 1].axvline(St_W, color=GRIGIO, ls="--", lw=0.9, label=f"Williamson: $\\mathrm{{St}}={St_W:.4f}$")
ax[0, 1].set_xlim(0, 1.0); ax[0, 1].set_ylim(1e-4, 1.5); ax[0, 1].set_xlabel("$fD/U$"); ax[0, 1].set_ylabel("spettro normalizzato"); ax[0, 1].legend(fontsize=6.5)
ax[0, 1].set_title(f"FFT: $\\mathrm{{St}}={St_Cl:.4f}$ ($C_L$), ${St_v:.4f}$ (sonda)", fontsize=9)
cf = campo2d(ax[1, 0], m, np.clip(om[:, 2], -3, 3), livelli=np.linspace(-3, 3, 25), cmap="RdBu_r", maschera_lato=1.0)
ax[1, 0].add_patch(plt.Circle((0, 0), 0.5, color="0.3")); ax[1, 0].set_xlim(-2, 12); ax[1, 0].set_ylim(-3, 3)
ax[1, 0].plot([3, 6], [0.5, 0.5], "k+", ms=6); ax[1, 0].set_xlabel("$x/D$"); ax[1, 0].set_ylabel("$y/D$"); ax[1, 0].set_title(f"vorticità $\\omega_z D/U$ a $tU/D={t[-1]:.0f}$", fontsize=9)
fig.colorbar(cf, ax=ax[1, 0], shrink=0.8, pad=0.02)
sel_s = ts > t[-1] - 30
ax[1, 1].plot(ts[sel_s], U1[sel_s, 1], color=BLU, lw=0.9, label="$v$ in $(3,\\,0{,}5)$"); ax[1, 1].plot(ts[sel_s], U2[sel_s, 1], color=VERDE, lw=0.9, label="$v$ in $(6,\\,0{,}5)$")
ax[1, 1].set_xlabel("$tU/D$"); ax[1, 1].set_ylabel("$v/U$"); ax[1, 1].legend(fontsize=7); ax[1, 1].set_title("sonde nella scia (ultimi 30 $D/U$)", fontsize=9)
fig.tight_layout(); salva(fig, "c16_cilindro")

ris = dict(St_Cl=float(St_Cl), St_sonda=float(St_v), St_Williamson=St_W, St_Roshko=St_R, Cd_medio=float(Cd_med), Cd_ampiezza=float(Cd_amp),
           Cl_ampiezza=float(Cl_amp), Cl_rms=float(Cl_rms), f_Cd_su_f_Cl=float(f_Cd / St_Cl), passi=int(len(log["campi"]["p"]["t"])),
           tempo=float(log["exec"][-1, 1]), Co_max=float(log["Co"][:, 2].max()), celle=int(m.nCells))
json.dump(ris, open(os.path.join(C, "risultati.json"), "w"), indent=1)
righe = ["\\begin{itemize}"]
righe.append(f"\\item Mesh a O di {m.nCells} celle; {ris['passi']} passi di $\\Delta t=0{{,}}005\\,D/U$ in {it(ris['tempo']/60,0)}~min, $\\mathrm{{Co}}_{{\\max}}={it(ris['Co_max'],2)}$ (PISO con \\kw{{backward}}, due correttori).")
righe.append(f"\\item Il distacco si instaura spontaneamente (nessuna perturbazione imposta) dopo circa $tU/D\\approx{it(t[(t > 20) & (np.abs(Cl) > 0.05)][0],0)}$ (ampiezza $0{{,}}05$) e satura a un ciclo limite: "
             f"$C_L$ oscilla con ampiezza $\\pm{it(Cl_amp,3)}$ (valore efficace {it(Cl_rms,3)}), $C_D={it(Cd_med,3)}\\pm{it(Cd_amp,3)}$; la resistenza oscilla alla frequenza doppia "
             f"({it(ris['f_Cd_su_f_Cl'],2)} volte quella di $C_L$), perché ogni vortice, da sopra o da sotto, la fa crescere allo stesso modo.")
righe.append(f"\\item Numero di Strouhal dalla FFT di $C_L$: $\\mathrm{{St}}={it(St_Cl,4)}$; dalla componente $v$ alla sonda in $(6D,\\,0{{,}}5D)$: {it(St_v,4)}. "
             f"Correlazioni sperimentali a $\\Rey=100$: Williamson $0{{,}}2175-5{{,}}1064/\\Rey={it(St_W,4)}$, Roshko $0{{,}}212(1-21{{,}}2/\\Rey)={it(St_R,4)}$: "
             f"scarto {it(100*abs(St_Cl/St_W-1),1)}\\%. Valori numerici di riferimento in letteratura: $C_D=1{{,}}33$--$1{{,}}37$, $\\mathrm{{St}}=0{{,}}164$--$0{{,}}167$.")
righe.append(f"\\item La risoluzione in frequenza della FFT è $\\Delta f=1/T={it(1/(t[-1]-T0),4)}$ (finestra di {t[-1]-T0:.0f}~$D/U$, circa {int((t[-1]-T0)*St_Cl)} periodi): "
             f"l'interpolazione parabolica del picco porta l'incertezza su $\\mathrm{{St}}$ sotto il $\\pm1$\\%; con una finestra più corta il risultato sarebbe illusoriamente preciso solo sulla terza cifra.")
righe.append("\\end{itemize}")
scrivi_tex("c16_risultati", "\n".join(righe))
