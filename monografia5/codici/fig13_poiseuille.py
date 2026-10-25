"""
fig13_poiseuille.py -- figura e risultati del caso c13_poiseuille (Cap. 11): profilo d'ingresso (codificato) e
d'uscita contro la parabola di Hagen-Poiseuille, pressione lungo l'asse, sforzo di parete; variante con ingresso
uniforme su un condotto lungo il doppio (L/D = 20) per la lunghezza di sviluppo.
Uso: python3 fig13_poiseuille.py   (dopo casi/c13_poiseuille/Allrun)
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
from stile_grafici import salva, BLU, ARANCIO, VERDE, GRIGIO
from foamcase import Caso
from figutil import caso, it, scrivi_tex

C = caso("c13_poiseuille")
NU = 1e-5; UM = 0.1; R = 0.01; L = 0.2; RE = UM * 2 * R / NU
DPDX = -8 * NU * UM / R**2           # pressione cinematica
TAU = 4 * NU * UM / R


def variante_uniforme():
    dest = C + "_uniforme"
    if not os.path.isdir(dest):
        shutil.copytree(C, dest, ignore=shutil.ignore_patterns("[1-9]*", "log.*", "postProcessing", "dynamicCode"))
        p = os.path.join(dest, "system", "blockMeshDict"); s = open(p).read()
        s = re.sub(r"\nL\s+20;", "\nL   40;", s); s = s.replace("(100 20 1)", "(200 20 1)"); open(p, "w").write(s)
        p = os.path.join(dest, "0", "U"); s = open(p).read()
        i = s.index("\n    ingresso") + 1; j = s.index("\n    uscita") + 1
        s = s[:i] + "ingresso    { type fixedValue; value uniform (0.1 0 0); }\n    " + s[j:]
        open(p, "w").write(s)
        subprocess.run("blockMesh > log.blockMesh && foamRun > log.foamRun", cwd=dest, shell=True, check=True)
    return Caso(dest)


c = Caso(C); m = c.mesh
z0 = 0.0
# profili radiali (y e' la coordinata radiale nel piano del cuneo, z ~ 0)
r_in, U_in = c.linea("U", (0.0005, 0, z0), (0.0005, R, z0), 20)
r_out, U_out = c.linea("U", (L - 0.0005, 0, z0), (L - 0.0005, R, z0), 20)
U_in_patch = c.valore_patch("U", "ingresso"); Cf_in = m.Cf[m.facce_patch("ingresso")]; r_patch = np.hypot(Cf_in[:, 1], Cf_in[:, 2])
rr = np.linspace(0, R, 100); u_es = 2 * UM * (1 - (rr / R)**2)
x_ax, p_ax = c.linea("p", (0, 0.0003, z0), (L, 0.0003, z0), 100)
xw, tau = c.sforzo_parete("parete", NU)
o = np.argsort(xw[:, 0]); xw, tau = xw[o], tau[o]
Fp, Fv = c.forze("parete", NU)
area = 2 * np.pi * R * L * (4.0 / 360.0)          # il cuneo copre 4 gradi
tau_medio_forza = Fv[0] / area                    # forza del fluido sulla parete, diretta come il moto

cu = variante_uniforme(); mu_ = cu.mesh
x_u, p_u = cu.linea("p", (0, 0.0003, z0), (2 * L, 0.0003, z0), 200)
_, Uc = cu.linea("U", (0, 0.0003, z0), (2 * L, 0.0003, z0), 200)      # velocità sull'asse
u_c = Uc[:, 0]
# lunghezza di sviluppo: dove la velocità sull'asse raggiunge il 99% di 2 Um
i99 = np.argmax(u_c >= 0.99 * 2 * UM); Le = x_u[i99] if u_c[i99] >= 0.99 * 2 * UM else float("nan")
# caduta di pressione in eccesso: p(0) - p(2L) meno la caduta di Poiseuille
dp_tot_u = p_u[0] - p_u[-1]; dp_pois_u = -DPDX * (x_u[-1] - x_u[0]); K = (dp_tot_u - dp_pois_u) / (0.5 * UM**2)

fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.2))
ax[0].plot(u_es / UM, rr / R, color=GRIGIO, lw=1.8, alpha=0.7, label="Hagen–Poiseuille $2(1-r^2/R^2)$")
ax[0].plot(U_in_patch[:, 0] / UM, r_patch / R, "s", color=VERDE, ms=3.5, mfc="none", label="ingresso (facce del patch, codedFixedValue)")
ax[0].plot(U_out[:, 0] / UM, r_out / R, "o", color=BLU, ms=3.5, mfc="none", label="uscita (celle)")
ax[0].set_xlabel("$u/U_m$"); ax[0].set_ylabel("$r/R$"); ax[0].legend(fontsize=6.5, loc="lower left")
ax[1].plot(x_ax / (2 * R), (p_ax - p_ax[-1]) / (0.5 * UM**2), "-", color=BLU, lw=1.2, label="ingresso parabolico, $L/D=10$")
ax[1].plot(x_u / (2 * R), (p_u - np.interp(L, x_u, p_u)) / (0.5 * UM**2), "-", color=ARANCIO, lw=1.2, label="ingresso uniforme, $L/D=20$")
ax[1].plot(x_ax / (2 * R), -DPDX * (L - x_ax) / (0.5 * UM**2), "k:", lw=0.9, label="$-\\mathrm{d}p/\\mathrm{d}x=8\\nu U_m/R^2$")
ax[1].axvline(Le / (2 * R), color=GRIGIO, lw=0.8, ls="--"); ax[1].text(Le / (2 * R) + 0.3, ax[1].get_ylim()[1] * 0.85, f"$L_e/D={Le/(2*R):.1f}$", fontsize=7, color=GRIGIO)
ax[1].set_xlabel("$x/D$"); ax[1].set_ylabel("$(p-p_{x=L})/(\\frac{1}{2}U_m^2)$"); ax[1].legend(fontsize=6.5)
fig.tight_layout(); salva(fig, "c13_poiseuille")

dp_num = p_ax[0] - p_ax[-1]; dp_es = -DPDX * (x_ax[-1] - x_ax[0])
f_num = dp_num * 2 * R / ((x_ax[-1] - x_ax[0]) * 0.5 * UM**2); f_es = 64 / RE
tau_svil = np.mean(tau[(xw[:, 0] > 0.05) & (xw[:, 0] < 0.19), 0])
err_prof = np.max(np.abs(U_out[:, 0] - 2 * UM * (1 - (r_out / R)**2))) / (2 * UM)
ris = dict(dp_num=float(dp_num), dp_es=float(dp_es), f_num=float(f_num), f_es=float(f_es), tau_svil=float(tau_svil), tau_es=TAU,
           tau_da_forza=float(tau_medio_forza), err_prof_uscita=float(err_prof), Le_D=float(Le / (2 * R)), K_imbocco=float(K))
json.dump(ris, open(os.path.join(C, "risultati.json"), "w"), indent=1)
righe = ["\\begin{itemize}"]
righe.append(f"\\item Il profilo codificato è esatto sulle facce d'ingresso (scarto massimo dalla parabola {it(np.max(np.abs(U_in_patch[:,0]-2*UM*(1-(r_patch/R)**2)))/(2*UM),2,True)} relativo); "
             f"all'uscita il profilo nelle celle differisce dalla parabola al più del {it(100*err_prof,2)}\\% di $u_{{\\max}}$: il flusso resta sviluppato lungo tutto il condotto.")
righe.append(f"\\item Caduta di pressione fra $x={it(x_ax[0]*100,2)}$ e $x={it(x_ax[-1]*100,2)}$~cm: {it(dp_num,3,True)} contro {it(dp_es,3,True)}~m$^2$/s$^2$ ({it(100*(dp_num/dp_es-1),2)}\\%); "
             f"fattore d'attrito $f={it(f_num,4)}$ contro $64/\\Rey={it(f_es,4)}$.")
righe.append(f"\\item Sforzo di parete (\\file{{foamcase.py}}, media nel tratto $5<x<19$~cm): $\\tau_w/\\rho=${it(tau_svil,3,True)} contro $4\\nu U_m/R=${it(TAU,3,True)}~m$^2$/s$^2$ "
             f"({it(100*(tau_svil/TAU-1),2)}\\%); la forza viscosa totale sulla parete del cuneo divisa per la sua area dà {it(tau_medio_forza,3,True)}: stesso valore, "
             f"stessa formula (gradiente della prima cella, $y_1={it(1000*(R-m.C[m.owner[m.facce_patch('parete')]][:,1].max()),3)}$~mm).")
righe.append(f"\\item Ingresso uniforme ($L/D=20$): la velocità sull'asse raggiunge il 99\\% di $2U_m$ in $L_e/D={it(Le/(2*R),1)}$ (le correlazioni classiche danno $0{{,}}05$--$0{{,}}065\\,\\Rey$, cioè 10--13 diametri: "
             f"la definizione al 99\\% è severa perché l'avvicinamento al profilo finale è esponenziale); la caduta di pressione in eccesso rispetto a Poiseuille vale $K={it(K,2)}$ in unità di $\\tfrac12U_m^2$ "
             f"(il valore classico della perdita d'imbocco laminare è $K\\approx1{{,}}2$--$1{{,}}3$).")
righe.append("\\end{itemize}")
scrivi_tex("c13_risultati", "\n".join(righe))
