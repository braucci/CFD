"""
fig01_cavita.py -- figure e risultati del caso c01_cavita (Cap. 1): campi di velocita' e pressione
letti con foamcase, profilo sulla mezzeria confrontato con una mesh 80 x 80, residui dal log.
Uso: python3 fig01_cavita.py   (dopo aver eseguito casi/c01_cavita/Allrun)
"""
import os
import sys
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stile_grafici import salva, BLU, ARANCIO, GRIGIO, VERDE
from foamcase import Caso
from foamlog import leggi_log, riassunto
from figutil import caso, campo2d, it, scrivi_tex

C = caso("c01_cavita")
c = Caso(C)
m = c.mesh
U = c.campo("U"); p = c.campo("p")
magU = np.linalg.norm(U, axis=1)

# --- confronto con una mesh 80 x 80 (eseguita qui in una copia temporanea) -------------------
fine = C + "_80"
if not os.path.isdir(fine):
    shutil.copytree(C, fine, ignore=shutil.ignore_patterns("0.*", "log.*", "postProcessing"))
    subprocess.run("foamDictionary -entry blocks -set '( hex (0 1 2 3 4 5 6 7) (80 80 1) simpleGrading (1 1 1) )' system/blockMeshDict "
                   "&& foamDictionary -entry deltaT -set 0.00125 system/controlDict && foamDictionary -entry writeInterval -set 80 system/controlDict "
                   "&& blockMesh > log.blockMesh && foamRun > log.foamRun", cwd=fine, shell=True, check=True)
cf = Caso(fine)
y20, U20 = c.linea("U", (0.05, 0, 0.005), (0.05, 0.1, 0.005), 20)
y80, U80 = cf.linea("U", (0.05, 0, 0.005), (0.05, 0.1, 0.005), 80)

fig, ax = plt.subplots(1, 3, figsize=(9.2, 3.0))
cfm = campo2d(ax[0], m, magU, livelli=np.linspace(0, 1, 21), cmap="viridis")
passo = 2
ax[0].quiver(m.C[::passo, 0], m.C[::passo, 1], U[::passo, 0], U[::passo, 1], color="w", scale=12, width=0.004)
fig.colorbar(cfm, ax=ax[0], label="$|\\mathbf{U}|$ [m/s]", shrink=0.85)
ax[0].set_title("velocità", fontsize=9); ax[0].set_xlabel("$x$ [m]"); ax[0].set_ylabel("$y$ [m]")
cfp = campo2d(ax[1], m, p, livelli=np.linspace(np.percentile(p, 2), np.percentile(p, 98), 21), cmap="RdBu_r", linee=True, colore_linee="k", lw=0.3, extend="both")
fig.colorbar(cfp, ax=ax[1], label="$p$ [m$^2$/s$^2$]", shrink=0.85)
ax[1].set_title("pressione cinematica", fontsize=9); ax[1].set_xlabel("$x$ [m]")
ax[2].plot(U20[:, 0], y20 / 0.1, "o", color=BLU, ms=3.5, mfc="none", label="$20\\times20$ (celle)")
ax[2].plot(U80[:, 0], y80 / 0.1, "-", color=ARANCIO, lw=1.2, label="$80\\times80$")
ax[2].set_xlabel("$u/U$ su $x=L/2$"); ax[2].set_ylabel("$y/L$"); ax[2].legend(fontsize=7)
fig.tight_layout()
salva(fig, "c01_cavita")

# --- residui ---------------------------------------------------------------------------------
d = leggi_log(os.path.join(C, "log.foamRun"))
fig, ax = plt.subplots(figsize=(5.2, 3.0))
for nome, col in (("Ux", BLU), ("Uy", ARANCIO), ("p", VERDE)):
    cc = d["campi"][nome]
    ax.semilogy(cc["t"], cc["res0"], color=col, lw=1.1, label=nome)
ax.set_xlabel("$t$ [s]"); ax.set_ylabel("residuo iniziale"); ax.legend(fontsize=8)
fig.tight_layout()
salva(fig, "c01_residui")

# --- numeri per il riquadro ------------------------------------------------------------------
umin20 = U20[:, 0].min(); umin80 = U80[:, 0].min()
i_cen = np.argmin(np.abs(m.C[:, 0] - 0.05) + np.abs(m.C[:, 1] - 0.05))
print(riassunto(d))
testo = f"""
\\begin{{itemize}}
\\item Mesh: {m.nCells} celle, {m.nFaces} facce ({m.nInternalFaces} interne), {m.nPoints} punti; \\kw{{checkMesh}}: \\emph{{Mesh OK}}, non ortogonalità nulla.
\\item Calcolo: {len(d['campi']['p']['t'])} passi temporali in {it(d['exec'][-1,1],2)}~s; numero di Courant massimo {it(d['Co'][:,2].max(),3)}; errore di continuità cumulativo {it(d['continuita'][-1,3],1,True)}.
\\item A $t=0{{,}}5$~s i residui iniziali sono {it(d['campi']['Ux']['res0'][-1],1,True)} ($U_x$), {it(d['campi']['Uy']['res0'][-1],1,True)} ($U_y$), {it(d['campi']['p']['res0'][-1],1,True)} ($p$): stato stazionario raggiunto.
\\item Velocità minima (controcorrente) sulla mezzeria: $u_{{\\min}}/U={it(umin20,4)}$ in $y/L={it(y20[np.argmin(U20[:,0])]/0.1,2)}$ su $20\\times20$, $u_{{\\min}}/U={it(umin80,4)}$ su $80\\times80$: la mesh del tutorial sottostima il ricircolo del {it(100*abs(umin80-umin20)/abs(umin80),0)}\\%, come si vede nel profilo di Fig.~\\ref{{fig:c01_campi}}.
\\item Pressione cinematica: minima {it(p.min(),3)} e massima {it(p.max(),3)}~m$^2$/s$^2$ negli spigoli superiori (dove il coperchio ``strappa'' e comprime il fluido); nella cella centrale $p={it(p[i_cen],4)}$.
\\end{{itemize}}
"""
scrivi_tex("c01_risultati", testo)
