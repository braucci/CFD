"""
prog4_5b_cavita_vento.py -- cavita' mossa dal vento (shear-driven cavity): sulla
superficie superiore, invece della velocita', e' imposto lo sforzo tangenziale
costante tau = rho_a u*^2 esercitato dall'aria. Con la velocita' d'attrito u* e
l'altezza L come scale, Re = u* L/nu e la condizione diventa
    du/dy = -(rho_a/rho) Re,   v = 0   (vento da destra verso sinistra),
con rho_a/rho = 1/1000 (aria su acqua). Si usa lo stesso risolutore MAC del
Programma 4.5 (prog4_5_cavita_mac.py) su una griglia 41 x 41, per Re = 100 e 2000.
"""
import numpy as np
import matplotlib.pyplot as plt
from prog4_5_cavita_mac import cavita
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

RAPPORTO = 1.0 / 1000.0          # rho_aria / rho_acqua
ris = {}
print("Cavita' mossa dal vento (sforzo imposto in superficie), griglia 41x41:")
for Re, tmax in ((100, 60.0), (2000, 400.0)):
    g = -RAPPORTO * Re
    r = cavita(Re, 40, tmax, tol=1e-6, sforzo=g, u_max=max(0.3, abs(g) * 0.5), stampa=False)
    psi = r["psi"]; N = r["N"]; h = r["h"]
    u_sup = 0.5 * (r["u"][:, N] + r["u"][:, N + 1])
    k_min = np.unravel_index(np.argmin(psi), psi.shape); k_max = np.unravel_index(np.argmax(psi), psi.shape)
    print(f"   Re = {Re:5d}: du/dy = {g:+.2f}; {r['passi']} passi, t = {r['t']:.1f};"
          f" psi_min = {psi.min():+.5f} in ({k_min[0]*h:.3f}, {k_min[1]*h:.3f}), psi_max = {psi.max():+.5f},"
          f" Delta psi = {psi.max()-psi.min():.5f}; velocita' in superficie: max |u| = {np.max(np.abs(u_sup)):.4f}")
    ris[Re] = r

fig, axs = plt.subplots(1, 3, figsize=(8.6, 2.8), gridspec_kw={"width_ratios": [1, 1, 1.1]})
for a_, Re in zip(axs[:2], (100, 2000)):
    r = ris[Re]; N = r["N"]; xg = np.linspace(0, 1, N + 1)
    m = np.max(np.abs(r["psi"]))
    lv = np.linspace(-m, m, 15); lv = lv[np.abs(lv) > 1e-9]
    a_.contour(xg, xg, r["psi"].T, levels=lv, colors=[ROSSO if v < 0 else BLU for v in lv],
               linestyles=["--" if v < 0 else "-" for v in lv], linewidths=0.8)
    a_.set_aspect("equal"); a_.set_title(f"$\\psi$, $Re={Re}$", fontsize=8)
    a_.set_xticks([]); a_.set_yticks([])
for Re, col in zip((100, 2000), (BLU, ROSSO)):
    r = ris[Re]; N = r["N"]; h = r["h"]
    yy = (np.arange(N + 2) - 0.5) * h
    axs[2].plot(r["u"][N // 2, 1:-1], yy[1:-1], color=col, label=f"$Re={Re}$")
axs[2].axvline(0, color="k", lw=0.6)
axs[2].set_xlabel("$u(0{,}5,y)/u_*$"); axs[2].set_ylabel("$y$"); axs[2].legend(fontsize=7)
fig.tight_layout()
salva(fig, "prog4_5b_cavita_vento")
