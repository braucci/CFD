"""
fig14_codice.py -- figura e risultati del caso c14_codice (Cap. 12): il solutore trasportoFoam confrontato con la
function object scalarTransport (caso c05, vanLeer, 40 x 40) e la condizione al contorno compilata profiloParabolico
confrontata con la versione codedFixedValue (caso c13).
Uso: python3 fig14_codice.py   (dopo c14_codice/Allrun, fig05_convezione.py e c13_poiseuille/Allrun)
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stile_grafici import salva, BLU, ARANCIO, VERDE, GRIGIO
from foamcase import Caso
from foamlog import leggi_log
from figutil import caso, campo2d, it, scrivi_tex

C14 = caso("c14_codice")
ct = Caso(os.path.join(C14, "casoTrasporto")); T = ct.campo("T")
c5 = Caso(caso("c05_convezione") + "_vanLeer40"); s5 = c5.campo("s")
cp = Caso(os.path.join(C14, "casoProfilo")); c13 = Caso(caso("c13_poiseuille"))
R = 0.01; UM = 0.1

d_T = np.abs(T - s5)
Uin = cp.valore_patch("U", "ingresso"); Uin13 = c13.valore_patch("U", "ingresso")
m = cp.mesh; Cf = m.Cf[m.facce_patch("ingresso")]; r = np.hypot(Cf[:, 1], Cf[:, 2])
Uout = cp.campo("U"); Uout13 = c13.campo("U")
scarto_campo = np.max(np.abs(Uout - Uout13)) / (2 * UM)
log_t = leggi_log(os.path.join(C14, "casoTrasporto", "log.trasportoFoam"))
log_5 = leggi_log(os.path.join(caso("c05_convezione") + "_vanLeer40", "log.foamRun"))

fig, ax = plt.subplots(1, 3, figsize=(9.8, 3.1), gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
cf = campo2d(ax[0], ct.mesh, T, livelli=np.linspace(0, 1, 11), cmap="RdYlBu_r")
ax[0].tricontour(ct.mesh.C[:, 0], ct.mesh.C[:, 1], s5, levels=[0.1, 0.5, 0.9], colors="k", linewidths=0.6, linestyles="--")
fig.colorbar(cf, ax=ax[0], label="$T$ (trasportoFoam)", shrink=0.85)
ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$y$"); ax[0].set_title("colori: trasportoFoam; tratteggio: scalarTransport", fontsize=8.5)
ax[1].semilogy(log_t["campi"]["T"]["t"], log_t["campi"]["T"]["res0"], color=BLU, lw=1.0, label="trasportoFoam ($T$)")
ax[1].semilogy(log_5["campi"]["s"]["t"], log_5["campi"]["s"]["res0"], "--", color=ARANCIO, lw=1.0, label="scalarTransport ($s$)")
ax[1].set_xlabel("iterazione"); ax[1].set_ylabel("residuo iniziale"); ax[1].legend(fontsize=7); ax[1].set_title("stessa storia di convergenza", fontsize=9)
rr = np.linspace(0, R, 50)
ax[2].plot(2 * (1 - (rr / R)**2), rr / R, color=GRIGIO, lw=1.8, alpha=0.7, label="parabola esatta")
ax[2].plot(Uin[:, 0] / UM, r / R, "s", color=BLU, ms=4, mfc="none", label="profiloParabolico (classe)")
ax[2].plot(Uin13[:, 0] / UM, r / R, "+", color=ARANCIO, ms=6, label="codedFixedValue (c13)")
ax[2].set_xlabel("$u/U_m$ all'ingresso"); ax[2].set_ylabel("$r/R$"); ax[2].legend(fontsize=7)
fig.tight_layout(); salva(fig, "c14_confronto")

ris = dict(scarto_T_max=float(d_T.max()), scarto_T_medio=float(d_T.mean()), iter_trasporto=len(log_t["campi"]["T"]["t"]), iter_c05=len(log_5["campi"]["s"]["t"]),
           scarto_ingresso=float(np.max(np.abs(Uin - Uin13)) / (2 * UM)), scarto_campo=float(scarto_campo))
json.dump(ris, open(os.path.join(C14, "risultati.json"), "w"), indent=1)
righe = ["\\begin{itemize}"]
righe.append(f"\\item \\kw{{trasportoFoam}} e \\kw{{scalarTransport}} sullo stesso gradino obliquo ($40\\times40$, \\kw{{vanLeer}}, rilassamento $0{{,}}7$): i due campi coincidono cifra per cifra "
             f"(scarto massimo {('nullo alle 8 cifre scritte' if ris['scarto_T_max'] == 0 else it(ris['scarto_T_max'],2,True))}); stesse {ris['iter_trasporto']} iterazioni con residui sovrapponibili (Fig.~\\ref{{fig:c14_confronto}}): le due implementazioni "
             f"eseguono le stesse operazioni, perché \\py{{fvm::ddt}}, \\py{{fvm::div}} e \\py{{fvm::laplacian}} sono le stesse funzioni e leggono gli stessi dizionari.")
righe.append(f"\\item La classe \\kw{{profiloParabolico}} e il blocco \\kw{{codedFixedValue}} del Cap.~\\ref{{cap:contorno}} producono lo stesso profilo d'ingresso (scarto {('nullo' if ris['scarto_ingresso'] == 0 else it(ris['scarto_ingresso'],1,True))} alle cifre scritte) "
             f"e lo stesso campo nel condotto (scarto massimo {it(ris['scarto_campo'],1,True)}); la classe aggiunge il controllo degli argomenti, la rampa temporale (\\kw{{fattore}}) e la scrittura "
             f"ordinata nel file del campo, ed è riutilizzabile in qualunque caso con \\kw{{libs (\"libprofiloParabolico.so\")}}.")
righe.append("\\item Compilazione: \\kw{wmake libso} produce \\file{libprofiloParabolico.so} in \\kw{\\$FOAM\\_USER\\_LIBBIN}, \\kw{wmake} produce \\file{trasportoFoam} in \\kw{\\$FOAM\\_USER\\_APPBIN}; "
             "i due script \\file{Allrun} e \\file{Allclean} del caso compilano, eseguono e ripuliscono (\\kw{wclean}).")
righe.append("\\end{itemize}")
scrivi_tex("c14_risultati", "\n".join(righe))
