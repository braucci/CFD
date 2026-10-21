"""
prog2_1_flusso_radiale.py -- PROGRAMMA 2.1
Problema ai limiti per un'equazione differenziale ordinaria del secondo ordine:
flusso radiale prodotto da una distribuzione di sorgenti e da una sorgente
lineare nell'origine.

    Phi'' + Phi'/R = 9 R ,   1 <= R <= 4
    Phi(1) = 1 ,  Phi(4) = 64 - 24 ln 4
    soluzione esatta:  Phi = R^3 - 24 ln R ,  U = Phi' = 3 R^2 - 24/R

Differenze centrate del secondo ordine + sistema tridiagonale (funzione trid).
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, GRIGIO, ROSSO
from metodi_numerici import trid, thomas

# ----- coefficienti dell'equazione nella forma f'' + A f' + B f = D ---------
A = lambda R: 1.0 / R
B = lambda R: 0.0 * R
D = lambda R: 9.0 * R
phi_esatta = lambda R: R**3 - 24.0 * np.log(R)
u_esatta = lambda R: 3.0 * R**2 - 24.0 / R

RMIN, RMAX = 1.0, 4.0
PHI0, PHINP1 = phi_esatta(RMIN), phi_esatta(RMAX)    # = 1 e 64 - 24 ln 4


def risolvi(intervalli, solutore="trid"):
    """Restituisce R, Phi, U nei punti interni per un numero dato di intervalli."""
    n = intervalli - 1                    # numero di incognite
    h = (RMAX - RMIN) / intervalli
    R = RMIN + h * np.arange(1, n + 1)
    C = np.zeros((n, 4))
    C[:, 0] = 1.0 - 0.5 * h * A(R)        # coefficiente di f_{i-1}
    C[:, 1] = -2.0 + h * h * B(R)         # coefficiente di f_i
    C[:, 2] = 1.0 + 0.5 * h * A(R)        # coefficiente di f_{i+1}
    C[:, 3] = h * h * D(R)                # termine noto
    # condizioni al contorno: i termini noti passano a secondo membro
    C[0, 3] -= C[0, 0] * PHI0
    C[0, 0] = 0.0
    C[-1, 3] -= C[-1, 2] * PHINP1
    C[-1, 2] = 0.0
    if solutore == "trid":
        phi = trid(C)
    else:
        phi = thomas(C[:, 0], C[:, 1], C[:, 2], C[:, 3])
    # velocita' radiale con la differenza centrata
    ff = np.concatenate([[PHI0], phi, [PHINP1]])
    U = (ff[2:] - ff[:-2]) / (2.0 * h)
    return R, phi, U


if __name__ == "__main__":
    R, phi, U = risolvi(30)
    print("   I     R       PHI        PHI esatto    err.%      U          U esatto")
    for i in range(len(R)):
        ep = phi_esatta(R[i])
        print(f"  {i+1:2d}  {R[i]:4.2f}  {phi[i]:10.5f}  {ep:10.5f}  "
              f"{100*abs(phi[i]-ep)/abs(ep):6.3f}  {U[i]:10.5f}  {u_esatta(R[i]):10.5f}")

    # ----- grafico della soluzione -----------------------------------------
    Rf = np.linspace(RMIN, RMAX, 300)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(Rf, u_esatta(Rf), color=BLU, lw=1.0, label="$U$ esatta")
    ax.plot(R, U, "o", color=BLU, ms=3.5, label="$U$ numerica")
    ax.plot(Rf, phi_esatta(Rf), "--", color=ARANCIO, lw=1.0, label=r"$\Phi$ esatta")
    ax.plot(R, phi, "s", color=ARANCIO, ms=3.2, label=r"$\Phi$ numerica")
    ax.axhline(0, color=GRIGIO, lw=0.6)
    ax.set_xlabel("$R = r/r_0$"); ax.set_ylabel(r"$U,\ \Phi$")
    ax.legend(ncol=2)
    salva(fig, "prog2_1_flusso_radiale")

    # ----- studio di convergenza (algoritmo di Thomas) ---------------------
    print("\n intervalli     h        max|err Phi|    max|err U|   ordine Phi  ordine U")
    NN = [15, 30, 60, 120, 240, 480, 960]
    hh, ep, eu = [], [], []
    for nint in NN:
        R, phi, U = risolvi(nint, solutore="thomas")
        hh.append(3.0 / nint)
        ep.append(np.max(np.abs(phi - phi_esatta(R))))
        eu.append(np.max(np.abs(U - u_esatta(R))))
    for k in range(len(NN)):
        op = np.log2(ep[k-1] / ep[k]) if k else np.nan
        ou = np.log2(eu[k-1] / eu[k]) if k else np.nan
        print(f"   {NN[k]:5d}   {hh[k]:8.5f}   {ep[k]:.4e}    {eu[k]:.4e}     {op:5.2f}     {ou:5.2f}")

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    hh = np.array(hh)
    ax.loglog(hh, ep, "o-", color=ARANCIO, label=r"errore su $\Phi$")
    ax.loglog(hh, eu, "s-", color=BLU, label="errore su $U$")
    ax.loglog(hh, 0.8 * ep[0] * (hh / hh[0])**2, "--", color=GRIGIO, label="pendenza 2")
    ax.set_xlabel("$h$"); ax.set_ylabel("errore massimo")
    ax.legend()
    salva(fig, "prog2_1_convergenza")

    # ----- confronto trid / thomas e crescita dei coefficienti -------------
    R, p1, _ = risolvi(30, "trid")
    R, p2, _ = risolvi(30, "thomas")
    print(f"\n differenza massima trid/thomas (30 intervalli): {np.max(np.abs(p1-p2)):.2e}")
