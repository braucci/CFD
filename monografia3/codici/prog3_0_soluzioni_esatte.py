"""
prog3_0_soluzioni_esatte.py -- soluzioni esatte delle equazioni di Navier-Stokes
per flussi paralleli: la famiglia di Couette-Poiseuille fra due lastre piane.

Canale 0 <= y <= h, lastra inferiore ferma, lastra superiore con velocita' U,
gradiente di pressione costante dp/dx. Con Y = y/h e il parametro adimensionale
    P = -(h^2 / (2 mu U)) dp/dx
il profilo di velocita' e'   u/U = Y + P Y (1 - Y).
Si calcolano: profili, vorticita' zeta = -du/dy, sforzo alle pareti,
portata e potenza dissipata per unita' di volume (funzione di dissipazione).
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ROSSO, GRIGIO, PALETTE

Y = np.linspace(0.0, 1.0, 201)
valori_P = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]


def profilo(P, Y):
    return Y + P * Y * (1.0 - Y)


def derivata(P, Y):                  # d(u/U)/dY
    return 1.0 + P * (1.0 - 2.0 * Y)


print("Famiglia di Couette-Poiseuille:  u/U = Y + P Y (1 - Y)")
print("     P    tau_inf/(mu U/h)  tau_sup/(mu U/h)  portata/(U h)  dissipazione/(mu U^2/h^2)")
for P in valori_P:
    tau_inf = derivata(P, 0.0)       # sforzo sulla lastra inferiore
    tau_sup = derivata(P, 1.0)       # sforzo esercitato DAL fluido... sulla superiore: -du/dy
    portata = 0.5 + P / 6.0          # integrale di u/U fra 0 e 1
    diss = np.trapezoid(derivata(P, Y) ** 2, Y)   # Phi/mu = (du/dy)^2, media sul canale
    print(f"  {P:5.1f}   {tau_inf:12.4f}   {tau_sup:14.4f}   {portata:13.4f}   {diss:16.4f}")

# controllo: il profilo soddisfa  mu u'' = dp/dx, cioe' (u/U)'' = -2P
Pt = 2.0
h_y = Y[1] - Y[0]
u = profilo(Pt, Y)
residuo = (u[2:] - 2 * u[1:-1] + u[:-2]) / h_y**2 + 2 * Pt
print(f"\nVerifica (P = {Pt}): max |(u/U)'' + 2P| = {np.max(np.abs(residuo)):.2e}")
print("Il flusso inverso alla parete inferiore compare per P < -1 (tau_inf < 0).")

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3))
for k, P in enumerate(valori_P):
    col = PALETTE[k % len(PALETTE)]
    ax[0].plot(profilo(P, Y), Y, color=col, label=f"$P={P:+.0f}$")
    ax[1].plot(-derivata(P, Y), Y, color=col)
ax[0].axvline(0, color=GRIGIO, lw=0.8)
ax[0].set_xlabel("$u/U$")
ax[0].set_ylabel("$y/h$")
ax[0].set_title("profili di velocita'")
ax[0].legend(loc="upper left", fontsize=7)
ax[1].axvline(0, color=GRIGIO, lw=0.8)
ax[1].set_xlabel(r"$\zeta h/U = -\,\mathrm{d}(u/U)/\mathrm{d}(y/h)$")
ax[1].set_title("vorticita'")
fig.tight_layout()
salva(fig, "prog3_0_couette_poiseuille")
