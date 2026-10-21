"""
prog2_8b_courant.py -- il ruolo del numero di Courant nello schema leapfrog.

1) Analisi di von Neumann: fattore di amplificazione G(kh) e rapporto fra la
   velocita' di fase numerica e quella esatta, per diversi Co.
2) Esperimenti: la stessa ondina del Programma 2.8 integrata con Co = 1, 0,5
   e 0,25 (dispersione) e con Co = 1,02 (instabilita').
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, ROSSO, GRIGIO, VERDE, VIOLA
from prog2_8_onda_tubo import leapfrog, esatta, A_SUONO

# ------------------------------------------------------ 1) von Neumann
kh = np.linspace(1e-3, np.pi, 400)
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.8))
for Co, col in ((0.25, VIOLA), (0.5, VERDE), (0.75, ARANCIO), (1.0, BLU)):
    Ah = 1 - Co**2 * (1 - np.cos(kh))                # A/2
    theta = np.arccos(np.clip(Ah, -1, 1))            # fase numerica per passo
    ax[1].plot(kh, theta / (Co * kh), color=col, label=f"Co = {Co:g}")
for Co, col in ((1.0, BLU), (1.05, ARANCIO), (1.2, ROSSO)):
    Ah = 1 - Co**2 * (1 - np.cos(kh))
    G = np.where(np.abs(Ah) <= 1, 1.0, np.abs(Ah) + np.sqrt(np.maximum(Ah**2 - 1, 0)))
    ax[0].plot(kh, G, color=col, label=f"Co = {Co:g}")
    print(f"Co = {Co:4.2f}: max |G| = {G.max():.4f} (a kh = pi)")
ax[0].set_xlabel("$kh$"); ax[0].set_ylabel(r"$\max|G|$"); ax[0].legend()
ax[0].set_xticks([0, np.pi / 2, np.pi]); ax[0].set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
ax[1].set_xlabel("$kh$"); ax[1].set_ylabel(r"$c_{num}/a$"); ax[1].legend()
ax[1].set_xticks([0, np.pi / 2, np.pi]); ax[1].set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
salva(fig, "prog2_8b_vonneumann")

# ------------------------------------------------------ 2) esperimenti
h = 0.02
t_fin = 40 * h / A_SUONO                      # stesso istante fisico per tutti
fig, ax = plt.subplots(2, 1, figsize=(5.6, 4.6), sharex=True)
print("\n  Co    passi   max|u - u_esatta| a t = 40 h/a")
for Co, col in ((1.0, BLU), (0.5, VERDE), (0.25, VIOLA)):
    n = int(round(40 / Co))
    x, t, U = leapfrog(h, Co, n)
    e = np.max(np.abs(U[-1] - esatta(x, t[-1])))
    print(f"  {Co:4.2f}   {n:4d}     {e:.3e}")
    ax[0].plot(x, U[-1], color=col, lw=1.0, label=f"Co = {Co:g}")
ax[0].plot(x, esatta(x, t_fin), "--", color=GRIGIO, lw=1.0, label="esatta")
ax[0].set_ylabel("$u$ [m/s]"); ax[0].legend(ncol=4, fontsize=7, loc="lower center"); ax[0].set_ylim(-0.75, 0.65)
ax[0].set_title(r"$t = 40\,h/a$", fontsize=8)
# instabilita': Co leggermente maggiore di 1
Co = 1.02
x, t, U = leapfrog(h, Co, 120)
amp = np.max(np.abs(U), axis=1)
print(f"\nCo = {Co}: ampiezza massima dopo 40, 80, 120 passi = "
      f"{amp[40]:.3g}, {amp[80]:.3g}, {amp[120]:.3g}")
ax[1].plot(x, U[60], color=ROSSO, lw=1.0)
ax[1].set_ylabel("$u$ [m/s]"); ax[1].set_xlabel("$x$ [m]")
ax[1].set_title(fr"Co = {Co}: soluzione dopo 60 passi", fontsize=8)
salva(fig, "prog2_8b_esperimenti")

fig, ax = plt.subplots(figsize=(4.4, 3.0))
ax.semilogy(amp, color=ROSSO)
Ah = 1 - 2 * Co**2
Gmax = abs(Ah) + np.sqrt(Ah**2 - 1)
ax.semilogy(np.arange(121), amp[20] * Gmax**(np.arange(121) - 20.0), "--", color=GRIGIO,
            label=fr"$|G|^n$, $|G|_{{max}}={Gmax:.3f}$")
ax.set_ylim(0.3, 1e20); ax.set_xlabel("passo $n$"); ax.set_ylabel(r"$\max_i |u_i^n|$")
ax.legend()
salva(fig, "prog2_8b_crescita")
