"""
prog3_4_rayleigh.py -- PROGRAMMA 3.4
Problema di Rayleigh generalizzato: acqua fra due lastre distanti 1 m, la
superiore messa bruscamente in moto (u0 = 1 m/s) all'istante t = 0.
    u_t = nu u_yy,   u(0, t) = 0,   u(1, t) = u0,   u(y, 0) = 0 per y < 1.
Metodo esplicito (avanti nel tempo, centrato nello spazio):
    u_i^(j+1) = u_i^j + R (u_(i-1)^j - 2 u_i^j + u_(i+1)^j),   R = nu tau / h^2 <= 1/2.
Confronto con la soluzione esatta in serie di Fourier
    u/u0 = y/d + sum_n 2 (-1)^n/(n pi) sin(n pi y/d) exp(-n^2 pi^2 nu t/d^2)
e, per tempi brevi, con la soluzione di Rayleigh per la lastra singola
    u/u0 = erfc[(d - y) / (2 sqrt(nu t))].
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from stile_grafici import salva, PALETTE, GRIGIO

M = 21
D = 1.0
H = D / (M - 1)
NU = 1.0e-6
R = 0.25
TAU = R * H**2 / NU
NPRINT, NPLOT = 20, 40
TMAX = 2.5e5
U0 = 1.0

y = np.linspace(0.0, D, M)


def esatta(y, t, nmax=200):
    u = y / D
    for n in range(1, nmax + 1):
        u = u + 2 * (-1) ** n / (n * np.pi) * np.sin(n * np.pi * y / D) * np.exp(-(n * np.pi) ** 2 * NU * t / D**2)
    return U0 * u


u = np.zeros(M)
u[-1] = U0
t, j = 0.0, 0
profili = []
print(f"tau = R h^2 / nu = {TAU:.0f} s")
print("   tempo (s)                velocita' (m/s) a y = 0; 0,1; ...; 1")
print(f"  {t:9.3e}  " + " ".join(f"{v:.4f}" for v in u[::2]))
while t < TMAX - 1e-9:
    j += 1
    t += TAU
    u[1:-1] = u[1:-1] + R * (u[:-2] - 2 * u[1:-1] + u[2:])     # tutto il vettore al livello j
    if j % NPRINT == 0:
        print(f"  {t:9.3e}  " + " ".join(f"{v:.4f}" for v in u[::2]))
    if j % NPLOT == 0 and len(profili) < 5:
        profili.append((t, u.copy()))

print("\nErrore massimo rispetto alla serie esatta:")
for tp, up in profili:
    print(f"   t = {tp:8.0f} s   max|u - u_es| = {np.max(np.abs(up - esatta(y, tp))):.2e} m/s")
print(f"   t = {t:8.0f} s   max|u - u_es| = {np.max(np.abs(u - esatta(y, t))):.2e} m/s;"
      f"  scarto dal profilo di Couette = {np.max(np.abs(u - y)):.3f} m/s")
tc = D**2 / (np.pi**2 * NU)
print(f"\nTempo caratteristico d^2/(pi^2 nu) = {tc:.3e} s = {tc/86400:.2f} giorni")
for frac in [0.5, 0.1, 0.01]:
    print(f"   lo scarto dal profilo lineare scende sotto {frac*100:.0f}% di u0 dopo circa "
          f"{tc*np.log(2/np.pi/frac):.3e} s")

# ------------------------------------------------------------------ figura
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3))
yy = np.linspace(0, D, 201)
simboli = ["+", "o", "s", "d", "v"]
for k, (tp, up) in enumerate(profili):
    col = PALETTE[k]
    ax[0].plot(esatta(yy, tp), yy, color=col, lw=1.0)
    ax[0].plot(up, y, simboli[k], ms=3.5, color=col, mfc="none", label=f"$t$ = {tp:,.0f} s".replace(",", "."))
ax[0].plot(yy, yy, ":", color=GRIGIO, label="Couette (regime)")
ax[0].set_xlabel("velocita' (m/s)")
ax[0].set_ylabel("quota (m)")
ax[0].set_xlim(0, 1)
ax[0].set_ylim(0, 1)
ax[0].legend(fontsize=6.5, loc="upper left")
ax[0].set_title("simboli: esplicito; linee: serie esatta")
# tempi brevi: confronto con Rayleigh (lastra singola)
for k, tp in enumerate([625.0 * 4, 625.0 * 16, 625.0 * 40]):
    col = PALETTE[k]
    nst = int(round(tp / TAU))
    uu = np.zeros(M)
    uu[-1] = U0
    for _ in range(nst):
        uu[1:-1] = uu[1:-1] + R * (uu[:-2] - 2 * uu[1:-1] + uu[2:])
    ax[1].plot(erfc((D - yy) / (2 * np.sqrt(NU * tp))), yy, color=col, lw=1.0)
    ax[1].plot(uu, y, "o", ms=3, mfc="none", color=col, label=f"$t$ = {tp:,.0f} s".replace(",", "."))
ax[1].set_xlabel("velocita' (m/s)")
ax[1].set_ylim(0.5, 1.0)
ax[1].set_xlim(0, 1)
ax[1].legend(fontsize=6.5, loc="lower right")
ax[1].set_title("tempi brevi: linee = erfc (Rayleigh)")
fig.tight_layout()
salva(fig, "prog3_4_rayleigh")
