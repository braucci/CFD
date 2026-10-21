"""
prog1_1b_problemi.py
Problemi collegati al Programma 1.1:
  (1) pallina da ping-pong rilasciata sul fondo di una vasca d'acqua
      (effetto del passo h sulla stabilita')
  (2) velocita' terminale: indipendenza dalla velocita' iniziale
  (3) velocita' terminale di gocce di pioggia (sfera rigida d'acqua)
"""
import numpy as np
import matplotlib.pyplot as plt
from prog1_1_caduta_sfera import caduta, costanti
from ode_solvers import cd_sfera
from stile_grafici import salva, PALETTE, BLU, ARANCIO, ROSSO

# ------------------------------------------------- (1) ping-pong in acqua
print("=== (1) Pallina da ping-pong in acqua ===")
d, rho, rhof, nu = 0.036, 1.22, 1000.0, 1.0e-6
A, B, C = costanti(d, rho, rhof)
print(f"A = {A:.2f}  B = {B:.1f} m/s^2  C = {C:.1f} 1/m   (accelerazione iniziale B/A = {B/A:.1f} m/s^2)")
fig, ax = plt.subplots(figsize=(4.4, 3.0))
for h, c in ((0.1, ROSSO), (0.01, BLU)):
    T, Z, V, RE = caduta(d, h=h, tmax=0.5, rho=rho, rhof=rhof, nu=nu)
    ax.plot(T, V, "o-", ms=2.5, color=c, label=f"$h = {h}$ s")
    print(f"  h = {h}: v(0.5 s) = {V[-1]:.3f} m/s   Re = {RE[-1]:.0f}")
# velocita' terminale teorica (c_d = 0.5 per 400 < Re < 3e5): v_t = sqrt(B/(C c_d)) (verso l'alto)
vt = -np.sqrt(-B / (C * 0.5))
ax.axhline(vt, color="gray", ls="--", lw=0.8, label=f"$v_t = {vt:.2f}$ m/s")
ax.set_xlabel("$t$ [s]"); ax.set_ylabel("$v$ [m/s] (positivo verso il basso)")
ax.set_ylim(-2.0, 0.5)
ax.legend()
salva(fig, "prog1_1b_pingpong")
print(f"  v_t teorica = {vt:.3f} m/s;  tempo caratteristico tau = A/(2 C c_d |v_t|) = "
      f"{A/(2*C*0.5*abs(vt)):.3f} s")

# ------------------------------------------------- (2) effetto di v0
print("\n=== (2) Velocita' iniziale diversa, sfera d'acciaio d = 0.01 m ===")
fig, ax = plt.subplots(figsize=(4.4, 3.0))
for v0, c in ((0.0, PALETTE[0]), (-30.0, PALETTE[1]), (80.0, PALETTE[2])):
    T, Z, V, RE = caduta(0.01, h=0.1, tmax=12.0, v0=v0)
    ax.plot(T, V, color=c, label=f"$v_0 = {v0:+.0f}$ m/s")
    print(f"  v0 = {v0:+5.1f} m/s -> v(12 s) = {V[-1]:.2f} m/s")
ax.axhline(41.39, color="gray", ls="--", lw=0.8, label="$v_t$")
ax.set_xlabel("$t$ [s]"); ax.set_ylabel("$v$ [m/s]")
ax.legend()
salva(fig, "prog1_1b_v0")

# ------------------------------------------------- (3) gocce di pioggia
print("\n=== (3) Velocita' terminale di gocce d'acqua in aria (sfera rigida) ===")
# valori misurati (Gunn & Kinzer, 1949) usati come riferimento sperimentale
d_mis = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 5.8]) * 1e-3
v_mis = np.array([2.06, 4.03, 5.40, 6.49, 7.41, 8.06, 8.53, 8.83, 9.00, 9.09, 9.14, 9.17])
d_calc = np.logspace(np.log10(0.1e-3), np.log10(6e-3), 30)
v_calc = []
for dd in d_calc:
    T, Z, V, RE = caduta(dd, h=0.005, tmax=5.0, rho=1000.0)
    v_calc.append(V[-1])
v_calc = np.array(v_calc)
for dd, vv in zip(d_calc[::6], v_calc[::6]):
    print(f"  d = {dd*1e3:5.2f} mm  v_t = {vv:6.2f} m/s  Re = {vv*dd/1.49e-5:8.0f}")
fig, ax = plt.subplots(figsize=(4.4, 3.0))
ax.plot(d_calc * 1e3, v_calc, color=BLU, label="sfera rigida (Programma 1.1)")
ax.plot(d_mis * 1e3, v_mis, "o", color=ARANCIO, ms=4, label="misure (Gunn-Kinzer)")
ax.set_xlabel("diametro [mm]"); ax.set_ylabel("$v_t$ [m/s]")
ax.legend(loc="lower right")
salva(fig, "prog1_1b_pioggia")
