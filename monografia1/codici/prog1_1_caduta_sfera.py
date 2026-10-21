"""
prog1_1_caduta_sfera.py  --  Programma 1.1
Caduta libera di una sfera in un fluido in quiete (Runge-Kutta 4).

    dz/dt = v
    dv/dt = F(v) = [ B - C v|v| c_d(Re) ] / A

con  A = 1 + rho_bar/2,  B = (1 - rho_bar) g,  C = 3 rho_bar/(4 d),
rho_bar = rho_f / rho  (rapporto fra densita' del fluido e del corpo).
"""
import numpy as np
import matplotlib.pyplot as plt
from ode_solvers import cd_sfera
from stile_grafici import salva, PALETTE

# ------------------------------------------------------------ dati
RHO, RHOF, NU, G = 8000.0, 1.22, 1.49e-5, 9.8   # acciaio in aria (livello del mare)


def costanti(d, rho=RHO, rhof=RHOF):
    rb = rhof / rho
    return 1.0 + 0.5 * rb, (1.0 - rb) * G, 3.0 * rb / (4.0 * d)


def caduta(d, h=0.1, tmax=10.0, z0=0.0, v0=0.0, rho=RHO, rhof=RHOF, nu=NU):
    """Integra il moto con RK4 e restituisce gli array t, z, v, Re."""
    A, B, C = costanti(d, rho, rhof)

    def F(v):                                    # accelerazione dv/dt
        return (B - C * v * abs(v) * cd_sfera(v * d / nu)) / A

    t, z, v = 0.0, z0, v0
    T, Z, V = [t], [z], [v]
    while t < tmax - 1e-12:
        d1z = h * v;              d1v = h * F(v)
        d2z = h * (v + d1v / 2);  d2v = h * F(v + d1v / 2)
        d3z = h * (v + d2v / 2);  d3v = h * F(v + d2v / 2)
        d4z = h * (v + d3v);      d4v = h * F(v + d3v)
        z += (d1z + 2 * d2z + 2 * d3z + d4z) / 6
        v += (d1v + 2 * d2v + 2 * d3v + d4v) / 6
        t += h
        T.append(t); Z.append(z); V.append(v)
    T, Z, V = map(np.array, (T, Z, V))
    return T, Z, V, V * d / nu


if __name__ == "__main__":
    # -------------------------------------------- tabella per d = 0.01 m
    d = 0.01
    T, Z, V, RE = caduta(d, h=0.1, tmax=10.0)
    ZV, VV = 0.5 * G * T ** 2, G * T                # soluzione nel vuoto
    print(f"Sfera d'acciaio, d = {d} m, h = 0.1 s")
    print(f"{'t [s]':>6} {'z_vuoto':>9} {'z_aria':>9} {'v_vuoto':>9} {'v_aria':>9} {'Re':>10}")
    for i in range(0, len(T), 10):
        print(f"{T[i]:6.1f} {ZV[i]:9.3f} {Z[i]:9.3f} {VV[i]:9.3f} {V[i]:9.3f} {RE[i]:10.0f}")

    # -------------------------------------------- verifica del passo
    print("\nVerifica della sensibilita' al passo (z, v a t = 10 s):")
    for h in (0.2, 0.1, 0.02):
        T_, Z_, V_, _ = caduta(d, h=h, tmax=10.0)
        print(f"  h = {h:4.2f}  z = {Z_[-1]:9.4f} m   v = {V_[-1]:8.4f} m/s")

    # -------------------------------------------- velocita' terminale
    A, B, C = costanti(d)
    vt = np.sqrt(B / (C * cd_sfera(RE[-1])))
    print(f"\nRe finale = {RE[-1]:.0f} -> c_d = {cd_sfera(RE[-1])}, "
          f"v_t = sqrt(B/(C c_d)) = {vt:.2f} m/s  (v a 10 s: {V[-1]:.2f} m/s)")

    # -------------------------------------------- figure per vari diametri
    diametri = (0.001, 0.005, 0.01, 0.02, 0.07)
    fig1, ax1 = plt.subplots(figsize=(4.4, 3.2))
    fig2, ax2 = plt.subplots(figsize=(4.4, 3.2))
    tt = np.linspace(0, 12, 200)
    for dd, c in zip(diametri, PALETTE):
        T, Z, V, RE = caduta(dd, h=0.1, tmax=12.0)
        ax1.plot(T, Z, color=c, label=f"$d={dd}$ m")
        ax2.plot(T, V, color=c, label=f"$d={dd}$ m")
        print(f"d = {dd:5.3f} m: v(12 s) = {V[-1]:6.2f} m/s,  Re = {RE[-1]:9.0f}, "
              f"t(z=56 m) = {np.interp(56.0, Z, T):.3f} s")
    ax1.plot(tt, 0.5 * G * tt ** 2, "k--", lw=1, label="nel vuoto")
    ax2.plot(tt, G * tt, "k--", lw=1, label="nel vuoto")
    ax1.set_xlabel("$t$ [s]"); ax1.set_ylabel("$z$ [m]"); ax1.set_ylim(0, 500)
    ax2.set_xlabel("$t$ [s]"); ax2.set_ylabel("$v$ [m/s]"); ax2.set_ylim(0, 100)
    ax1.legend(fontsize=7); ax2.legend(fontsize=7, loc="upper left")
    salva(fig1, "prog1_1_z")
    salva(fig2, "prog1_1_v")

    # -------------------------------------------- curva c_d(Re)
    Re = np.logspace(-1, 7, 500)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.loglog(Re, [cd_sfera(r) for r in Re], color=PALETTE[0])
    for r, lab in ((1, "a"), (400, "b"), (3e5, "c-d"), (2e6, "e")):
        ax.axvline(r, color="gray", lw=0.5, ls=":")
        ax.text(r, 300, lab, ha="center", fontsize=7)
    ax.set_xlabel("$Re = v d/\\nu$"); ax.set_ylabel("$c_d$")
    salva(fig, "prog1_1_cd")
