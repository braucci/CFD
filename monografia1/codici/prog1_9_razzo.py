"""
prog1_9_razzo.py  --  Problema 5.1
Volo verticale di un razzo con massa variabile (RK2 a passo 0.1 s, confronto con solve_ivp).

    dz/dt = v
    (m_c + m_p) dv/dt = -(m_c + m_p) g + mdot_p v_e - 1/2 rho v|v| A_s C_D

Ipotesi sulla legge di combustione: mdot_p = m_p0/4 costante per 0 <= t <= 4 s,
nulla dopo (il propellente, pari a 1000 N di peso, si esaurisce in 4 s).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from ode_solvers import integra
from stile_grafici import salva, BLU, ARANCIO

g, rho, As, ve, CD = 9.8, 1.23, 0.1, 360.0, 0.15
mc, mp0 = 500.0 / g, 1000.0 / g
t_burn = 4.0


def mdot(t):
    return mp0 / t_burn if t <= t_burn else 0.0


def mp(t):
    return max(mp0 - mp0 * min(t, t_burn) / t_burn, 0.0)


def f(t, s):
    z, v = s
    m = mc + mp(t)
    spinta = mdot(t) * ve
    drag = 0.5 * rho * v * abs(v) * As * CD
    return np.array([v, (-m * g + spinta - drag) / m])


if __name__ == "__main__":
    h = 0.1
    tocca_terra = lambda t, s: s[0] < 0.0 and t > 1.0
    T, S = integra(f, 0.0, [0.0, 0.0], h, 2000, metodo="rk2", stop=tocca_terra)
    z, v = S[:, 0], S[:, 1]
    i_vmax, i_zmax = np.argmax(v), np.argmax(z)
    print(f"Spinta iniziale = {mdot(0)*ve:.1f} N, peso iniziale = {(mc+mp0)*g:.0f} N "
          f"(rapporto spinta/peso = {mdot(0)*ve/((mc+mp0)*g):.2f})")
    print(f"a) v_max = {v[i_vmax]:.2f} m/s a t = {T[i_vmax]:.1f} s, z = {z[i_vmax]:.1f} m")
    print(f"b) z_max = {z[i_zmax]:.1f} m a t = {T[i_zmax]:.1f} s")
    print(f"c) impatto a t ~ {T[-1]:.1f} s con v = {v[-1]:.2f} m/s")

    # confronto con solve_ivp (RK45 adattivo) -- analogo di ODE45
    ev = lambda t, s: s[0]
    ev.terminal, ev.direction = True, -1
    sol = solve_ivp(f, [0.5, 200.0], S[5], method="RK45", events=ev,
                    max_step=0.1, rtol=1e-8, atol=1e-10, dense_output=True)
    print(f"d) solve_ivp: impatto a t = {sol.t_events[0][0]:.3f} s, "
          f"v = {sol.y_events[0][0][1]:.3f} m/s, z_max = {sol.y[0].max():.2f} m")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.6))
    a1.plot(T, z, color=BLU, label="RK2, $h=0.1$ s")
    a1.plot(sol.t, sol.y[0], "--", color=ARANCIO, label="solve\\_ivp (RK45)")
    a2.plot(T, v, color=BLU); a2.plot(sol.t, sol.y[1], "--", color=ARANCIO)
    a1.set_xlabel("$t$ [s]"); a1.set_ylabel("$z$ [m]"); a1.legend(fontsize=7)
    a2.set_xlabel("$t$ [s]"); a2.set_ylabel("$v$ [m/s]")
    salva(fig, "prog1_9_razzo")
