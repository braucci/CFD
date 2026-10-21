"""
prog1_4_proiettile.py  --  Programma 1.4
Moto bidimensionale di un proiettile sferico in un fluido (RK4 per sistemi).

Stato s = [x, y, u, v];  vento (uf, vf) assegnato dalla funzione vento(x, y, t).

    d2x/dt2 = FX = C c_d (uf - u) w_r / A
    d2y/dt2 = FY = [ -B + C c_d (vf - v) w_r ] / A ,   w_r = |w_f - w|
"""
import numpy as np
import matplotlib.pyplot as plt
from ode_solvers import passo_rk4, cd_sfera
from stile_grafici import salva, PALETTE, BLU, ARANCIO, VERDE

RHO, RHOF, NU, G = 8000.0, 1.22, 1.49e-5, 9.8


def modello(d, rhof=RHOF, vento=lambda x, y, t: (0.0, 0.0)):
    """Costruisce f(t, s) per una sfera di diametro d in un fluido di densita' rhof."""
    rb = rhof / RHO
    A, B, C = 1 + rb / 2, (1 - rb) * G, 3 * rb / (4 * d)

    def f(t, s):
        x, y, u, v = s
        uf, vf = vento(x, y, t)
        wr = np.hypot(uf - u, vf - v)
        cd = cd_sfera(wr * d / NU)
        return np.array([u, v,
                         C * cd * (uf - u) * wr / A,
                         (-B + C * cd * (vf - v) * wr) / A])
    return f


def traiettoria(f, w0, theta0_deg, h=0.1):
    """Integra finche' il proiettile non ricade a y <= 0; restituisce la storia e i punti P, Q."""
    th = np.radians(theta0_deg)
    t, s = 0.0, np.array([0.0, 0.0, w0 * np.cos(th), w0 * np.sin(th)])
    storia = [(t, *s)]
    while s[1] >= 0.0:
        s = passo_rk4(f, t, s, h)
        t += h
        storia.append((t, *s))
    sQ = s
    sP = passo_rk4(f, t, s, -h)            # un passo all'indietro: punto P
    return np.array(storia), sP, sQ


def gittata(sP, sQ):
    """Eq. (1.5.3): intersezione della corda PQ con l'asse x."""
    xP, yP, xQ, yQ = sP[0], sP[1], sQ[0], sQ[1]
    return (xQ * yP - xP * yQ) / (yP - yQ)


if __name__ == "__main__":
    d, w0 = 0.05, 50.0
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    print(f"Sfera d'acciaio d = {d} m, w0 = {w0} m/s, h = 0.1 s")
    for th, c in zip((30, 45, 60), (BLU, ARANCIO, VERDE)):
        S, sP, sQ = traiettoria(modello(d), w0, th)
        Sv, sPv, sQv = traiettoria(modello(d, rhof=0.0), w0, th)
        ax.plot(S[:, 1], S[:, 2], color=c, label=f"$\\theta_0 = {th}^\\circ$")
        ax.plot(Sv[:, 1], Sv[:, 2], "--", color=c, lw=0.9)
        xr_teor = w0 ** 2 * np.sin(2 * np.radians(th)) / G
        print(f"  theta0 = {th:2d}: in aria  x_r = {gittata(sP, sQ):7.2f} m, "
              f"y_max = {S[:,2].max():6.2f} m, t_volo = {S[-1,0]:5.2f} s | "
              f"nel vuoto x_r = {gittata(sPv, sQv):7.2f} m (teorico {xr_teor:.2f})")
    ax.plot([], [], "k--", lw=0.9, label="nel vuoto")
    ax.set_xlabel("$x$ [m]"); ax.set_ylabel("$y$ [m]")
    ax.set_xlim(0, 300); ax.set_ylim(0, 100); ax.set_aspect("equal")
    ax.legend(fontsize=7)
    salva(fig, "prog1_4_traiettorie")

    # Problema 4.1: vento di 20 m/s a favore e contro (theta0 = 45)
    print("\nProblema 4.1 -- vento orizzontale, theta0 = 45 deg:")
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for uf, c in ((20.0, VERDE), (0.0, BLU), (-20.0, ARANCIO)):
        S, sP, sQ = traiettoria(modello(d, vento=lambda x, y, t, uf=uf: (uf, 0.0)), w0, 45)
        ax.plot(S[:, 1], S[:, 2], color=c, label=f"$u_f = {uf:+.0f}$ m/s")
        print(f"  u_f = {uf:+5.1f} m/s -> x_r = {gittata(sP, sQ):7.2f} m")
    ax.set_xlabel("$x$ [m]"); ax.set_ylabel("$y$ [m]"); ax.set_aspect("equal")
    ax.legend(fontsize=7)
    salva(fig, "prog1_4_vento")

    # verifica di reversibilita': integrazione all'indietro fino a t = 0
    S, sP, sQ = traiettoria(modello(d), w0, 45)
    f = modello(d)
    t, s = S[-1, 0], S[-1, 1:].copy()
    n = len(S) - 1
    for _ in range(n):
        s = passo_rk4(f, t, s, -0.1); t -= 0.1
    print(f"\nIntegrazione all'indietro di {n} passi: t = {t:.2e}, "
          f"x = {s[0]:.2e} m, y = {s[1]:.2e} m, u = {s[2]:.6f}, v = {s[3]:.6f}")
