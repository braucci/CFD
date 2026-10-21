"""
prog1_8_cannone.py  --  Problema 4.3
Proietto d'artiglieria (sfera d'acciaio equivalente, d = 0.3 m, c_d = 0.4 costante,
w0 = 800 m/s) in atmosfera a densita' costante e in atmosfera esponenziale
rho_f = 1.22 exp(-0.000118 y). Angolo ottimo e gittata massima con il metodo
del dimezzamento dell'intervallo.
"""
import numpy as np
import matplotlib.pyplot as plt
from ode_solvers import passo_rk4
from prog1_4_proiettile import traiettoria, gittata
from prog1_5_gittata_max import gittata_massima
from stile_grafici import salva, BLU, ARANCIO

RHO, G, d, CD, w0 = 8000.0, 9.8, 0.3, 0.4, 800.0


def modello_cannone(densita):
    """f(t, s) con c_d costante e densita' dell'aria funzione della quota y."""
    def f(t, s):
        x, y, u, v = s
        rb = densita(y) / RHO
        A, B, C = 1 + rb / 2, (1 - rb) * G, 3 * rb / (4 * d)
        w = np.hypot(u, v)
        return np.array([u, v, -C * CD * u * w / A, (-B - C * CD * v * w) / A])
    return f


if __name__ == "__main__":
    atmosfere = (("densita' costante", lambda y: 1.22, BLU),
                 ("atmosfera esponenziale", lambda y: 1.22 * np.exp(-0.000118 * y), ARANCIO))
    h = 0.05
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    for nome, dens, c in atmosfere:
        f = modello_cannone(dens)
        th, xr, n = gittata_massima(f, w0, h=h, delta=10.0, eps=1e-2)
        S, sP, sQ = traiettoria(f, w0, th, h)
        ax.plot(S[:, 1] / 1e3, S[:, 2] / 1e3, color=c, label=f"{nome}: $\\theta_0={th:.1f}^\\circ$")
        print(f"{nome:24s}: theta_ott = {th:6.2f} deg, gittata max = {xr/1e3:6.2f} km, "
              f"quota max = {S[:,2].max()/1e3:5.2f} km, tempo di volo ~ {S[-1,0]:5.1f} s")
    # confronto a parita' di angolo (45 gradi)
    for nome, dens, c in atmosfere:
        S, sP, sQ = traiettoria(modello_cannone(dens), w0, 45.0, h)
        print(f"  a 45 deg, {nome:22s}: gittata = {gittata(sP, sQ)/1e3:6.2f} km")
    ax.set_xlabel("$x$ [km]"); ax.set_ylabel("$y$ [km]"); ax.set_aspect("equal")
    ax.legend(fontsize=7)
    salva(fig, "prog1_8_cannone")
