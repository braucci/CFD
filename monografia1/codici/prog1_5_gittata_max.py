"""
prog1_5_gittata_max.py  --  Programma 1.5
Angolo di tiro ottimo (gittata massima) di un proiettile sferico
con il metodo del dimezzamento dell'intervallo.
"""
import numpy as np
import matplotlib.pyplot as plt
from prog1_4_proiettile import modello, traiettoria, gittata
from stile_grafici import salva, BLU, ARANCIO, VERDE


def gittata_massima(f, w0, h=0.1, theta_old=0.0, xr_old=0.0, delta=10.0, eps=1e-3,
                    verbose=False):
    """
    Metodo del dimezzamento dell'intervallo (Sez. 1.5):
    theta_new = theta_old + delta; se la gittata non cresce, delta <- -delta/2.
    Restituisce (theta_ottimo, x_r massima, numero di valutazioni).
    """
    n = 0
    while abs(delta) >= eps:
        theta_new = theta_old + delta
        n += 1
        _, sP, sQ = traiettoria(f, w0, theta_new, h)
        xr_new = gittata(sP, sQ)
        if verbose:
            print(f"    n={n:2d}  theta={theta_new:9.4f}  x_r={xr_new:9.4f}  delta={delta:+9.4f}")
        if xr_old >= xr_new:
            delta = -delta / 2.0
        else:
            theta_old, xr_old = theta_new, xr_new
    if xr_old >= xr_new:
        return theta_old, xr_old, n
    return theta_new, xr_new, n


if __name__ == "__main__":
    d, w0 = 0.01, 50.0
    print(f"Proiettile sferico d'acciaio d = {d} m, w0 = {w0} m/s")
    print(f"{'u_f [m/s]':>10} {'theta_ott [deg]':>16} {'x_r max [m]':>12} {'N':>4}")
    for uf in (20.0, 0.0, -20.0):
        f = modello(d, vento=lambda x, y, t, uf=uf: (uf, 0.0))
        th, xr, n = gittata_massima(f, w0, verbose=(uf == 0.0))
        print(f"{uf:10.1f} {th:16.3f} {xr:12.2f} {n:4d}")

    # ---------------- Fig. 1.5.4: angolo ottimo in funzione del diametro
    print("\nAngolo ottimo in funzione del diametro (puo' richiedere qualche secondo)...")
    diam = np.concatenate([np.arange(0.0025, 0.0101, 0.0005),
                           np.arange(0.01125, 0.0201, 0.00125),
                           np.arange(0.025, 0.1001, 0.005)])
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for uf, c, mk, lab in ((20.0, VERDE, "s", "$u_f = +20$ m/s"), (0.0, BLU, "o", "$u_f = 0$"),
                           (-20.0, ARANCIO, "+", "$u_f = -20$ m/s")):
        angoli = []
        for dd in diam:
            f = modello(dd, vento=lambda x, y, t, uf=uf: (uf, 0.0))
            th, xr, n = gittata_massima(f, w0)
            angoli.append(th)
        ax.semilogx(diam, angoli, mk, color=c, ms=4, mfc="none", label=lab)
        print(f"  u_f = {uf:+5.1f}: theta_ott da {angoli[0]:.2f} (d={diam[0]}) a {angoli[-1]:.2f} (d={diam[-1]})")
    ax.axhline(45, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("diametro $d$ [m]"); ax.set_ylabel("$\\theta_0$ ottimo [gradi]")
    ax.set_ylim(25, 50); ax.legend(fontsize=7, loc="lower right")
    salva(fig, "prog1_5_angolo_ottimo")

    # ---------------- Problema 4.2: sfera di 0.09 m
    print("\nProblema 4.2 -- sfera d = 0.09 m:")
    for uf in (20.0, 0.0, -20.0):
        f = modello(0.09, vento=lambda x, y, t, uf=uf: (uf, 0.0))
        th, xr, n = gittata_massima(f, w0)
        print(f"  u_f = {uf:+5.1f}: theta_ott = {th:6.2f} deg, x_r max = {xr:7.2f} m, "
              f"Re iniziale = {abs(w0*np.cos(np.radians(th)) - uf)*0.09/1.49e-5:.0f}")
