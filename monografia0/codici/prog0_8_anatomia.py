"""
prog0_8_anatomia.py -- Programma 0.8: anatomia di un programma delle monografie.

Il "programma modello" del corso, nella forma che tutti i programmi delle monografie
seguono: (1) docstring con il problema, le equazioni e lo schema; (2) import;
(3) costanti in maiuscolo; (4) funzioni piccole e riutilizzabili con docstring;
(5) blocco  if __name__ == "__main__"  con le sezioni: calcolo, verifica contro una
soluzione esatta, ordine di convergenza, tabella stampata, figure salvate.

Problema: equazione della diffusione u_t = nu u_xx su 0 <= x <= 1, u = 0 agli estremi,
u(x, 0) = sin(pi x); soluzione esatta u = sin(pi x) exp(-nu pi^2 t). Schema esplicito
FTCS, u_i^(n+1) = u_i^n + R (u_(i+1)^n - 2 u_i^n + u_(i-1)^n), R = nu dt/h^2 <= 1/2.
Verifiche: errore O(h^2) a R fissato (l'ordine misurato e' 2), estrapolazione di
Richardson, instabilita' per R > 1/2, caso speciale R = 1/6 (ordine 4).
Uso:  python3 prog0_8_anatomia.py [R]      (R predefinito 0,4)
"""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

NU = 1.0                      # diffusivita'
T_FIN = 0.1                   # istante finale
R_DEFAULT = 0.4               # numero di diffusione nu dt/h^2


def esatta(x, t, nu=NU):
    """Soluzione esatta del problema modello."""
    return np.sin(np.pi * x) * np.exp(-nu * np.pi**2 * t)


def passo_ftcs(u, R):
    """Un passo dello schema esplicito; u e' modificato sul posto nei nodi interni."""
    u[1:-1] = u[1:-1] + R * (u[2:] - 2 * u[1:-1] + u[:-2])
    return u


def integra(n, R, t_fin=T_FIN, nu=NU):
    """Integra su n intervalli (n+1 nodi) con numero di diffusione R fino a t_fin.
    Restituisce x, u(x, t_fin), il tempo effettivo raggiunto e il numero di passi."""
    x = np.linspace(0.0, 1.0, n + 1)
    h = x[1] - x[0]
    dt = R * h**2 / nu
    n_passi = int(np.ceil(t_fin / dt))           # ultimo passo accorciato per arrivare esattamente a t_fin
    u = np.sin(np.pi * x)
    for k in range(n_passi):
        dt_k = min(dt, t_fin - k * dt)
        passo_ftcs(u, nu * dt_k / h**2)
    return x, u, t_fin, n_passi


def errore(n, R):
    """Errore massimo a t_fin rispetto alla soluzione esatta."""
    x, u, t, _ = integra(n, R)
    return np.max(np.abs(u - esatta(x, t)))


if __name__ == "__main__":
    R = float(sys.argv[1]) if len(sys.argv) > 1 else R_DEFAULT
    print(f"Diffusione 1D con lo schema FTCS, R = {R}, t_fin = {T_FIN}")

    # --- 1) un calcolo e la sua verifica
    n = 40
    t0 = time.perf_counter()
    x, u, t, n_passi = integra(n, R)
    durata = time.perf_counter() - t0
    err = np.max(np.abs(u - esatta(x, t)))
    print(f"   n = {n}: {n_passi} passi in {durata*1e3:.1f} ms; errore massimo {err:.3e}")
    print(f"   {'x':>6} {'numerica':>10} {'esatta':>10} {'errore':>10}")
    for i in range(0, n + 1, 8):
        print(f"   {x[i]:6.2f} {u[i]:10.6f} {esatta(x[i], t):10.6f} {u[i] - esatta(x[i], t):10.2e}")

    # --- 2) ordine di convergenza (R fissato: dt ~ h^2, l'errore totale e' O(h^2))
    print("\n   Ordine di convergenza (R fissato):")
    ns = [10, 20, 40, 80, 160]
    errs = [errore(n, R) for n in ns]
    for n, e, e_prec in zip(ns, errs, [None] + errs[:-1]):
        print(f"   n = {n:4d}: errore {e:.3e}" + (f", rapporto {e_prec / e:.2f}" if e_prec else ""))
    ordine = np.polyfit(np.log(ns), np.log(errs), 1)[0]
    print(f"   pendenza nel piano log-log: {-ordine:.2f} (atteso 2; con R = 1/6 si ottiene 4)")

    # --- 3) estrapolazione di Richardson con le soluzioni n = 80 e n = 160
    x80, u80, _, _ = integra(80, R)
    x160, u160, _, _ = integra(160, R)
    u_rich = (4 * u160[::2] - u80) / 3            # combinazione che elimina il termine O(h^2)
    print(f"\n   Richardson: errore n = 160: {np.max(np.abs(u160 - esatta(x160, T_FIN))):.3e}, "
          f"estrapolato: {np.max(np.abs(u_rich - esatta(x80, T_FIN))):.3e}")

    # --- 4) stabilita'
    print("\n   Stabilita' (n = 40): errore per vari R")
    for Rr in (0.1, 1 / 6, 0.3, 0.5, 0.51, 0.6):
        with np.errstate(over="ignore", invalid="ignore"):
            e = errore(40, Rr)
        print(f"   R = {Rr:.3f}: " + ("instabile" if not np.isfinite(e) or e > 1 else f"errore {e:.3e}"))

    # --- 5) figura
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
    xe = np.linspace(0, 1, 200)
    ax[0].plot(xe, esatta(xe, T_FIN), color=GRIGIO, label="esatta")
    ax[0].plot(x, u, "o", color=BLU, ms=3, mfc="none", label=f"FTCS, $n={n}$")
    ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$u(x, t_{\\mathrm{fin}})$"); ax[0].legend(fontsize=7)
    ax[1].loglog(ns, errs, "o-", color=BLU, label=f"$R={R}$")
    ax[1].loglog(ns, [errore(n, 1 / 6) for n in ns], "s-", color=ARANCIO, label="$R=1/6$")
    ax[1].loglog(ns, errs[0] * (np.array(ns) / ns[0]) ** -2.0, ":", color=GRIGIO, label="pendenza $-2$")
    ax[1].set_xlabel("$n$"); ax[1].set_ylabel("errore massimo"); ax[1].legend(fontsize=7)
    fig.tight_layout()
    salva(fig, "prog0_8_anatomia")
