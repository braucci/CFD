"""
prog4_4_burgers_abcn.py -- Programma 4.4: equazione di Burgers viscosa
    u_t + u u_x = nu u_xx,   -6 <= x <= 6,
con lo schema semi-implicito AB-CN: Adams-Bashforth del secondo ordine sul termine
convettivo (forma conservativa H = -(u^2/2)_x, differenze centrate) e
Crank-Nicolson sul termine diffusivo; a ogni passo un sistema tridiagonale
    -d/2 u_(i-1) + (1 + d) u_i - d/2 u_(i+1) = RHS_i,    d = nu dt/dx^2,
risolto con l'algoritmo di Thomas oppure, poiche' la matrice e' costante,
con la fattorizzazione LU calcolata una volta sola (Esercizio 6).
Soluzione esatta di Benton-Platzman (nu = 1): u = -2 sinh x/(cosh x - e^{-t}),
condizioni di Dirichlet agli estremi date dalla soluzione esatta.
Il programma confronta i tre schemi (Eulero esplicito, AB esplicito, AB-CN),
misura l'ordine di convergenza nel tempo e il costo dei risolutori.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from metodi_numerici import thomas
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

XL, XR = -6.0, 6.0
NU = 1.0


def esatta(x, t):
    return -2.0 * np.sinh(x) / (np.cosh(x) - np.exp(-t))


def lu_tridiag(a, b, c):
    """Fattorizzazione LU di una matrice tridiagonale (a sotto, b diag, c sopra):
    L bidiagonale inferiore (diag l, sotto a), U bidiagonale superiore unitaria (sopra u)."""
    n = len(b)
    l = np.empty(n); u = np.empty(n - 1)
    l[0] = b[0]
    for j in range(1, n):
        u[j - 1] = c[j - 1] / l[j - 1]
        l[j] = b[j] - a[j] * u[j - 1]
    return l, u


def lu_risolvi(a, l, u, f):
    n = len(l)
    z = np.empty(n); x = np.empty(n)
    z[0] = f[0] / l[0]
    for j in range(1, n):
        z[j] = (f[j] - a[j] * z[j - 1]) / l[j]
    x[-1] = z[-1]
    for j in range(n - 2, -1, -1):
        x[j] = z[j] - u[j] * x[j + 1]
    return x


def H(u, dx):
    """H = -(u^2/2)_x con differenze centrate (nodi interni)."""
    return -(u[2:]**2 - u[:-2]**2) / (4 * dx)


def integra(schema, nx, dt, t_fin, t0=0.01, risolutore="thomas"):
    """Integra da t0 (dato iniziale esatto, regolare) a t_fin. schema: 'eulero', 'ab', 'abcn'."""
    x = np.linspace(XL, XR, nx); dx = x[1] - x[0]
    u = esatta(x, t0)
    d = NU * dt / dx**2
    n = int(round((t_fin - t0) / dt))
    t = t0
    Hprec = None
    n_int = nx - 2
    if schema == "abcn":
        a = np.full(n_int, -d / 2); b = np.full(n_int, 1 + d); c = np.full(n_int, -d / 2)
        if risolutore == "lu":
            l, uu = lu_tridiag(a, b, c)
    for _ in range(n):
        Hn = H(u, dx)
        lap = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        if Hprec is None or schema == "eulero":
            conv = Hn                                       # Eulero esplicito (primo passo)
        else:
            conv = 1.5 * Hn - 0.5 * Hprec                   # Adams-Bashforth
        t_new = t + dt
        if schema in ("eulero", "ab"):
            un = u.copy()
            un[1:-1] = u[1:-1] + dt * (conv + NU * lap)
        else:
            rhs = u[1:-1] + dt * conv + 0.5 * NU * dt * lap
            rhs[0] += 0.5 * d * esatta(XL, t_new)           # condizioni di Dirichlet al nuovo livello
            rhs[-1] += 0.5 * d * esatta(XR, t_new)
            un = np.empty_like(u)
            un[1:-1] = thomas(a, b, c, rhs) if risolutore == "thomas" else lu_risolvi(a, l, uu, rhs)
        un[0] = esatta(XL, t_new); un[-1] = esatta(XR, t_new)
        Hprec = Hn
        u = un
        t = t_new
        if not np.all(np.isfinite(u)) or np.max(np.abs(u)) > 1e3:
            return x, None, t
    return x, u, t


if __name__ == "__main__":
    T_FIN = 1.0
    nx = 101
    x = np.linspace(XL, XR, nx); dx = x[1] - x[0]
    print(f"Burgers viscosa, nu = {NU}, {nx} nodi (dx = {dx:.3f}), da t = 0,01 a t = {T_FIN}")
    print("   schema    dt       C=u dt/dx   d=nu dt/dx^2   errore max")
    for schema in ("eulero", "ab", "abcn"):
        for dt in (0.002, 0.005, 0.01, 0.02, 0.05):
            x, u, t = integra(schema, nx, dt, T_FIN)
            C = 2 * dt / dx; d = NU * dt / dx**2
            err = "instabile" if u is None else f"{np.max(np.abs(u - esatta(x, t))):.3e}"
            print(f"   {schema:7s} {dt:6.3f}    {C:6.3f}       {d:6.3f}       {err}")

    print("\nOrdine di convergenza nel tempo (AB-CN, 401 nodi; errore rispetto alla soluzione con dt = 1e-4")
    print("sulla stessa griglia, che isola l'errore temporale da quello spaziale, pari a 1,4e-4):")
    x_rif, u_rif, _ = integra("abcn", 401, 1e-4, T_FIN)
    err_prec = None
    errs = []
    for dt in (0.02, 0.01, 0.005, 0.0025, 0.00125):
        x, u, t = integra("abcn", 401, dt, T_FIN)
        err = np.max(np.abs(u - u_rif))
        errs.append(err)
        print(f"   dt = {dt:8.5f}: errore {err:.3e}" + (f", ordine {np.log2(err_prec/err):.2f}" if err_prec else ""))
        err_prec = err
    print(f"   errore spaziale residuo (dt = 1e-4 contro soluzione esatta): {np.max(np.abs(u_rif - esatta(x_rif, T_FIN))):.2e}")

    print("\nThomas contro LU (fattorizzazione una volta sola), AB-CN con 2001 nodi, 1000 passi:")
    for ris in ("thomas", "lu"):
        t0 = time.perf_counter()
        x, u, t = integra("abcn", 2001, 0.001, T_FIN, risolutore=ris)
        print(f"   {ris:7s}: {time.perf_counter()-t0:6.2f} s, errore {np.max(np.abs(u - esatta(x, t))):.3e}")

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    xe = np.linspace(XL, XR, 400)
    for t, col in zip((0.01, 0.25, 1.0), (GRIGIO, ARANCIO, BLU)):
        ax[0].plot(xe, esatta(xe, t), color=col, label=f"$t={t:g}$".replace(".", "{,}"))
    x, u, t = integra("abcn", 51, 0.02, T_FIN)
    ax[0].plot(x, u, "o", color=BLU, ms=2.5, mfc="none", label="AB--CN, 51 nodi")
    ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$u$"); ax[0].legend(fontsize=7)
    dts = np.array([0.02, 0.01, 0.005, 0.0025, 0.00125])
    ax[1].loglog(dts, errs, "o-", color=BLU, label="AB--CN")
    ax[1].loglog(dts, errs[0] * (dts / dts[0])**2, ":", color=GRIGIO, label="$\\Delta t^2$")
    ax[1].set_xlabel("$\\Delta t$"); ax[1].set_ylabel("errore massimo a $t=1$"); ax[1].legend(fontsize=7)
    fig.tight_layout()
    salva(fig, "prog4_4_burgers_abcn")
