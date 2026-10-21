"""
prog2_5b_convergenza.py -- confronto fra i metodi iterativi di Jacobi
(Richardson), Gauss-Seidel (Liebmann) e del sovrarilassamento successivo (SOR)
sul problema del Programma 2.5.

1) storia dell'errore vero ||psi_k - psi*||_inf, con psi* soluzione esatta del
   sistema discreto (ottenuta con un solutore diretto per matrici sparse);
2) numero di iterazioni in funzione di omega (ricerca dell'ottimo);
3) numero di iterazioni al raffinarsi della griglia;
4) errore di discretizzazione rispetto alla soluzione del problema continuo.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO, VIOLA
from metodi_numerici import (jacobi_passo, gauss_seidel_passo, sor_passo,
                             omega_ottimo, nodi_interni_rettangolo)
import prog2_5_vortice_rettangolo as p25

GAMMA = p25.ZETA0 * p25.H**2          # circolazione del vortice (fissa)


def problema(h):
    x, y = p25.griglia(h)
    M, N = len(x), len(y)
    zeta = np.zeros((M, N))
    i0 = int(round((p25.X0 - p25.XMIN) / h)); j0 = int(round((p25.Y0 - p25.YMIN) / h))
    zeta[i0, j0] = GAMMA / h**2       # stessa circolazione su ogni griglia
    return x, y, zeta


def soluzione_diretta(zeta, h):
    """Assembla la matrice a 5 punti e risolve con un metodo diretto sparso."""
    M, N = zeta.shape
    mi, ni = M - 2, N - 2
    Tx = sp.diags([1, -2, 1], [-1, 0, 1], shape=(mi, mi), dtype=float)
    Ty = sp.diags([1, -2, 1], [-1, 0, 1], shape=(ni, ni), dtype=float)
    L = (sp.kron(Tx, sp.identity(ni)) + sp.kron(sp.identity(mi), Ty)) / h**2
    b = -zeta[1:-1, 1:-1].ravel()
    psi = np.zeros((M, N))
    psi[1:-1, 1:-1] = spla.spsolve(L.tocsc(), b).reshape(mi, ni)
    return psi


def itera(metodo, zeta, h, psi_star, tol=1e-6, omega=1.0, kmax=20000, guess=0.5):
    M, N = zeta.shape
    psi = np.zeros((M, N)); psi[1:-1, 1:-1] = guess
    interni = nodi_interni_rettangolo(M, N)
    e0 = np.max(np.abs(psi - psi_star))
    storia, somme = [e0], []
    for k in range(1, kmax + 1):
        if metodo == "jacobi":
            s = jacobi_passo(psi, zeta, h, interni)
        elif metodo == "gs":
            s = gauss_seidel_passo(psi, zeta, h, interni)
        else:
            s = sor_passo(psi, zeta, h, interni, omega)
        somme.append(s)
        storia.append(np.max(np.abs(psi - psi_star)))
        if storia[-1] < tol * e0:
            break
    return k, np.array(storia), np.array(somme)


if __name__ == "__main__":
    h = p25.H
    x, y, zeta = problema(h)
    psi_star = soluzione_diretta(zeta, h)
    m, n = len(x) - 1, len(y) - 1                 # numero di intervalli
    rJ = 0.5 * (np.cos(np.pi / m) + np.cos(np.pi / n))
    wopt = omega_ottimo(m, n)
    print(f"griglia {m} x {n} intervalli: raggio spettrale di Jacobi = {rJ:.5f}")
    print(f"  Gauss-Seidel: {rJ**2:.5f};   omega_ott = {wopt:.4f};  SOR: {wopt-1:.4f}")

    # -------- 1) storie dell'errore --------------------------------------
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    print("\n metodo      iterazioni (errore / 1e6)   previste = ln(1e6)/(-ln rho)")
    for nome, met, om, rho, col in (("Jacobi", "jacobi", 1.0, rJ, GRIGIO),
                                    ("Gauss-Seidel", "gs", 1.0, rJ**2, BLU),
                                    (f"SOR, omega = {wopt:.3f}", "sor", wopt, wopt - 1, ARANCIO)):
        k, st, so = itera(met, zeta, h, psi_star, omega=om)
        prev = np.log(1e6) / (-np.log(rho))
        print(f" {nome:24s}  {k:5d}               {prev:7.0f}")
        ax.semilogy(st / st[0], color=col,
                    label=nome.replace("omega", r"$\omega$"))
        if met == "gs":
            # quante iterazioni avrebbe fatto il criterio del Programma 2.5?
            k25 = np.argmax(so <= p25.ERRMAX) + 1
            print(f"   (criterio 'somma |delta| <= 0,001' soddisfatto all'iterazione {k25}: "
                  f"errore vero = {st[k25]:.2e}, somma = {so[k25-1]:.2e})")
    ax.set_xlabel("iterazione $k$"); ax.set_ylabel(r"$\|\psi^{(k)}-\psi^*\|_\infty / \|\psi^{(0)}-\psi^*\|_\infty$")
    ax.set_xlim(0, 1000); ax.legend()
    salva(fig, "prog2_5b_storie")

    # -------- 2) ricerca dell'omega ottimo --------------------------------
    omegas = np.arange(1.0, 1.96, 0.05)
    its = [itera("sor", zeta, h, psi_star, omega=w)[0] for w in omegas]
    k_opt = itera("sor", zeta, h, psi_star, omega=wopt)[0]
    print("\n omega   iterazioni")
    for w, k in zip(omegas, its):
        print(f"  {w:.2f}    {k:5d}")
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(omegas, its, "o-", color=BLU, ms=3.5)
    ax.plot([wopt], [k_opt], "*", color=ROSSO, ms=10, label=fr"$\omega_{{ott}}={wopt:.3f}$")
    ax.set_yscale("log"); ax.set_xlabel(r"$\omega$"); ax.set_ylabel("iterazioni")
    ax.legend()
    salva(fig, "prog2_5b_omega")

    # -------- 3) iterazioni al raffinarsi della griglia ---------------------
    print("\n    h    nodi interni   Gauss-Seidel    SOR(omega_ott)   omega_ott")
    for hh in (0.5, 0.25, 0.125):
        xg, yg, zg = problema(hh)
        ps = soluzione_diretta(zg, hh)
        mg, ng = len(xg) - 1, len(yg) - 1
        w = omega_ottimo(mg, ng)
        kgs = itera("gs", zg, hh, ps, tol=1e-4)[0]
        ksor = itera("sor", zg, hh, ps, tol=1e-4, omega=w)[0]
        print(f"  {hh:5.3f}    {(mg-1)*(ng-1):6d}        {kgs:6d}          {ksor:5d}         {w:.4f}")

    # -------- 4) errore di discretizzazione ---------------------------------
    print("\n    h     psi(0,0) discreto   psi(0,0) continuo   errore    rapporto")
    esatto = p25.psi_serie(0.0, 0.0, GAMMA, mmax=400)
    prec = None
    for hh in (0.5, 0.25, 0.125, 0.0625, 0.03125):
        xg, yg, zg = problema(hh)
        ps = soluzione_diretta(zg, hh)
        i, j = int(round(3.0 / hh)), int(round(2.0 / hh))
        e = ps[i, j] - esatto
        r = prec / e if prec else np.nan
        print(f"  {hh:7.5f}     {ps[i, j]:.6f}          {esatto:.6f}        {e:+.2e}   {r:5.2f}")
        prec = e
