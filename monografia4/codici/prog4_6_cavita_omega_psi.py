"""
prog4_6_cavita_omega_psi.py -- Programma 4.6: cavita' con coperchio mobile nella
formulazione vorticita'-funzione di corrente, griglia collocata, Eulero esplicito
per l'equazione del trasporto della vorticita',
    omega_t + u omega_x + v omega_y = lap omega / Re,     lap psi = -omega,
vorticita' di parete al primo ordine (omega_w = -2 psi_1/h^2 - 2 u_w/h sul coperchio).
L'equazione di Poisson per psi e' risolta a ogni passo con il SOR oppure con il metodo
ADI di Peaceman-Rachford usato come iterazione (pseudo-tempo), con parametro fisso
rho ~ 8/N oppure con la successione di Wachspress. Griglia con M = 32 intervalli
(31 x 31 nodi interni), Re = 1, 100, 400; confronto con Ghia et al. e con il
Programma 4.5 (variabili primitive).
"""
import time
import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

GHIA = {100: (-0.103423, -3.16646), 400: (-0.113909, -2.29469), 1000: (-0.117929, -2.04968)}


def wachspress(M):
    """Successione di parametri r_k = 1/rho_k di Wachspress per il quadrato con M intervalli."""
    k0 = 2
    while (np.sqrt(2) - 1)**(k0 - 1) > np.tan(np.pi / (2 * M)):
        k0 += 1
    return [1 / (2 * np.cos(np.pi / (2 * M))**2) * (1 / np.tan(np.pi / (2 * M)))**(2 * k / (k0 - 1)) for k in range(k0)]


def tridiag_cost(a, b, c, D):
    """Risolve, per ogni colonna di D, il sistema tridiagonale a coefficienti costanti (a sotto, b diag, c sopra)."""
    n = D.shape[0]
    ab = np.zeros((3, n))
    ab[0, 1:] = c; ab[1, :] = b; ab[2, :-1] = a
    return solve_banded((1, 1), ab, D)


def poisson_adi(psi, f, h, rho, tol, it_max=500):
    """Iterazione ADI (Peaceman-Rachford) per lap psi = f (f = -omega), psi assegnata sul contorno.
    rho = h^2/dt_pseudo, scalare oppure successione ciclica. Restituisce psi e il numero di doppi passi."""
    n = psi.shape[0] - 2
    seq = np.atleast_1d(rho)
    for it in range(1, it_max + 1):
        r = seq[(it - 1) % len(seq)]
        vecchio = psi.copy()
        # mezzo passo implicito in x (sistemi lungo le righe j fissato): psi_(i+1) - (2+r) psi_i + psi_(i-1) = rhs
        rhs = -(psi[1:-1, 2:] + psi[1:-1, :-2]) + (2 - r) * psi[1:-1, 1:-1] + h * h * f[1:-1, 1:-1]
        rhs[0, :] -= psi[0, 1:-1]; rhs[-1, :] -= psi[-1, 1:-1]
        psi_s = psi.copy()
        psi_s[1:-1, 1:-1] = tridiag_cost(1.0, -(2 + r), 1.0, rhs)
        # mezzo passo implicito in y (sistemi lungo le colonne i fissato)
        rhs = -(psi_s[2:, 1:-1] + psi_s[:-2, 1:-1]) + (2 - r) * psi_s[1:-1, 1:-1] + h * h * f[1:-1, 1:-1]
        rhs[:, 0] -= psi[1:-1, 0]; rhs[:, -1] -= psi[1:-1, -1]
        psi[1:-1, 1:-1] = tridiag_cost(1.0, -(2 + r), 1.0, rhs.T).T
        if np.max(np.abs(psi - vecchio)) < tol:
            return psi, it
    return psi, it_max


def poisson_sor(psi, f, h, omega_rf, tol, it_max=5000):
    n = psi.shape[0]
    for it in range(1, it_max + 1):
        err = 0.0
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                nuovo = (1 - omega_rf) * psi[i, j] + omega_rf / 4 * (
                    psi[i - 1, j] + psi[i + 1, j] + psi[i, j - 1] + psi[i, j + 1] - h * h * f[i, j])
                err = max(err, abs(nuovo - psi[i, j]))
                psi[i, j] = nuovo
        if err < tol:
            return psi, it
    return psi, it_max


def cavita_omega_psi(Re, M, t_max, risolutore="adi", rho=None, tol_psi=1e-6, tol_stat=1e-5, cfl=0.4):
    h = 1.0 / M
    n = M + 1
    x = np.linspace(0, 1, n)
    psi = np.zeros((n, n)); om = np.zeros((n, n)); u = np.zeros((n, n)); v = np.zeros((n, n))
    u[:, -1] = 1.0
    dt = min(cfl * 0.25 * Re * h * h, cfl * h)
    if rho is None:
        rho = 8.0 / M
    a = 2 * np.cos(np.pi / M)
    om_sor = (8 - 4 * np.sqrt(4 - a * a)) / (a * a)
    t, passo, res, it_tot = 0.0, 0, np.inf, 0
    t0 = time.perf_counter()
    while t < t_max and res > tol_stat:
        # vorticita' di parete (primo ordine)
        om[1:-1, 0] = -2 * psi[1:-1, 1] / h**2
        om[1:-1, -1] = -2 * psi[1:-1, -2] / h**2 - 2 * 1.0 / h
        om[0, 1:-1] = -2 * psi[1, 1:-1] / h**2
        om[-1, 1:-1] = -2 * psi[-2, 1:-1] / h**2
        # trasporto della vorticita' (Eulero esplicito, differenze centrate)
        omn = om.copy()
        omn[1:-1, 1:-1] = om[1:-1, 1:-1] + dt * (
            -u[1:-1, 1:-1] * (om[2:, 1:-1] - om[:-2, 1:-1]) / (2 * h)
            - v[1:-1, 1:-1] * (om[1:-1, 2:] - om[1:-1, :-2]) / (2 * h)
            + (om[2:, 1:-1] + om[:-2, 1:-1] + om[1:-1, 2:] + om[1:-1, :-2] - 4 * om[1:-1, 1:-1]) / (h * h * Re))
        res = np.max(np.abs(omn[1:-1, 1:-1] - om[1:-1, 1:-1])) / dt
        om = omn
        # funzione di corrente
        if risolutore == "adi":
            psi, it = poisson_adi(psi, -om, h, rho, tol_psi)
        else:
            psi, it = poisson_sor(psi, -om, h, om_sor, tol_psi)
        it_tot += it
        u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * h)
        v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * h)
        t += dt; passo += 1
    durata = time.perf_counter() - t0
    return dict(Re=Re, M=M, x=x, psi=psi, om=om, u=u, v=v, t=t, passi=passo, it_medie=it_tot / max(passo, 1),
                durata=durata, res=res, dt=dt)


if __name__ == "__main__":
    M = 32
    print(f"Cavita' in formulazione omega-psi, M = {M} intervalli ({M-1}x{M-1} nodi interni)")
    print(f"   parametri di Wachspress (M = {M}): r_k = " + ", ".join(f"{r:.4f}" for r in wachspress(M)))
    print(f"   esempio del testo, M = 11: r_k = " + ", ".join(f"{r:.8f}" for r in wachspress(11)))
    ris = {}
    for Re, tmax in ((1, 2.0), (100, 30.0), (400, 60.0)):
        r = cavita_omega_psi(Re, M, tmax, "adi")
        ris[Re] = r
        k = np.unravel_index(np.argmin(r["psi"]), r["psi"].shape)
        print(f"   Re = {Re:4d}: {r['passi']} passi (dt = {r['dt']:.2e}), t = {r['t']:.1f}, residuo {r['res']:.1e},"
              f" {r['it_medie']:.1f} doppi passi ADI per passo, {r['durata']:.1f} s")
        print(f"      psi_min = {r['psi'].min():.5f} in ({k[0]/M:.4f}, {k[1]/M:.4f}), omega_c = {r['om'][k]:.4f}"
              + (f";  Ghia: {GHIA[Re][0]:.5f}, {GHIA[Re][1]:.4f}" if Re in GHIA else ""))
    # confronto fra i risolutori per l'equazione di Poisson (un singolo problema: omega finale di Re = 100)
    print("\nRisolutori per lap psi = -omega (omega finale di Re = 100, tolleranza 1e-6 sulla variazione massima):")
    om = ris[100]["om"]; h = 1.0 / M
    for nome, fun in (("SOR, omega ottimo", lambda p: poisson_sor(p, -om, h, (8 - 4 * np.sqrt(4 - (2 * np.cos(np.pi / M))**2)) / (2 * np.cos(np.pi / M))**2, 1e-6)),
                      ("ADI, rho = 8/M fisso", lambda p: poisson_adi(p, -om, h, 8.0 / M, 1e-6)),
                      ("ADI, rho = 1 fisso", lambda p: poisson_adi(p, -om, h, 1.0, 1e-6)),
                      ("ADI, successione di Wachspress", lambda p: poisson_adi(p, -om, h, 1 / np.array(wachspress(M)), 1e-6))):
        t0 = time.perf_counter()
        p, it = fun(np.zeros((M + 1, M + 1)))
        print(f"   {nome:32s}: {it:4d} iterazioni, {time.perf_counter()-t0:.3f} s, psi_min = {p.min():.6f}")
    # la stessa cavita' con il SOR nel ciclo temporale (Re = 100) per confronto dei tempi
    r = cavita_omega_psi(100, M, 30.0, "sor")
    print(f"\n   Re = 100 con SOR nel ciclo temporale: {r['it_medie']:.1f} iterazioni per passo, {r['durata']:.1f} s"
          f" (ADI: {ris[100]['it_medie']:.1f}, {ris[100]['durata']:.1f} s); psi_min = {r['psi'].min():.5f}")

    fig, axs = plt.subplots(2, 3, figsize=(8.0, 5.0))
    for col, Re in enumerate((1, 100, 400)):
        r = ris[Re]; x = r["x"]
        lv = [-0.115, -0.11, -0.1, -0.09, -0.07, -0.05, -0.03, -0.01, -1e-4, 1e-5, 1e-4, 5e-4, 1e-3, 1.5e-3]
        axs[0, col].contour(x, x, r["psi"].T, levels=lv, colors=[ROSSO if v > 0 else BLU for v in lv], linewidths=0.7)
        lo = [-5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5]
        axs[1, col].contour(x, x, r["om"].T, levels=lo, colors=[ROSSO if v > 0 else BLU for v in lo], linewidths=0.7)
        axs[0, col].set_title(f"$\\psi$, $Re={Re}$", fontsize=8); axs[1, col].set_title(f"$\\omega$, $Re={Re}$", fontsize=8)
        for a_ in axs[:, col]:
            a_.set_aspect("equal"); a_.set_xticks([]); a_.set_yticks([])
    fig.tight_layout()
    salva(fig, "prog4_6_cavita_omega_psi")
