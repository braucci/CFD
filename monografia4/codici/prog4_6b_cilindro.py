"""
prog4_6b_cilindro.py -- corrente uniforme intorno a un cilindro circolare a basso
numero di Reynolds (2 <= Re <= 40, Re = U d/nu), formulazione vorticita'-funzione
di corrente nelle coordinate trasformate di Ta Phuoc Loc
    r/a = exp(pi xi),   theta = pi eta,          g(xi) = pi^2 exp(2 pi xi),
    omega_t + (psi_eta omega_xi - psi_xi omega_eta)/g = (2/Re) (omega_xixi + omega_etaeta)/g,
    psi_xixi + psi_etaeta = -g omega.
Meta' superiore del dominio (0 <= eta <= 1), 0 <= xi <= xi_inf = 1 (r_inf = e^pi = 23,1 a).
Condizioni: psi = omega = 0 sull'asse di simmetria; sulla superficie psi = 0 e
omega = -2 psi_(i,2)/(pi^2 h^2); sul contorno esterno omega = 0 e psi = corrente
potenziale 2 sin(pi eta) sinh(pi xi_inf). Avviamento impulsivo (psi potenziale,
omega = 0). Eulero esplicito con differenze centrate per omega; l'equazione di
Poisson, a coefficienti costanti, e' risolta a ogni passo con la fattorizzazione
LU sparsa calcolata una volta sola.
Risultati: Re di separazione, lunghezza della bolla L/d, angolo di separazione,
coefficiente di resistenza (attrito + pressione) dalla (4.6.60), confronto con
Dennis e Chang (1970) e Fornberg (1980).
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

XI_INF = 1.0
# Dennis e Chang (1970): Re, C_D, lunghezza della scia L/a dal punto posteriore, angolo di separazione (gradi)
DENNIS = {5: (4.116, None, None), 7: (3.421, None, None), 10: (2.846, 0.53, 29.6),
          20: (2.045, 1.88, 43.7), 40: (1.522, 4.69, 53.8)}
FORNBERG = {20: (2.0001, 1.82), 40: (1.4980, 4.48)}   # Re: (C_D, L/a)


def matrice_poisson(N, Nx, h):
    """psi_xixi + psi_etaeta sui nodi interni (i = 1..N-1 in eta, j = 1..Nx-1 in xi), Dirichlet ovunque."""
    n, nx = N - 1, Nx - 1
    idx = lambda i, j: (i - 1) * nx + (j - 1)
    A = sp.lil_matrix((n * nx, n * nx))
    for i in range(1, N):
        for j in range(1, Nx):
            p = idx(i, j)
            A[p, p] = -4.0 / h**2
            for ii, jj in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 1 <= ii <= N - 1 and 1 <= jj <= Nx - 1:
                    A[p, idx(ii, jj)] = 1.0 / h**2
    return spla.splu(A.tocsc())


def cilindro(Re, N=60, t_max=200.0, tol=1e-5, cfl=0.5, stampa=True, xi_inf=XI_INF):
    """N intervalli in eta (h = 1/N); in xi lo stesso passo h fino a xi_inf (Nx = xi_inf N intervalli)."""
    h = 1.0 / N
    Nx = int(round(xi_inf * N))
    eta = np.linspace(0, 1, N + 1); xi = h * np.arange(Nx + 1)
    g = np.pi**2 * np.exp(2 * np.pi * xi)[None, :]           # g(xi), forma (1, Nx+1)
    psi_pot = 2 * np.sin(np.pi * eta)[:, None] * np.sinh(np.pi * xi)[None, :]
    psi = psi_pot.copy()
    om = np.zeros((N + 1, Nx + 1))
    LU = matrice_poisson(N, Nx, h)
    # passo temporale: limite diffusivo alla parete (g = pi^2) e limite convettivo
    dt_diff = Re * np.pi**2 * h**2 / 8
    t, passo, res = 0.0, 0, np.inf
    t0 = time.perf_counter()
    storia = []
    while t < t_max and res > tol:
        # vorticita' sulla superficie (primo ordine) e sul contorno esterno / asse
        om[:, 0] = -2 * psi[:, 1] / (np.pi**2 * h**2)
        om[0, :] = om[-1, :] = 0.0; om[:, -1] = 0.0
        psi_xi = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * h)
        psi_eta = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * h)
        gi = g[:, 1:-1]
        vel = max(np.max(np.abs(psi_xi / gi)), np.max(np.abs(psi_eta / gi)))
        dt = cfl * min(dt_diff, h / max(vel, 1e-12))
        om_xi = (om[1:-1, 2:] - om[1:-1, :-2]) / (2 * h)
        om_eta = (om[2:, 1:-1] - om[:-2, 1:-1]) / (2 * h)
        lap = (om[1:-1, 2:] + om[1:-1, :-2] + om[2:, 1:-1] + om[:-2, 1:-1] - 4 * om[1:-1, 1:-1]) / h**2
        incr = (-(psi_eta * om_xi - psi_xi * om_eta) + 2.0 / Re * lap) / gi
        om[1:-1, 1:-1] += dt * incr
        res = np.max(np.abs(incr))
        # funzione di corrente: psi_xixi + psi_etaeta = -g omega, contorno esterno potenziale
        rhs = -(g * om)[1:-1, 1:-1].copy()
        rhs[:, -1] -= psi_pot[1:-1, -1] / h**2
        psi[1:-1, 1:-1] = LU.solve(rhs.ravel()).reshape(N - 1, Nx - 1)
        t += dt; passo += 1
        if passo % 200 == 0:
            storia.append((t, coeff_resistenza(om, Re, h)[0], bolla(psi, xi)))
    r = analisi(psi, om, Re, h, eta, xi)
    r.update(t=t, passi=passo, res=res, durata=time.perf_counter() - t0, storia=storia, psi=psi, om=om,
             eta=eta, xi=xi, N=N, dt=dt, xi_inf=xi_inf)
    if stampa:
        print(f"   Re = {Re:5.1f}: {passo:6d} passi, t = {t:6.1f}, residuo {res:.1e} ({r['durata']:.0f} s);"
              f" C_D = {r['cd']:.3f} (attrito {r['cds']:.3f}, pressione {r['cdp']:.3f});"
              f" theta_s = {r['theta_s']:5.1f} gradi, L/a = {r['L']:.3f} (L/d = {r['L']/2:.3f})")
    return r


def coeff_resistenza(om, Re, h):
    """(4.6.60): C_D = -(4 pi/Re) int omega sin(pi eta) deta - 2 pi int P cos(pi eta) deta,
    con dP/deta = (2/Re) domega/dxi sulla superficie (4.6.61)-(4.6.62), P(eta = 0) = 0 (la costante
    additiva non contribuisce alla resistenza). Integrazione con la regola dei trapezi."""
    N = om.shape[0] - 1
    eta = np.linspace(0, 1, N + 1)
    om_w = om[:, 0]
    dom_dxi = (-3 * om[:, 0] + 4 * om[:, 1] - om[:, 2]) / (2 * h)
    P = np.concatenate([[0.0], np.cumsum(h / Re * (dom_dxi[1:] + dom_dxi[:-1]))])
    cds = -4 * np.pi / Re * np.trapezoid(om_w * np.sin(np.pi * eta), eta)
    cdp = -2 * np.pi * np.trapezoid(P * np.cos(np.pi * eta), eta)
    return cds + cdp, cds, cdp, P


def bolla(psi, xi):
    """Lunghezza della bolla di ricircolo L/a lungo l'asse posteriore (eta = 0): la velocita' sull'asse
    e' u = psi_eta/(pi r) ~ psi_(1,j)/(h pi r); la bolla termina dove psi_(1,j) cambia segno."""
    s = psi[1, :]
    j = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    if len(j) == 0:
        return 0.0
    j = j[0]
    xs = xi[j] + (xi[j + 1] - xi[j]) * (-s[j]) / (s[j + 1] - s[j])
    return np.exp(np.pi * xs) - 1.0


def analisi(psi, om, Re, h, eta, xi):
    cd, cds, cdp, P = coeff_resistenza(om, Re, h)
    om_w = om[:, 0]
    # angolo di separazione dal punto posteriore: omega_w cambia segno (omega_w > 0 nella bolla, eta piccolo)
    k = np.where((om_w[1:-1] > 0) & (om_w[2:] <= 0))[0]
    if len(k):
        k = k[0] + 1
        es = eta[k] + h * om_w[k] / (om_w[k] - om_w[k + 1])
        theta_s = 180.0 * es
    else:
        theta_s = 0.0
    return dict(Re=Re, cd=cd, cds=cds, cdp=cdp, P=P, om_w=om_w, theta_s=theta_s, L=bolla(psi, xi))


def disegna_linee(ax, r, titolo):
    eta, xi, psi = r["eta"], r["xi"], r["psi"]
    R = np.exp(np.pi * xi)[None, :]; TH = np.pi * eta[:, None]
    X = R * np.cos(TH); Y = R * np.sin(TH)
    lv_pos = [0.05, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
    lv_neg = [-0.08, -0.04, -0.02, -0.01, -0.004, -0.001]
    ax.contour(X, Y, psi, levels=lv_pos, colors=BLU, linewidths=0.7)
    ax.contour(X, Y, psi, levels=lv_neg, colors=ROSSO, linewidths=0.7, linestyles="--")
    ax.contour(X, -Y, psi, levels=lv_pos, colors=BLU, linewidths=0.7)
    ax.contour(X, -Y, psi, levels=lv_neg, colors=ROSSO, linewidths=0.7, linestyles="--")
    ax.add_patch(plt.Circle((0, 0), 1.0, color=GRIGIO))
    ax.set_xlim(-3, 10); ax.set_ylim(-3, 3); ax.set_aspect("equal")
    ax.set_title(titolo, fontsize=8); ax.set_xlabel("$x/a$"); ax.set_ylabel("$y/a$")


if __name__ == "__main__":
    N = 60
    print(f"Cilindro circolare: griglia {N+1}x{N+1} (h = 1/{N}), xi_inf = {XI_INF} (r_inf/a = {np.exp(np.pi*XI_INF):.2f})")
    print("   spaziatura radiale alla parete: Delta r/a = pi h =", f"{np.pi/N:.4f}")
    ris = {}
    for Re, tmax in ((2, 40.0), (10, 80.0), (20, 150.0), (36, 250.0)):
        ris[Re] = cilindro(Re, N, tmax)
        if Re in DENNIS:
            d = DENNIS[Re]
            print(f"             Dennis-Chang: C_D = {d[0]:.3f}" + (f", theta_s = {d[2]} gradi, L/a = {d[1]}" if d[1] else ""))
    print("\nRicerca del numero di Reynolds di separazione (prima comparsa di omega_w > 0 sul punto posteriore):")
    sep = {}
    for Re in (4.0, 5.0, 6.0, 7.0, 8.0):
        r = cilindro(Re, N, 60.0, stampa=False)
        sep[Re] = (r["om_w"][1], r["L"], r["cd"])
        print(f"   Re = {Re:3.0f}: omega_w vicino al punto posteriore = {r['om_w'][1]:+.4f}, L/a = {r['L']:.3f}, C_D = {r['cd']:.3f}"
              + (f" (Dennis-Chang {DENNIS[int(Re)][0]:.3f})" if int(Re) in DENNIS else ""))
    Res = sorted(sep); vals = [sep[R][0] for R in Res]
    k = np.where((np.array(vals[:-1]) < 0) & (np.array(vals[1:]) >= 0))[0]
    if len(k):
        k = k[0]; Re_s = Res[k] + (Res[k + 1] - Res[k]) * (-vals[k]) / (vals[k + 1] - vals[k])
        print(f"   -> Re di separazione (interpolazione lineare): {Re_s:.1f}  (esperimenti: 5-7; Dennis-Chang: 6,2)")
    print("\nPosizione del contorno esterno (Re = 20, h = 1/60):")
    for xinf in (0.75, 1.0, 1.25, 1.5):
        r = cilindro(20, N, 150.0, stampa=False, xi_inf=xinf) if xinf != 1.0 else ris[20]
        print(f"   xi_inf = {xinf:.2f} (r_inf/a = {np.exp(np.pi*xinf):6.1f}): C_D = {r['cd']:.3f}, theta_s = {r['theta_s']:.1f}, L/a = {r['L']:.3f}")
    print("\nSensibilita' alla griglia (Re = 20):")
    r40 = cilindro(20, 40, 150.0, stampa=False); r80 = cilindro(20, 80, 150.0, stampa=False)
    for r in (r40, ris[20], r80):
        print(f"   N = {r['N']:3d}: C_D = {r['cd']:.3f}, theta_s = {r['theta_s']:.1f}, L/a = {r['L']:.3f}, {r['durata']:.0f} s")

    # ------------------------------------------------------------------ figure
    fig, axs = plt.subplots(2, 2, figsize=(8.0, 4.6))
    for a_, Re in zip(axs.ravel(), (2, 10, 20, 36)):
        r = ris[Re]
        disegna_linee(a_, r, f"$Re={Re}$: $C_D={r['cd']:.2f}$, $L/d={r['L']/2:.2f}$".replace(".", "{,}"))
    fig.tight_layout()
    salva(fig, "prog4_6b_cilindro_linee")

    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.7))
    for Re, col in zip((2, 10, 20, 36), (GRIGIO, VERDE, ARANCIO, BLU)):
        r = ris[Re]
        th = 180 * r["eta"]
        ax[0].plot(th, -r["om_w"] * 2 / Re, color=col, label=f"$Re={Re}$")
        ax[1].plot(th, 2 * (r["P"] - r["P"][-1]) + 1.0 + 0 * th, color=col)   # C_p riferito al punto anteriore (ipotesi C_p = 1 ... vedi testo)
    ax[0].axhline(0, color="k", lw=0.5); ax[0].set_xlabel("$\\theta$ dal punto posteriore [gradi]")
    ax[0].set_ylabel("$c_f = -2\\omega_w/Re$"); ax[0].legend(fontsize=7); ax[0].set_title("attrito superficiale", fontsize=8)
    ax[1].set_xlabel("$\\theta$ [gradi]"); ax[1].set_ylabel("$C_p$")
    ax[1].set_title("pressione superficiale", fontsize=8)
    Re_d = sorted(DENNIS); ax[2].plot(Re_d, [DENNIS[R][0] for R in Re_d], "s", color=GRIGIO, mfc="none", label="Dennis--Chang (1970)")
    ax[2].plot(sorted(FORNBERG), [FORNBERG[R][0] for R in sorted(FORNBERG)], "^", color=GRIGIO, mfc="none", label="Fornberg (1980)")
    Re_all = sorted(list(ris) + list(sep))
    cd_all = [ris[R]["cd"] if R in ris else sep[R][2] for R in Re_all]
    ax[2].plot(Re_all, cd_all, "o-", color=BLU, ms=3, label="presente calcolo")
    ax[2].set_xscale("log"); ax[2].set_xlabel("$Re$"); ax[2].set_ylabel("$C_D$"); ax[2].legend(fontsize=7)
    ax[2].set_title("resistenza", fontsize=8)
    fig.tight_layout()
    salva(fig, "prog4_6b_cilindro_cd")
