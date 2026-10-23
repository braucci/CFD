"""
prog4_5c_cavita_abcn.py -- cavita' con coperchio mobile su griglia MAC con lo schema
semi-implicito AB-CN (Adams-Bashforth del secondo ordine per i termini convettivi,
Crank-Nicolson per quelli viscosi) e metodo a passi frazionari (progetto 1a):

  (I - dt/(2Re) L) u^ = u^n + dt (3/2 H^n - 1/2 H^(n-1)) + dt/(2Re) L u^n,
  lap p = div u^ / dt,      u^(n+1) = u^ - dt grad p,

con L il laplaciano discreto comprensivo delle condizioni di aderenza imposte con i
nodi fittizi (u_0 = -u_1 sul fondo, u_(N+1) = 2 U - u_N sul coperchio). I tre sistemi
lineari (u^, v^, p) hanno matrici costanti: fattorizzazione LU sparsa una volta sola.
Il passo temporale e' limitato soltanto dalla condizione convettiva (CFL), non da
quella diffusiva dt <= Re h^2/4: il vantaggio e' grande a basso Re e su griglie fini.
Confronto con il Programma 4.5 (esplicito) e con Ghia et al. (1982).
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from prog4_5_cavita_mac import matrice_pressione, GHIA, cavita
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

TABELLA = {1: (-0.099, -3.207), 100: (-0.101, -3.157), 400: (-0.107, -2.609), 1000: (-0.113, -2.191)}  # 65x65 (81x81)


def matrice_viscosa(N, h, coef, comp):
    """I - coef * L sui nodi interni della componente comp ('u': (N-1) x N nodi, 'v': N x (N-1)).
    Le pareti ortogonali alla componente sono nodi di Dirichlet (contributo nullo); su quelle
    parallele il nodo fittizio vale -(nodo interno) [+ termine noto], cioe' -1/h^2 in diagonale."""
    nx, ny = (N - 1, N) if comp == "u" else (N, N - 1)
    idx = lambda i, j: i * ny + j
    A = sp.lil_matrix((nx * ny, nx * ny))
    for i in range(nx):
        for j in range(ny):
            p = idx(i, j)
            diag = -4.0 / h**2
            for (a, b) in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < nx and 0 <= b < ny:
                    A[p, idx(a, b)] = coef * (-1.0 / h**2)
                else:
                    tangente = (comp == "u" and (b < 0 or b >= ny)) or (comp == "v" and (a < 0 or a >= nx))
                    if tangente:
                        diag -= 1.0 / h**2          # nodo fittizio = -nodo interno (+ termine noto)
            A[p, p] = 1.0 - coef * diag
    return spla.splu(A.tocsc())


def convettivi(u, v, N, h):
    """Termini convettivi conservativi H_u = -((u^2)_x + (uv)_y), H_v = -((uv)_x + (v^2)_y) nei nodi interni."""
    ue = 0.5 * (u[1:, 1:-1] + u[:-1, 1:-1]); duudx = (ue[1:, :]**2 - ue[:-1, :]**2) / h
    uc = 0.5 * (u[1:-1, 1:] + u[1:-1, :-1]); vc = 0.5 * (v[1:-1, :][1:, :] + v[1:-1, :][:-1, :])
    duvdy = (uc[:, 1:] * vc[:, 1:] - uc[:, :-1] * vc[:, :-1]) / h
    ve = 0.5 * (v[1:-1, 1:] + v[1:-1, :-1]); dvvdy = (ve[:, 1:]**2 - ve[:, :-1]**2) / h
    uc2 = 0.5 * (u[:, 1:-1][:, 1:] + u[:, 1:-1][:, :-1]); vc2 = 0.5 * (v[1:, 1:-1] + v[:-1, 1:-1])
    duvdx = (uc2[1:, :] * vc2[1:, :] - uc2[:-1, :] * vc2[:-1, :]) / h
    return -duudx - duvdy, -duvdx - dvvdy


def lap(w, h):
    return (w[2:, 1:-1] + w[:-2, 1:-1] + w[1:-1, 2:] + w[1:-1, :-2] - 4 * w[1:-1, 1:-1]) / h**2


def cavita_abcn(Re, N, t_max, dt=None, tol=1e-6, cfl=0.5, stampa=True, incrementale=False):
    """incrementale = True: forma a correzione di pressione incrementale (Kim-Moin, Van Kan): nel
    predittore compare -dt grad p^n e si risolve per l'incremento phi, p^(n+1) = p^n + phi; a regime
    phi -> 0 e l'errore di separazione (splitting) O(dt) dello schema non incrementale scompare."""
    h = 1.0 / N
    u = np.zeros((N + 1, N + 2)); v = np.zeros((N + 2, N + 1)); p = np.zeros((N, N))
    if dt is None:
        dt = cfl * h                                     # solo il limite convettivo (u_max = 1)
    coef = dt / (2 * Re)
    LUp = matrice_pressione(N, h)
    LUu = matrice_viscosa(N, h, coef, "u"); LUv = matrice_viscosa(N, h, coef, "v")
    Hu_prec = Hv_prec = None
    t, passo, res = 0.0, 0, np.inf
    t0 = time.perf_counter()
    while t < t_max and res > tol:
        u[:, 0] = -u[:, 1]; u[:, N + 1] = 2.0 - u[:, N]
        v[0, :] = -v[1, :]; v[N + 1, :] = -v[N, :]
        Hu, Hv = convettivi(u, v, N, h)
        if Hu_prec is None:
            cu, cv = Hu, Hv                               # primo passo: Eulero esplicito
        else:
            cu, cv = 1.5 * Hu - 0.5 * Hu_prec, 1.5 * Hv - 0.5 * Hv_prec
        # termini noti: parte esplicita di Crank-Nicolson (con i nodi fittizi correnti) e, nella parte
        # implicita, il contributo noto del coperchio 2U/h^2 sui nodi adiacenti (j = N)
        rhs_u = u[1:-1, 1:-1] + dt * cu + coef * lap(u, h)
        rhs_u[:, -1] += coef * 2.0 / h**2
        rhs_v = v[1:-1, 1:-1] + dt * cv + coef * lap(v, h)
        if incrementale:
            rhs_u -= dt * (p[1:, :] - p[:-1, :]) / h
            rhs_v -= dt * (p[:, 1:] - p[:, :-1]) / h
        us = u.copy(); vs = v.copy()
        us[1:-1, 1:-1] = LUu.solve(rhs_u.ravel()).reshape(N - 1, N)
        vs[1:-1, 1:-1] = LUv.solve(rhs_v.ravel()).reshape(N, N - 1)
        div = (us[1:, 1:-1] - us[:-1, 1:-1]) / h + (vs[1:-1, 1:] - vs[1:-1, :-1]) / h
        rhs = (div / dt).ravel(); rhs[0] = 0.0
        phi = LUp.solve(rhs).reshape(N, N)
        un = us.copy(); vn = vs.copy()
        un[1:-1, 1:-1] -= dt * (phi[1:, :] - phi[:-1, :]) / h
        vn[1:-1, 1:-1] -= dt * (phi[:, 1:] - phi[:, :-1]) / h
        p = p + phi if incrementale else phi
        res = max(np.max(np.abs(un - u)), np.max(np.abs(vn - v))) / dt
        Hu_prec, Hv_prec = Hu, Hv
        u, v = un, vn
        t += dt; passo += 1
        if not np.isfinite(res):
            break
    durata = time.perf_counter() - t0
    psi = np.zeros((N + 1, N + 1))
    for j in range(1, N + 1):
        psi[:, j] = psi[:, j - 1] + h * u[:, j]
    om = np.zeros((N + 1, N + 1))
    om[1:-1, 1:-1] = (v[2:-1, 1:-1] - v[1:-2, 1:-1]) / h - (u[1:-1, 2:-1] - u[1:-1, 1:-2]) / h
    k = np.unravel_index(np.argmin(psi), psi.shape)
    if stampa:
        print(f"   Re = {Re:5.0f}, griglia {N+1}x{N+1}, dt = {dt:.4f} (d = dt/(Re h^2) = {dt/(Re*h*h):.2f},"
              f" limite esplicito 0,25): {passo} passi, t = {t:.1f}, residuo {res:.1e}, {durata:.1f} s")
        print(f"      psi_min = {psi.min():.5f} in ({k[0]*h:.3f}, {k[1]*h:.3f}), omega_c = {om[k]:.4f}"
              + (f";  Tabella (65x65): {TABELLA[Re][0]:.3f}, {TABELLA[Re][1]:.3f}" if Re in TABELLA else "")
              + (f";  Ghia: {GHIA[Re][0]:.4f}, {GHIA[Re][1]:.4f}" if Re in GHIA else ""))
    return dict(Re=Re, N=N, h=h, u=u, v=v, p=p, psi=psi, om=om, t=t, passi=passo, dt=dt, durata=durata, res=res)


if __name__ == "__main__":
    print("Cavita' con coperchio mobile, griglia MAC, schema semi-implicito AB-CN a passi frazionari")
    print("Griglia 21x21 (progetto):")
    ris = {}
    for Re in (100, 400):
        ris[(Re, 20)] = cavita_abcn(Re, 20, 60.0)
    print("\nGriglia 65x65, confronto con lo schema esplicito (Programma 4.5) a parita' di griglia:")
    for Re, tmax, dt in ((1, 3.0, 0.002), (100, 30.0, None), (400, 60.0, None)):
        r = cavita_abcn(Re, 64, tmax, dt=dt)
        ris[(Re, 64)] = r
        t0 = time.perf_counter()
        e = cavita(Re, 64, tmax, stampa=False)
        print(f"      esplicito: {e['passi']} passi, {time.perf_counter()-t0:.1f} s, psi_min = {e['psi'].min():.5f}"
              f"  (rapporto fra i passi: {e['passi']/r['passi']:.0f})")
    print("\nErrore di separazione (splitting) a regime, Re = 1, 65x65 (riferimento esplicito psi_min = -0,10002):")
    print("   lo schema non incrementale produce uno strato limite numerico di spessore ~ sqrt(dt/Re)")
    for dt in (0.0005, 0.001, 0.002, 0.004, 0.008):
        with np.errstate(all="ignore"):
            r1 = cavita_abcn(1, 64, 3.0, dt=dt, stampa=False)
            r2 = cavita_abcn(1, 64, 3.0, dt=dt, stampa=False, incrementale=True)
        print(f"   dt = {dt:.4f} (d = {dt*4096:5.1f}, sqrt(dt/Re)/h = {np.sqrt(dt)*64:4.1f} celle): non incrementale psi_min = {r1['psi'].min():.5f}"
              f" (residuo {r1['res']:.1e}); incrementale psi_min = {r2['psi'].min():.5f} (residuo {r2['res']:.1e})")
    print("\nStabilita' al crescere di dt (Re = 100, 65x65): d = dt/(Re h^2), C = dt/h")
    for dt in (0.005, 0.01, 0.02, 0.03, 0.04):
        with np.errstate(all="ignore"):
            r = cavita_abcn(100, 64, 30.0, dt=dt, stampa=False)
        print(f"   dt = {dt:.3f}: d = {dt/(100/64**2):5.2f}, C = {dt*64:.2f} -> "
              + ("instabile" if not np.isfinite(r['res']) else f"psi_min = {r['psi'].min():.5f} ({r['passi']} passi, {r['durata']:.1f} s)"))

    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.9))
    for a_, key in zip(axs[:2], ((100, 20), (400, 20))):
        r = ris[key]; x = np.linspace(0, 1, r["N"] + 1)
        lv = [-0.11, -0.1, -0.09, -0.07, -0.05, -0.03, -0.01, -1e-4, 1e-5, 1e-4, 5e-4, 1e-3]
        a_.contour(x, x, r["psi"].T, levels=lv, colors=[ROSSO if v > 0 else BLU for v in lv], linewidths=0.7)
        a_.set_aspect("equal"); a_.set_title(f"$\\psi$, $Re={key[0]}$, $21\\times21$", fontsize=8)
        a_.set_xticks([0, 0.5, 1]); a_.set_yticks([0, 0.5, 1])
    from prog4_5_cavita_mac import GHIA_U
    for Re, col in ((100, BLU), (400, ARANCIO)):
        for N, ls in ((20, "--"), (64, "-")):
            r = ris[(Re, N)]
            u = r["u"]; h = r["h"]
            uc = 0.5 * (u[N // 2, :] + u[N // 2, :]) if N % 2 == 0 else None
            yv = np.linspace(0, 1, N + 1)
            u_mid = 0.5 * (u[N // 2, 1:] + u[N // 2, :-1])       # u(0,5, y) nei vertici
            axs[2].plot(u_mid, yv, ls, color=col, lw=0.9, label=f"$Re={Re}$, ${N+1}\\times{N+1}$")
        axs[2].plot(GHIA_U[Re], GHIA_U["y"], "o", color=col, ms=2.5, mfc="none")
    axs[2].set_xlabel("$u(0{,}5,y)$"); axs[2].set_ylabel("$y$"); axs[2].legend(fontsize=6)
    axs[2].set_title("profili sulla mezzeria (cerchi: Ghia et al.)", fontsize=8)
    fig.tight_layout()
    salva(fig, "prog4_5c_cavita_abcn")
