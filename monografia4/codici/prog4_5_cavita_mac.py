"""
prog4_5_cavita_mac.py -- Programma 4.5: cavita' quadrata con coperchio mobile,
equazioni di Navier-Stokes in variabili primitive, griglia sfalsata MAC e metodo
a passi frazionari con avanzamento esplicito di Eulero.

Passo 1 (predizione): u^ = u^n + dt [ -( (u^2)_x + (uv)_y ) + lap u / Re ],  v^ analoga.
Passo 2 (pressione):  lap p = div u^ / dt,  dp/dn = 0 sul contorno (equazione di Poisson di
                      Neumann: matrice singolare, si fissa p = 0 in una cella; la griglia MAC
                      soddisfa automaticamente la condizione di compatibilita').
Passo 3 (correzione): u^(n+1) = u^ - dt grad p,  con div u^(n+1) = 0 a meno dell'arrotondamento.
Le quantita' non definite nel nodo in cui servono sono ottenute per media; le condizioni di
aderenza sulle pareti parallele alla velocita' si impongono con i nodi fittizi
(u_(-1) = 2 u_parete - u_1). L'equazione di Poisson e' risolta con un metodo diretto sparso
(fattorizzazione LU calcolata una sola volta). Si calcolano funzione di corrente e vorticita'
nei vertici delle celle e si confrontano i valori al centro del vortice principale con
i dati di Ghia, Ghia e Shin (1982).
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

GHIA = {   # psi_min, omega al centro, x_c, y_c (Ghia, Ghia e Shin 1982, griglia 129x129)
    100: (-0.103423, -3.16646, 0.6172, 0.7344),
    400: (-0.113909, -2.29469, 0.5547, 0.6055),
    1000: (-0.117929, -2.04968, 0.5313, 0.5625),
}
# profilo u(0,5, y) di Ghia et al. per Re = 100, 400, 1000 (alcuni punti)
GHIA_U = {
    "y": [0.0, 0.0547, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9766, 1.0],
    100: [0.0, -0.03717, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.84123, 1.0],
    400: [0.0, -0.08186, -0.14612, -0.24299, -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093, 0.55892, 0.75837, 1.0],
    1000: [0.0, -0.18109, -0.29730, -0.38289, -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304, 0.46604, 0.65928, 1.0],
}


def matrice_pressione(N, h):
    """Laplaciano a cinque punti sulle celle di pressione con dp/dn = 0 (nodi fittizi
    uguali a quelli interni); la prima cella e' fissata a p = 0."""
    n = N * N
    idx = lambda i, j: i * N + j
    A = sp.lil_matrix((n, n))
    for i in range(N):
        for j in range(N):
            p = idx(i, j)
            diag = 0.0
            for (a, b) in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < N and 0 <= b < N:
                    A[p, idx(a, b)] = 1.0 / h**2
                    diag -= 1.0 / h**2
            A[p, p] = diag
    A[0, :] = 0.0; A[0, 0] = 1.0                     # livello di riferimento della pressione
    return spla.splu(A.tocsc())


def cavita(Re, N, t_max, tol=1e-6, u_cop=1.0, cfl=0.4, stampa=True, sforzo=None, u_max=1.0):
    """Griglia MAC con N celle per lato. u: (N+1) x (N+2) (colonne = nodi u in x, righe
    fittizie j = 0 e j = N+1); v: (N+2) x (N+1); p: N x N.
    Se 'sforzo' non e' None, sulla superficie superiore si impone du/dy = sforzo (superficie
    libera mossa dal vento) invece della velocita' u_cop; u_max serve per il passo temporale."""
    h = 1.0 / N
    u = np.zeros((N + 1, N + 2)); v = np.zeros((N + 2, N + 1)); p = np.zeros((N, N))
    LU = matrice_pressione(N, h)
    dt = min(cfl * 0.25 * Re * h * h, cfl * 0.5 * h / max(u_max, 1e-9))
    t, passo, res = 0.0, 0, np.inf
    storia = []
    t0 = time.perf_counter()
    while t < t_max and res > tol:
        # nodi fittizi: aderenza sulle pareti orizzontali (per u) e verticali (per v)
        u[:, 0] = -u[:, 1]                           # fondo fermo
        u[:, N + 1] = (2 * u_cop - u[:, N]) if sforzo is None else (u[:, N] + h * sforzo)   # coperchio mobile / sforzo imposto
        v[0, :] = -v[1, :]; v[N + 1, :] = -v[N, :]   # pareti laterali ferme
        # --- predizione (Eulero esplicito, forma conservativa)
        # u-momento nei nodi u interni i = 1..N-1 (x), j = 1..N (y)
        ue = 0.5 * (u[1:, 1:-1] + u[:-1, 1:-1])          # u sui centri cella (N, N)
        duudx = (ue[1:, :]**2 - ue[:-1, :]**2) / h       # (N-1, N)
        uc = 0.5 * (u[1:-1, 1:] + u[1:-1, :-1])          # u sui vertici (N-1, N+1)
        vc = 0.5 * (v[1:-1, :][1:, :] + v[1:-1, :][:-1, :])   # v sui vertici (N-1, N+1)
        duvdy = (uc[:, 1:] * vc[:, 1:] - uc[:, :-1] * vc[:, :-1]) / h
        lapu = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / h**2
        us = u.copy()
        us[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (-duudx - duvdy + lapu / Re)
        # v-momento nei nodi v interni i = 1..N (x), j = 1..N-1 (y)
        ve = 0.5 * (v[1:-1, 1:] + v[1:-1, :-1])          # v sui centri cella (N, N)
        dvvdy = (ve[:, 1:]**2 - ve[:, :-1]**2) / h       # (N, N-1)
        uc2 = 0.5 * (u[:, 1:-1][:, 1:] + u[:, 1:-1][:, :-1])  # u sui vertici (N+1, N-1)
        vc2 = 0.5 * (v[1:, 1:-1] + v[:-1, 1:-1])         # v sui vertici (N+1, N-1)
        duvdx = (uc2[1:, :] * vc2[1:, :] - uc2[:-1, :] * vc2[:-1, :]) / h
        lapv = (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1] + v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / h**2
        vs = v.copy()
        vs[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * (-duvdx - dvvdy + lapv / Re)
        # --- pressione
        div = (us[1:, 1:-1] - us[:-1, 1:-1]) / h + (vs[1:-1, 1:] - vs[1:-1, :-1]) / h
        rhs = (div / dt).ravel(); rhs[0] = 0.0
        p = LU.solve(rhs).reshape(N, N)
        # --- correzione
        un = us.copy(); vn = vs.copy()
        un[1:-1, 1:-1] -= dt * (p[1:, :] - p[:-1, :]) / h
        vn[1:-1, 1:-1] -= dt * (p[:, 1:] - p[:, :-1]) / h
        res = max(np.max(np.abs(un - u)), np.max(np.abs(vn - v))) / dt
        u, v = un, vn
        t += dt; passo += 1
        if passo % 200 == 0:
            storia.append((t, res))
    divf = np.max(np.abs((u[1:, 1:-1] - u[:-1, 1:-1]) / h + (v[1:-1, 1:] - v[1:-1, :-1]) / h))
    durata = time.perf_counter() - t0
    # funzione di corrente e vorticita' nei vertici (N+1) x (N+1): psi_(j+1) = psi_j + h u_(i+1/2, j+1/2)
    psi = np.zeros((N + 1, N + 1))
    for j in range(1, N + 1):
        psi[:, j] = psi[:, j - 1] + h * u[:, j]
    om = np.zeros((N + 1, N + 1))
    om[1:-1, 1:-1] = (v[2:-1, 1:-1] - v[1:-2, 1:-1]) / h - (u[1:-1, 2:-1] - u[1:-1, 1:-2]) / h
    if stampa:
        k = np.unravel_index(np.argmin(psi), psi.shape)
        print(f"   Re = {Re:5.0f}, griglia {N+1}x{N+1}: {passo} passi (dt = {dt:.2e}), t = {t:.1f}, residuo {res:.1e},"
              f" max|div u| = {divf:.1e}, {durata:.1f} s")
        print(f"      psi_min = {psi.min():.5f} in ({k[0]*h:.4f}, {k[1]*h:.4f}), omega_c = {om[k]:.4f}, "
              f"psi sulla parete superiore: max|psi| = {np.max(np.abs(psi[:, N])):.1e}")
        if Re in GHIA:
            g = GHIA[Re]
            print(f"      Ghia et al. (129x129): psi_min = {g[0]:.5f} in ({g[2]}, {g[3]}), omega_c = {g[1]:.4f}")
    return dict(Re=Re, N=N, h=h, u=u, v=v, p=p, psi=psi, om=om, t=t, passi=passo, storia=storia)


if __name__ == "__main__":
    print("Cavita' quadrata con coperchio mobile, griglia MAC, passi frazionari espliciti")
    casi = [(1, 64, 5.0), (100, 64, 30.0), (400, 64, 60.0), (1000, 80, 120.0)]
    ris = {}
    for Re, N, tmax in casi:
        ris[Re] = cavita(Re, N, tmax)
    # profili di velocita' sulla mezzeria verticale
    print("\nVelocita' u sulla mezzeria x = 0,5 (griglia MAC: colonna i = N/2):")
    print("      y     " + "".join(f"  Re={Re:4d} (calc./Ghia)" for Re in (100, 400, 1000)))
    for k, y in enumerate(GHIA_U["y"]):
        riga = f"   {y:6.4f} "
        for Re in (100, 400, 1000):
            r = ris[Re]; N = r["N"]; h = r["h"]
            yy = (np.arange(N + 2) - 0.5) * h          # quote dei nodi u (con fittizi)
            uu = np.interp(y, yy, r["u"][N // 2, :])
            riga += f"   {uu:+.4f}/{GHIA_U[Re][k]:+.4f}   "
        print(riga)

    fig, axs = plt.subplots(2, 4, figsize=(8.8, 4.4))
    for col, Re in enumerate((1, 100, 400, 1000)):
        r = ris[Re]; N = r["N"]
        xg = np.linspace(0, 1, N + 1)
        lv = [-0.1175, -0.115, -0.11, -0.1, -0.09, -0.07, -0.05, -0.03, -0.01, -1e-4, 1e-5, 1e-4, 5e-4, 1e-3, 1.5e-3, 3e-3]
        axs[0, col].contour(xg, xg, r["psi"].T, levels=lv, colors=[ROSSO if v > 0 else BLU for v in lv], linewidths=0.7)
        axs[0, col].set_aspect("equal"); axs[0, col].set_title(f"$\\psi$, $Re={Re}$", fontsize=8)
        lo = [-5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5]
        axs[1, col].contour(xg, xg, r["om"].T, levels=lo, colors=[ROSSO if v > 0 else BLU for v in lo], linewidths=0.7)
        axs[1, col].set_aspect("equal"); axs[1, col].set_title(f"$\\omega$, $Re={Re}$", fontsize=8)
        for a_ in axs[:, col]:
            a_.set_xticks([]); a_.set_yticks([])
    fig.tight_layout()
    salva(fig, "prog4_5_cavita_mac")

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for Re, col in zip((100, 400, 1000), (BLU, VERDE, ROSSO)):
        r = ris[Re]; N = r["N"]; h = r["h"]
        yy = (np.arange(N + 2) - 0.5) * h
        ax[0].plot(r["u"][N // 2, 1:-1], yy[1:-1], color=col, label=f"$Re={Re}$")
        ax[0].plot(GHIA_U[Re], GHIA_U["y"], "o", color=col, ms=3, mfc="none")
    ax[0].set_xlabel("$u(0{,}5,y)$"); ax[0].set_ylabel("$y$"); ax[0].legend(fontsize=7)
    ax[0].set_title("profili sulla mezzeria (cerchi: Ghia et al.)", fontsize=8)
    for Re, col in zip((100, 400, 1000), (BLU, VERDE, ROSSO)):
        st = np.array(ris[Re]["storia"])
        ax[1].semilogy(st[:, 0], st[:, 1], color=col, label=f"$Re={Re}$")
    ax[1].set_xlabel("$t$"); ax[1].set_ylabel("$\\max|u^{n+1}-u^n|/\\Delta t$"); ax[1].legend(fontsize=7)
    ax[1].set_title("avvicinamento allo stato stazionario", fontsize=8)
    fig.tight_layout()
    salva(fig, "prog4_5_cavita_profili")
