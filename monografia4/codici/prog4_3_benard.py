"""
prog4_3_benard.py -- Programma 4.3: convezione di Benard non lineare in una
scatola rettangolare riscaldata dal basso (formulazione vorticita'-funzione di
corrente, schema upwind di Torrance, SOR per la funzione di corrente).

Variabili adimensionali (lunghezza H, velocita' nu/H, tempo H^2/nu):
    U = dPsi/dY,  V = -dPsi/dX,  Omega = -(Psi_XX + Psi_YY),
    dOmega/dT + d(U Omega)/dX + d(V Omega)/dY = Gr dtheta/dX + lap Omega,
    dtheta/dT + d(U theta)/dX + d(V theta)/dY = (1/Pr) lap theta.
Si calcola la meta' sinistra della scatola (simmetria rispetto all'asse Y):
m = 21 linee verticali (i = m sull'asse), n = 9 orizzontali, h = 1/(n-1);
theta = 1 sul fondo, 0 sul coperchio e sulla parete laterale, dtheta/dX = 0
sull'asse; Psi = 0 su tutto il contorno; vorticita' di parete dalle formule
unilatere del secondo ordine. Acqua: Pr = 6,75; Gr = 1000; dT = 0,0025.
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

M, N = 21, 9
H = 1.0 / (N - 1)
PR, GR = 6.75, 1000.0
DT = 0.0025
MAXSTP = 2000
ERRMAX = 1e-4
istanti = [0.025, 0.05, 0.125, 0.2, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0]


def sorlx(psi, omega, h, errmax, omega_rf=None):
    """SOR per lap Psi = -Omega nei nodi interni (i = 2..M-1, j = 2..N-1), Psi = 0 sul contorno
    e sull'asse i = M. Restituisce Psi e il numero di iterazioni."""
    m, n = psi.shape
    if omega_rf is None:
        a = np.cos(np.pi / (m - 1)) + np.cos(np.pi / (n - 1))
        omega_rf = (8 - 4 * np.sqrt(4 - a * a)) / (a * a)
    for it in range(1, 10001):
        err = 0.0
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                nuovo = (1 - omega_rf) * psi[i, j] + omega_rf / 4 * (
                    psi[i - 1, j] + psi[i + 1, j] + psi[i, j - 1] + psi[i, j + 1] + h * h * omega[i, j])
                err += abs(nuovo - psi[i, j])
                psi[i, j] = nuovo
        if err < errmax:
            return psi, it
    raise RuntimeError("SORLX: convergenza non raggiunta")


def pnew(P, Q, U, V, A, B, dt, h):
    """Un passo esplicito dell'equazione generale (4.3.20)
        dP/dT = -d(UP)/dX - d(VP)/dY + A dQ/dX + B lap P
    con lo schema upwind di Torrance per i termini convettivi, nei nodi interni
    (i = 1..m-2, j = 1..n-2 in indici Python). Restituisce il nuovo P nei nodi interni."""
    Uf = 0.5 * (U[2:, 1:-1] + U[1:-1, 1:-1]);  Ub = 0.5 * (U[1:-1, 1:-1] + U[:-2, 1:-1])
    Vf = 0.5 * (V[1:-1, 2:] + V[1:-1, 1:-1]);  Vb = 0.5 * (V[1:-1, 1:-1] + V[1:-1, :-2])
    Pc = P[1:-1, 1:-1]
    P1 = (Uf - abs(Uf)) * P[2:, 1:-1] + (Uf + abs(Uf) - Ub + abs(Ub)) * Pc - (Ub + abs(Ub)) * P[:-2, 1:-1]
    P2 = (Vf - abs(Vf)) * P[1:-1, 2:] + (Vf + abs(Vf) - Vb + abs(Vb)) * Pc - (Vb + abs(Vb)) * P[1:-1, :-2]
    P3 = Q[2:, 1:-1] - Q[:-2, 1:-1]
    P4 = P[2:, 1:-1] + P[:-2, 1:-1] + P[1:-1, 2:] + P[1:-1, :-2] - 4 * Pc
    return Pc + dt / (2 * h) * (-P1 - P2 + A * P3 + 2 * B * P4 / h)


def dt_massimo(U, V, B, h):
    """Limite di stabilita' (4.3.33): coefficiente del nodo centrale non negativo."""
    Uf = 0.5 * (U[2:, 1:-1] + U[1:-1, 1:-1]);  Ub = 0.5 * (U[1:-1, 1:-1] + U[:-2, 1:-1])
    Vf = 0.5 * (V[1:-1, 2:] + V[1:-1, 1:-1]);  Vb = 0.5 * (V[1:-1, 1:-1] + V[:-2, 1:-1])
    s = (Uf + abs(Uf) - Ub + abs(Ub) + Vf + abs(Vf) - Vb + abs(Vb)) / (2 * h) + 4 * B / h**2
    return 1.0 / np.max(s)


def simula(maxstp=MAXSTP, dt=DT, istanti=istanti, stampa=True):
    psi = np.zeros((M, N)); omega = np.zeros((M, N)); theta = np.zeros((M, N))
    U = np.zeros((M, N)); V = np.zeros((M, N))
    theta[:, 0] = 1.0                                  # fondo caldo
    theta[0, 0] = theta[M - 1, 0] = 1.0
    theta[0, :] = 0.0; theta[0, 0] = 1.0                 # parete laterale fredda (spigolo caldo)
    salvati, storia = {}, []
    dtmin = np.inf
    for nstep in range(1, maxstp + 1):
        # funzione di corrente e velocita'
        psi, it = sorlx(psi, omega, H, ERRMAX)
        U[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * H)
        V[1:-1, :] = -(psi[2:, :] - psi[:-2, :]) / (2 * H)
        V[M - 1, :] = psi[M - 2, :] / H                  # asse: Psi_(m+1) = -Psi_(m-1)
        # vorticita' e temperatura nei nodi interni (esclusa la colonna dell'asse)
        omnew = omega.copy(); thnew = theta.copy()
        omnew[1:-1, 1:-1] = pnew(omega, theta, U, V, GR, 1.0, dt, H)
        thnew[1:-1, 1:-1] = pnew(theta, theta, U, V, 0.0, 1.0 / PR, dt, H)
        # temperatura sull'asse (i = M): nodo specchio i = M+1 -> M-1
        Ue = np.vstack([U[M - 2:M, :], -U[M - 2, :][None, :]])
        Ve = np.vstack([V[M - 2:M, :], V[M - 2, :][None, :]])
        The = np.vstack([theta[M - 2:M, :], theta[M - 2, :][None, :]])
        thnew[M - 1, 1:-1] = pnew(The, The, Ue, Ve, 0.0, 1.0 / PR, dt, H)[0]
        omnew[M - 1, :] = 0.0
        # vorticita' di parete
        omnew[0, 1:-1] = (4 * V[1, 1:-1] - V[2, 1:-1]) / (2 * H)
        omnew[1:-1, 0] = (-4 * U[1:-1, 1] + U[1:-1, 2]) / (2 * H)
        omnew[1:-1, N - 1] = (4 * U[1:-1, N - 2] - U[1:-1, N - 3]) / (2 * H)
        omnew[0, 0] = omnew[0, N - 1] = 0.0
        dtmin = min(dtmin, dt_massimo(U, V, 1.0 / PR, H), dt_massimo(U, V, 1.0, H))
        omega, theta = omnew, thnew
        T = nstep * dt
        storia.append((T, np.max(np.abs(psi)), it))
        for Ts in istanti:
            if abs(T - Ts) < 1e-9:
                salvati[Ts] = (psi.copy(), theta.copy(), omega.copy())
                if stampa:
                    # numero di Nusselt medio sul fondo: -dtheta/dY mediato su X (differenza in avanti del 2o ordine)
                    dth = (-3 * theta[:, 0] + 4 * theta[:, 1] - theta[:, 2]) / (2 * H)
                    Nu = -np.mean(dth[1:])
                    print(f"   T = {T:5.3f} (passo {nstep:4d}): max|Psi| = {np.max(np.abs(psi)):.4f},"
                          f" min Psi = {psi.min():+.4f}, max Psi = {psi.max():+.4f}, iter. SOR = {it:3d}, Nu = {Nu:.3f}")
    return salvati, storia, dtmin


if __name__ == "__main__":
    print(f"Benard non lineare: Pr = {PR}, Gr = {GR} (Ra = {PR*GR:.0f}), griglia {M}x{N}, h = {H}, dT = {DT}")
    salvati, storia, dtmin = simula()
    print(f"   passo temporale massimo consentito dalla (4.3.33) lungo tutta la simulazione: {dtmin:.4f} (usato {DT})")

    # --------------------------------------------------------- figure: storia temporale delle linee di corrente
    X = np.arange(M) * H
    Y = np.arange(N) * H
    Xf = np.concatenate([X, 2 * X[-1] - X[-2::-1]])     # scatola intera per simmetria
    fig, axs = plt.subplots(5, 2, figsize=(8.4, 7.2))
    for a_, Ts in zip(axs.ravel(), istanti):
        psi, theta, omega = salvati[Ts]
        psif = np.vstack([psi, -psi[-2::-1, :]])         # Psi antisimmetrica
        lv = np.linspace(-np.max(np.abs(psif)), np.max(np.abs(psif)), 13)
        lv = lv[np.abs(lv) > 1e-6]
        a_.contour(Xf, Y, psif.T, levels=lv, colors=[ROSSO if v < 0 else BLU for v in lv],
                   linestyles=["--" if v < 0 else "-" for v in lv], linewidths=0.8)
        a_.set_aspect("equal")
        a_.set_xticks([]); a_.set_yticks([])
        a_.set_title(f"$T={Ts:g}$".replace(".", "{,}") + f",  $|\\Psi|_{{\\max}}={np.max(np.abs(psi)):.2f}$".replace(".", "{,}"), fontsize=8)
    fig.tight_layout()
    salva(fig, "prog4_3_benard_storia")

    # --------------------------------------------------------- figure: isoterme e storia di |Psi|max
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 2.8), gridspec_kw={"width_ratios": [2.2, 1]})
    psi, theta, omega = salvati[1.0]
    thf = np.vstack([theta, theta[-2::-1, :]])
    cs = ax[0].contour(Xf, Y, thf.T, levels=np.linspace(0.1, 0.9, 9), cmap="coolwarm", linewidths=0.9)
    ax[0].clabel(cs, fmt=lambda v: f"{v:.1f}".replace(".", ","), fontsize=6)
    ax[0].set_aspect("equal"); ax[0].set_title("isoterme a $T=1$")
    ax[0].set_xlabel("$X$"); ax[0].set_ylabel("$Y$")
    Ts_ = [s[0] for s in storia]
    ax[1].plot(Ts_, [s[1] for s in storia], color=BLU)
    ax[1].set_xlabel("$T$"); ax[1].set_ylabel("$|\\Psi|_{\\max}$")
    ax[1].set_title("intensità del moto".replace("à", "a'"))
    fig.tight_layout()
    salva(fig, "prog4_3_benard_isoterme")
