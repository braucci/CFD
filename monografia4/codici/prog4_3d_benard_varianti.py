"""
prog4_3d_benard_varianti.py -- due varianti del problema di Benard risolte con lo
stesso schema del Programma 4.3 (upwind di Torrance, formulazione Omega-Psi):

(a) scatola riscaldata dall'alto: theta = 1 sulla superficie superiore, che e'
    una superficie libera piana e priva di sforzi (Omega = 0); theta = 0 sul fondo
    e sulla parete laterale; fluido inizialmente in quiete. La stratificazione
    media e' stabile; il moto nasce dal gradiente orizzontale di temperatura
    negli spigoli superiori (modello di una regione di oceano fra due iceberg
    riscaldata dal sole). Meta' sinistra della scatola, asse di simmetria i = m.
(b) strato fra due piastre orizzontali infinite con temperatura periodica sulla
    piastra superiore, theta = 1 + 0,5 sin(2 pi X/lambda), lambda = 2,5, theta = 0
    sulla piastra inferiore; condizioni periodiche in X (modello della circolazione
    generata da gradienti termici orizzontali). Pr = 6,75, Gr = 1000.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from prog4_3_benard import pnew, sorlx, dt_massimo, M, N, H, PR, GR
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO


# ----------------------------------------------------------------- (a) riscaldamento dall'alto
def scatola_dall_alto(maxstp=2000, dt=0.0025, istanti=(0.05, 0.25, 1.0, 5.0), errmax=1e-4):
    psi = np.zeros((M, N)); omega = np.zeros((M, N)); theta = np.zeros((M, N))
    U = np.zeros((M, N)); V = np.zeros((M, N))
    theta[:, N - 1] = 1.0                                # superficie superiore calda
    theta[0, :] = 0.0                                    # parete laterale fredda (spigolo superiore freddo)
    salvati, storia = {}, []
    for nstep in range(1, maxstp + 1):
        psi, it = sorlx(psi, omega, H, errmax)
        U[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * H)
        V[1:-1, :] = -(psi[2:, :] - psi[:-2, :]) / (2 * H)
        V[M - 1, :] = psi[M - 2, :] / H
        omnew = omega.copy(); thnew = theta.copy()
        omnew[1:-1, 1:-1] = pnew(omega, theta, U, V, GR, 1.0, dt, H)
        thnew[1:-1, 1:-1] = pnew(theta, theta, U, V, 0.0, 1.0 / PR, dt, H)
        Ue = np.vstack([U[M - 2:M, :], -U[M - 2, :][None, :]])
        Ve = np.vstack([V[M - 2:M, :], V[M - 2, :][None, :]])
        The = np.vstack([theta[M - 2:M, :], theta[M - 2, :][None, :]])
        thnew[M - 1, 1:-1] = pnew(The, The, Ue, Ve, 0.0, 1.0 / PR, dt, H)[0]
        omnew[M - 1, :] = 0.0
        omnew[0, 1:-1] = (4 * V[1, 1:-1] - V[2, 1:-1]) / (2 * H)         # parete laterale
        omnew[1:-1, 0] = (-4 * U[1:-1, 1] + U[1:-1, 2]) / (2 * H)         # fondo
        omnew[:, N - 1] = 0.0                                             # superficie libera senza sforzi
        omega, theta = omnew, thnew
        T = nstep * dt
        storia.append((T, np.max(np.abs(psi))))
        for Ts in istanti:
            if abs(T - Ts) < 1e-9:
                salvati[Ts] = (psi.copy(), theta.copy())
                print(f"   (a) T = {T:5.2f}: max|Psi| = {np.max(np.abs(psi)):.4f}, min Psi = {psi.min():+.4f},"
                      f" max Psi = {psi.max():+.4f}, iter. SOR = {it}")
    return salvati, storia


# ----------------------------------------------------------------- (b) gradiente orizzontale periodico
LAM = 2.5
NX = int(round(LAM / H))                 # nodi periodici in X (X_i = i h, i = 0..NX-1)


def matrice_periodica():
    """lap Psi nei nodi j = 1..N-2 (tutti gli i, periodici), Psi = 0 su j = 0 e j = N-1."""
    nj = N - 2
    idx = lambda i, j: (i % NX) * nj + (j - 1)
    A = sp.lil_matrix((NX * nj, NX * nj))
    for i in range(NX):
        for j in range(1, N - 1):
            p = idx(i, j)
            A[p, p] = -4 / H**2
            A[p, idx(i + 1, j)] += 1 / H**2
            A[p, idx(i - 1, j)] += 1 / H**2
            if j + 1 <= N - 2: A[p, idx(i, j + 1)] += 1 / H**2
            if j - 1 >= 1: A[p, idx(i, j - 1)] += 1 / H**2
    return spla.splu(A.tocsc())


def estesa(S):
    """Aggiunge una colonna fantasma per lato (periodicita' in X)."""
    return np.vstack([S[-1:, :], S, S[:1, :]])


def strato_periodico(maxstp=2000, dt=0.0025, istanti=(0.05, 0.25, 1.0, 5.0)):
    X = H * np.arange(NX)
    LU = matrice_periodica()
    psi = np.zeros((NX, N)); omega = np.zeros((NX, N)); theta = np.zeros((NX, N))
    U = np.zeros((NX, N)); V = np.zeros((NX, N))
    theta[:, N - 1] = 1.0 + 0.5 * np.sin(2 * np.pi * X / LAM)
    salvati, storia = {}, []
    for nstep in range(1, maxstp + 1):
        psi[:, 1:-1] = LU.solve((-omega[:, 1:-1]).ravel()).reshape(NX, N - 2)
        U[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * H)
        V = -(np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2 * H)
        omnew = omega.copy(); thnew = theta.copy()
        omnew[:, 1:-1] = pnew(estesa(omega), estesa(theta), estesa(U), estesa(V), GR, 1.0, dt, H)
        thnew[:, 1:-1] = pnew(estesa(theta), estesa(theta), estesa(U), estesa(V), 0.0, 1.0 / PR, dt, H)
        omnew[:, 0] = (-4 * U[:, 1] + U[:, 2]) / (2 * H)                  # piastra inferiore
        omnew[:, N - 1] = (4 * U[:, N - 2] - U[:, N - 3]) / (2 * H)      # piastra superiore
        omega, theta = omnew, thnew
        T = nstep * dt
        storia.append((T, np.max(np.abs(psi))))
        for Ts in istanti:
            if abs(T - Ts) < 1e-9:
                salvati[Ts] = (psi.copy(), theta.copy())
                print(f"   (b) T = {T:5.2f}: max|Psi| = {np.max(np.abs(psi)):.4f}, min Psi = {psi.min():+.4f},"
                      f" max Psi = {psi.max():+.4f}")
    return salvati, storia, X


if __name__ == "__main__":
    print(f"Varianti del problema di Benard: Pr = {PR}, Gr = {GR}, h = {H}, dT = 0.0025")
    print("(a) scatola riscaldata dall'alto con superficie libera (meta' sinistra, 21x9):")
    sa, storia_a = scatola_dall_alto()
    print(f"(b) strato periodico, lambda = {LAM} ({NX} nodi in X, periodici):")
    sb, storia_b, X = strato_periodico()

    Xa = np.arange(M) * H; Y = np.arange(N) * H
    Xf = np.concatenate([Xa, 2 * Xa[-1] - Xa[-2::-1]])
    fig, axs = plt.subplots(4, 2, figsize=(8.4, 6.0))
    for riga, Ts in enumerate((0.05, 0.25, 1.0, 5.0)):
        psi, theta = sa[Ts]
        psif = np.vstack([psi, -psi[-2::-1, :]]); thf = np.vstack([theta, theta[-2::-1, :]])
        m = np.max(np.abs(psif)); lv = np.linspace(-m, m, 13); lv = lv[np.abs(lv) > 1e-7]
        a_ = axs[riga, 0]
        a_.contour(Xf, Y, psif.T, levels=lv, colors=[ROSSO if v < 0 else BLU for v in lv],
                   linestyles=["--" if v < 0 else "-" for v in lv], linewidths=0.7)
        a_.contour(Xf, Y, thf.T, levels=np.linspace(0.1, 0.9, 5), colors=GRIGIO, linewidths=0.5)
        a_.set_aspect("equal"); a_.set_xticks([]); a_.set_yticks([])
        a_.set_title(f"(a) $T={Ts:g}$, $|\\Psi|_{{\\max}}={m:.3f}$".replace(".", "{,}"), fontsize=8)
        psi, theta = sb[Ts]
        Xe = np.concatenate([X, [LAM]]); pe = np.vstack([psi, psi[:1]]); te = np.vstack([theta, theta[:1]])
        m = np.max(np.abs(pe)); lv = np.linspace(-m, m, 13); lv = lv[np.abs(lv) > 1e-7]
        a_ = axs[riga, 1]
        a_.contour(Xe, Y, pe.T, levels=lv, colors=[ROSSO if v < 0 else BLU for v in lv],
                   linestyles=["--" if v < 0 else "-" for v in lv], linewidths=0.7)
        a_.contour(Xe, Y, te.T, levels=np.linspace(0.2, 1.4, 7), colors=GRIGIO, linewidths=0.5)
        a_.set_aspect("equal"); a_.set_xticks([]); a_.set_yticks([])
        a_.set_title(f"(b) $T={Ts:g}$, $|\\Psi|_{{\\max}}={m:.3f}$".replace(".", "{,}"), fontsize=8)
    fig.tight_layout()
    salva(fig, "prog4_3d_benard_varianti")
