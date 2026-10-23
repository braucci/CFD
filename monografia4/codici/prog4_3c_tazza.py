"""
prog4_3c_tazza.py -- avviamento in rotazione (spin-up) del caffe' in una tazza
cilindrica messa improvvisamente in rotazione con velocita' angolare omega0
(progetto "You'll find it's already stirred").

Moto assialsimmetrico in coordinate (R, Z) riferite a r0, velocita' a r0 omega0,
tempo a 1/omega0; tazza di raggio 1 e altezza 2, superficie libera piana e priva
di sforzi tangenziali. Formulazione Omega-Psi-W della corrente di Taylor:
    U = (1/R) dPsi/dR,  V = -(1/R) dPsi/dZ,  W = r u_theta/(r0^2 omega0),
    Psi_RR - Psi_R/R + Psi_ZZ = R Omega,
    dOmega/dT = -d(U Omega)/dZ - d(V Omega)/dR - (2W/R^3) dW/dZ + (1/Re) L1 Omega,
    dW/dT     = -d(U W)/dZ - d(V W)/dR - V W/R + (1/Re) L2 W,
con L1 = d_RR + d_R/R - 1/R^2 + d_ZZ e L2 = d_RR - d_R/R + d_ZZ, Re = r0^2 omega0/nu = 500.
Condizioni: asse R = 0: Psi = Omega = W = 0; fondo e parete (rotanti con la
tazza): Psi = 0, W = R^2, Omega dalla formula di parete; superficie libera Z = 2:
Psi = 0, Omega = 0, dW/dZ = 0. Fluido inizialmente in quiete (W = 0). Schema
upwind di Torrance con passo temporale scelto a ogni passo dal criterio di
stabilita' (4.3.33), funzione di corrente con fattorizzazione LU sparsa.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

RE = 500.0
H = 0.025
ALT = 2.0
NR = int(round(1.0 / H)) + 1
NZ = int(round(ALT / H)) + 1
R = H * np.arange(NR)
Z = H * np.arange(NZ)
T_FIN = 8.0


def matrice_psi():
    """Psi_RR - Psi_R/R + Psi_ZZ nei nodi interni (j = 1..NR-2, k = 1..NZ-2), Psi = 0 sul contorno."""
    nr, nz = NR - 2, NZ - 2
    idx = lambda j, k: j * nz + k
    A = sp.lil_matrix((nr * nz, nr * nz))
    for j in range(nr):
        r = R[j + 1]
        for k in range(nz):
            p = idx(j, k)
            A[p, p] = -4 / H**2
            if k + 1 < nz: A[p, idx(j, k + 1)] = 1 / H**2
            if k - 1 >= 0: A[p, idx(j, k - 1)] = 1 / H**2
            if j + 1 < nr: A[p, idx(j + 1, k)] = 1 / H**2 - 1 / (2 * r * H)
            if j - 1 >= 0: A[p, idx(j - 1, k)] = 1 / H**2 + 1 / (2 * r * H)
    return spla.splu(A.tocsc())


def torrance(P, U, V):
    """-d(UP)/dZ - d(VP)/dR (schema upwind di Torrance) nei nodi interni; array (NR, NZ)."""
    Pc = P[1:-1, 1:-1]
    Uf = 0.5 * (U[1:-1, 2:] + U[1:-1, 1:-1]); Ub = 0.5 * (U[1:-1, 1:-1] + U[1:-1, :-2])
    Vf = 0.5 * (V[2:, 1:-1] + V[1:-1, 1:-1]); Vb = 0.5 * (V[1:-1, 1:-1] + V[:-2, 1:-1])
    Pz = (Uf - abs(Uf)) * P[1:-1, 2:] + (Uf + abs(Uf) - Ub + abs(Ub)) * Pc - (Ub + abs(Ub)) * P[1:-1, :-2]
    Pr = (Vf - abs(Vf)) * P[2:, 1:-1] + (Vf + abs(Vf) - Vb + abs(Vb)) * Pc - (Vb + abs(Vb)) * P[:-2, 1:-1]
    return -(Pz + Pr) / (2 * H)


def dt_massimo(U, V, B):
    """Criterio (4.3.33) per il termine convettivo + diffusivo con coefficiente B."""
    Uf = 0.5 * (U[1:-1, 2:] + U[1:-1, 1:-1]); Ub = 0.5 * (U[1:-1, 1:-1] + U[1:-1, :-2])
    Vf = 0.5 * (V[2:, 1:-1] + V[1:-1, 1:-1]); Vb = 0.5 * (V[1:-1, 1:-1] + V[:-2, 1:-1])
    s = (Uf + abs(Uf) - Ub + abs(Ub) + Vf + abs(Vf) - Vb + abs(Vb)) / (2 * H) + 4 * B / H**2
    return 1.0 / np.max(s)


def simula(t_fin=T_FIN, istanti=(0.5, 1.0, 2.0, 4.0, 8.0), stampa=True, sicurezza=0.8):
    LU = matrice_psi()
    Rc = R[:, None]
    Psi = np.zeros((NR, NZ)); Om = np.zeros((NR, NZ)); W = np.zeros((NR, NZ))
    U = np.zeros((NR, NZ)); V = np.zeros((NR, NZ))
    W[-1, :] = 1.0; W[:, 0] = R**2                       # parete e fondo rotanti
    T, salvati, storia = 0.0, {}, []
    prossimo = 0
    while T < t_fin - 1e-12:
        # funzione di corrente e velocita'
        Psi[1:-1, 1:-1] = LU.solve((Rc * Om)[1:-1, 1:-1].ravel()).reshape(NR - 2, NZ - 2)
        U[1:-1, 1:-1] = (Psi[2:, 1:-1] - Psi[:-2, 1:-1]) / (2 * H) / Rc[1:-1]
        V[1:-1, 1:-1] = -(Psi[1:-1, 2:] - Psi[1:-1, :-2]) / (2 * H) / Rc[1:-1]
        # vorticita' sulle pareti solide (Psi = Psi_n = 0 -> Omega = 2 Psi_1/(R h^2)); asse e superficie libera: 0
        Om[-1, 1:-1] = 2 * Psi[-2, 1:-1] / (R[-1] * H**2)
        Om[1:-1, 0] = 2 * Psi[1:-1, 1] / (R[1:-1] * H**2)
        Om[0, :] = 0.0; Om[:, -1] = 0.0
        # superficie libera: dW/dZ = 0
        W[:, -1] = W[:, -2]
        # passo temporale dal criterio di stabilita' (il piu' restrittivo fra le due equazioni)
        dt = sicurezza * min(dt_massimo(U, V, 1.0 / RE), dt_massimo(U, V, 1.0 / RE))
        if prossimo < len(istanti):
            dt = min(dt, istanti[prossimo] - T)
        Rm = Rc[1:-1]
        lapOm = ((Om[2:, 1:-1] - 2 * Om[1:-1, 1:-1] + Om[:-2, 1:-1]) / H**2 + (Om[2:, 1:-1] - Om[:-2, 1:-1]) / (2 * H * Rm)
                 - Om[1:-1, 1:-1] / Rm**2 + (Om[1:-1, 2:] - 2 * Om[1:-1, 1:-1] + Om[1:-1, :-2]) / H**2)
        lapW = ((W[2:, 1:-1] - 2 * W[1:-1, 1:-1] + W[:-2, 1:-1]) / H**2 - (W[2:, 1:-1] - W[:-2, 1:-1]) / (2 * H * Rm)
                + (W[1:-1, 2:] - 2 * W[1:-1, 1:-1] + W[1:-1, :-2]) / H**2)
        dWdZ = (W[1:-1, 2:] - W[1:-1, :-2]) / (2 * H)
        Om_n = Om.copy(); W_n = W.copy()
        Om_n[1:-1, 1:-1] = Om[1:-1, 1:-1] + dt * (torrance(Om, U, V) - 2 * W[1:-1, 1:-1] / Rm**3 * dWdZ + lapOm / RE)
        W_n[1:-1, 1:-1] = W[1:-1, 1:-1] + dt * (torrance(W, U, V) - V[1:-1, 1:-1] * W[1:-1, 1:-1] / Rm + lapW / RE)
        Om, W = Om_n, W_n
        T += dt
        # momento angolare totale del fluido rispetto a quello della rotazione rigida (W = R^2)
        L = np.sum((W * Rc)[1:-1, 1:-1]) / np.sum((Rc**3)[1:-1] * np.ones((1, NZ - 2)))
        storia.append((T, dt, np.max(np.abs(Psi)), L))
        if prossimo < len(istanti) and abs(T - istanti[prossimo]) < 1e-10:
            salvati[istanti[prossimo]] = (Psi.copy(), W.copy(), Om.copy())
            if stampa:
                print(f"   T = {T:4.1f}: dT = {dt:.5f}, max|Psi| = {np.max(np.abs(Psi)):.5f},"
                      f" momento angolare / rotazione rigida = {L:.3f}, u_theta sull'asse di meta' altezza (R = 0,5): "
                      f"{W[NR//2, NZ//2]/R[NR//2]:.3f} (rigido: {R[NR//2]:.3f})")
            prossimo += 1
    return salvati, storia


if __name__ == "__main__":
    print(f"Spin-up nella tazza: Re = {RE:.0f}, griglia {NR}x{NZ} (h = {H}), altezza {ALT}")
    print(f"   tempo di spin-up di Ekman stimato: sqrt(Re) = {np.sqrt(RE):.1f} unita' di 1/omega0;"
          f" tempo di diffusione viscosa Re = {RE:.0f}")
    print(f"   discesa della scala: T = 8 corrisponde a {8/(0.1*np.pi):.1f} s con omega0 = 0,1 pi rad/s")
    salvati, storia = simula()
    Ts = np.array([s[0] for s in storia]); Ls = np.array([s[3] for s in storia])
    print(f"   frazione di momento angolare acquisita a T = 8: {Ls[-1]:.3f}")

    fig, axs = plt.subplots(1, 5, figsize=(8.6, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.6]})
    for a_, Tk in zip(axs[:4], (1.0, 2.0, 4.0, 8.0)):
        Psi, W, Om = salvati[Tk]
        m = np.max(np.abs(Psi))
        lv = np.linspace(-m, m, 11); lv = lv[np.abs(lv) > 1e-12]
        a_.contour(R, Z, Psi.T, levels=lv, colors=[ROSSO if v < 0 else BLU for v in lv],
                   linestyles=["--" if v < 0 else "-" for v in lv], linewidths=0.7)
        cs = a_.contour(R, Z, (W / np.maximum(R, 1e-9)[:, None]).T, levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors=GRIGIO, linewidths=0.6)
        a_.clabel(cs, fontsize=5, fmt=lambda v: f"{v:.1f}".replace(".", ","))
        a_.set_aspect("equal"); a_.set_xlabel("$R$")
        a_.set_title(f"$T={Tk:g}$, $|\\Psi|_{{\\max}}={m:.4f}$".replace(".", "{,}"), fontsize=7)
        a_.set_xticks([0, 0.5, 1]); a_.set_yticks([0, 1, 2])
    axs[0].set_ylabel("$Z$")
    axs[4].plot(Ts, Ls, color=BLU)
    axs[4].set_xlabel("$T$"); axs[4].set_ylabel("momento angolare / rotazione rigida")
    axs[4].set_title("avviamento", fontsize=7)
    fig.tight_layout()
    salva(fig, "prog4_3c_tazza")
