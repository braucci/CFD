"""
prog4_3b_taylor.py -- instabilita' di Taylor fra cilindri coassiali rotanti
(moto assialsimmetrico, formulazione vorticita'-funzione di corrente-momento
angolare), con lo stesso schema upwind del problema di Benard.

Variabili (lunghezze riferite a r_i, velocita' a r_i omega_i, tempo a 1/omega_i):
    U = (1/R) dPsi/dR (assiale),  V = -(1/R) dPsi/dZ (radiale),  W = r u_theta/(r_i^2 omega_i),
    Psi_RR - Psi_R/R + Psi_ZZ = R Omega,
    dOmega/dT = -d(U Omega)/dZ - d(V Omega)/dR - (2W/R^3) dW/dZ + (1/Re)[Omega_RR + Omega_R/R - Omega/R^2 + Omega_ZZ],
    dW/dT     = -d(U W)/dZ - d(V W)/dR - V W/R + (1/Re)[W_RR - W_R/R + W_ZZ].
Cilindro interno R = 1 rotante (W = 1), esterno R = A = 2 fermo (W = 0); dominio
periodico in Z di lunghezza lambda = 4; Re = r_i^2 omega_i/nu = 1000; h = 0,125;
dT = 0,02. Stato iniziale: corrente di Couette circolare W = (A^2 - R^2)/(A^2 - 1)
perturbata con Omega = 0,01 sin(2 pi Z/lambda). Il programma segue la crescita
delle celle di Taylor e calcola il momento torcente sul cilindro interno.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

A, B = 2.0, 0.0
LAM, RE = 4.0, 1000.0
H, DT = 0.125, 0.02
NR = int(round((A - 1) / H)) + 1          # nodi radiali (pareti comprese)
NZ = int(round(LAM / H))                  # nodi assiali (periodici)
R = 1.0 + H * np.arange(NR)
Z = H * np.arange(NZ)
T_FIN = 60.0


def matrice_psi():
    """Operatore Psi_RR - Psi_R/R + Psi_ZZ sui nodi interni (j = 1..NR-2) x (tutti i k), periodico in Z."""
    n_int = NR - 2
    idx = lambda j, k: j * NZ + (k % NZ)
    Am = sp.lil_matrix((n_int * NZ, n_int * NZ))
    for j in range(n_int):
        r = R[j + 1]
        for k in range(NZ):
            p = idx(j, k)
            Am[p, p] = -4 / H**2
            Am[p, idx(j, k + 1)] += 1 / H**2
            Am[p, idx(j, k - 1)] += 1 / H**2
            if j + 1 < n_int:
                Am[p, idx(j + 1, k)] += 1 / H**2 - 1 / (2 * r * H)
            if j - 1 >= 0:
                Am[p, idx(j - 1, k)] += 1 / H**2 + 1 / (2 * r * H)
    return spla.splu(Am.tocsc())


LU = matrice_psi()


def torrance(P, U, V):
    """-d(UP)/dZ - d(VP)/dR con lo schema upwind di Torrance, nei nodi radiali interni;
    P, U, V hanno forma (NR, NZ) con Z periodico (asse 1)."""
    Pc = P[1:-1, :]
    Uf = 0.5 * (np.roll(U, -1, axis=1) + U)[1:-1, :]; Ub = 0.5 * (U + np.roll(U, 1, axis=1))[1:-1, :]
    Vf = 0.5 * (V[2:, :] + V[1:-1, :]);           Vb = 0.5 * (V[1:-1, :] + V[:-2, :])
    Pz1 = (Uf - abs(Uf)) * np.roll(P, -1, axis=1)[1:-1, :] + (Uf + abs(Uf) - Ub + abs(Ub)) * Pc \
        - (Ub + abs(Ub)) * np.roll(P, 1, axis=1)[1:-1, :]
    Pr1 = (Vf - abs(Vf)) * P[2:, :] + (Vf + abs(Vf) - Vb + abs(Vb)) * Pc - (Vb + abs(Vb)) * P[:-2, :]
    return -(Pz1 + Pr1) / (2 * H)


def simula(t_fin=T_FIN, dt=DT, ampiezza=0.01, modo=1, stampa=True):
    Rc = R[:, None]
    W = np.tile(((A * A * B - 1) * R**2 + A * A * (1 - B)) / (A * A - 1), (NZ, 1)).T
    Om = ampiezza * np.sin(2 * np.pi * modo * Z / LAM)[None, :] * np.ones((NR, 1))
    Om[0, :] = Om[-1, :] = 0.0
    Psi = np.zeros((NR, NZ)); U = np.zeros((NR, NZ)); V = np.zeros((NR, NZ))
    storia, salvati = [], {}
    istanti = (5.0, 10.0, 15.0, 20.0, 30.0, 60.0)
    n = int(round(t_fin / dt))
    for passo in range(1, n + 1):
        # funzione di corrente (Psi = 0 sulle pareti) e velocita'
        rhs = (Rc * Om)[1:-1, :].ravel()
        Psi[1:-1, :] = LU.solve(rhs).reshape(NR - 2, NZ)
        U[1:-1, :] = (Psi[2:, :] - Psi[:-2, :]) / (2 * H) / Rc[1:-1]
        V[1:-1, :] = -(np.roll(Psi, -1, axis=1) - np.roll(Psi, 1, axis=1))[1:-1, :] / (2 * H) / Rc[1:-1]
        # vorticita' alle pareti: Omega = Psi_RR/R con Psi = Psi_R = 0 -> 2 Psi_1/(R h^2)
        Om[0, :] = 2 * Psi[1, :] / (R[0] * H**2)
        Om[-1, :] = 2 * Psi[-2, :] / (R[-1] * H**2)
        # termini diffusivi e sorgenti nei nodi interni
        Rm = Rc[1:-1]
        lapOm = ((Om[2:, :] - 2 * Om[1:-1, :] + Om[:-2, :]) / H**2 + (Om[2:, :] - Om[:-2, :]) / (2 * H * Rm)
                 - Om[1:-1, :] / Rm**2 + (np.roll(Om, -1, axis=1) - 2 * Om + np.roll(Om, 1, axis=1))[1:-1, :] / H**2)
        lapW = ((W[2:, :] - 2 * W[1:-1, :] + W[:-2, :]) / H**2 - (W[2:, :] - W[:-2, :]) / (2 * H * Rm)
                + (np.roll(W, -1, axis=1) - 2 * W + np.roll(W, 1, axis=1))[1:-1, :] / H**2)
        dWdZ = (np.roll(W, -1, axis=1) - np.roll(W, 1, axis=1))[1:-1, :] / (2 * H)
        Om_new = Om.copy(); W_new = W.copy()
        Om_new[1:-1, :] = Om[1:-1, :] + dt * (torrance(Om, U, V) - 2 * W[1:-1, :] / Rm**3 * dWdZ + lapOm / RE)
        W_new[1:-1, :] = W[1:-1, :] + dt * (torrance(W, U, V) - V[1:-1, :] * W[1:-1, :] / Rm + lapW / RE)
        Om, W = Om_new, W_new
        T = passo * dt
        # momento torcente sul cilindro interno (per unita' di lunghezza, in unita' di mu r_i^2 omega_i):
        # tau = mu r d(u_theta/r)/dr = mu [dW/dR - 2W/R]/R  -> coppia = 2 pi R^2 tau; media su Z
        dWdR = (-3 * W[0, :] + 4 * W[1, :] - W[2, :]) / (2 * H)
        coppia = 2 * np.pi * np.mean(dWdR - 2 * W[0, :] / R[0])
        storia.append((T, np.max(np.abs(Psi)), coppia))
        if any(abs(T - ti) < 1e-9 for ti in istanti):
            salvati[round(T, 6)] = (Psi.copy(), W.copy(), Om.copy())
            if stampa:
                print(f"   T = {T:5.1f}: max|Psi| = {np.max(np.abs(Psi)):.5f}, coppia/(mu r_i^2 omega_i) = {coppia:+.4f}")
        if not np.isfinite(np.max(np.abs(Psi))):
            raise RuntimeError(f"instabilita' numerica al passo {passo}")
    return salvati, storia


if __name__ == "__main__":
    coppia_couette = 2 * np.pi * (-2 * A * A / (A * A - 1))      # dW/dR - 2W/R in R = 1 per il moto di Couette
    print(f"Taylor-Couette: A = {A}, B = {B}, lambda = {LAM}, Re = {RE}, griglia {NR}x{NZ}, dT = {DT}")
    print(f"   coppia della corrente di Couette circolare (laminare, senza celle): {coppia_couette:+.4f}")
    salvati, storia = simula()
    Ts = np.array([s[0] for s in storia]); amp = np.array([s[1] for s in storia]); cop = np.array([s[2] for s in storia])
    # tasso di crescita lineare (fra T = 2 e T = 8)
    sel = (Ts > 2) & (Ts < 8)
    pend = np.polyfit(Ts[sel], np.log(amp[sel]), 1)[0]
    print(f"   tasso di crescita iniziale di |Psi|max: sigma = {pend:.4f} (unita' omega_i); ampiezza finale {amp[-1]:.4f}")
    print(f"   coppia a regime {cop[-1]:+.4f}, cioe' {cop[-1]/coppia_couette:.2f} volte quella del moto di Couette")
    # perturbazione di lunghezza d'onda meta' (due coppie di celle nel dominio)
    print("\nPerturbazione iniziale con lunghezza d'onda lambda/2 = 2 (celle di sezione quasi quadrata):")
    salvati2, storia2 = simula(modo=2, stampa=False)
    Ts2 = np.array([s[0] for s in storia2]); amp2 = np.array([s[1] for s in storia2]); cop2 = np.array([s[2] for s in storia2])
    sel2 = (Ts2 > 2) & (Ts2 < 8)
    print(f"   tasso di crescita iniziale sigma = {np.polyfit(Ts2[sel2], np.log(amp2[sel2]), 1)[0]:.4f};"
          f" ampiezza finale {amp2[-1]:.4f}; coppia a regime {cop2[-1]:+.4f} ({cop2[-1]/coppia_couette:.2f} volte Couette)")

    fig, axs = plt.subplots(2, 3, figsize=(8.6, 4.6))
    for a_, Tk in zip(axs.ravel()[:5], (5.0, 15.0, 20.0, 60.0, -60.0)):
        Psi, W, Om = salvati[Tk] if Tk > 0 else salvati2[-Tk]
        Ze = np.concatenate([Z, [LAM]])
        Pe = np.concatenate([Psi, Psi[:, :1]], axis=1)
        m = np.max(np.abs(Pe))
        lv = np.linspace(-m, m, 11); lv = lv[np.abs(lv) > 1e-9]
        a_.contour(Ze, R, Pe, levels=lv, colors=[ROSSO if v < 0 else BLU for v in lv],
                   linestyles=["--" if v < 0 else "-" for v in lv], linewidths=0.8)
        a_.set_aspect("equal")
        a_.set_title((f"$T={abs(Tk):g}$, $|\\Psi|_{{\\max}}={m:.3f}$" + (" ($\\lambda/2$)" if Tk < 0 else "")).replace(".", "{,}"), fontsize=8)
        a_.set_xlabel("$Z$"); a_.set_ylabel("$R$")
    a_ = axs.ravel()[5]
    a_.semilogy(Ts, amp, color=BLU, label="$\\lambda=4$")
    a_.semilogy(Ts2, amp2, "--", color=ARANCIO, label="$\\lambda/2=2$")
    a_.set_xlabel("$T$"); a_.set_ylabel("$|\\Psi|_{\\max}$"); a_.set_title("crescita delle celle", fontsize=8); a_.legend(fontsize=7)
    fig.tight_layout()
    salva(fig, "prog4_3b_taylor")
