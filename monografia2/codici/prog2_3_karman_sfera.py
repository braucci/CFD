"""
prog2_3_karman_sfera.py -- PROGRAMMA 2.3
Metodo di von Karman per il flusso assialsimmetrico attorno a un corpo di
rivoluzione: n segmenti di sorgenti uniformi sull'asse, n-1 punti di
controllo sulla superficie + condizione di chiusura (portata totale nulla).
Verifica sulla sfera di raggio a, per cui la soluzione esatta e'
   V = 1,5 U r/a ,   C_p = 1 - (9/4)(r/a)^2.
Le funzioni karman_* sono riutilizzate da prog2_3b_ellissoide.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, ROSSO, GRIGIO, VERDE


def karman_risolvi(zz, r, z):
    """zz: n+1 estremi dei segmenti; (r, z): n-1 punti sulla superficie.
    Restituisce le intensita' adimensionali q_j/U e la matrice del sistema."""
    n = len(zz) - 1
    d = np.sqrt(r[:, None]**2 + (z[:, None] - zz[None, :])**2)   # (n-1) x (n+1)
    M = np.zeros((n, n))
    M[:-1, :] = d[:, :-1] - d[:, 1:]             # Delta d_ij = d_ij - d_i,j+1
    M[-1, :] = np.diff(zz)                       # chiusura: sum s_j q_j = 0
    b = np.concatenate([0.5 * r**2, [0.0]])
    q = np.linalg.solve(M, b)
    return q, M


def karman_velocita(q, zz, r, z, U=1.0):
    """Componenti (u_r, u_z) della velocita' nei punti (r, z), r > 0."""
    d = np.sqrt(r[:, None]**2 + (z[:, None] - zz[None, :])**2)
    g = (z[:, None] - zz[None, :]) / d
    ur = -U / r * ((g[:, 1:] - g[:, :-1]) @ q)
    uz = U * (1.0 + (1.0 / d[:, 1:] - 1.0 / d[:, :-1]) @ q)
    return ur, uz


def karman_psi(q, zz, R, Z, U=1.0):
    """Funzione di corrente (2D array) per il disegno delle linee di corrente."""
    psi = 0.5 * U * R**2
    for j in range(len(q)):
        psi = psi + U * q[j] * (np.sqrt(R**2 + (Z - zz[j + 1])**2) -
                                np.sqrt(R**2 + (Z - zz[j])**2))
    return psi


if __name__ == "__main__":
    A, U = 1.0, 1.0
    zz = -0.8 * A + 0.16 * A * np.arange(11)       # 10 segmenti tra -0,8a e 0,8a
    i = np.arange(1, 10)
    fi = (10 - i) * np.pi / 10                     # 9 punti sulla superficie
    r, z = A * np.sin(fi), A * np.cos(fi)

    q, M = karman_risolvi(zz, r, z)
    np.set_printoptions(precision=4, suppress=True, linewidth=110)
    print("Matrice del sistema (10 x 10) e termine noto:")
    print(np.column_stack([M, np.concatenate([0.5 * r**2, [0]])]))
    print("\nIntensita' q_j/U dei segmenti:")
    print(q)
    print(f"numero di condizionamento della matrice: {np.linalg.cond(M):.3e}")

    ur, uz = karman_velocita(q, zz, r, z, U)
    V = np.hypot(ur, uz)
    Vex = 1.5 * U * r / A
    cp, cpex = 1 - (V / U)**2, 1 - (Vex / U)**2
    print("\n   r/a     z/a      V/U      V/U esatta    C_p       C_p esatto   |errore C_p|")
    for k in range(9):
        print(f"  {r[k]:.4f}  {z[k]:+.4f}  {V[k]:.5f}   {Vex[k]:.5f}   {cp[k]:+.5f}   "
              f"{cpex[k]:+.5f}    {abs(cp[k]-cpex[k]):.1e}")

    # tangenza della velocita' (controllo indipendente): V . n = 0 sulla sfera
    print(f"\nmax |V.n|/U nei punti di controllo: "
          f"{np.max(np.abs(ur*np.sin(fi) + uz*np.cos(fi))):.2e}")

    # --- C_p su tutta la superficie (anche fra i punti di controllo) -------
    fic = np.linspace(0.02, np.pi - 0.02, 300)
    rc, zc = A * np.sin(fic), A * np.cos(fic)
    urc, uzc = karman_velocita(q, zz, rc, zc, U)
    vn = urc * np.sin(fic) + uzc * np.cos(fic)
    print(f"max |V.n|/U su tutta la superficie: {np.max(np.abs(vn)):.2e}")
    cpc = 1 - (urc**2 + uzc**2) / U**2

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
    # (a) geometria e linee di corrente
    zg = np.linspace(-2.2, 2.2, 441); rg = np.linspace(1e-3, 2.4, 241)
    ZG, RG = np.meshgrid(zg, rg, indexing="ij")
    P = karman_psi(q, zz, RG, ZG, U)
    P[np.hypot(ZG, RG) < A] = np.nan
    ax[0].contour(ZG, RG, P, levels=[0.02, 0.1, 0.25, 0.45, 0.7, 1.0, 1.35],
                  colors=BLU, linewidths=0.7, linestyles="solid")
    t = np.linspace(0, np.pi, 200)
    ax[0].fill_between(A * np.cos(t), 0, A * np.sin(t), color=GRIGIO, alpha=0.3)
    for j in range(10):
        col = ROSSO if q[j] > 0 else BLU
        ax[0].plot(zz[j:j + 2], [0.012, 0.012], color=col, lw=3.0, solid_capstyle="butt")
    ax[0].plot(zz, 0 * zz + 0.012, "|", color="k", ms=6)
    ax[0].plot(z, r, "o", color=VERDE, ms=4)
    ax[0].set_aspect("equal"); ax[0].set_xlabel("$z/a$"); ax[0].set_ylabel("$r/a$")
    ax[0].set_ylim(0, 1.8); ax[0].grid(False)
    # (b) C_p
    ax[1].plot(np.degrees(fic), 1 - 2.25 * np.sin(fic)**2, color=GRIGIO, label="esatto")
    ax[1].plot(np.degrees(fic), cpc, "--", color=ARANCIO, label="von Karman")
    ax[1].plot(np.degrees(fi), cp, "o", color=VERDE, ms=4, label="punti di controllo")
    ax[1].invert_yaxis(); ax[1].set_xlabel(r"$\varphi$ [gradi]"); ax[1].set_ylabel("$C_p$")
    ax[1].set_xticks([0, 45, 90, 135, 180]); ax[1].legend()
    salva(fig, "prog2_3_sfera")
