"""
prog2_4c_pannelli.py -- metodo dei pannelli di vortice a intensita' lineare
per profili di forma qualsiasi (metodo diretto).

Il contorno e' approssimato da N pannelli rettilinei percorsi in senso orario
(bordo d'uscita -> ventre -> bordo d'attacco -> dorso -> bordo d'uscita).
Su ogni pannello la vorticita' gamma varia linearmente; le N+1 incognite sono i
valori di gamma nei nodi. Equazioni: velocita' normale nulla nei N punti di
controllo (punti medi) + condizione di Kutta  gamma_1 + gamma_{N+1} = 0.
Coefficienti di influenza in forma chiusa (formulazione classica a vortici
lineari; le incognite sono gamma' = gamma / (2 pi V_inf)).

1) Validazione: profilo di Joukowski del Programma 2.4 (soluzione esatta nota).
2) Verifica con dati: NACA 0009 a incidenza nulla (velocita' sul dorso).
3) Applicazione: NACA 2412 e 4415, C_L e C_m,c/4 in funzione di alfa.
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, ROSSO, GRIGIO, VERDE, VIOLA
import prog2_4_joukowski as jk


def naca4(codice, n_pannelli=100, corda=1.0):
    """Nodi di un profilo NACA a 4 cifre (spaziatura a coseno, bordo d'uscita
    chiuso), ordinati in senso orario a partire dal bordo d'uscita."""
    m = int(codice[0]) / 100.0
    p = int(codice[1]) / 10.0
    t = int(codice[2:]) / 100.0
    nlato = n_pannelli // 2
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, nlato + 1)))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 - 0.1036 * x**4)          # -0.1036: b.u. chiuso
    yc = np.zeros_like(x); dyc = np.zeros_like(x)
    if m > 0:
        a = x < p
        yc[a] = m / p**2 * (2 * p * x[a] - x[a]**2)
        dyc[a] = 2 * m / p**2 * (p - x[a])
        yc[~a] = m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x[~a] - x[~a]**2)
        dyc[~a] = 2 * m / (1 - p)**2 * (p - x[~a])
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    XB = np.concatenate([xl[::-1], xu[1:]]) * corda     # b.u. -> ventre -> b.a. -> dorso
    YB = np.concatenate([yl[::-1], yu[1:]]) * corda
    return XB, YB


def pannelli_vortice(XB, YB, alfa):
    """Restituisce i punti di controllo, gli angoli dei pannelli, la velocita'
    tangenziale V/V_inf, C_p e i valori gamma' nei nodi."""
    M = len(XB) - 1
    X = 0.5 * (XB[:-1] + XB[1:]); Y = 0.5 * (YB[:-1] + YB[1:])
    S = np.hypot(np.diff(XB), np.diff(YB))
    TH = np.arctan2(np.diff(YB), np.diff(XB))
    sn, cs = np.sin(TH), np.cos(TH)
    # matrici dei coefficienti (i = punto di controllo, j = pannello)
    dx = X[:, None] - XB[None, :-1]
    dy = Y[:, None] - YB[None, :-1]
    Ai = -dx * cs[None, :] - dy * sn[None, :]
    Bi = dx**2 + dy**2
    Ci = np.sin(TH[:, None] - TH[None, :])
    Di = np.cos(TH[:, None] - TH[None, :])
    Ei = dx * sn[None, :] - dy * cs[None, :]
    Sj = S[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.log(1.0 + Sj * (Sj + 2.0 * Ai) / Bi)
        G = np.arctan2(Ei * Sj, Bi + Ai * Sj)
        P = dx * np.sin(TH[:, None] - 2 * TH[None, :]) + dy * np.cos(TH[:, None] - 2 * TH[None, :])
        Q = dx * np.cos(TH[:, None] - 2 * TH[None, :]) - dy * np.sin(TH[:, None] - 2 * TH[None, :])
        CN2 = Di + 0.5 * Q * F / Sj - (Ai * Ci + Di * Ei) * G / Sj
        CN1 = 0.5 * Di * F + Ci * G - CN2
        CT2 = Ci + 0.5 * P * F / Sj + (Ai * Di - Ci * Ei) * G / Sj
        CT1 = 0.5 * Ci * F - Di * G - CT2
    k = np.arange(M)
    CN1[k, k], CN2[k, k] = -1.0, 1.0           # auto-influenza del pannello
    CT1[k, k], CT2[k, k] = 0.5 * np.pi, 0.5 * np.pi
    AN = np.zeros((M + 1, M + 1)); AT = np.zeros((M, M + 1))
    AN[:M, :M] += CN1; AN[:M, 1:] += CN2
    AT[:, :M] += CT1; AT[:, 1:] += CT2
    AN[M, 0] = AN[M, M] = 1.0                  # condizione di Kutta
    rhs = np.concatenate([np.sin(TH - alfa), [0.0]])
    gam = np.linalg.solve(AN, rhs)
    V = np.cos(TH - alfa) + AT @ gam
    cp = 1.0 - V**2
    return X, Y, TH, S, V, cp, gam


def coefficienti(XB, YB, X, Y, TH, S, cp, gam, alfa, corda=1.0):
    """C_L dall'integrale della pressione e dalla circolazione; C_m,c/4."""
    nx, ny = -np.sin(TH), np.cos(TH)            # normale esterna (contorno orario)
    fx = np.sum(-cp * nx * S) / corda
    fy = np.sum(-cp * ny * S) / corda
    CL = fy * np.cos(alfa) - fx * np.sin(alfa)
    CD = fx * np.cos(alfa) + fy * np.sin(alfa)
    Gam = 2 * np.pi * np.sum(0.5 * (gam[:-1] + gam[1:]) * S)   # in unita' di V_inf
    CL_gamma = 2 * Gam / corda
    xr = X - 0.25 * corda
    # momento rispetto a c/4, positivo se cabrante (a naso in su)
    Cm = np.sum(cp * (xr * ny - Y * nx) * S) / corda**2
    return CL, CD, CL_gamma, Cm


if __name__ == "__main__":
    # ------------------------------------------------ 1) validazione Joukowski
    print("1) Profilo di Joukowski (a=1, b=0,8, y_P=0,199): confronto con l'esatto")
    print("   N     C_L pannelli   C_L esatto   max|C_p - C_p,ex| per x/c < 0,9")
    for alfa_g in (0.0,):
        alfa = np.radians(alfa_g)
        for N in (40, 80, 160, 320):
            fi = -jk.BETA - np.linspace(0, 2 * np.pi, N + 1)       # senso orario
            zb = jk.joukowski(jk.A * np.exp(1j * fi) + jk.ZP)
            XB, YB = zb.real, zb.imag
            X, Y, TH, S, V, cp, gam = pannelli_vortice(XB, YB, alfa)
            corda = XB.max() - XB.min()
            CL, CD, CLg, Cm = coefficienti(XB, YB, X, Y, TH, S, cp, gam, alfa, corda)
            # soluzione esatta nei punti di controllo: anomalia media del pannello
            fim = 0.5 * (fi[:-1] + fi[1:])
            Vex = jk._velocita_cerchio(fim, alfa)
            cpex = 1 - Vex**2
            xs = (X - XB.min()) / (XB.max() - XB.min())
            err = np.max(np.abs(cp - cpex)[xs < 0.9])
            CLex = 2 * jk.circolazione(alfa) / corda
            print(f"  {N:4d}     {CL:8.5f}      {CLex:8.5f}      {err:.2e}")
            if N == 160:
                salvati = (X, cp, fim, cpex)
    X, cp, fim, cpex = salvati
    fi_f, z_f, V_f = jk.superficie(400, 0.0)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(z_f.real, 1 - V_f**2, color=GRIGIO, lw=2.0, label="trasformazione conforme (esatta)")
    ax.plot(X, cp, "o", color=BLU, ms=3, mfc="none", label="pannelli di vortice, $N=160$")
    ax.invert_yaxis(); ax.set_xlabel("$x$ [m]"); ax.set_ylabel("$C_p$"); ax.legend()
    salva(fig, "prog2_4c_validazione")

    # ------------------------------------------------ 2) NACA 0009, alfa = 0
    tab_x = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    tab_V = np.array([1.140, 1.144, 1.137, 1.119, 1.100, 1.082, 1.061, 1.043, 1.018, 0.982])
    XB, YB = naca4("0009", 160)
    X, Y, TH, S, V, cp, gam = pannelli_vortice(XB, YB, 0.0)
    dorso = Y > 0
    Vd = np.interp(tab_x, X[dorso], np.abs(V[dorso]))
    print("\n2) NACA 0009, alfa = 0: velocita' sul dorso")
    print("   x/c    V/U tabella   V/U pannelli   scarto")
    for xx, vt, vp in zip(tab_x, tab_V, Vd):
        print(f"  {xx:4.2f}     {vt:5.3f}        {vp:6.4f}      {vp-vt:+.4f}")

    # ------------------------------------------------ 3) NACA 2412 e 4415
    print("\n3) Curve C_L(alfa) e C_m,c/4(alfa) (N = 160)")
    alfas = np.radians(np.arange(-4, 11, 2))
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for codice, col in (("0012", GRIGIO), ("2412", BLU), ("4415", ARANCIO)):
        XB, YB = naca4(codice, 160)
        CLs, Cms = [], []
        for al in alfas:
            X, Y, TH, S, V, cp, gam = pannelli_vortice(XB, YB, al)
            CL, CD, CLg, Cm = coefficienti(XB, YB, X, Y, TH, S, cp, gam, al)
            CLs.append(CL); Cms.append(Cm)
        CLs = np.array(CLs)
        pend = np.polyfit(alfas, CLs, 1)
        a0L = -pend[1] / pend[0]
        print(f"  NACA {codice}: dC_L/dalfa = {pend[0]:.3f} /rad ({pend[0]/(2*np.pi):.3f} x 2 pi), "
              f"alfa_0L = {np.degrees(a0L):+.2f} gradi, C_L(4 gradi) = {CLs[4]:.3f}, "
              f"C_m,c/4 medio = {np.mean(Cms):+.4f}")
        ax[0].plot(np.degrees(alfas), CLs, "o-", ms=3, color=col, label=f"NACA {codice}")
        ax[1].plot(np.degrees(alfas), Cms, "o-", ms=3, color=col)
    ax[0].set_xlabel(r"$\alpha$ [gradi]"); ax[0].set_ylabel("$C_L$"); ax[0].legend()
    ax[1].set_xlabel(r"$\alpha$ [gradi]"); ax[1].set_ylabel(r"$C_{m,c/4}$")
    salva(fig, "prog2_4c_naca")

    # distribuzione di C_p per NACA 2412 a 4 gradi
    XB, YB = naca4("2412", 160)
    X, Y, TH, S, V, cp, gam = pannelli_vortice(XB, YB, np.radians(4.0))
    fig, ax = plt.subplots(2, 1, figsize=(5.4, 4.4), sharex=True,
                           gridspec_kw={"height_ratios": [1.2, 3]})
    ax[0].plot(XB, YB, color="k", lw=0.8)
    ax[0].set_ylim(-0.08, 0.16); ax[0].set_yticks([]); ax[0].grid(False)
    sup = np.arange(len(X)) >= len(X) // 2
    ax[1].plot(X[sup], cp[sup], "-", color=BLU, label="dorso")
    ax[1].plot(X[~sup], cp[~sup], "-", color=ARANCIO, label="ventre")
    ax[1].invert_yaxis(); ax[1].set_xlabel("$x/c$"); ax[1].set_ylabel("$C_p$"); ax[1].legend()
    salva(fig, "prog2_4c_cp2412")
