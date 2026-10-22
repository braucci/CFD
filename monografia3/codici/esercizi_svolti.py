"""
esercizi_svolti.py -- codice degli esercizi svolti della monografia 3 (flussi viscosi).

Ogni esercizio e' una funzione es_NN(); senza argomenti si eseguono tutti,
altrimenti solo quelli indicati:   python3 esercizi_svolti.py 3 7
Le figure vengono salvate come es_NN.pdf nella cartella figure/.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp, simpson
from scipy.linalg import eig
from scipy.optimize import brentq, minimize_scalar
from scipy.special import jv, jn_zeros, erfc
from metodi_numerici import trid, thomas, rk4_integra, cheb, numeri_onda
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, VIOLA, GIALLO, GRIGIO, PALETTE

Q0_BLASIUS = 0.332057336215


def blasius(eta, z):
    f, p, q = z
    return np.array([p, q, -0.5 * f * q])


def soluzione_blasius(etamax, h=0.01):
    """f, f', f'' su [0, etamax] (oltre eta = 10 si prolunga con f = eta - 1,7208)."""
    e10, Z = rk4_integra(blasius, np.array([0.0, 0.0, Q0_BLASIUS]), min(10.0, etamax), h)
    if etamax <= 10.0:
        return e10, Z[:, 0], Z[:, 1], Z[:, 2]
    eta = np.arange(0.0, etamax + h / 2, h)
    f = np.where(eta <= 10, np.interp(eta, e10, Z[:, 0]), Z[-1, 0] + (eta - 10.0))
    p = np.where(eta <= 10, np.interp(eta, e10, Z[:, 1]), 1.0)
    q = np.where(eta <= 10, np.interp(eta, e10, Z[:, 2]), 0.0)
    return eta, f, p, q


# =========================================================================
def es_01():
    """Trasformazione di Topfer: f''(0) con una sola integrazione."""
    print("\n=== Esercizio 1: f''(0) di Blasius senza iterazioni")
    def g_eq(z, y):
        g, gp, gpp = y
        return [gp, gpp, -0.5 * g * gpp]
    for zmax in (6.0, 8.0, 10.0, 15.0):
        s = solve_ivp(g_eq, (0, zmax), [0, 0, 1.0], rtol=1e-12, atol=1e-13, method="DOP853")
        k = np.sqrt(s.y[1, -1])
        print(f"   z_max = {zmax:5.1f}:  g'(z_max) = {s.y[1,-1]:.10f}  k = {k:.8f}  "
              f"f''(0) = k^-3 = {k**-3:.10f}")
    # profili: f(eta) = g(z)/k con eta = k z, confronto con l'integrazione diretta
    s = solve_ivp(g_eq, (0, 10.0), [0, 0, 1.0], rtol=1e-12, atol=1e-13, dense_output=True)
    k = np.sqrt(s.y[1, -1])
    eta = np.linspace(0, 8, 81)
    fp_topfer = s.sol(eta / k)[1] / k**2          # f' = g'/k^2
    _, Z = rk4_integra(blasius, np.array([0, 0, k**-3]), 8.0, 0.1)
    print(f"   scarto massimo fra f' di Topfer e quella integrata con f''(0) = k^-3: "
          f"{np.max(np.abs(fp_topfer - Z[:, 1])):.1e}")
    zz = np.linspace(0, 6, 200)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax[0].plot(s.sol(zz)[1], zz, color=ARANCIO, label="$g'(z)$")
    ax[0].axvline(k**2, color=GRIGIO, ls=":", label=f"$k^2={k**2:.4f}$".replace(".", "{,}"))
    ax[0].set_xlabel("$g'$")
    ax[0].set_ylabel("$z$")
    ax[0].legend(fontsize=7)
    ax[0].set_title("problema ausiliario, $g''(0)=1$")
    ax[1].plot(fp_topfer, eta, color=BLU, label="da $g$: $f'=g'/k^2$, $\\eta=kz$")
    ax[1].plot(Z[::4, 1], eta[::4], "o", ms=3, mfc="none", color=ROSSO, label="integrazione diretta")
    ax[1].set_xlabel("$f'=u/U$")
    ax[1].set_ylabel("$\\eta$")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    salva(fig, "es_01")


# =========================================================================
def _tiro_dimezzamento(fw, etamax=10.0, h=0.05, q0=1.0, dq0=0.2, eps=1e-6, itmax=400):
    """Dimezzamento dell'intervallo con il criterio generale: si inverte e si
    dimezza il passo quando error*error' < 0 oppure |error'| - |error| < 0."""
    err_prec = None
    for it in range(1, itmax + 1):
        q0 += dq0
        _, Z = rk4_integra(blasius, np.array([fw, 0.0, q0]), etamax, h)
        err = 1.0 - Z[-1, 1]
        if not np.isfinite(err):
            err = -1e3
        if 0.0 <= err <= eps:
            return q0, it
        if err_prec is not None and (err * err_prec < 0 or abs(err_prec) - abs(err) < 0):
            dq0 = -dq0 / 2.0
        err_prec = err
    raise RuntimeError("dimezzamento non convergente")


def _tiro_newton(fw, q0, etamax, h=0.02, tol=1e-12):
    def F(eta, z):
        f, p, q, Fs, P, Q = z
        return np.array([p, q, -0.5 * f * q, P, Q, -0.5 * (Fs * q + f * Q)])
    for _ in range(60):
        eta, Z = rk4_integra(F, np.array([fw, 0, q0, 0, 0, 1.0]), etamax, h)
        r = Z[-1, 1] - 1.0
        if abs(r) < tol:
            return q0, eta, Z
        q0 = q0 - r / Z[-1, 4]
    raise RuntimeError("Newton non convergente")


def es_02():
    """Iniezione e aspirazione: f(0) = -2C."""
    print("\n=== Esercizio 2: lamina piana con iniezione (C > 0) o aspirazione (C < 0)")
    print("     C     f''(0) dimezz.  iter   f''(0) Newton   delta_99   eta_flesso")
    Cs = np.round(np.arange(-0.6, 0.61, 0.1), 2)
    q0s, deltas, profili = [], [], {}
    q_guess = 0.33
    for C in Cs:
        q_dim, it = _tiro_dimezzamento(-2 * C)
        q_n, eta, Z = _tiro_newton(-2 * C, q_dim, etamax=14.0)
        p, q = Z[:, 1], Z[:, 2]
        d99 = np.interp(0.99, p[: np.argmax(p > 0.995)], eta[: np.argmax(p > 0.995)])
        # punto di flesso del profilo: f''' = -(1/2) f f'' = 0  <=>  f = 0 (con f'' > 0)
        f = Z[:, 0]
        fl = eta[np.argmax(f >= 0)] if C > 0 else np.nan
        q0s.append(q_n)
        deltas.append(d99)
        profili[C] = (eta, p)
        print(f"   {C:+.1f}   {q_dim:11.6f}   {it:4d}    {q_n:11.6f}    {d99:7.3f}    "
              + (f"{fl:6.3f}" if C > 0 else "   -"))
    # limite di "soffiamento" (blowing-off): f''(0) -> 0
    print("   continuazione in C con eta_max = 40 (Newton):")
    q = q0s[-1]
    ultimi = []
    for C in np.arange(0.60, 0.6301, 0.002):
        try:
            q, _, _ = _tiro_newton(-2 * C, q, etamax=40.0, h=0.02)
        except RuntimeError:
            break
        ultimi.append((C, q))
        if q < 2e-4:
            break
    for C, q in ultimi[-4:]:
        print(f"      C = {C:.3f}   f''(0) = {q:.6f}")
    (C1, q1), (C2, q2) = ultimi[-2], ultimi[-1]
    print(f"   estrapolazione lineare: f''(0) = 0 per C = {C2 - q2 * (C2 - C1) / (q2 - q1):.4f}")
    fig, ax = plt.subplots(1, 3, figsize=(7.8, 2.8))
    for C, col in zip([-0.6, -0.3, 0.0, 0.3, 0.6], [VERDE, BLU, "k", ARANCIO, ROSSO]):
        eta, p = profili[C]
        ax[0].plot(p, eta, color=col, label=f"$C={C:+.1f}$")
    ax[0].set_ylim(0, 12)
    ax[0].set_xlabel("$f'=u/U$")
    ax[0].set_ylabel("$\\eta$")
    ax[0].legend(fontsize=6.5)
    ax[1].plot(Cs, q0s, "o-", color=BLU, ms=3)
    ax[1].set_xlabel("$C$")
    ax[1].set_ylabel("$f''(0)$ (sforzo alla parete)")
    ax[2].plot(Cs, deltas, "s-", color=ARANCIO, ms=3)
    ax[2].set_xlabel("$C$")
    ax[2].set_ylabel("$\\eta_{99}$ (spessore)")
    fig.tight_layout()
    salva(fig, "es_02")


# =========================================================================
def _falkner_skan(beta, q0, etamax=10.0):
    """f''' + f f'' + beta (1 - f'^2) = 0; Newton sul valore iniziale f''(0)."""
    def F(eta, z):
        f, p, q, Fs, P, Q = z
        return [p, q, -f * q - beta * (1 - p * p), P, Q, -(Fs * q + f * Q) + 2 * beta * p * P]
    for _ in range(50):
        s = solve_ivp(F, (0, etamax), [0, 0, q0, 0, 0, 1.0], rtol=1e-11, atol=1e-12, method="DOP853")
        r = s.y[1, -1] - 1.0
        if abs(r) < 1e-10:
            return q0
        q0 -= r / s.y[4, -1]
    raise RuntimeError(f"Falkner-Skan: Newton non converge per beta = {beta}")


def es_03():
    """Equazione di Falkner-Skan."""
    print("\n=== Esercizio 3: soluzioni di Falkner-Skan, U = U0 x^m, beta = 2m/(m+1)")
    print("      m        beta      f''(0)    angolo del cuneo")
    ms = [1.0, 0.5, 0.0, -0.05, -0.09]
    q = 1.2
    ris = {}
    for m in ms:
        beta = 2 * m / (m + 1)
        q = _falkner_skan(beta, q, etamax=10.0 if m >= 0 else 14.0)
        ris[m] = q
        cuneo = f"{beta*180:7.1f} gradi" if m >= 0 else "  (gradiente avverso)"
        print(f"   {m:+6.3f}   {beta:+7.4f}   {q:9.5f}    {cuneo}")
    # distacco: si impone f''(0) = 0 e si cerca beta tale che f'(eta_max) = 1
    def residuo(beta, etamax=12.0):
        s = solve_ivp(lambda e, z: [z[1], z[2], -z[0] * z[2] - beta * (1 - z[1]**2)],
                      (0, etamax), [0, 0, 0.0], rtol=1e-12, atol=1e-13, method="DOP853")
        return s.y[1, -1] - 1.0
    for emax in (10.0, 12.0, 14.0):
        beta_s = brentq(lambda b: residuo(b, emax), -0.21, -0.18, xtol=1e-12)
        print(f"   distacco (f''(0) = 0), eta_max = {emax:4.1f}: beta_s = {beta_s:.6f},"
              f"  m_s = beta/(2 - beta) = {beta_s/(2 - beta_s):.6f}")
    betas = np.linspace(beta_s + 1e-4, 0.0, 24)
    qs = []
    q = 0.05
    for b in betas:
        q = _falkner_skan(b, max(q, 0.02), etamax=14.0)
        qs.append(q)
    qs = np.array(qs)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    eta = np.linspace(0, 8, 300)
    for m, col in zip(ms, [ROSSO, ARANCIO, "k", BLU, VERDE]):
        beta = 2 * m / (m + 1)
        s = solve_ivp(lambda e, z: [z[1], z[2], -z[0] * z[2] - beta * (1 - z[1]**2)],
                      (0, 8), [0, 0, ris[m]], rtol=1e-10, atol=1e-12, dense_output=True)
        ax[0].plot(s.sol(eta)[1], eta, color=col, label=f"$m={m:+.2f}$".replace(".", "{,}"))
    ax[0].set_xlabel("$f'=u/U$")
    ax[0].set_ylabel("$\\eta$")
    ax[0].legend(fontsize=6.5)
    ax[1].plot(betas, qs, "o-", ms=3, color=BLU)
    ax[1].plot([beta_s], [0.0], "o", color=ROSSO, label="distacco")
    ax[1].axvline(beta_s, color=GRIGIO, ls=":")
    ax[1].set_xlabel("$\\beta=2m/(m+1)$")
    ax[1].set_ylabel("$f''(0)$")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    salva(fig, "es_03")


# =========================================================================
def _theta_adiabatica(Pr, etamax, n):
    eta, f, p, q = soluzione_blasius(etamax, h=etamax / n)
    h = eta[1]
    C = np.zeros((n, 4))
    C[:, 0] = 1 - h / 4 * Pr * f[:n]
    C[:, 1] = -2.0
    C[:, 2] = 1 + h / 4 * Pr * f[:n]
    C[:, 3] = -2 * h * h * Pr * q[:n] ** 2
    C[0, 2] += C[0, 0]
    C[0, 0] = 0.0
    C[n - 1, 2] = 0.0
    th = np.zeros(n + 1)
    th[:n] = trid(C)
    return eta, p, th


def es_04():
    """Distribuzione dell'entalpia totale nello strato limite."""
    print("\n=== Esercizio 4: entalpia totale (H - H1)/(U^2/2) = theta + f'^2 - 1")
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for (nome, Pr), col in zip([("acqua", 6.75), ("aria", 0.714), ("mercurio", 0.044), ("Pr = 1", 1.0)],
                               [BLU, ARANCIO, VERDE, GRIGIO]):
        etamax = 12.0 if Pr > 0.5 else 60.0
        eta, p, th = _theta_adiabatica(Pr, etamax, int(etamax / 0.02))
        dH = th + p**2 - 1
        bilancio = simpson(p * dH, x=eta)
        i_min, i_max = np.argmin(dH), np.argmax(dH)
        print(f"   {nome:9s}: parete {dH[0]:+.4f}; min {dH[i_min]:+.4f} (eta = {eta[i_min]:.2f}),"
              f" max {dH[i_max]:+.4f} (eta = {eta[i_max]:.2f});  int f'(H-H1) deta = {bilancio:+.1e}")
        ax.plot(dH, eta, color=col, label=f"{nome}" + (f" ($Pr={Pr}$)" if "Pr" not in nome else ""))
    ax.axvline(0, color="k", lw=0.6)
    ax.set_ylim(0, 10)
    ax.set_xlabel("$(H-H_1)/(U^2/2)=\\theta+f'^2-1$")
    ax.set_ylabel("$\\eta$")
    ax.legend(fontsize=7)
    salva(fig, "es_04")


# =========================================================================
def _cilindro_rotante(Ra, n, condizione="dV/dR"):
    """V'' + V'/R - V/R^2 = 0 su [1, Ra], V(1) = 1; in Ra:
       'dV/dR'  : dV/dR = 0;      'sforzo' : d(V/R)/dR = 0, cioe' dV/dR = V/R."""
    R = np.linspace(1.0, Ra, n + 1)
    h = R[1] - R[0]
    # incognite V_1..V_n (V_0 = 1 noto); nodo fittizio V_(n+1)
    C = np.zeros((n, 4))
    Ri = R[1:]
    C[:, 0] = 1 / h**2 - 1 / (2 * h * Ri)
    C[:, 1] = -2 / h**2 - 1 / Ri**2
    C[:, 2] = 1 / h**2 + 1 / (2 * h * Ri)
    C[0, 3] = -C[0, 0] * 1.0
    C[0, 0] = 0.0
    if condizione == "dV/dR":                       # V_(n+1) = V_(n-1)
        C[n - 1, 0] += C[n - 1, 2]
    else:                                           # V_(n+1) = V_(n-1) + 2h V_n / Ra
        C[n - 1, 0] += C[n - 1, 2]
        C[n - 1, 1] += C[n - 1, 2] * 2 * h / Ra
    C[n - 1, 2] = 0.0
    V = np.concatenate([[1.0], trid(C)])
    return R, V


def es_05():
    """Fluido attorno a un cilindro rotante, con superficie libera."""
    print("\n=== Esercizio 5: cilindro rotante con strato di fluido a superficie libera (Ra = 2)")
    Ra, n = 2.0, 50
    R, V = _cilindro_rotante(Ra, n)
    Ves = Ra / (1 + Ra**2) * (Ra / R + R / Ra)
    err = 100 * (V - Ves) / Ves
    print(f"   condizione dV/dR = 0: errore percentuale massimo {np.max(np.abs(err)):.4f} %"
          f" (in R = {R[np.argmax(np.abs(err))]:.2f}); V(Ra) = {V[-1]:.5f} (esatto {Ves[-1]:.5f})")
    for i in range(0, n + 1, 10):
        print(f"      R = {R[i]:.2f}   V = {V[i]:.6f}   V_es = {Ves[i]:.6f}   errore = {err[i]:+.4f} %")
    R2, V2 = _cilindro_rotante(Ra, n, "sforzo")
    print(f"   condizione di sforzo nullo d(V/R)/dR = 0: max |V - R| = {np.max(np.abs(V2 - R2)):.1e}"
          " (rotazione rigida)")
    tau_a = (np.gradient(V, R) - V / R)[-1]
    print(f"   con dV/dR = 0 lo sforzo sulla superficie libera NON e' nullo: tau/(mu v_b/r_b) = {tau_a:+.4f}")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax[0].plot(R, Ves, color=GRIGIO, lw=2, alpha=0.6, label="esatta (3.32)")
    ax[0].plot(R[::5], V[::5], "o", ms=3.5, color=BLU, mfc="none", label="TRID, $dV/dR=0$")
    ax[0].plot(R2, V2, "--", color=ROSSO, label="sforzo nullo: $V=R$")
    ax[0].set_xlabel("$R=r/r_b$")
    ax[0].set_ylabel("$V=v_\\theta/v_b$")
    ax[0].legend(fontsize=7)
    ax[1].plot(R, err, color=BLU)
    ax[1].set_xlabel("$R$")
    ax[1].set_ylabel("errore percentuale")
    fig.tight_layout()
    salva(fig, "es_05")


# =========================================================================
def _laplace_canale(n, dirichlet_sup, neumann_sup):
    """lap U = 0 nel quadrato [0,1]^2 (nodi i: z, j: y), U = 0 su fondo e pareti;
    sulla riga superiore j = n: dirichlet_sup[i] (valore) dove neumann_sup[i] e' False,
    altrimenti dU/dy = 0 (nodo fittizio). Metodo diretto per matrici sparse."""
    nz = n - 1
    incognite = {}
    for j in range(1, n + 1):
        for i in range(1, n):
            if j < n or neumann_sup[i]:
                incognite[(i, j)] = len(incognite)
    A = sp.lil_matrix((len(incognite), len(incognite)))
    b = np.zeros(len(incognite))
    def val(i, j):
        if j == n and not neumann_sup[i]:
            return dirichlet_sup[i]
        return 0.0
    for (i, j), r in incognite.items():
        A[r, r] = -4.0
        vicini = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        if j == n:                                   # riga superiore libera: fittizio = (i, n-1)
            vicini[3] = (i, j - 1)
        for (a, c) in vicini:
            if (a, c) in incognite:
                A[r, incognite[(a, c)]] += 1.0
            elif 0 < a < n and 0 < c <= n:
                b[r] -= val(a, c)
    u = spla.spsolve(A.tocsr(), b)
    U = np.zeros((n + 1, n + 1))
    for i in range(1, n):
        U[i, n] = val(i, n)
    for (i, j), r in incognite.items():
        U[i, j] = u[r]
    return U


def _portata(U, h):
    w = np.ones_like(U)
    w[0, :] *= 0.5
    w[-1, :] *= 0.5
    w[:, 0] *= 0.5
    w[:, -1] *= 0.5
    return np.sum(w * U) * h * h


def es_06():
    """Canale quadrato con parete superiore mobile."""
    print("\n=== Esercizio 6: canale quadrato con parete superiore mobile (velocita' u0)")
    def serie_Q(nmax=4001):
        Q = 0.0
        for k in range(1, nmax, 2):
            Q += 8 / (k * np.pi) ** 3 * np.tanh(k * np.pi / 2)   # (cosh x - 1)/sinh x = tanh(x/2)
        return Q
    print(f"   portata esatta (serie): Q/(u0 L^2) = {serie_Q():.6f}  (argomento di simmetria: 1/4)")
    for n in (20, 40, 80, 160):
        sup = np.ones(n + 1)
        sup[0] = sup[-1] = 0.0                   # spigoli a velocita' nulla
        U = _laplace_canale(n, sup, np.zeros(n + 1, dtype=bool))
        Q = _portata(U, 1.0 / n)
        print(f"   h = 1/{n:<4d}  Q = {Q:.6f}   errore = {Q - 0.25:+.2e};  U al centro = {U[n//2, n//2]:.5f}")
    zz = np.linspace(0, 1, n + 1)
    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    cs = ax.contour(zz, zz, U.T, levels=[0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9], colors=BLU,
                    linewidths=0.9)
    ax.clabel(cs, fmt=lambda v: f"{v:.2f}".replace(".", ","), fontsize=6)
    ax.plot([0, 1], [1, 1], color=ROSSO, lw=3)
    ax.set_aspect("equal")
    ax.set_xlabel("$z/L$")
    ax.set_ylabel("$y/L$")
    ax.set_title("$u/u_0$, parete superiore mobile")
    salva(fig, "es_06")


# =========================================================================
def es_07():
    """Lastra mobile larga L/2 su un canale a pelo libero."""
    print("\n=== Esercizio 7: lastra mobile di larghezza L/2, resto della superficie libero")
    n = 80
    h = 1.0 / n
    pos = np.linspace(0.0, 0.5, 11)              # ascissa del bordo sinistro della lastra
    Qs = []
    for s in pos:
        z = np.linspace(0, 1, n + 1)
        sotto = (z >= s - 1e-12) & (z <= s + 0.5 + 1e-12)
        sup = np.where(sotto, 1.0, 0.0)
        neu = ~sotto
        neu[0] = neu[-1] = False
        sup[0] = sup[-1] = 0.0
        U = _laplace_canale(n, sup, neu)
        Qs.append(_portata(U, h))
        if s in (0.0, 0.25):
            print(f"   lastra fra z = {s:.2f} e {s+0.5:.2f}: Q/(u0 L^2) = {Qs[-1]:.5f}")
    Qs = np.array(Qs)
    print(f"   massimo per lastra centrata (s = {pos[np.argmax(Qs)]:.2f}): Q = {Qs.max():.5f};"
          f" minimo con la lastra contro una parete: Q = {Qs.min():.5f}")
    # confronto: lastra larga quanto il canale (Esercizio 6) -> 0,25
    s = 0.25
    z = np.linspace(0, 1, n + 1)
    sotto = (z >= s - 1e-12) & (z <= s + 0.5 + 1e-12)
    sup = np.where(sotto, 1.0, 0.0)
    neu = ~sotto
    neu[0] = neu[-1] = False
    U = _laplace_canale(n, sup, neu)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax[0].plot(pos, Qs, "o-", ms=3.5, color=BLU)
    ax[0].set_xlabel("posizione $s/L$ del bordo sinistro della lastra")
    ax[0].set_ylabel("$Q/(u_0L^2)$")
    cs = ax[1].contour(z, z, U.T, levels=[0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8], colors=BLU,
                       linewidths=0.9)
    ax[1].clabel(cs, fmt=lambda v: f"{v:.2f}".replace(".", ","), fontsize=6)
    ax[1].plot([0.25, 0.75], [1, 1], color=ROSSO, lw=3)
    ax[1].plot([0, 0.25], [1, 1], "--", color=GRIGIO)
    ax[1].plot([0.75, 1], [1, 1], "--", color=GRIGIO)
    ax[1].set_aspect("equal")
    ax[1].set_title("lastra centrata")
    fig.tight_layout()
    salva(fig, "es_07")


# =========================================================================
def es_08():
    """Instabilita' dello schema esplicito per R > 1/2."""
    print("\n=== Esercizio 8: schema esplicito con R > 1/2")
    M = 21
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for R, col in ((0.5, BLU), (0.55, ARANCIO), (0.6, ROSSO)):
        u = np.zeros(M)
        u[-1] = 1.0
        storia = []
        for j in range(1, 201):
            u[1:-1] = u[1:-1] + R * (u[:-2] - 2 * u[1:-1] + u[2:])
            storia.append(np.max(np.abs(u)))
        print(f"   R = {R:.2f}: max|u| dopo 50, 100, 200 passi = {storia[49]:.3e}, {storia[99]:.3e},"
              f" {storia[199]:.3e};  |G|max = |1 - 4R sin^2(pi/2 * 19/20)| ="
              f" {abs(1 - 4*R*np.sin(np.pi*19/40)**2):.4f}")
        ax[0].semilogy(np.arange(1, 201), storia, color=col, label=f"$R={R}$")
        if R == 0.6:
            for j in (40, 60):
                u2 = np.zeros(M)
                u2[-1] = 1.0
                for _ in range(j):
                    u2[1:-1] = u2[1:-1] + R * (u2[:-2] - 2 * u2[1:-1] + u2[2:])
                ax[1].plot(u2, np.linspace(0, 1, M), "o-", ms=2.5, label=f"{j} passi")
    ax[0].set_xlabel("passo temporale")
    ax[0].set_ylabel("$\\max|u|$ (m/s)")
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("$u$ (m/s)")
    ax[1].set_ylabel("$y$ (m)")
    ax[1].set_title("$R=0{,}6$: oscillazione a dente di sega")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    salva(fig, "es_08")



# =========================================================================
def _spin_up(Ra, n, T_fin, condizione, istanti, Rd=0.25):
    """Schema esplicito per V_T = V_RR + V_R/R - V/R^2 su [1, Ra]."""
    R = np.linspace(1.0, Ra, n + 1)
    h = R[1] - R[0]
    tau = Rd * h * h
    V = np.zeros(n + 1)
    V[0] = 1.0                                       # il cilindro parte bruscamente
    out = {}
    t = 0.0
    k = 0
    istanti = sorted(istanti)
    while t < T_fin - 1e-12:
        Vg = np.empty(n + 2)                         # con il nodo fittizio V_(n+1)
        Vg[:n + 1] = V
        if condizione == "dV/dR":
            Vg[n + 1] = V[n - 1]
        else:                                        # dV/dR = V/R in R = Ra
            Vg[n + 1] = V[n - 1] + 2 * h * V[n] / Ra
        Ri = R[1:]
        lap = (Vg[2:] - 2 * Vg[1:-1] + Vg[:-2]) / h**2 + (Vg[2:] - Vg[:-2]) / (2 * h * Ri) - Vg[1:-1] / Ri**2
        V[1:] = V[1:] + tau * lap
        t += tau
        while k < len(istanti) and t >= istanti[k] - 1e-12:
            out[istanti[k]] = V.copy()
            k += 1
    return R, out


def es_09():
    """Messa in rotazione del fluido attorno al cilindro (spin-up)."""
    print("\n=== Esercizio 9: avviamento della rotazione (Ra = 2, T = nu t / r_b^2)")
    istanti = [0.01, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for a_, cond, titolo in zip(ax, ("dV/dR", "sforzo"), ("condizione $dV/dR=0$", "sforzo nullo")):
        R, out = _spin_up(2.0, 50, 4.0, cond, istanti)
        fin = (2.0 / 5.0 * (2.0 / R + R / 2.0)) if cond == "dV/dR" else R
        for T, col in zip(istanti, PALETTE):
            a_.plot(out[T], R, color=col, label=f"$T={T:g}$".replace(".", "{,}"))
        a_.plot(fin, R, "k--", lw=1.0, label="regime")
        a_.set_xlabel("$V=v_\\theta/v_b$")
        a_.set_ylabel("$R$")
        a_.set_title(titolo)
        scarti = [np.max(np.abs(out[T] - fin)) for T in istanti]
        print(f"   {cond:6s}: scarto dal regime " + ", ".join(f"T={T:g}: {s:.2e}" for T, s in zip(istanti, scarti)))
    ax[1].legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    salva(fig, "es_09")
    # costante di tempo: primo autovalore del problema omogeneo (decadimento esponenziale)
    for cond in ("dV/dR", "sforzo"):
        R, out = _spin_up(2.0, 50, 4.0, cond, [2.0, 3.0])
        fin = (2.0 / 5.0 * (2.0 / R + R / 2.0)) if cond == "dV/dR" else R
        e2, e3 = np.max(np.abs(out[2.0] - fin)), np.max(np.abs(out[3.0] - fin))
        lam = np.log(e2 / e3)
        print(f"   {cond:6s}: decadimento exp(-lam T) con lam = {lam:.3f};"
              f"  scarto < 1% per T > {np.log(e3/0.01)/lam + 3.0:.2f}")


# =========================================================================
def es_10():
    """Lastra superiore oscillante: secondo problema di Stokes in un canale."""
    print("\n=== Esercizio 10: lastra superiore con velocita' u0 sin(omega t)")
    nu, d, u0, om = 1.0e-6, 1.0, 1.0, 1.0e-3
    delta = np.sqrt(2 * nu / om)
    periodo = 2 * np.pi / om
    print(f"   spessore dello strato di Stokes sqrt(2 nu/omega) = {delta*100:.2f} cm; periodo = {periodo:.0f} s")
    kap = (1 + 1j) / delta
    def esatta(y, t):                                # soluzione periodica di regime
        return u0 * np.imag(np.exp(1j * om * t) * np.sinh(kap * y) / np.sinh(kap * d))
    risultati = {}
    for h in (0.05, 0.01, 0.005):
        M = int(round(d / h)) + 1
        y = np.linspace(0, d, M)
        R = 0.25
        tau = R * h * h / nu
        u = np.zeros(M)
        t = 0.0
        errori = []
        for per in (10, 20, 40):
            while t < per * periodo - 1e-9:
                t += tau
                u[1:-1] = u[1:-1] + R * (u[:-2] - 2 * u[1:-1] + u[2:])
                u[-1] = u0 * np.sin(om * t)
            errori.append(np.max(np.abs(u - esatta(y, t))))
            if per == 10:
                risultati[h] = (y, u.copy(), t)
            if h == 0.05:
                break
        print(f"   h = {h:5.3f} m  tau = {tau:7.2f} s ({periodo/tau:6.1f} passi per periodo):"
              f"  max|u - u_es| dopo 10, 20, 40 periodi = " + ", ".join(f"{e:.2e}" for e in errori))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    y, u, t = risultati[0.005]
    yy = np.linspace(0.75, 1.0, 400)
    for k, col in enumerate(PALETTE[:4]):
        tt = t + k * periodo / 4
        ax[0].plot(esatta(yy, tt), yy, color=col, label=f"$\\omega t$ = {k*90}$^\\circ$ (dopo 10 periodi)")
    ax[0].set_ylim(0.75, 1.0)
    ax[0].set_xlabel("$u/u_0$")
    ax[0].set_ylabel("$y$ (m)")
    ax[0].legend(fontsize=6)
    ax[0].set_title("soluzione periodica esatta")
    yc, uc, tc = risultati[0.05]
    ax[1].plot(esatta(yy, tc), yy, color=GRIGIO, lw=2, alpha=0.6, label="esatta")
    ax[1].plot(uc, yc, "o--", ms=3, color=ROSSO, label="$h=5$ cm")
    yf, uf, tf = risultati[0.005]
    ax[1].plot(esatta(yy, tf), yy, color=GRIGIO, lw=2, alpha=0.6)
    ax[1].plot(uf, yf, "-", color=BLU, lw=1.0, label="$h=5$ mm")
    ax[1].set_ylim(0.75, 1.0)
    ax[1].set_xlabel("$u/u_0$")
    ax[1].legend(fontsize=7)
    ax[1].set_title("dopo 10 periodi, due griglie")
    fig.tight_layout()
    salva(fig, "es_10")


# =========================================================================
def es_11():
    """Schema di DuFort-Frankel: stabilita' e consistenza."""
    print("\n=== Esercizio 11: DuFort-Frankel su u_t = u_yy, u(y,0) = sin(pi y)")
    # fattore di amplificazione: radici di (1+2R) G^2 - 4R cos(theta) G - (1-2R) = 0
    Gmax = 0.0
    for R in np.logspace(-2, 3, 60):
        for th in np.linspace(0, np.pi, 91):
            r = np.roots([1 + 2 * R, -4 * R * np.cos(th), -(1 - 2 * R)])
            Gmax = max(Gmax, np.max(np.abs(r)))
    print(f"   max |G| su 0,01 <= R <= 1000 e 0 <= kh <= pi: {Gmax:.6f}")
    def df(h, tau, T):
        n = int(round(1 / h))
        y = np.linspace(0, 1, n + 1)
        R = tau / h**2
        u_old = np.sin(np.pi * y)
        u = np.exp(-np.pi**2 * tau) * np.sin(np.pi * y)  # primo passo dalla soluzione esatta
        passi = int(round(T / tau))
        for _ in range(passi - 1):
            u_new = np.zeros_like(u)
            u_new[1:-1] = (2 * R * (u[:-2] + u[2:]) + (1 - 2 * R) * u_old[1:-1]) / (1 + 2 * R)
            u_old, u = u, u_new
        tt = passi * tau
        return np.max(np.abs(u - np.exp(-np.pi**2 * tt) * np.sin(np.pi * y)))
    print("        h      tau = h^2/4 (R fisso)     tau = h/10 (tau/h fisso)")
    passi_h = [1 / 10, 1 / 20, 1 / 40, 1 / 80, 1 / 160]
    e1, e2 = [], []
    for h in passi_h:
        e1.append(df(h, h * h / 4, 0.2))
        e2.append(df(h, h / 10, 0.2))
        print(f"    1/{int(round(1/h)):<4d}       {e1[-1]:.3e}                {e2[-1]:.3e}")
    # equazione modificata: u_t + (tau/h)^2 u_tt = u_yy  ->  decadimento del modo sin(pi y)
    c = 0.1
    s = (-1 + np.sqrt(1 - 4 * c**2 * np.pi**2)) / (2 * c**2)
    print(f"   con tau/h = {c}: il metodo converge a u_t + {c**2:.2f} u_tt = u_yy, il cui modo decade come exp({s:.4f} t)"
          f" invece di exp({-np.pi**2:.4f} t)")
    s2 = (-1 - np.sqrt(1 - 4 * c**2 * np.pi**2)) / (2 * c**2)
    A = (-np.pi**2 - s2) / (s - s2)                 # da u(0) = 1 e u'(0) = -pi^2
    print(f"   secondo modo (spurio) exp({s2:.1f} t); ampiezza del modo lento A = {A:.4f}")
    print(f"   errore previsto a t = 0,2: {abs(A*np.exp(s*0.2) - np.exp(-np.pi**2*0.2)):.3e}")
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    hh = np.array(passi_h)
    ax.loglog(hh, e1, "o-", color=BLU, label="$\\tau=h^2/4$")
    ax.loglog(hh, e2, "s-", color=ROSSO, label="$\\tau=h/10$")
    ax.loglog(hh, e1[0] * (hh / hh[0]) ** 2, ":", color="k", label="$h^2$")
    ax.set_xlabel("$h$")
    ax.set_ylabel("errore massimo a $t=0{,}2$")
    ax.legend(fontsize=7)
    salva(fig, "es_11")


# =========================================================================
def es_12():
    """Avviamento della corrente in un tubo circolare (metodo implicito)."""
    print("\n=== Esercizio 12: tubo circolare, gradiente di pressione applicato bruscamente")
    N = 40
    h = 1.0 / N
    Rn = np.linspace(0, 1, N + 1)
    lam = jn_zeros(0, 200)
    def W_es(R, T):
        return sum(8 * jv(0, l * R) / (l**3 * jv(1, l)) * np.exp(-l**2 * T) for l in lam)
    for tau in (0.01, 0.002):
        r = tau / h**2
        # nodi 0..N-1 incogniti (W_N = 0); in R = 0: W_T = 4 (W_1 - W_0)/h^2
        a = np.zeros(N); b = np.zeros(N); c = np.zeros(N)
        b[0] = 1 + 4 * r
        c[0] = -4 * r
        for i in range(1, N):
            a[i] = -r * (1 - 1 / (2 * i))
            b[i] = 1 + 2 * r
            c[i] = -r * (1 + 1 / (2 * i))
        W = 1 - Rn**2
        T = 0.0
        storia = []
        while T < 1.0 - 1e-12:
            W[:N] = thomas(a, b, c, W[:N])
            W[N] = 0.0
            T += tau
            storia.append((T, 1 - W[0]))
        err = np.max(np.abs(W - W_es(Rn, T)))
        print(f"   tau = {tau}: U al centro per T = 1: {1 - W[0]:.5f} (esatta {1 - W_es(np.array([0.0]), 1.0)[0]:.5f});"
              f" errore massimo {err:.1e}")
    Ts = np.linspace(0.005, 3.0, 1200)
    Uc = np.array([1 - W_es(np.array([0.0]), T)[0] for T in Ts])
    # canale piano (Programma 3.5) per confronto
    def Wc(T):
        return sum(4 * (-1) ** (m + 1) / ((2*m-1)*np.pi/2) ** 3 * np.exp(-((2*m-1)*np.pi/2) ** 2 * T)
                   for m in range(1, 200))
    Ucan = np.array([1 - Wc(T) for T in Ts])
    for f in (0.9, 0.99):
        print(f"   U_centro = {f}: tubo T = {Ts[np.argmax(Uc >= f)]:.3f};  canale piano T = {Ts[np.argmax(Ucan >= f)]:.3f}")
    print(f"   primo autovalore: tubo lam_1^2 = {lam[0]**2:.3f}, canale (pi/2)^2 = {(np.pi/2)**2:.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for T, col in zip([0.05, 0.1, 0.2, 0.4, 0.8], PALETTE):
        ax[0].plot(1 - Rn**2 - W_es(Rn, T), Rn, color=col, label=f"$T={T:g}$".replace(".", "{,}"))
    ax[0].plot(1 - Rn**2, Rn, "k--", lw=1, label="Poiseuille")
    ax[0].set_xlabel("$U=u/[-(A^2/4\\mu)\\,dp/dx]$")
    ax[0].set_ylabel("$R=r/A$")
    ax[0].legend(fontsize=6.5)
    sel = Ts <= 1.5
    ax[1].plot(Ts[sel], Uc[sel], color=BLU, label="tubo circolare")
    ax[1].plot(Ts[sel], Ucan[sel], "--", color=ARANCIO, label="canale piano")
    ax[1].plot([s[0] for s in storia][::20], [s[1] for s in storia][::20], "o", ms=2.5, color=BLU, mfc="none")
    ax[1].set_xlabel("$T$")
    ax[1].set_ylabel("velocita' al centro / regime")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    salva(fig, "es_12")


# =========================================================================
def _cavita_generale(n, fondo_u=0.0, coperchio_u=1.0, superficie_libera=False):
    """Cavita' di Stokes con il solutore diretto: fondo e coperchio con velocita'
    tangenziale assegnata, oppure coperchio sostituito da una superficie libera
    (psi = 0, zeta = 0)."""
    h = 1.0 / n
    ip = -np.ones((n + 1, n + 1), dtype=int)
    iz = -np.ones((n + 1, n + 1), dtype=int)
    k = 0
    for i in range(1, n):
        for j in range(1, n):
            ip[i, j] = k; k += 1
    for i in range(n + 1):
        for j in range(n + 1):
            if not ((i in (0, n)) and (j in (0, n))):
                iz[i, j] = k; k += 1
    rr, cc, vv = [], [], []
    b = np.zeros(k)
    r = 0
    def add(a_, c_, v_):
        rr.append(a_); cc.append(c_); vv.append(v_)
    for i in range(1, n):
        for j in range(1, n):
            add(r, ip[i, j], -4 / h**2)
            for (a_, c_) in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                if ip[a_, c_] >= 0:
                    add(r, ip[a_, c_], 1 / h**2)
            add(r, iz[i, j], 1.0); r += 1
            add(r, iz[i, j], -4.0)
            for (a_, c_) in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                add(r, iz[a_, c_], 1.0)
            r += 1
    pareti = []
    for t_ in range(1, n):
        pareti += [((0, t_), (1, 0), 0.0), ((n, t_), (-1, 0), 0.0), ((t_, 0), (0, 1), fondo_u)]
        pareti.append(((t_, n), (0, -1), coperchio_u))
    for (i, j), (ni, nj), uw in pareti:
        add(r, iz[i, j], 1.0)
        if superficie_libera and j == n:
            r += 1                                    # zeta = 0 sulla superficie libera
            continue
        ti, tj = nj, ni
        i1, j1, i2, j2 = i + ni, j + nj, i + 2 * ni, j + 2 * nj
        for (a_, c_, w) in ((i1 - ti, j1 - tj, -1.0), (i1, j1, 8 / 3), (i1 + ti, j1 + tj, -1.0), (i2, j2, -2 / 3)):
            if ip[a_, c_] >= 0:
                add(r, ip[a_, c_], -w / h**2)
        # termine di velocita': -2/(3h) dpsi/dy sul coperchio (normale -y), +2/(3h) dpsi/dy sul fondo
        b[r] = (-2 / (3 * h) * uw) if nj == -1 else (2 / (3 * h) * uw if nj == 1 else 0.0)
        r += 1
    A = sp.csr_matrix((vv, (rr, cc)), shape=(k, k))
    x = spla.spsolve(A, b)
    psi = np.zeros((n + 1, n + 1)); zeta = np.zeros((n + 1, n + 1))
    m = ip >= 0; psi[m] = x[ip[m]]
    m = iz >= 0; zeta[m] = x[iz[m]]
    return psi, zeta


def es_13():
    """Cavita' con superficie libera superiore e fondo mobile."""
    print("\n=== Esercizio 13: cavita' di Stokes con fondo mobile (u = 1) e superficie libera")
    for n in (10, 40, 160):
        psi, zeta = _cavita_generale(n, fondo_u=1.0, coperchio_u=0.0, superficie_libera=True)
        i, j = np.unravel_index(np.argmax(np.abs(psi)), psi.shape)
        u_sup = (psi[n // 2, n] - psi[n // 2, n - 1]) / (1.0 / n)   # stima di u in superficie (O(h))
        print(f"   griglia {n+1}x{n+1}: psi estremo = {psi[i, j]:+.5f} in x = {i/n:.3f}, y = {j/n:.3f}")
    psiL, _ = _cavita_generale(160, fondo_u=0.0, coperchio_u=1.0)
    print(f"   per confronto, cavita' con coperchio mobile: psi_min = {psiL.min():+.5f}")
    n = 160
    x = np.linspace(0, 1, n + 1)
    # velocita' orizzontale lungo la verticale di mezzeria
    u_mid = np.gradient(psi[n // 2, :], x)
    print(f"   velocita' sulla superficie libera (x = 0,5): u = {u_mid[-1]:+.4f};"
          f" inversione del moto a y = {x[np.argmax(np.diff(np.sign(u_mid)) != 0)]:.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.2))
    lv = np.linspace(psi.max(), 0, 9)[:-1][::-1] if psi.max() > 0 else np.linspace(psi.min(), 0, 9)[:-1]
    ax[0].contour(x, x, psi.T, levels=np.sort(lv), colors=BLU, linewidths=0.9, linestyles="solid")
    ax[0].plot([0, 1], [0, 0], color=ROSSO, lw=3)
    ax[0].plot([0, 1], [1, 1], "--", color=GRIGIO)
    ax[0].set_aspect("equal")
    ax[0].set_title("fondo mobile, superficie libera")
    ax[1].plot(u_mid, x, color=BLU, label="superficie libera")
    ax[1].plot(np.gradient(psiL[n // 2, :], x), x, "--", color=ARANCIO, label="coperchio mobile")
    ax[1].axvline(0, color="k", lw=0.6)
    ax[1].set_xlabel("$u$ lungo $x=0{,}5$")
    ax[1].set_ylabel("$y$")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    salva(fig, "es_13")


# =========================================================================
def _benard_cheb(N, k):
    """Benard con pareti rigide, collocazione di Chebyshev su z = (1 + y)/2."""
    D, y = cheb(N)
    Dz = 2 * D
    D2 = Dz @ Dz
    I = np.eye(N + 1)
    L = D2 - k**2 * I
    Z = np.zeros((N + 1, N + 1))
    A = np.block([[L, I], [Z, L @ L]])
    B = np.block([[Z, Z], [k**2 * I, Z]])
    n1 = N + 1
    for rr, riga in ((0, None), (N, None)):          # theta = 0
        A[rr, :] = 0; A[rr, rr] = 1; B[rr, :] = 0
    A[n1 + 0, :] = 0; A[n1 + 0, n1 + 0] = 1; B[n1 + 0, :] = 0           # w(1) = 0
    A[n1 + N, :] = 0; A[n1 + N, n1 + N] = 1; B[n1 + N, :] = 0           # w(0) = 0
    A[n1 + 1, :] = 0; A[n1 + 1, n1:] = Dz[0]; B[n1 + 1, :] = 0          # w'(1) = 0
    A[n1 + N - 1, :] = 0; A[n1 + N - 1, n1:] = Dz[N]; B[n1 + N - 1, :] = 0  # w'(0) = 0
    lam = eig(A, B, right=False)
    ok = np.isfinite(lam) & (np.abs(lam.imag) < 1e-6 * np.abs(lam)) & (lam.real > 0)
    return lam[ok].real.min()


def es_14():
    """Curva di stabilita' neutra di Rayleigh-Benard (pareti rigide)."""
    print("\n=== Esercizio 14: curva neutra di Benard, pareti rigide")
    from prog3_7a_benard import Ra_minimo
    ks = np.round(np.arange(0.5, 10.01, 0.1), 2)
    Ra40 = np.array([Ra_minimo(40, k) for k in ks])
    i = np.argmin(Ra40)
    print(f"   n = 40: minimo della curva per k = {ks[i]:.1f}, Ra = {Ra40[i]:.1f}")
    ris = minimize_scalar(lambda k: Ra_minimo(150, k), bracket=(2.9, 3.1, 3.3), tol=1e-6)
    print(f"   n = 150, minimo raffinato: k_c = {ris.x:.4f}, Ra_c = {ris.fun:.2f}")
    for N in (10, 16, 20, 30, 40):
        print(f"   Chebyshev N = {N:2d}: Ra(k = 3,1163) = {_benard_cheb(N, 3.1163):.4f}")
    rc = minimize_scalar(lambda k: _benard_cheb(30, k), bracket=(3.0, 3.1, 3.2), tol=1e-8)
    print(f"   Chebyshev N = 30: k_c = {rc.x:.4f}, Ra_c = {rc.fun:.4f}")
    def jeffreys(k):
        kap = np.tanh(k / 2) / (8 * k) + (1 / np.cosh(k / 2)) ** 2 / 16
        return (np.pi**2 + k**2) ** 3 / ((1 - np.pi**2 / ((np.pi**2 + k**2) ** 2 * kap)) * k**2)
    kj = minimize_scalar(jeffreys, bracket=(2.5, 3.1, 3.8))
    print(f"   formula approssimata: minimo Ra = {kj.fun:.1f} per k = {kj.x:.3f}")
    Ra_exact = np.array([_benard_cheb(30, k) for k in ks])
    print(f"   scarto massimo fra la formula approssimata e la curva 'esatta' (Chebyshev) per 1 < k < 8: "
          f"{np.max(np.abs(jeffreys(ks[5:76]) / Ra_exact[5:76] - 1))*100:.2f} %")
    Ra_c, vec = Ra_minimo(150, ris.x, vettori=True)
    n = 150
    th, w = vec[:n], vec[n:]
    z = np.linspace(0, 1, n)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax[0].plot(ks, Ra_exact, color=GRIGIO, lw=2.2, alpha=0.6, label="Chebyshev $N=30$")
    ax[0].plot(ks[::3], Ra40[::3], "o", ms=3, color=BLU, mfc="none", label="diff. finite $n=40$")
    ax[0].plot(ks, jeffreys(ks), "--", color=ROSSO, label="formula approssimata")
    ax[0].plot(ris.x, ris.fun, "k*", ms=8)
    ax[0].set_ylim(0, 12000)
    ax[0].set_xlabel("$k$")
    ax[0].set_ylabel("$Ra$")
    ax[0].legend(fontsize=6.5)
    ax[1].plot(w / np.max(np.abs(w)), z, color=BLU, label="$w'$")
    ax[1].plot(th / np.max(np.abs(th)), z, "--", color=ROSSO, label="$\\theta'$")
    ax[1].set_xlabel("autofunzioni normalizzate")
    ax[1].set_ylabel("$z$")
    ax[1].legend(fontsize=7)
    ax[1].set_title(f"$k_c={ris.x:.3f}$".replace(".", "{,}"))
    fig.tight_layout()
    salva(fig, "es_14")


# =========================================================================
def _os_cp(ReP, alfa, r, N=64):
    """Orr-Sommerfeld per la corrente di Couette-Poiseuille scritta nella scala
    della velocita' massima della parte di Poiseuille:
        U~ = (r/2)(1 + y) + (1 - y^2),   r = U_parete / U_P = 2/B,   U~'' = -2,
    con Re_P = U_P h / nu = Re B / 2. Restituisce max lam_R (= max c_I)."""
    D, y = cheb(N)
    D2 = D @ D
    D4 = D2 @ D2
    I = np.eye(N + 1)
    U = 0.5 * r * (1 + y) + (1 - y**2)
    A = (D4 - 2 * alfa**2 * D2 + alfa**4 * I) / (ReP * alfa) - 2j * I \
        - 1j * np.diag(U) @ (D2 - alfa**2 * I)
    Bm = (D2 - alfa**2 * I).astype(complex)
    A[0, :] = 0; A[0, 0] = 1; A[1, :] = D[0]
    A[N, :] = 0; A[N, N] = 1; A[N - 1, :] = D[N]
    Bm[[0, 1, N - 1, N], :] = 0
    lam = eig(A, Bm, right=False)
    lam = lam[np.isfinite(lam) & (np.abs(lam) < 1e3)]
    return lam.real.max()


def es_15():
    """Curve neutre della corrente di Couette-Poiseuille."""
    print("\n=== Esercizio 15: corrente di Couette-Poiseuille U = (1+y)/2 + (B/2)(1-y^2)")
    # B = 0: Couette pura (nella scala della parete: U = (1+y)/2, U'' = 0)
    def os_couette(Re, alfa, N=80):
        D, y = cheb(N); D2 = D @ D; D4 = D2 @ D2; I = np.eye(N + 1)
        A = (D4 - 2 * alfa**2 * D2 + alfa**4 * I) / (Re * alfa) - 1j * np.diag(0.5 * (1 + y)) @ (D2 - alfa**2 * I)
        Bm = (D2 - alfa**2 * I).astype(complex)
        A[0, :] = 0; A[0, 0] = 1; A[1, :] = D[0]; A[N, :] = 0; A[N, N] = 1; A[N - 1, :] = D[N]
        Bm[[0, 1, N - 1, N], :] = 0
        lam = eig(A, Bm, right=False)
        return lam[np.isfinite(lam) & (np.abs(lam) < 1e3)].real.max()
    mx = max(os_couette(Re, a) for Re in (1e3, 1e4, 1e5) for a in (0.3, 0.6, 1.0, 1.5))
    print(f"   B = 0 (Couette pura): max lam_R per Re = 1e3..1e5, alfa = 0,3..1,5: {mx:+.4f} -> stabile")
    risultati = {0.0: (5772.22, 1.0206)}
    griglia_Re = np.logspace(np.log10(2e3), np.log10(1e6), 30)
    def Re_n(a, r):
        """Ramo inferiore della curva neutra: primo Re_P instabile sulla griglia, poi Brent."""
        prec = None
        for Re in griglia_Re:
            N = 90 if Re < 2e5 else 150
            g = _os_cp(Re, a, r, N)
            if g > 0:
                if prec is None:
                    return Re
                return np.exp(brentq(lambda lr: _os_cp(np.exp(lr), a, r, N), np.log(prec), np.log(Re),
                                     xtol=1e-6))
            prec = Re
        return np.inf
    for B in (100.0, 25.0, 7.0, 3.0):
        r = 2.0 / B
        al = np.arange(0.3, 1.21, 0.05)
        vals = np.array([Re_n(a, r) for a in al])
        if np.all(~np.isfinite(vals)):
            print(f"   B = {B:5.1f} (r = {r:.3f}): nessun modo instabile per Re_P <= 1e6")
            continue
        j = int(np.argmin(vals))
        ris = minimize_scalar(lambda a: Re_n(a, r), bounds=(al[max(j - 1, 0)], al[min(j + 1, len(al) - 1)]),
                              method="bounded", options={"xatol": 1e-4})
        risultati[r] = (ris.fun, ris.x)
        print(f"   B = {B:5.1f} (r = U_par/U_P = {r:.3f}): Re_P,c = {ris.fun:9.1f}, alfa_c = {ris.x:.3f};"
              f"  nella scala della parete Re_c = 2 Re_P,c / B = {2*ris.fun/B:8.1f}")
    r = 2.0
    mxB = max(_os_cp(Re, a, r, 100 if Re < 2e5 else 160)
              for Re in (1e4, 3e4, 1e5, 3e5, 1e6) for a in np.linspace(0.1, 1.2, 12))
    print(f"   B =   1.0 (r = 2,000): max lam_R per Re_P = 1e4..1e6, alfa = 0,1..1,2: {mxB:+.5f}")
    # curve neutre nel piano (Re_P, alfa)
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    lRP = np.linspace(np.log10(3000), np.log10(3.0e5), 26)
    al = np.linspace(0.4, 1.25, 22)
    for r, col, et in ((0.0, "k", "Poiseuille"), (0.02, BLU, "$B=100$"), (0.08, VERDE, "$B=25$"),
                       (2 / 7, ARANCIO, "$B=7$")):
        G = np.array([[_os_cp(10**l, a, r, 64) for l in lRP] for a in al])
        ax[0].contour(10**lRP, al, G, levels=[0.0], colors=[col], linewidths=1.2)
        ax[0].plot([], [], color=col, label=et)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("$Re_P=U_Ph/\\nu$")
    ax[0].set_ylabel("$\\alpha$")
    ax[0].legend(fontsize=7)
    rr = sorted(risultati)
    ax[1].plot(rr, [risultati[x][0] for x in rr], "o-", color=BLU)
    ax[1].set_xlabel("$r=U_{\\mathrm{parete}}/U_P=2/B$")
    ax[1].set_ylabel("$Re_{P,c}$")
    ax[1].set_yscale("log")
    fig.tight_layout()
    salva(fig, "es_15")


# =========================================================================
def es_16():
    """Convezione-diffusione con schema misto esplicito-implicito."""
    print("\n=== Esercizio 16: convezione esplicita, diffusione implicita (modo per modo in y)")
    import prog3_10_convezione_diffusione as cd
    x, NX, NY, dx, k, U, V = cd.x, cd.NX, cd.NY, cd.dx, cd.k, cd.U, cd.V
    import time
    def imex(Pr, dt, toll=1e-4, nmax=200000):
        a = 1.0 / (Pr * cd.RE)
        T = np.tile(x[:, None], (1, NY))
        Th_bordo = np.zeros(NY, dtype=complex)
        Th_bordo[0] = NY                               # trasformata di T = 1 su x = 1
        n_int = NX - 2
        for n in range(1, nmax + 1):
            Tx = np.zeros_like(T)
            Tx[1:-1] = (T[2:] - T[:-2]) / (2 * dx)
            conv = -U * Tx - V * cd.dy(T)
            rhs = np.fft.fft(T + dt * conv, axis=1)     # (T^n + dt N^n) nello spazio di Fourier
            Tn = np.empty_like(rhs)
            Tn[0] = 0.0
            Tn[-1] = Th_bordo
            r = a * dt / dx**2
            # algoritmo di Thomas eseguito contemporaneamente su tutti i modi m
            bb = 1 + 2 * r + a * dt * k**2              # diagonale (dipende dal modo)
            d = rhs[1:-1].copy()
            d[-1] += r * Th_bordo
            cp = np.zeros((n_int, NY), dtype=complex)
            dp = np.zeros((n_int, NY), dtype=complex)
            cp[0] = -r / bb
            dp[0] = d[0] / bb
            for i in range(1, n_int):
                den = bb + r * cp[i - 1]
                cp[i] = -r / den
                dp[i] = (d[i] + r * dp[i - 1]) / den
            sol = np.zeros((n_int, NY), dtype=complex)
            sol[-1] = dp[-1]
            for i in range(n_int - 2, -1, -1):
                sol[i] = dp[i] - cp[i] * sol[i + 1]
            Tn[1:-1] = sol
            Tnew = np.fft.ifft(Tn, axis=1).real
            var = np.max(np.abs(Tnew - T)) / dt
            T = Tnew
            if not np.isfinite(var) or var > 1e6:
                return None, n, n * dt
            if var < toll:
                return T, n, n * dt
        return T, nmax, nmax * dt
    for Pr, dts in ((0.71, [1e-3, 1e-2, 0.05, 0.5, 2.0]), (25.0, [1e-2, 0.1, 0.3, 1.0])):
        Ts = cd.stazionario(Pr)
        a = 1 / (Pr * cd.RE)
        print(f"   Pr = {Pr}: limite di stabilita' stimato dt < 2a/|V|max^2 = {2*a/2.0:.3f}")
        for dt in dts:
            t0 = time.perf_counter()
            T, n, tf = imex(Pr, dt)
            durata = time.perf_counter() - t0
            if T is None:
                print(f"      dt = {dt:6.3f}: INSTABILE dopo {n} passi")
            else:
                print(f"      dt = {dt:6.3f}: {n:6d} passi, t = {tf:6.2f}, {durata:5.2f} s,"
                      f" scarto dallo stazionario {np.max(np.abs(T - Ts)):.1e}")
    T, _, _ = imex(25.0, 0.01)
    xe = np.concatenate([cd.y, [cd.LY]])
    fig, ax = plt.subplots(figsize=(3.0, 4.2))
    Te = np.concatenate([T, T[:, :1]], axis=1)
    cs = ax.contour(x, xe, Te.T, levels=np.linspace(0.05, 0.95, 10), cmap="coolwarm", linewidths=1.0)
    ax.clabel(cs, fmt=lambda v: f"{v:.2f}".replace(".", ","), fontsize=5.5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("IMEX, $Pr=25$, $\\Delta t=0{,}01$")
    salva(fig, "es_16")


# =========================================================================
def _poisson_maschera(dentro, h, bracci=None, simmetria=()):
    """Risolve lap U = -1 sui nodi 'dentro' (matrice booleana), U = 0 altrove.
    bracci[(i,j)] = (aE, aW, aN, aS) frazioni di h per i nodi irregolari
    (formula di Shortley-Weller). simmetria: 'x' e/o 'y' -> dU/dn = 0 sulle
    linee i = 0 / j = 0 (nodi fittizi speculari)."""
    idx = -np.ones(dentro.shape, dtype=int)
    idx[dentro] = np.arange(dentro.sum())
    A = sp.lil_matrix((dentro.sum(), dentro.sum()))
    b = -np.ones(dentro.sum())
    for (i, j) in zip(*np.nonzero(dentro)):
        r = idx[i, j]
        aE, aW, aN, aS = bracci.get((i, j), (1, 1, 1, 1)) if bracci else (1, 1, 1, 1)
        cx = 2 / (h * h * (aE + aW)); cy = 2 / (h * h * (aN + aS))
        A[r, r] = -cx * (1 / aE + 1 / aW) * (aE + aW) / 2 * 0 - 2 / (h * h * aE * aW) - 2 / (h * h * aN * aS)
        for (a_, c_, coef) in ((i + 1, j, cx / aE), (i - 1, j, cx / aW), (i, j + 1, cy / aN), (i, j - 1, cy / aS)):
            if a_ < 0 and "x" in simmetria:
                a_ = -a_
            if c_ < 0 and "y" in simmetria:
                c_ = -c_
            if 0 <= a_ < dentro.shape[0] and 0 <= c_ < dentro.shape[1] and dentro[a_, c_] and \
                    not ((a_, c_) != (i + 1, j) and coef == cx / aE and aE < 1):
                A[r, idx[a_, c_]] += coef
    u = spla.spsolve(A.tocsr(), b)
    Uf = np.zeros(dentro.shape)
    Uf[dentro] = u
    return Uf


def _cerchio_quarto(n):
    """Quarto di cerchio di raggio 1 con simmetria sugli assi (Shortley-Weller)."""
    h = 1.0 / n
    xs = np.arange(n + 1) * h
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    dentro = X**2 + Y**2 < 1 - 1e-12
    bracci = {}
    for (i, j) in zip(*np.nonzero(dentro)):
        aE = aN = 1.0
        if not (i + 1 <= n and dentro[i + 1, j]):
            aE = (np.sqrt(1 - Y[i, j]**2) - X[i, j]) / h
        if not (j + 1 <= n and dentro[i, j + 1]):
            aN = (np.sqrt(1 - X[i, j]**2) - Y[i, j]) / h
        if aE < 1 or aN < 1:
            bracci[(i, j)] = (aE, 1.0, aN, 1.0)
    # assemblaggio diretto (piu' leggibile della routine generale)
    idx = -np.ones(dentro.shape, dtype=int)
    idx[dentro] = np.arange(dentro.sum())
    A = sp.lil_matrix((dentro.sum(), dentro.sum()))
    b = -np.ones(dentro.sum())
    for (i, j) in zip(*np.nonzero(dentro)):
        r = idx[i, j]
        aE, aW, aN, aS = bracci.get((i, j), (1.0, 1.0, 1.0, 1.0))
        A[r, r] = -2 / (h * h * aE * aW) - 2 / (h * h * aN * aS)
        vicini = (((i + 1, j), 2 / (h * h * aE * (aE + aW)), aE),
                  ((i - 1, j), 2 / (h * h * aW * (aE + aW)), aW),
                  ((i, j + 1), 2 / (h * h * aN * (aN + aS)), aN),
                  ((i, j - 1), 2 / (h * h * aS * (aN + aS)), aS))
        for (a_, c_), coef, braccio in vicini:
            if braccio < 1.0:
                continue                             # il vicino e' il contorno: U = 0
            a_, c_ = abs(a_), abs(c_)                # simmetria: U(-x) = U(x), U(-y) = U(y)
            A[r, idx[a_, c_]] += coef
    u = spla.spsolve(A.tocsr(), b)
    U = np.zeros(dentro.shape)
    U[dentro] = u
    # portata del quarto: integrazione in coordinate polari del campo interpolato
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((xs, xs), U)
    rr = np.linspace(0, 1, 401)
    tt = np.linspace(0, np.pi / 2, 201)
    Rg, Tg = np.meshgrid(rr, tt, indexing="ij")
    vals = f(np.stack([np.clip(Rg * np.cos(Tg), 0, 1), np.clip(Rg * np.sin(Tg), 0, 1)], axis=-1))
    Q = simpson(simpson(vals * Rg, x=tt, axis=1), x=rr)
    return xs, U, Q


def _dominio_maschera(dentro, h):
    idx = -np.ones(dentro.shape, dtype=int)
    idx[dentro] = np.arange(dentro.sum())
    A = sp.lil_matrix((dentro.sum(), dentro.sum()))
    for (i, j) in zip(*np.nonzero(dentro)):
        r = idx[i, j]
        A[r, r] = -4 / h**2
        for (a_, c_) in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if dentro[a_, c_]:
                A[r, idx[a_, c_]] = 1 / h**2
    u = spla.spsolve(A.tocsr(), -np.ones(dentro.sum()))
    U = np.zeros(dentro.shape)
    U[dentro] = u
    return U


def es_17():
    """Progetto: condotti di sezione circolare, triangolare, rettangolare con tubo interno."""
    print("\n=== Esercizio 17: correnti di Poiseuille in condotti di varie sezioni")
    print("   (a) tubo circolare, quarto di sezione con simmetria; U = u/[-(dp/dx) R^2/mu]")
    for n in (10, 20, 40, 80):
        xs, Uc, Q = _cerchio_quarto(n)
        print(f"      n = {n:3d}: U(0) = {Uc[0,0]:.6f} (esatto 0,25), Q_tubo = 4 Q_quarto = {4*Q:.6f}"
              f" (esatto pi/8 = {np.pi/8:.6f})")
    print("   (b) triangolo rettangolo isoscele (ipotenusa L = 2 in alto, vertice in basso)")
    ris_t = []
    for n in (20, 40, 80, 160):
        h = 1.0 / n
        xs = np.arange(2 * n + 1) * h
        ys = np.arange(n + 1) * h
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        dentro = (Y > np.abs(X - 1) + 1e-12) & (Y < 1 - 1e-12)
        U = _dominio_maschera(dentro, h)
        Q = np.sum(U) * h * h                        # U = 0 sul contorno: regola dei trapezi
        ris_t.append(Q)
        print(f"      n = {n:3d}: Q = {Q:.6f},  U_max = {U.max():.6f}")
    Qt = ris_t[-1] + (ris_t[-1] - ris_t[-2]) / 3
    A_t, P_t = 1.0, 2 + 2 * np.sqrt(2)
    Dh = 4 * A_t / P_t
    print(f"      Richardson: Q = {Qt:.6f};  D_h = {Dh:.4f};  f Re = 2 D_h^2 A / Q = {2*Dh**2*A_t/Qt:.2f}")
    print("   (c) rettangolo 3a x 2a con tubo interno a x a (a = 1)")
    ris_r = []
    for n in (8, 16, 32, 64):
        h = 1.0 / n
        xs = np.arange(3 * n + 1) * h
        ys = np.arange(2 * n + 1) * h
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        interno = (X >= 1.5 - 1e-12) & (X <= 2.5 + 1e-12) & (Y >= 0.25 - 1e-12) & (Y <= 1.25 + 1e-12)
        dentro = (X > 1e-12) & (X < 3 - 1e-12) & (Y > 1e-12) & (Y < 2 - 1e-12) & ~interno
        U = _dominio_maschera(dentro, h)
        pieno = (X > 1e-12) & (X < 3 - 1e-12) & (Y > 1e-12) & (Y < 2 - 1e-12)
        U0 = _dominio_maschera(pieno, h)
        ris_r.append((np.sum(U) * h * h, np.sum(U0) * h * h))
        print(f"      n = {n:3d}: Q = {ris_r[-1][0]:.6f}  (senza tubo interno {ris_r[-1][1]:.6f})")
    Qr = ris_r[-1][0] + (ris_r[-1][0] - ris_r[-2][0]) / 3
    Q0 = ris_r[-1][1] + (ris_r[-1][1] - ris_r[-2][1]) / 3
    print(f"      Richardson: Q = {Qr:.5f}, senza tubo {Q0:.5f}: il tubo interno riduce la portata del"
          f" {100*(1-Qr/Q0):.1f} % occupando il {100/6:.1f} % della sezione")
    xs_c, Uc, _ = _cerchio_quarto(80)
    fig, ax = plt.subplots(1, 3, figsize=(8.0, 2.9), gridspec_kw={"width_ratios": [1, 1.6, 1.4]})
    ax[0].contour(xs_c, xs_c, Uc.T, levels=np.linspace(0.025, 0.225, 9), colors=BLU, linewidths=0.8)
    tt = np.linspace(0, np.pi / 2, 100)
    ax[0].plot(np.cos(tt), np.sin(tt), "k", lw=1.5)
    ax[0].set_aspect("equal")
    ax[0].set_title("(a) quarto di tubo")
    n = 80; h = 1 / n
    xs = np.arange(2 * n + 1) * h; ys = np.arange(n + 1) * h
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Ut = _dominio_maschera((Y > np.abs(X - 1) + 1e-12) & (Y < 1 - 1e-12), h)
    ax[1].contour(xs, ys, Ut.T, levels=np.linspace(0.005, Ut.max() * 0.97, 7), colors=BLU, linewidths=0.8)
    ax[1].plot([0, 1, 2, 0], [1, 0, 1, 1], "k", lw=1.5)
    ax[1].set_aspect("equal")
    ax[1].set_title("(b) triangolo")
    n = 32; h = 1 / n
    xs = np.arange(3 * n + 1) * h; ys = np.arange(2 * n + 1) * h
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    interno = (X >= 1.5 - 1e-12) & (X <= 2.5 + 1e-12) & (Y >= 0.25 - 1e-12) & (Y <= 1.25 + 1e-12)
    Ur = _dominio_maschera((X > 1e-12) & (X < 3 - 1e-12) & (Y > 1e-12) & (Y < 2 - 1e-12) & ~interno, h)
    ax[2].contour(xs, ys, Ur.T, levels=np.linspace(0.01, Ur.max() * 0.97, 8), colors=BLU, linewidths=0.8)
    ax[2].plot([0, 3, 3, 0, 0], [0, 0, 2, 2, 0], "k", lw=1.5)
    ax[2].fill([1.5, 2.5, 2.5, 1.5], [0.25, 0.25, 1.25, 1.25], color=GRIGIO, alpha=0.5)
    ax[2].set_aspect("equal")
    ax[2].set_title("(c) con tubo interno")
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
        a_.spines[:].set_visible(False)
    fig.tight_layout()
    salva(fig, "es_17")


# =========================================================================
def es_18():
    """Progetto: corrente pulsante in un tubo (soluzione di Womersley)."""
    print("\n=== Esercizio 18: gradiente di pressione alternato, Omega U_T = 4 cos T + U_RR + U_R/R")
    N = 60
    h = 1.0 / N
    Rn = np.linspace(0, 1, N + 1)
    def esatta(R, T, Om):
        k = np.exp(3j * np.pi / 4) * np.sqrt(Om)      # i^(3/2) sqrt(Omega)
        F = 4 / (1j * Om) * (1 - jv(0, k * R) / jv(0, k))
        return np.real(F * np.exp(1j * T))
    fig, ax = plt.subplots(1, 3, figsize=(8.0, 2.9))
    for a_, Om in zip(ax, (0.1, 7.0, 35.0)):
        dT = 2 * np.pi / 400
        r = dT / (Om * h * h)
        # Crank-Nicolson: (I - r/2 L) U^(n+1) = (I + r/2 L) U^n + (dT/Om) 4 cos(T_(n+1/2))
        a = np.zeros(N); b = np.zeros(N); c = np.zeros(N)
        b[0] = 1 + 2 * r; c[0] = -2 * r                  # in R = 0: L U = 4 (U_1 - U_0)/h^2 (x h^2)
        for i in range(1, N):
            a[i] = -r / 2 * (1 - 1 / (2 * i)); b[i] = 1 + r; c[i] = -r / 2 * (1 + 1 / (2 * i))
        U = np.zeros(N + 1)
        T = 0.0
        periodi = 12 if Om < 10 else 25
        for nstep in range(400 * periodi):
            LU = np.zeros(N)
            LU[0] = 4 * (U[1] - U[0])
            i = np.arange(1, N)
            LU[1:] = (U[2:] - 2 * U[1:-1] + U[:-2]) + (U[2:] - U[:-2]) / (2 * i)
            rhs = U[:N] + r / 2 * LU + dT / Om * 4 * np.cos(T + dT / 2)
            U[:N] = thomas(a, b, c, rhs)
            U[N] = 0.0
            T += dT
        err = np.max(np.abs(U - esatta(Rn, T, Om)))
        # ampiezza e sfasamento della portata rispetto al gradiente (cos T)
        fasi = np.linspace(0, 2 * np.pi, 361)
        Qs = np.array([2 * simpson(esatta(Rn, T0, Om) * Rn, x=Rn) for T0 in fasi])
        sfas = np.degrees(fasi[np.argmax(Qs)])
        print(f"   Omega = {Om:5.1f} (alfa_W = {np.sqrt(Om):.2f}): errore CN dopo {periodi} periodi = {err:.1e};"
              f" portata max {Qs.max():.4f} (quasi stazionaria: {2*0.25:.4f}), ritardo di fase {sfas:.1f} gradi")
        for k, col in zip(range(8), PALETTE + [GRIGIO]):
            T0 = k * np.pi / 4
            a_.plot(esatta(Rn, T0, Om), Rn, color=col, lw=1.0)
        a_.axvline(0, color="k", lw=0.5)
        a_.set_title(f"$\\Omega={Om:g}$".replace(".", "{,}"))
        a_.set_xlabel("$U$")
    ax[0].set_ylabel("$R=r/A$")
    fig.tight_layout()
    salva(fig, "es_18")


if __name__ == "__main__":
    scelti = [int(a) for a in sys.argv[1:]] or range(1, 19)
    for k in scelti:
        nome = f"es_{k:02d}"
        if nome in globals():
            globals()[nome]()
