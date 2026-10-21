"""
esercizi_svolti.py -- codice degli esercizi svolti della monografia 2.
Ogni esercizio e' la funzione es_NN(); senza argomenti si eseguono tutti,
altrimenti solo quelli indicati:   python3 esercizi_svolti.py 3 7 12
Le figure vengono salvate come figure/es_NN*.pdf.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from stile_grafici import salva, BLU, ARANCIO, ROSSO, GRIGIO, VERDE, VIOLA, GIALLO
from metodi_numerici import cerca_x, sor_passo, gauss_seidel_passo, omega_ottimo, thomas


def titolo(n, testo):
    print("\n" + "=" * 72 + f"\nEsercizio {n} -- {testo}\n" + "=" * 72)


# ============================================================ esercizio 1
def es_01():
    """Combinazioni di flussi elementari (metodo inverso)."""
    titolo(1, "combinazioni di flussi elementari")
    U = 1.0
    x = np.linspace(-3, 3, 601); y = np.linspace(-2, 2, 401)
    X, Y = np.meshgrid(x, y, indexing="ij")
    th = lambda X, Y, x0=0.0: np.arctan2(Y, X - x0) / np.pi      # theta/pi in (-1, 1]
    sig = 2.0                                                   # sigma/(U l)
    kap = 0.6                                                   # kappa/(2 pi U l^2)
    with np.errstate(divide="ignore", invalid="ignore"):
        casi = {
            "(a) corrente + sorgente": U * Y + 0.5 * sig * th(X, Y),
            "(b) corrente + pozzo": U * Y - 0.5 * sig * th(X, Y),
            "(c) sorgente in $(-1,0)$ + doppietta": 0.5 * sig * th(X, Y, -1.0)
                                                    - kap * Y / (X**2 + Y**2),
        }
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.6))
    for ax, (nome, P) in zip(axs, casi.items()):
        ax.contour(X, Y, P, levels=np.linspace(-2.5, 2.5, 21), colors=BLU,
                   linewidths=0.6, linestyles="solid")
        ax.set_aspect("equal"); ax.set_title(nome, fontsize=7.5); ax.grid(False)
        ax.set_xticks([-3, 0, 3]); ax.set_yticks([-2, 0, 2])
    # linee divisorie psi = +-sigma/2 (contorno del semicorpo; arctan normalizzato con pi)
    for ax, P in zip(axs[:2], list(casi.values())[:2]):
        ax.contour(X, Y, P, levels=[-1.0, 1.0], colors=ROSSO, linewidths=1.2,
                   linestyles="solid")
    axs[0].plot(0, 0, "o", color=ROSSO, ms=3); axs[1].plot(0, 0, "o", color=ROSSO, ms=3)
    axs[2].plot([-1, 0], [0, 0], "o", color=ROSSO, ms=3)
    salva(fig, "es_01a")
    xs = -sig / (2 * np.pi * U)
    print(f"(a) punto di ristagno x_s = -sigma/(2 pi U) = {xs:.4f}; semispessore a valle = "
          f"sigma/(2U) = {sig/2:.3f}")
    print(f"(b) ristagno in x = +{-xs:.4f} (a valle del pozzo)")
    # (c) ristagni sull'asse: u = sig/(2 pi (x+1)) + kap / x^2 ... (doppietta: u = -kap (x^2-y^2)/r^4)
    u_asse = lambda xx: sig / (2 * np.pi * (xx + 1.0)) - kap / xx**2
    xr = brentq(u_asse, 0.05, 5.0)
    print(f"(c) punto di ristagno sull'asse x > 0: x = {xr:.4f}")
    # (d) due doppiette uguali a distanza d
    k = 0.25
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.4))
    print("(d) corrente + due doppiette di intensita' c = 0,25 in x = -+d/2:")
    for ax, d in zip(axs, (0.4, 1.0, 1.6)):
        with np.errstate(invalid="ignore"):
            P = Y - k * Y / ((X + d / 2)**2 + Y**2) - k * Y / ((X - d / 2)**2 + Y**2)
        ax.contour(X, Y, P, levels=np.linspace(-1.5, 1.5, 13), colors=BLU,
                   linewidths=0.6, linestyles="solid")
        ax.contour(X, Y, P, levels=[0.0], colors=ROSSO, linewidths=1.2)
        ax.plot([-d / 2, d / 2], [0, 0], "k+", ms=5)
        ax.set_aspect("equal"); ax.set_title(f"$d = {d}\\,l$", fontsize=8); ax.grid(False)
        ax.set_xticks([-3, 0, 3]); ax.set_yticks([-2, 0, 2])
        # spessore in x = 0: psi(0, y) = 0  ->  y^2 = 2c - d^2/4
        y0 = np.sqrt(max(2 * k - d**2 / 4, 0.0))
        print(f"   d = {d:.1f}: semispessore in x = 0 -> {y0:.4f}"
              f"   (corpo unico se d^2/4 < 2c, cioe' d < {2*np.sqrt(2*k):.3f})")
    salva(fig, "es_01b")


# ============================================================ esercizio 2
def es_02():
    """Metodo delle immagini: vortice vicino a una parete; sorgente e parete."""
    titolo(2, "metodo delle immagini")
    G, d = 2 * np.pi, 1.0
    x = np.linspace(-3, 3, 601); y = np.linspace(-3, 3, 601)
    X, Y = np.meshgrid(x, y, indexing="ij")
    with np.errstate(divide="ignore"):
        P1 = G / (2 * np.pi) * (np.log(np.hypot(X, Y - d)) - np.log(np.hypot(X, Y + d)))
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.2))
    axs[0].contour(X, Y, P1, levels=np.linspace(-1.5, 1.5, 25), colors=BLU,
                   linewidths=0.6, linestyles="solid")
    axs[0].axhline(0, color=ROSSO, lw=1.5)
    axs[0].plot([0, 0], [d, -d], "o", color="k", ms=3)
    axs[0].set_title(r"vortici $\pm\Gamma$ in $(0,\pm d)$", fontsize=8)
    # la funzione di corrente della sorgente e' a piu' valori (taglio di arctan):
    # per il secondo caso si tracciano le linee di corrente dal campo di velocita'
    xs_ = np.linspace(-3, 0, 151); ys_ = np.linspace(-3, 3, 301)
    XS, YS = np.meshgrid(xs_, ys_)
    r1 = (XS + d)**2 + YS**2; r2 = (XS - d)**2 + YS**2
    with np.errstate(invalid="ignore", divide="ignore"):
        us = (XS + d) / r1 + (XS - d) / r2
        vs = YS / r1 + YS / r2
    semi = [(-d + 0.08 * np.cos(a), 0.08 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    axs[1].streamplot(XS, YS, us, vs, color=BLU, linewidth=0.6, density=1.2,
                      start_points=semi, arrowsize=0.6, broken_streamlines=False)
    axs[1].axvline(0, color=ROSSO, lw=1.5)
    axs[1].plot([-d, d], [0, 0], "o", color="k", ms=3)
    axs[1].plot([d], [0], "o", color=GRIGIO, ms=3, mfc="none")
    axs[1].set_xlim(-3, 3); axs[1].set_ylim(-3, 3)
    axs[1].set_title(r"sorgente in $(-d,0)$ e immagine in $(d,0)$", fontsize=8)
    for ax in axs:
        ax.set_aspect("equal"); ax.grid(False); ax.set_xticks([-3, 0, 3]); ax.set_yticks([-3, 0, 3])
    salva(fig, "es_02")
    # velocita' indotta dall'immagine sul vortice reale e velocita' sulla parete
    print(f"velocita' con cui il vortice si muove parallelamente alla parete: "
          f"Gamma/(4 pi d) = {G/(4*np.pi*d):.4f}")
    xw = np.array([0.0, 0.5, 1.0, 2.0])
    uw = G / np.pi * d / (xw**2 + d**2)
    print("velocita' sulla parete u(x,0) = Gamma d/(pi (x^2+d^2)):", np.round(uw, 4))
    print("sulla parete v = 0 per simmetria: la parete e' una linea di corrente (psi = 0)")


# ============================================================ esercizio 3
def es_03():
    """Corpo chiuso vicino a una lastra piana (doppietta + immagine)."""
    titolo(3, "doppietta e immagine: forza fra corpo e parete")
    U, kap, hd = 1.0, 0.5, 1.5        # kappa/(2 pi) e quota della doppietta
    x = np.linspace(-4, 4, 801); y = np.linspace(0, 3.5, 351)
    X, Y = np.meshgrid(x, y, indexing="ij")
    with np.errstate(invalid="ignore"):
        P = U * Y - kap * (Y - hd) / (X**2 + (Y - hd)**2) - kap * (Y + hd) / (X**2 + (Y + hd)**2)
    # velocita' sulla parete y = 0: u = dpsi/dy
    # velocita' sulla parete y = 0: u = U + u',  u' = 2 kap (h^2 - x^2)/(x^2 + h^2)^2
    up = lambda xx, hh: 2 * kap * (hh**2 - xx**2) / (xx**2 + hh**2)**2
    xw = np.linspace(-4, 4, 2001)
    cpw = 1 - (U + up(xw, hd))**2
    print(f"doppietta kappa/(2 pi) = {kap}, a quota h = {hd}")
    print(f"C_p minimo sulla parete = {cpw.min():.4f} in x = {xw[np.argmin(cpw)]:.3f}")
    # -C_p = 2u' + u'^2: il primo termine ha integrale nullo sulla lastra infinita
    # ma decade solo come 1/x^2: integrarlo su un tratto finito falsa il risultato
    for X_ in (15.0, 150.0, 1500.0):
        xx = np.linspace(-X_, X_, 600001)
        Fn = np.trapezoid(2 * up(xx, hd) + up(xx, hd)**2, xx)
        print(f"   integrale numerico di -C_p su [-{X_:g}, {X_:g}]: {Fn:.5f}")
    for hh in (1.0, 1.5, 2.0, 3.0):
        xx = np.linspace(-200, 200, 400001)
        F2 = np.trapezoid(up(xx, hh)**2, xx)
        print(f"   h = {hh:.1f}:  F_y/(rho U^2/2) = int u'^2 dx = {F2:.5f}"
              f"   (formula pi kap^2/h^3 = {np.pi*kap**2/hh**3:.5f})")
    print("=> la lastra e' attratta verso il corpo; sul corpo agisce la forza opposta")
    fig, axs = plt.subplots(2, 1, figsize=(5.6, 4.4), sharex=True,
                            gridspec_kw={"height_ratios": [1.6, 1]})
    axs[0].contour(X, Y, P, levels=np.linspace(0.1, 3.0, 15), colors=BLU,
                   linewidths=0.6, linestyles="solid")
    axs[0].contour(X, Y, P, levels=[0.0], colors=ROSSO, linewidths=1.2)
    axs[0].axhline(0, color="k", lw=2)
    axs[0].set_aspect("equal"); axs[0].set_ylim(0, 3.5); axs[0].grid(False)
    axs[0].set_ylabel("$y$")
    axs[1].plot(xw, cpw, color=ARANCIO); axs[1].invert_yaxis()
    axs[1].axhline(0, color=GRIGIO, lw=0.5)
    axs[1].set_xlim(-4, 4); axs[1].set_xlabel("$x$"); axs[1].set_ylabel("$C_p$ sulla parete")
    salva(fig, "es_03")


# ============================================================ esercizio 4
def es_04():
    """von Karman con sorgenti puntiformi per la sfera."""
    titolo(4, "sfera con sorgenti puntiformi sull'asse")
    A, U = 1.0, 1.0
    for n in (4, 6, 10):
        zs = np.linspace(-0.6, 0.6, n)                   # posizioni delle sorgenti
        fi = np.linspace(np.pi, 0, n + 1)[1:-1]          # n-1 punti di controllo
        r, z = A * np.sin(fi), A * np.cos(fi)
        R = np.sqrt(r[:, None]**2 + (z[:, None] - zs[None, :])**2)
        M = np.zeros((n, n))
        M[:-1] = -(z[:, None] - zs[None, :]) / R        # psi di una sorgente: -m (z-zs)/R
        M[-1] = 1.0                                       # somma delle intensita' nulla
        b = np.concatenate([-0.5 * U * r**2, [0.0]])
        m = np.linalg.solve(M, b)
        ft = np.linspace(0.05, np.pi - 0.05, 200)
        rt, zt = A * np.sin(ft), A * np.cos(ft)
        Rt = np.sqrt(rt[:, None]**2 + (zt[:, None] - zs[None, :])**2)
        uz = U + ((zt[:, None] - zs[None, :]) / Rt**3) @ m
        ur = (rt[:, None] / Rt**3) @ m
        V = np.hypot(ur, uz)
        err = np.max(np.abs(V - 1.5 * U * rt / A))
        print(f"n = {n:2d}: intensita' m_j/U = {np.array2string(m, precision=4)}")
        print(f"        max |V - V_esatta|/U sulla sfera = {err:.2e};  cond = {np.linalg.cond(M):.1e}")
        if n == 6:
            Vs, fs = V.copy(), ft.copy()
    mu_eq = np.sum(m * zs) * -1
    print(f"momento di doppietta equivalente -sum m_j z_j = {mu_eq:.4f}  (esatto U a^3/2 = 0,5)")
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(np.degrees(fs), 1 - 2.25 * np.sin(fs)**2, color=GRIGIO, lw=2, label="esatto")
    ax.plot(np.degrees(fs), 1 - Vs**2, "--", color=ARANCIO, label="6 sorgenti puntiformi")
    ax.invert_yaxis(); ax.set_xlabel(r"$\varphi$ [gradi]"); ax.set_ylabel("$C_p$"); ax.legend()
    salva(fig, "es_04")


# ============================================================ esercizio 5
def _naca00(x, t=0.09):
    return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                    + 0.2843 * x**3 - 0.1036 * x**4)


def _G2d(x, y, xi):
    """Primitiva di int theta d(xi): s theta + (y/2) ln(s^2 + y^2), s = x - xi."""
    s = x - xi
    return s * np.arctan2(y, s) + 0.5 * y * np.log(s**2 + y**2)


def es_05():
    """Progetto: metodo di von Karman piano per il profilo NACA 0009."""
    titolo(5, "von Karman piano: NACA 0009 a incidenza nulla")
    tab_x = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    tab_V = np.array([1.140, 1.144, 1.137, 1.119, 1.100, 1.082, 1.061, 1.043, 1.018, 0.982])
    risultati = {}
    print("  n   spaziatura   cond(M)     max|V - V_tab|/U")
    for n in (10, 20, 40):
        for nome in ("uniforme", "coseno"):
            xx = np.linspace(0, 1, n + 1) if nome == "uniforme" \
                else 0.5 * (1 - np.cos(np.linspace(0, np.pi, n + 1)))
            xc = xx[1:-1]; yc = _naca00(xc)
            M = np.zeros((n, n))
            for j in range(n):
                M[:-1, j] = (_G2d(xc, yc, xx[j]) - _G2d(xc, yc, xx[j + 1])) / (2 * np.pi)
            M[-1] = np.diff(xx)
            q = np.linalg.solve(M, np.concatenate([-yc, [0.0]]))
            def velocita(x, y, xx=xx, q=q):
                u = np.ones_like(x); v = np.zeros_like(x)
                for j in range(len(q)):
                    u += q[j] / (4 * np.pi) * np.log(((x - xx[j])**2 + y**2) /
                                                     ((x - xx[j + 1])**2 + y**2))
                    v += q[j] / (2 * np.pi) * (np.arctan2(y, x - xx[j + 1]) - np.arctan2(y, x - xx[j]))
                return u, v
            u, v = velocita(tab_x, _naca00(tab_x))
            V = np.hypot(u, v)
            print(f"  {n:2d}   {nome:9s}   {np.linalg.cond(M):.1e}      {np.max(np.abs(V-tab_V)):.4f}")
            risultati[(n, nome)] = (xx, q, velocita)
    xx, q, velocita = risultati[(20, "coseno")]
    xf = np.linspace(0.005, 0.995, 300)
    u, v = velocita(xf, _naca00(xf)); Vf = np.hypot(u, v)
    u, v = velocita(tab_x, _naca00(tab_x)); Vt = np.hypot(u, v)
    print("\n  x/c    V/U tabella   V/U von Karman (n = 20, coseno)")
    for a, b, c in zip(tab_x, tab_V, Vt):
        print(f"  {a:4.2f}     {b:5.3f}          {c:6.4f}")
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ax[0].plot(xf, Vf, color=BLU, label="von Karman, $n=20$")
    ax[0].plot(tab_x, tab_V, "o", color=ARANCIO, ms=4, label="valori tabulati")
    ax[0].set_xlabel("$x/c$"); ax[0].set_ylabel("$V/U$"); ax[0].legend(fontsize=7)
    ax[0].set_ylim(0.9, 1.25)
    xm = 0.5 * (xx[1:] + xx[:-1])
    dydx = np.gradient(_naca00(xf), xf)
    ax[1].step(xx, np.concatenate([q, [q[-1]]]), where="post", color=BLU, label="$q_j/U$")
    ax[1].plot(xf, 2 * dydx, "--", color=GRIGIO, label=r"$2\,\mathrm{d}y_t/\mathrm{d}x$")
    ax[1].set_ylim(-0.3, 1.2); ax[1].set_xlabel("$x/c$"); ax[1].legend(fontsize=7)
    salva(fig, "es_05")


# ============================================================ esercizi 6 e 7
def _lastra(alfa, con_kutta, npunti=400):
    a, U = 1.0, 1.0
    gam = 4 * np.pi * a * U * np.sin(alfa) if con_kutta else 0.0
    fi = np.linspace(0, 2 * np.pi, npunti, endpoint=False) + 1e-4
    zeta = a * np.exp(1j * fi)
    dw = U * (np.exp(-1j * alfa) - a**2 * np.exp(1j * alfa) / zeta**2) + 1j * gam / (2 * np.pi * zeta)
    dz = 1 - 1 / zeta**2
    V = np.abs(dw / dz)
    x = (zeta + 1 / zeta).real
    return fi, x, V, gam


def _linee_lastra(ax, alfa, con_kutta):
    a, U = 1.0, 1.0
    gam = 4 * np.pi * a * U * np.sin(alfa) if con_kutta else 0.0
    r = np.linspace(1.0, 5.0, 300); t = np.linspace(0, 2 * np.pi, 721)
    Z = r[:, None] * np.exp(1j * t[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        W = U * (Z * np.exp(-1j * alfa) + np.exp(1j * alfa) / Z) + 1j * gam / (2 * np.pi) * np.log(Z)
    figt, axt = plt.subplots()
    cs = axt.contour(Z.real, Z.imag, W.imag, levels=np.linspace(-3, 3, 25))
    for segs in cs.allsegs:
        for s in segs:
            zz = s[:, 0] + 1j * s[:, 1]
            zm = zz + 1 / zz
            ax.plot(zm.real, zm.imag, color=BLU, lw=0.6)
    plt.close(figt)
    ax.plot([-2, 2], [0, 0], color="k", lw=2.0)
    ax.set_xlim(-4, 4); ax.set_ylim(-2.5, 2.5); ax.set_aspect("equal"); ax.grid(False)


def es_06():
    """Lastra piana a incidenza, senza circolazione."""
    titolo(6, "lastra piana senza circolazione")
    alfa = np.radians(10.0)
    fi, x, V, gam = _lastra(alfa, False)
    cp = 1 - V**2
    dorso = np.sin(fi) > 0
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.8))
    _linee_lastra(ax[0], alfa, False)
    ax[0].set_title(r"$\alpha = 10^\circ$, $\Gamma = 0$", fontsize=8)
    ax[1].plot(x[dorso], cp[dorso], color=BLU, label="dorso")
    ax[1].plot(x[~dorso], cp[~dorso], color=ARANCIO, label="ventre")
    ax[1].set_ylim(4, -12); ax[1].set_xlabel("$x$"); ax[1].set_ylabel("$C_p$"); ax[1].legend(fontsize=7)
    salva(fig, "es_06")
    # punti di ristagno sul cerchio: fi = alfa e fi = pi + alfa
    zs = np.exp(1j * np.array([alfa, np.pi + alfa]))
    print(f"ristagni sul cerchio in fi = {np.degrees(alfa):.1f} e {180+np.degrees(alfa):.1f} gradi")
    print(f"-> sulla lastra in x = {(zs + 1/zs).real.round(4)} (ristagno posteriore sul dorso)")
    print("velocita' infinita ai due spigoli (dz/dzeta = 0 in zeta = +-1)")
    # forza: per simmetria della distribuzione la risultante e' nulla
    print("Gamma = 0: per il teorema di Kutta-Joukowski la portanza e' nulla")


def es_07():
    """Lastra piana con condizione di Kutta: portanza e suzione al bordo d'attacco."""
    titolo(7, "lastra piana con condizione di Kutta")
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8))
    alfa = np.radians(10.0)
    _linee_lastra(axs[0], alfa, True)
    axs[0].set_title(r"$\alpha = 10^\circ$, $\Gamma = 4\pi U a\sin\alpha$", fontsize=8)
    fi, x, V, gam = _lastra(alfa, True, 4000)
    cp = 1 - V**2
    dorso = np.sin(fi) > 0
    axs[1].plot(x[dorso], cp[dorso], color=BLU, label="dorso")
    axs[1].plot(x[~dorso], cp[~dorso], color=ARANCIO, label="ventre")
    axs[1].set_ylim(1.5, -8); axs[1].set_xlabel("$x$"); axs[1].set_ylabel("$C_p$")
    axs[1].legend(fontsize=7)
    salva(fig, "es_07")
    c = 4.0                                 # corda della lastra (da -2a a 2a)
    print("  alfa   C_L=2pi sin(a)   C_N (integrale di C_p)   2pi sin(a)cos(a)   C_S=2pi sin^2(a)"
          "   C_N cos(a) + C_S sin(a)")
    for ag in (2.0, 5.0, 10.0):
        al = np.radians(ag)
        # C_N = (1/c) int (C_p,ventre - C_p,dorso) dx, con x = 2 cos(fi), dx = -2 sin(fi) dfi
        fi = np.linspace(0, np.pi, 200001)[1:-1]
        cp = lambda f: 1 - _V_lastra(f, al)**2
        CN = np.trapezoid((cp(2 * np.pi - fi) - cp(fi)) * 2 * np.sin(fi), fi) / c
        CL = 2 * np.pi * np.sin(al)
        CS = 2 * np.pi * np.sin(al)**2
        print(f"  {ag:4.1f}     {CL:.5f}            {CN:.5f}              {2*np.pi*np.sin(al)*np.cos(al):.5f}"
              f"           {CS:.5f}              {CN*np.cos(al) + CS*np.sin(al):.5f}")
    print("la pressione sulle due facce da' solo la forza normale C_N = 2 pi sin(a) cos(a);")
    print("la portanza 2 pi sin(a) richiede anche la suzione C_S concentrata nel bordo d'attacco")


def _V_lastra(fi, alfa):
    """Velocita' sulla lastra (z = zeta + 1/zeta, a = 1) con circolazione di Kutta."""
    zeta = np.exp(1j * fi)
    gam = 4 * np.pi * np.sin(alfa)
    dw = (np.exp(-1j * alfa) - np.exp(1j * alfa) / zeta**2) + 1j * gam / (2 * np.pi * zeta)
    return np.abs(dw / (1 - 1 / zeta**2))


# ============================================================ esercizio 8
def es_08():
    """SOR contro Liebmann sul problema del Programma 2.5."""
    titolo(8, "SOR e Liebmann a confronto")
    import prog2_5_vortice_rettangolo as p25
    import prog2_5b_convergenza as p25b
    x, y = p25.griglia(); M, N = len(x), len(y)
    zeta = np.zeros((M, N)); zeta[16, 12] = p25.ZETA0
    psi_star = p25b.soluzione_diretta(zeta, p25.H)
    interni = [(i, j) for i in range(1, M - 1) for j in range(1, N - 1)]
    print("  metodo          omega    iterazioni (somma|delta| <= 0,001)   errore vero max")
    for nome, w in (("Liebmann", 1.0), ("SOR, formula", omega_ottimo(M - 1, N - 1)),
                    ("SOR", 1.5), ("SOR", 1.9)):
        psi = np.zeros((M, N)); psi[1:-1, 1:-1] = 0.5
        it, err = 0, 1.0
        while err > 0.001 and it < 2000:
            it += 1
            err = sor_passo(psi, zeta, p25.H, interni, w)
        print(f"  {nome:14s}  {w:5.3f}        {it:4d}                             "
              f"{np.max(np.abs(psi-psi_star)):.2e}")


# ============================================================ esercizio 9
def es_09():
    """Due vortici concentrati nel rettangolo."""
    titolo(9, "due vortici nel rettangolo")
    import prog2_5_vortice_rettangolo as p25
    import prog2_5b_convergenza as p25b
    x, y = p25.griglia(0.125); M, N = len(x), len(y); h = 0.125
    G = p25.ZETA0 * 0.25**2
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for ax, segno, nome in zip(axs, (-1, 1), ("segni opposti", "stesso segno")):
        zeta = np.zeros((M, N))
        for (xv, yv, s) in ((-1.0, 0.0, 1.0), (1.0, 0.0, segno)):
            zeta[int(round((xv + 3) / h)), int(round((yv + 2) / h))] = s * G / h**2
        psi = p25b.soluzione_diretta(zeta, h)
        if segno < 0:
            liv = np.concatenate([-np.array([1.5, 1.0, 0.5, 0.25, 0.1]), [0.1, 0.25, 0.5, 1.0, 1.5]])
            ax.contour(x, y, psi.T, levels=np.sort(liv), colors=BLU, linewidths=0.6, linestyles="solid")
            ax.contour(x, y, psi.T, levels=[0.0], colors=ROSSO, linewidths=1.2)
        else:
            # separatrice a otto: livello di psi nel punto di sella (x = 0, y = 0)
            ps = psi[int(round(3 / h)), int(round(2 / h))]
            print(f"stesso segno: psi nel punto di sella = {ps:.4f}")
            liv = np.array([0.1, 0.3, 0.6, 1.0, 1.4, 2.0, 2.6])
            ax.contour(x, y, psi.T, levels=liv, colors=BLU, linewidths=0.6, linestyles="solid")
            ax.contour(x, y, psi.T, levels=[ps], colors=ROSSO, linewidths=1.2)
        ax.plot([-1, 1], [0, 0], "k+", ms=6)
        ax.plot([-3, 3, 3, -3, -3], [-2, -2, 2, 2, -2], "k", lw=1.2)
        ax.set_aspect("equal"); ax.set_title(nome, fontsize=8); ax.grid(False)
    salva(fig, "es_09")
    print("segni opposti: psi = 0 sul contorno e lungo l'asse x = 0, che separa i due vortici")


# ============================================================ esercizio 10
def es_10():
    """Canale con un cilindro a sezione rettangolare (un quarto del dominio)."""
    titolo(10, "cilindro rettangolare in un canale")
    h = 0.25
    x = np.arange(-6.0, 0.0 + 1e-9, h); y = np.arange(0.0, 4.0 + 1e-9, h)
    M, N = len(x), len(y)
    solido = (x[:, None] >= -2.0 - 1e-9) & (y[None, :] <= 1.0 + 1e-9)
    psi = np.zeros((M, N))
    psi[0, :] = y / 4.0                       # AB: corrente uniforme in ingresso
    psi[:, -1] = 1.0                          # BC: parete del canale
    interni = [(i, j) for i in range(1, M) for j in range(1, N - 1) if not solido[i, j]]
    for i, j in interni:
        psi[i, j] = y[j] / 4.0
    w = omega_ottimo(M - 1, N - 1)
    it, err = 0, 1.0
    while err > 1e-6 and it < 5000:
        it += 1; err = 0.0
        for i, j in interni:
            if i == M - 1:                    # CD: dpsi/dx = 0 (nodo fittizio)
                gs = 0.25 * (2 * psi[i - 1, j] + psi[i, j - 1] + psi[i, j + 1])
            else:
                gs = 0.25 * (psi[i - 1, j] + psi[i + 1, j] + psi[i, j - 1] + psi[i, j + 1])
            nuovo = (1 - w) * psi[i, j] + w * gs
            err = max(err, abs(nuovo - psi[i, j])); psi[i, j] = nuovo
    print(f"SOR con omega = {w:.4f}: {it} iterazioni (max|delta| < 1e-6)")
    # velocita' sulla faccia superiore DE (y = 1) e sulla faccia frontale EF (x = -2)
    j1 = int(round(1.0 / h)); i2 = int(round(4.0 / h))
    u_top = (-3 * psi[i2:, j1] + 4 * psi[i2:, j1 + 1] - psi[i2:, j1 + 2]) / (2 * h)
    # psi e' adimensionale con U H (H = 4): u/U = H dpsi/dy
    print("velocita' u/U sulla faccia superiore (y = 1):")
    for xx, uu in zip(x[i2:], 4 * u_top):
        print(f"   x = {xx:5.2f}   u/U = {uu:.4f}")
    print(f"velocita' media u/U nella sezione x = 0, 1 < y < 4 (continuita': 4/3 = 1,333): "
          f"{4*(psi[-1, -1]-psi[-1, j1])/3:.4f}")
    print(f"u/U in x = 0 sull'asse di simmetria, appena sopra il corpo (y = 1,25): "
          f"{4*(psi[-1, j1+2]-psi[-1, j1])/(2*h):.4f};  sotto la parete (y = 3,75): "
          f"{4*(psi[-1, -1]-psi[-1, -3])/(2*h):.4f}")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    P = psi.copy(); P[solido] = 0.0
    xf = np.concatenate([x, -x[::-1][1:]])
    Pf = np.concatenate([P, P[::-1][1:]], axis=0)
    yf = np.concatenate([-y[::-1], y[1:]])
    Pf = np.concatenate([-Pf[:, ::-1], Pf[:, 1:]], axis=1)
    ax.contour(xf, yf, Pf.T, levels=np.linspace(-0.9, 0.9, 19), colors=BLU,
               linewidths=0.7, linestyles="solid")
    ax.fill([-2, 2, 2, -2], [-1, -1, 1, 1], color="#d5d8dc", zorder=3)
    ax.plot([-2, 2, 2, -2, -2], [-1, -1, 1, 1, -1], "k", lw=1, zorder=4)
    ax.plot([-6, 6], [4, 4], "k", lw=2); ax.plot([-6, 6], [-4, -4], "k", lw=2)
    ax.set_aspect("equal"); ax.grid(False); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    salva(fig, "es_10")


# ============================================================ esercizio 11
def es_11():
    """Flusso assialsimmetrico in un tubo con diaframmi forati ripetuti."""
    titolo(11, "tubo con diaframmi (flusso assialsimmetrico)")
    h = 0.1
    z = np.arange(0.0, 2.0 + 1e-9, h); r = np.arange(0.0, 1.0 + 1e-9, h)
    M, N = len(z), len(r)
    psi = np.zeros((M, N))
    psi[:, -1] = 1.0                                  # BC: parete del tubo
    diaframma = r >= 0.5 - 1e-9                       # AB: diaframma in z = 0
    psi[0, diaframma] = 1.0
    interni = [(i, j) for i in range(M) for j in range(1, N - 1)
               if not (i == 0 and diaframma[j])]
    for i, j in interni:
        psi[i, j] = (r[j] / 1.0)**2
    w = 1.8
    it, err = 0, 1.0
    while err > 1e-7 and it < 20000:
        it += 1; err = 0.0
        for i, j in interni:
            sx = psi[i + 1, j] if i == 0 else psi[i - 1, j]       # nodi fittizi in z = 0
            dx_ = psi[i - 1, j] if i == M - 1 else psi[i + 1, j]  # ... e in z = 2
            gs = 0.25 * (sx + dx_ + (1 + h / (2 * r[j])) * psi[i, j - 1]
                         + (1 - h / (2 * r[j])) * psi[i, j + 1])
            nuovo = (1 - w) * psi[i, j] + w * gs
            err = max(err, abs(nuovo - psi[i, j])); psi[i, j] = nuovo
    print(f"SOR (omega = {w}): {it} iterazioni")
    # velocita' assiale sull'asse u_z = (1/r) dpsi/dr ~ 2 psi/r^2 per r -> 0
    uz_asse = 2 * psi[:, 1] / r[1]**2
    print(f"velocita' assiale sull'asse: nel foro (z = 0) {uz_asse[0]:.3f}, "
          f"a meta' cella (z = 2) {uz_asse[-1]:.3f};  media nel tubo pieno 2 psi_parete/R^2 = 2")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    zz = np.concatenate([z - 2, z[1:]]); zz = np.concatenate([zz, zz[1:] + 4])
    P = np.concatenate([psi[::-1], psi[1:]], axis=0); P = np.concatenate([P, P[1:]], axis=0)
    rr = np.concatenate([-r[::-1], r[1:]])
    PP = np.concatenate([P[:, ::-1], P[:, 1:]], axis=1)
    ax.contour(zz, rr, PP.T, levels=np.linspace(0.1, 0.9, 9), colors=BLU,
               linewidths=0.7, linestyles="solid")
    for zd in (0.0, 4.0):
        ax.plot([zd, zd], [0.5, 1.0], "k", lw=2.5); ax.plot([zd, zd], [-0.5, -1.0], "k", lw=2.5)
    ax.plot([-2, 6], [1, 1], "k", lw=2); ax.plot([-2, 6], [-1, -1], "k", lw=2)
    ax.axhline(0, color=GRIGIO, lw=0.6, ls="-.")
    ax.set_aspect("equal"); ax.set_xlabel("$z$"); ax.set_ylabel("$r$"); ax.grid(False)
    salva(fig, "es_11")


# ============================================================ esercizio 12
def es_12():
    """Progetto: tubo d'urto linearizzato (leapfrog per la densita')."""
    titolo(12, "tubo d'urto linearizzato")
    h, Co = 0.01, 1.0
    x = np.arange(0.0, 1.0 + 1e-9, h); m = len(x)
    def integra(n, f, aperto_a_destra=False):
        R = np.zeros((n + 1, m)); R[0] = f
        # partenza: rho^1 = (f_{i-1} + f_{i+1})/2 (g = 0); estremi chiusi: nodi fittizi
        R[1, 1:-1] = 0.5 * (f[:-2] + f[2:])
        R[1, 0] = f[1]
        R[1, -1] = 0.0 if aperto_a_destra else f[-2]
        for j in range(1, n):
            R[j + 1, 1:-1] = R[j, :-2] + R[j, 2:] - R[j - 1, 1:-1]
            R[j + 1, 0] = 2 * R[j, 1] - R[j - 1, 0]           # chiuso: drho/dx = 0
            R[j + 1, -1] = 0.0 if aperto_a_destra else 2 * R[j, -2] - R[j - 1, -1]
        return R
    # (a) tubo chiuso alle due estremita', rho' = 1 a destra del diaframma
    R = integra(200, np.where(x >= 0.5 - 1e-12, 1.0, 0.0))
    # (b) estremo destro aperto: il gas a destra ha la densita' esterna (rho' = 0)
    Ra = integra(200, np.where(x < 0.5 - 1e-12, 1.0, 0.0), aperto_a_destra=True)
    fig, axs = plt.subplots(5, 2, figsize=(6.6, 6.0), sharex=True, sharey=True)
    for k, n in enumerate((0, 25, 45, 70, 100)):
        for c, (RR, nome) in enumerate(((R, "(a) estremi chiusi"), (Ra, "(b) estremo destro aperto"))):
            ax = axs[k, c]
            ax.plot(x, RR[n], color=BLU if c == 0 else VERDE, lw=1.1)
            ax.set_ylim(-0.6, 1.6); ax.set_yticks([0, 1]); ax.set_xticks([0, 0.5, 1])
            ax.text(0.98, 0.75, fr"$t={n}\tau$", transform=ax.transAxes, ha="right", fontsize=7)
            if k == 0:
                ax.set_title(nome, fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("$x$ [m]")
    fig.subplots_adjust(hspace=0.1, wspace=0.08)
    salva(fig, "es_12")
    print("(a) estremi chiusi: rho' in x = 0,25 e x = 0,75;   (b) estremo aperto: idem")
    for n in (0, 20, 40, 60, 80, 100, 150, 200):
        print(f"   t = {n:3d} tau:  (a) {R[n, 25]:5.2f}  {R[n, 75]:5.2f}     (b) {Ra[n, 25]:5.2f}  {Ra[n, 75]:5.2f}")


# ============================================================ esercizio 13
def es_13():
    """Riflessione di onde di ampiezza finita agli estremi del tubo."""
    titolo(13, "onde di ampiezza finita: riflessione")
    import prog2_9_onda_finita as p29
    x = p29.H * np.arange(p29.M)
    u0 = p29.u_iniziale()
    us = np.zeros_like(u0); us[50:] = u0[:p29.M - 50]          # spostata di 1 m
    U1, A1 = p29.cir(40, u0, verso=-1)                         # onda verso sinistra
    U2, A2 = p29.cir(40, us, verso=+1)                         # onda verso destra, estremo aperto
    fig, axs = plt.subplots(5, 2, figsize=(6.6, 6.2), sharex=True, sharey=True)
    for k, n in enumerate((0, 10, 20, 30, 40)):
        axs[k, 0].plot(x, U1[n], color=BLU, lw=1.1)
        axs[k, 1].plot(x, U2[n], color=VERDE, lw=1.1)
        for ax in axs[k]:
            ax.axhline(0, color=GRIGIO, lw=0.4); ax.set_ylim(-200, 200)
            ax.set_yticks([-150, 0, 150]); ax.set_xticks([0, 1, 2])
            ax.text(0.98, 0.75, fr"$t={n}\tau$", transform=ax.transAxes, ha="right", fontsize=7)
    axs[0, 0].set_title("onda verso sinistra (estremo chiuso)", fontsize=8)
    axs[0, 1].set_title("onda verso destra (estremo aperto)", fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("$x$ [m]")
    fig.subplots_adjust(hspace=0.1, wspace=0.08)
    salva(fig, "es_13")
    g = p29.GAMMA
    print("estremo chiuso (x = 0): u = 0 sempre; velocita' del suono e pressione (isentropica)")
    for n in (0, 10, 20, 30, 40):
        print(f"   t = {n:2d} tau:  a(0) = {A1[n, 0]:6.1f} m/s,  p/p0 = {(A1[n, 0]/340)**(2*g/(g-1)):.3f}")
    print("estremo aperto (x = 2 m): a = a0 sempre; velocita' del gas")
    for n in (0, 10, 20, 30, 40):
        print(f"   t = {n:2d} tau:  u(2) = {U2[n, -1]:7.1f} m/s")
    print(f"   velocita' minima nell'onda incidente: {us.min():.1f} m/s")


# ============================================================ esercizio 14
def es_14():
    """Verifica simbolica della soluzione di Benton e Platzman (SymPy)."""
    titolo(14, "soluzione esatta dell'equazione di Burgers")
    import sympy as sp
    x, t = sp.symbols("x t", real=True)
    u = -2 * sp.sinh(x) / (sp.cosh(x) - sp.exp(-t))
    residuo = sp.simplify(sp.diff(u, t) + u * sp.diff(u, x) - sp.diff(u, x, 2))
    print("u_t + u u_x - u_xx =", residuo)
    print("limite per t -> infinito:", sp.limit(u, t, sp.oo))
    print("pendenza in x = 0:  du/dx =", sp.simplify(sp.diff(u, x).subs(x, 0)))
    print("limite - (-2 tanh x) =", sp.simplify((sp.limit(u, t, sp.oo) + 2 * sp.tanh(x)).rewrite(sp.exp)))
    # trasformazione di Cole-Hopf: u = -2 phi_x / phi con phi soluzione di phi_t = phi_xx
    phi = sp.exp(t) * sp.cosh(x) - 1
    print("phi = e^t cosh x - 1:  phi_t - phi_xx =",
          sp.simplify(sp.diff(phi, t) - sp.diff(phi, x, 2)))
    print("-2 phi_x/phi - u =", sp.simplify(-2 * sp.diff(phi, x) / phi - u))


# ============================================================ esercizi 15 e 16
def es_15():
    """Burgers con MacCormack e condizioni di Neumann."""
    titolo(15, "Burgers: MacCormack esplicito")
    import prog2_12_burgers as pb
    istanti = (0.2, 0.5, 1.0, 2.0)
    x, u, dt, sal = pb.maccormack_burgers(2.0, istanti=istanti)
    dx = x[1] - x[0]
    u0 = pb.esatta(x, 0.0)
    print(f"dx = {dx:.4f}; |u|max(t=0) = {np.max(np.abs(u0)):.2f}")
    print(f"limiti: convettivo dx/|u|max = {dx/np.max(np.abs(u0)):.5f}, viscoso dx^2/2 = {dx**2/2:.5f};"
          f" dt usato = {dt:.5f}")
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    xf = np.linspace(-6, 6, 600)
    print("   t      max|u - u_esatta|   (|x| < 5)")
    for (tr, (ts, us)), col in zip(sorted(sal.items()), (VIOLA, VERDE, ARANCIO, BLU)):
        ax.plot(xf, pb.esatta(xf, ts), color=GRIGIO, lw=1.8)
        ax.plot(x, us, "o", ms=2.2, color=col, label=f"$t={ts:.2f}$")
        e = np.abs(us - pb.esatta(x, ts))
        print(f"  {ts:5.3f}     {e.max():.3e}        {e[np.abs(x) < 5].max():.3e}")
    ax.set_xlim(-6, 6); ax.set_ylim(-4, 4); ax.set_xlabel("$x$"); ax.set_ylabel("$u$")
    ax.legend(fontsize=7)
    salva(fig, "es_15")
    print("raffinamento della griglia (t = 1): errore massimo totale, per |x| < 2, e in x = 6")
    for nx in (100, 200, 400):
        xg, _, _, sg = pb.maccormack_burgers(1.0, nx=nx, istanti=(1.0,))
        ts, us = sg[1.0]
        e = us - pb.esatta(xg, ts)
        print(f"   nx = {nx:3d}:  {np.max(np.abs(e)):.3e}   {np.max(np.abs(e[np.abs(xg) < 2])):.3e}"
              f"   {e[-1]:+.3e}")


def es_16():
    """Burgers con Beam-Warming e condizioni di Dirichlet."""
    titolo(16, "Burgers: Beam-Warming implicito")
    import prog2_12_burgers as pb
    istanti = (0.2, 0.5, 1.0, 2.0)
    x = pb.griglia(); dx = x[1] - x[0]
    umax = np.max(np.abs(pb.esatta(x, 0.0)))
    print("   dt      Co iniziale   mu_e     errore massimo a t = 0,2   0,5   1   2")
    risultati = {}
    for dt, mu in ((0.002, 0.0), (0.01, 0.0), (0.01, 0.1), (0.05, 0.0)):
        _, _, sal = pb.beam_warming_burgers(2.0, dt, mu_e=mu, istanti=istanti)
        errs = [np.max(np.abs(us - pb.esatta(x, ts))) for _, (ts, us) in sorted(sal.items())]
        risultati[(dt, mu)] = sal
        print(f"  {dt:5.3f}     {umax*dt/dx:6.2f}      {mu:4.2f}      " +
              "  ".join(f"{e:.2e}" for e in errs))
    _, _, dtm, salm = pb.maccormack_burgers(2.0, istanti=istanti)
    errs = [np.max(np.abs(us - pb.esatta(x, ts))) for _, (ts, us) in sorted(salm.items())]
    print(f"  MacCormack (dt = {dtm:.4f}):                 " + "  ".join(f"{e:.2e}" for e in errs))
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
    xf = np.linspace(-6, 6, 600)
    for ax, chiave, tit in zip(axs, ((0.01, 0.0), (0.05, 0.0)), ("dt = 0,01", "dt = 0,05")):
        for (tr, (ts, us)), col in zip(sorted(risultati[chiave].items()), (VIOLA, VERDE, ARANCIO, BLU)):
            ax.plot(xf, pb.esatta(xf, ts), color=GRIGIO, lw=1.6)
            ax.plot(x, us, "o", ms=2.0, color=col, label=f"$t={ts:.2f}$")
        ax.set_title(tit.replace(",", "{,}") if False else tit, fontsize=8)
        ax.set_xlim(-6, 6); ax.set_ylim(-4, 4); ax.set_xlabel("$x$")
    axs[0].set_ylabel("$u$"); axs[0].legend(fontsize=7)
    salva(fig, "es_16")


# ============================================================ esercizio 17
def es_17():
    """Tubo elastico: frequenza, ampiezza della forzante e attrito turbolento."""
    titolo(17, "tubo elastico: studio parametrico")
    import prog2_13_tubo_elastico as pt
    casi = (("riferimento", dict()),
            ("frequenza doppia", dict(periodo=0.35)),
            ("frequenza dimezzata", dict(periodo=1.4)),
            ("ampiezza doppia", dict(p_min=40.0, p_max=100.0)),
            ("ampiezza dimezzata", dict(p_min=62.5, p_max=77.5)),
            ("attrito turbolento", dict(tipo_attrito="turbolento")))
    fig, axs = plt.subplots(len(casi), 2, figsize=(6.6, 8.4), sharex=True)
    print("  caso                  p(z=20) [mmHg]      V(z=20) [cm/s]     portata media [cm^3/s]")
    for k, (nome, par) in enumerate(casi):
        T = par.get("periodo", pt.PERIODO)
        z, Hs, n, CV, _ = pt.simula(t_fin=6 * T, z_reg=(20.0,), **par)
        t = Hs[:, 0]; ult = t >= t[-1] - T
        P, V, Q = Hs[ult, 1], Hs[ult, 2], Hs[ult, 3]
        q = np.trapezoid(Q, t[ult]) / T
        print(f"  {nome:20s}  [{P.min():5.1f}, {P.max():5.1f}]     [{V.min():6.1f}, {V.max():6.1f}]"
              f"         {q:.3f}")
        fase = (t[ult] - t[ult][0]) / T
        axs[k, 0].plot(fase, P, color=ROSSO, lw=1.0)
        axs[k, 1].plot(fase, V, color=BLU, lw=1.0)
        axs[k, 1].axhline(0, color=GRIGIO, lw=0.4)
        axs[k, 0].text(0.98, 0.8, nome, transform=axs[k, 0].transAxes, ha="right", fontsize=7)
        axs[k, 0].set_ylim(0, 210); axs[k, 1].set_ylim(-40, 320)
    axs[0, 0].set_title("$p(z=20)$ [mmHg]", fontsize=8)
    axs[0, 1].set_title("$V(z=20)$ [cm/s]", fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("$t/T$ (ultimo periodo)")
    fig.subplots_adjust(hspace=0.12, wspace=0.18)
    salva(fig, "es_17")


if __name__ == "__main__":
    tutti = {k: v for k, v in globals().items() if k.startswith("es_")}
    scelti = [f"es_{int(a):02d}" for a in sys.argv[1:]] or sorted(tutti)
    for nome in scelti:
        tutti[nome]()
