"""
esercizi_svolti.py -- codice degli esercizi svolti dell'Appendice B.
Ogni esercizio e' una funzione es_NN(); eseguendo il file si svolgono tutti,
stampando i risultati e salvando le figure es_NN*.pdf nella cartella figure/.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from ode_solvers import integra, passo_rk4, cd_sfera, METODI
from stile_grafici import salva, PALETTE, BLU, ARANCIO, VERDE, VIOLA, ROSSO, GRIGIO

G = 9.8


def titolo(n, testo):
    print(f"\n{'=' * 70}\nEsercizio {n}: {testo}\n{'=' * 70}")


# ====================================================================== 1
def es_01():
    titolo(1, "metodi di Heun e Ralston")
    f = lambda t, x: x - t ** 2 + 1.0
    esatta = lambda t: (t + 1) ** 2 - 0.5 * np.exp(t)
    passi = [0.4, 0.2, 0.1, 0.05, 0.025]
    print(f"{'h':>6} {'punto medio':>12} {'Heun':>12} {'Ralston':>12}")
    err = {m: [] for m in ("rk2", "heun", "ralston")}
    for h in passi:
        n = int(round(2 / h))
        for m in err:
            t, X = integra(f, 0.0, 0.5, h, n, metodo=m)
            err[m].append(abs(X[-1, 0] - esatta(2.0)))
        print(f"{h:6.3f} {err['rk2'][-1]:12.3e} {err['heun'][-1]:12.3e} {err['ralston'][-1]:12.3e}")
    for m in err:
        e = np.array(err[m])
        print(f"  {m:11s}: ordine osservato {np.log2(e[:-1]/e[1:]).round(2)},  K = E/h^2 a h=0.025: {e[-1]/0.025**2:.3f}")
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for m, lab, c in (("rk2", "punto medio", BLU), ("heun", "Heun", ARANCIO), ("ralston", "Ralston", VERDE)):
        ax.loglog(passi, err[m], "o-", color=c, label=lab)
    ax.set_xlabel("$h$"); ax.set_ylabel("errore globale in $t=2$"); ax.legend()
    salva(fig, "es_01")


# ====================================================================== 2
def es_02():
    titolo(2, "stabilita' di RK4 su dx/dt = -x")
    R = lambda z: 1 + z + z**2/2 + z**3/6 + z**4/24
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for hl, c in ((2.5, VERDE), (2.785, ARANCIO), (3.0, ROSSO)):
        n = int(30 / hl)
        t, X = integra(lambda t, x: -x, 0.0, 1.0, hl, n, metodo="rk4")
        print(f"  h*lambda = {hl:5.3f}: R = {R(-hl):+.4f}, |R|^N = {abs(R(-hl))**n:.3e}, x_N = {X[-1,0]:+.3e} (N = {n})")
        ax.plot(t, X[:, 0], "o-", ms=3, color=c, label=f"$h\\lambda={hl}$")
    tt = np.linspace(0, 30, 300)
    ax.plot(tt, np.exp(-tt), "k-", lw=0.8, label="esatta")
    ax.set_ylim(-3, 3); ax.set_xlabel("$t$"); ax.set_ylabel("$x$"); ax.legend(ncol=2, fontsize=7)
    salva(fig, "es_02")


# ====================================================================== 3
def es_03():
    titolo(3, "errore di arrotondamento di RK4")
    f = lambda t, x: x - t ** 2 + 1.0
    esatta = (2 + 1) ** 2 - 0.5 * np.exp(2.0)
    print(f"{'h':>8} {'N':>8} {'errore':>12}")
    hh, ee = [], []
    for h in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        n = int(round(2 / h))
        t, X = integra(f, 0.0, 0.5, h, n, metodo="rk4")
        e = abs(X[-1, 0] - esatta); hh.append(h); ee.append(e)
        print(f"{h:8.0e} {n:8d} {e:12.3e}")
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.loglog(hh, ee, "o-", color=BLU, label="errore misurato")
    ax.loglog(hh, 0.03 * np.array(hh) ** 4, ":", color=GRIGIO, label="$\\propto h^4$")
    ax.loglog(hh, 1e-16 * 2 / np.array(hh), "--", color=ARANCIO, label="$\\propto \\epsilon_m N$")
    ax.set_xlabel("$h$"); ax.set_ylabel("errore globale"); ax.legend(fontsize=7)
    salva(fig, "es_03")


# ====================================================================== 4
def es_04():
    titolo(4, "sfera d'acciaio in acqua")
    from prog1_1_caduta_sfera import caduta, costanti
    d, rho, rhof, nu = 0.01, 8000.0, 1000.0, 1e-6
    A, B, C = costanti(d, rho, rhof)
    print(f"  rho_bar = {rhof/rho}, A = {A:.4f}, B = {B:.3f} m/s^2, C = {C:.3f} 1/m")
    # stima della velocita' terminale (ipotesi c_d = 0.5) e della costante di tempo
    vt = np.sqrt(B / (C * 0.5)); Re = vt * d / nu; tau = A / (2 * C * 0.5 * vt)
    print(f"  stima: v_t = {vt:.3f} m/s, Re = {Re:.0f} (c_d = 0.5 coerente), tau = {tau:.4f} s -> h <= {2.785*tau:.3f} s")
    for h in (0.1, 0.02, 0.005):
        T, Z, V, RE = caduta(d, h=h, tmax=2.0, rho=rho, rhof=rhof, nu=nu)
        print(f"  h = {h:5.3f}: v(2 s) = {V[-1]:.4f} m/s, z(2 s) = {Z[-1]:.4f} m, Re = {RE[-1]:.0f}")
    T, Z, V, RE = caduta(d, h=0.005, tmax=2.0, rho=rho, rhof=rhof, nu=nu)
    Ta, Za, Va, REa = caduta(d, h=0.1, tmax=2.0)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(T, V, color=BLU, label="in acqua"); ax.plot(Ta, Va, color=ARANCIO, label="in aria")
    ax.axhline(vt, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("$t$ [s]"); ax.set_ylabel("$v$ [m/s]"); ax.legend()
    salva(fig, "es_04")


# ====================================================================== 5
def es_05():
    titolo(5, "goccia che gela")
    from prog1_1_caduta_sfera import caduta
    d_w, rho_w, rho_i = 0.2e-3, 1000.0, 917.0
    d_i = d_w * (rho_w / rho_i) ** (1 / 3)
    for nome, d, rho in (("acqua", d_w, rho_w), ("ghiaccio", d_i, rho_i)):
        T, Z, V, RE = caduta(d, h=0.002, tmax=1.0, rho=rho)
        print(f"  {nome:9s}: d = {d*1e3:.4f} mm, v_t = {V[-1]:.4f} m/s, Re = {RE[-1]:.2f}, "
              f"formula di Stokes (fuori validita', Re>1): {(1-1.22/rho)*G*d**2/(18*1.22/rho*1.49e-5):.4f} m/s")


# ====================================================================== 6
def es_06():
    titolo(6, "caduta da 20 km in atmosfera stratificata")
    d, rho, mu, y0 = 0.10, 8000.0, 1.818e-5, 20000.0
    rhof = lambda y: 1.22 * np.exp(-0.000118 * y)

    def f(t, s):
        z, v = s
        y = y0 - z
        rb = rhof(y) / rho
        A, B, C = 1 + rb / 2, (1 - rb) * G, 3 * rb / (4 * d)
        nu = mu / rhof(y)
        return np.array([v, (B - C * v * abs(v) * cd_sfera(v * d / nu)) / A])

    T, S = integra(f, 0.0, [0.0, 0.0], 0.05, 4000, stop=lambda t, s: s[0] >= y0)
    z, v = S[:, 0], S[:, 1]
    i = np.argmax(v)
    print(f"  v_max = {v[i]:.1f} m/s a t = {T[i]:.1f} s, quota {y0 - z[i]:.0f} m; "
          f"v al suolo = {v[-1]:.1f} m/s, t di caduta = {T[-1]:.1f} s")
    # velocita' terminale locale (c_d = 0.18 oltre Re=2e6, altrimenti 0.5) per confronto
    yy = np.linspace(0, y0, 200)
    vt_loc = [np.sqrt((1 - rhof(y)/rho) * G / (3 * rhof(y)/rho / (4*d) * 0.5)) for y in yy]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(y0 - z, v, color=BLU, label="velocit\\`a calcolata")
    ax.plot(yy, vt_loc, "--", color=ARANCIO, label="$v_t$ locale ($c_d=0{,}5$)")
    ax.set_xlabel("quota [m]"); ax.set_ylabel("$v$ [m/s]"); ax.invert_xaxis(); ax.legend(fontsize=7)
    salva(fig, "es_06")


# ====================================================================== 7
def es_07():
    titolo(7, "guscio pieno d'aria in acqua (pendolo invertito)")
    d, rho, rhof, nu, l = 0.01, 1.22, 1000.0, 1e-6, 2.0
    rb = rhof / rho
    A, B, C = 1 + rb / 2, (1 - rb) * G, 3 * rb / (4 * d)
    print(f"  A = {A:.1f}, B = {B:.0f} m/s^2, C = {C:.0f} 1/m")

    def f(t, s):
        y, v = s
        return np.array([v, (-B * np.sin(y / l) - C * v * abs(v) * cd_sfera(v * d / nu)) / A])

    y0 = l * np.radians(5.0)
    tau = A / (2 * C * 0.5 * 0.5)                    # stima con |v| ~ 0.5 m/s
    print(f"  costante di tempo stimata tau = {tau:.4f} s -> h < 2.785 tau = {2.785*tau:.3f} s")
    for h in (0.1, 0.01):
        with np.errstate(all="ignore"):
            T, S = integra(f, 0.0, [y0, 0.0], h, int(40 / h))
        th = np.degrees(S[:, 0] / l)
        print(f"  h = {h}: theta(40 s) = {th[-1]:.2f} deg, |v| max = {abs(S[:,1]).max():.3f} m/s")
    T, S = integra(f, 0.0, [y0, 0.0], 0.01, 4000)
    th = np.degrees(S[:, 0] / l)
    for soglia in (90, 170, 179):
        i = np.where(th >= soglia)[0]
        print(f"  theta = {soglia} gradi raggiunto a t = {T[i[0]]:.2f} s" if len(i) else f"  theta = {soglia} non raggiunto in 40 s")
    print(f"  equilibrio (theta=180) stabile? B<0 -> si; velocita' massima lungo l'arco stimata con eq. (v_t): "
          f"{np.sqrt(-B/(C*0.5)):.3f} m/s")
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(T, th, color=BLU); ax.axhline(180, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("$t$ [s]"); ax.set_ylabel("$\\theta$ [gradi]")
    salva(fig, "es_07")


# ====================================================================== 8
def es_08():
    titolo(8, "linea elettrica (cilindro) nel vento oscillante")
    rho, rhof, d, l, cd = 8950.0, 1.22, 0.005, 0.3, 1.2
    k = 2 * rhof * cd / (np.pi * rho * d)          # coefficiente della forza tangenziale per unita' di massa
    print(f"  k = 2 rho_f c_d/(pi rho d) = {k:.5f} 1/m")

    def modello(om):
        def f(t, s):
            th, v = s[0] / l, s[1]
            u = 30.0 * (1 + 0.2 * np.sin(om * t))
            ur = u * np.cos(th) - v
            Ft = k * ur * np.hypot(u * np.sin(th), ur)
            return np.array([v, -G * np.sin(th) + Ft])
        return f

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for om, c in zip((0.0, 0.5, 1.0, 2.0), PALETTE):
        T, S = integra(modello(om), 0.0, [0.0, 0.0], 0.005, 4000)
        th = np.degrees(S[:, 0] / l)
        ax.plot(T, th, color=c, label=f"$\\omega={om}$")
        print(f"  omega = {om}: theta medio (ultimi 5 s) = {th[T>15].mean():.2f} deg, "
              f"escursione = {th[T>15].max()-th[T>15].min():.2f} deg")
    # equilibrio statico per omega = 0 (v = 0): g sin(theta) = k u cos(theta) * u  ->  tan(theta) = k u^2 / g
    th_eq = np.arctan(k * 30.0 ** 2 / G)
    print(f"  equilibrio statico (omega=0): tan(theta_eq) = k u^2/g = {k*900/G:.3f} -> theta_eq = {np.degrees(th_eq):.2f} deg")
    ax.set_xlabel("$t$ [s]"); ax.set_ylabel("$\\theta$ [gradi]"); ax.legend(ncol=4, fontsize=7)
    salva(fig, "es_08")


# ====================================================================== 9
def es_09():
    titolo(9, "vento critico per l'ala su molla")
    from prog1_3_ala_molla import sistema, beta, cl, alpha0
    print(f"  stima lineare: U_crit = 2/beta = {2/beta:.0f}")
    h, n = 0.01, 2000
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for U, c in zip((400, 800, 1000, 1100, 1300), PALETTE):
        T, S = integra(sistema(lambda T, U=U: U), 0.0, [0.0, 0.0], h, n)
        Zeq = beta * cl(alpha0) * U ** 2 / (2 * np.pi) ** 2
        over = S[:, 0].max() / Zeq - 1
        amin = np.degrees(alpha0 - np.arctan2(S[:, 1], U)).min()
        ax.plot(T, S[:, 0] / Zeq, color=c, label=f"$U={U}$")
        print(f"  U = {U:5d}: Z_eq = {Zeq:7.3f}, sovraelongazione = {100*over:5.1f} %, alpha min = {amin:5.2f} deg")
    ax.set_xlabel("$T$"); ax.set_ylabel("$Z/Z_{\\rm eq}$"); ax.set_xlim(0, 6); ax.legend(fontsize=7)
    salva(fig, "es_09")


# ====================================================================== 10
def es_10():
    titolo(10, "ricerca della frequenza di massima risposta")
    from prog1_3_ala_molla import sistema
    U0, a, h = 100.0, 0.2, 0.01
    omegas = np.linspace(0.2, 3.0, 29)
    amp_h, amp_v = [], []
    for om in omegas:
        T, S = integra(sistema(lambda T, om=om: U0 * (1 + a * np.sin(om * T))), 0.0, [0, 0], h, 8000)
        Z = S[T > 40, 0]; amp_h.append(0.5 * (Z.max() - Z.min()))     # ampiezza a regime (T > 40)
        T, S = integra(sistema(lambda T: U0, lambda T, om=om: -a * U0 * np.sin(om * T)), 0.0, [0, 0], h, 8000)
        Z = S[T > 40, 0]; amp_v.append(0.5 * (Z.max() - Z.min()))
    amp_h, amp_v = np.array(amp_h), np.array(amp_v)
    print(f"  raffica orizzontale: ampiezza max {amp_h.max():.3f} a omega = {omegas[amp_h.argmax()]:.2f}")
    print(f"  raffica verticale  : ampiezza max {amp_v.max():.3f} a omega = {omegas[amp_v.argmax()]:.2f}")
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(omegas, amp_h, "o-", ms=3, color=BLU, label="raffica orizzontale")
    ax.plot(omegas, amp_v, "s-", ms=3, color=ARANCIO, label="raffica verticale")
    ax.set_xlabel("$\\omega$"); ax.set_ylabel("ampiezza a regime di $Z$"); ax.legend(fontsize=7)
    salva(fig, "es_10")


# ====================================================================== 11
def es_11():
    titolo(11, "effetto Magnus su un proiettile sferico")
    from prog1_4_proiettile import traiettoria, gittata, RHO, RHOF, NU
    d, w0 = 0.05, 50.0
    rb = RHOF / RHO
    A, B, C = 1 + rb / 2, (1 - rb) * G, 3 * rb / (4 * d)

    def modello_magnus(cl):
        def f(t, s):
            x, y, u, v = s
            w = np.hypot(u, v)                      # aria ferma: w_rel = w
            cd = cd_sfera(w * d / NU)
            fx = (-C * cd * u * w - C * cl * w * v) / A
            fy = (-B - C * cd * v * w + C * cl * w * u) / A
            return np.array([u, v, fx, fy])
        return f

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for cl, c, lab in ((0.0, BLU, "senza rotazione"), (0.2, VERDE, "backspin, $c_l=0{,}2$"), (-0.2, ARANCIO, "topspin, $c_l=-0{,}2$")):
        S, sP, sQ = traiettoria(modello_magnus(cl), w0, 45.0)
        ax.plot(S[:, 1], S[:, 2], color=c, label=lab)
        print(f"  c_l = {cl:+.1f}: gittata = {gittata(sP, sQ):7.2f} m, quota max = {S[:,2].max():6.2f} m, t_volo ~ {S[-1,0]:.1f} s")
    ax.set_xlabel("$x$ [m]"); ax.set_ylabel("$y$ [m]"); ax.set_aspect("equal"); ax.legend(fontsize=7)
    salva(fig, "es_11")


# ====================================================================== 12
def es_12():
    titolo(12, "angoli di tiro per colpire un bersaglio")
    from prog1_8_cannone import modello_cannone, w0
    from prog1_4_proiettile import traiettoria, gittata
    f = modello_cannone(lambda y: 1.22 * np.exp(-0.000118 * y))
    xr = lambda th: gittata(*traiettoria(f, w0, th, 0.05)[1:])

    def angoli_bersaglio(xt, delta=5.0, eps=0.01):
        soluzioni, n = [], 0
        th_old, xr_old = 0.0, 0.0            # a theta = 0 la gittata e' nulla
        th_new = th_old + delta; xr_new = xr(th_new); n += 1
        d = delta
        while th_new < 90.0:
            if (xt - xr_old) * (xt - xr_new) < 0:  # bersaglio scavalcato
                if abs(d) < eps:                      # soluzione isolata con precisione eps
                    soluzioni.append(0.5 * (th_old + th_new))
                    d = delta                         # riparte verso destra, oltre la soluzione
                    th_old = max(th_old, th_new) + d; xr_old = xr(th_old); n += 1
                    th_new = th_old + d; xr_new = xr(th_new); n += 1
                    continue
                d = -d / 2
            th_old, xr_old = th_new, xr_new
            th_new = th_old + d; xr_new = xr(th_new); n += 1
        return soluzioni, n

    for xt in (8000.0, 15000.0):
        sol, n = angoli_bersaglio(xt)
        if sol:
            print(f"  x_t = {xt/1e3:.0f} km: angoli = {', '.join(f'{s:.2f}' for s in sol)} deg "
                  f"(verifica gittate: {', '.join(f'{xr(s)/1e3:.3f}' for s in sol)} km), {n} traiettorie")
        else:
            print(f"  x_t = {xt/1e3:.0f} km: FUORI PORTATA (nessun angolo entro 90 gradi), {n} traiettorie")
    thetas = np.arange(2, 89, 3)
    xrs = [xr(th) / 1e3 for th in thetas]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(thetas, xrs, "o-", ms=3, color=BLU)
    for xt, c in ((8.0, VERDE), (15.0, ROSSO)):
        ax.axhline(xt, color=c, ls="--", lw=0.8, label=f"$x_t={xt:.0f}$ km")
    ax.set_xlabel("$\\theta_0$ [gradi]"); ax.set_ylabel("gittata [km]"); ax.legend(fontsize=7)
    salva(fig, "es_12")


# ====================================================================== 13
def es_13():
    titolo(13, "aliante con richiamata (A variabile)")
    B, W0, h = 0.06, 1.0, 0.05

    def aliante(A_di_T):
        def f(T, s):
            X, Y, U, V = s
            A = A_di_T(T); W = np.hypot(U, V)
            return np.array([U, V, -A * W * (B * U + V), -1.0 + A * W * (U - B * V)])
        return f

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for lab, Af, c in (("$A=1{,}5$ costante", lambda T: 1.5, BLU),
                       ("richiamata: $A$ da 1,5 a 2,5 per $T\\in[2,3]$", lambda T: 1.5 + np.clip(T - 2.0, 0, 1), ARANCIO)):
        T, S = integra(aliante(Af), 0.0, [0, 0, 0.0, -W0], h, 200)
        ax.plot(S[:, 0], S[:, 1], color=c, label=lab)
        W = np.hypot(S[:, 2], S[:, 3])
        print(f"  {lab}: Y min = {S[:,1].min():.3f}, Y a T=10: {S[-1,1]:.3f}, W min = {W.min():.3f}")
    ax.set_xlabel("$X$"); ax.set_ylabel("$Y$"); ax.set_aspect("equal"); ax.legend(fontsize=7)
    salva(fig, "es_13")


# ====================================================================== 14
def es_14():
    titolo(14, "razzo: nodo allineato allo spegnimento")
    from prog1_9_razzo import f
    ev = lambda t, s: s[0]; ev.terminal, ev.direction = True, -1
    rif = solve_ivp(f, [0.5, 200], integra(f, 0.0, [0, 0], 0.1, 5, metodo="rk2")[1][-1], method="RK45",
                    events=ev, max_step=0.05, rtol=1e-9, atol=1e-11)
    t_rif, v_rif = rif.t_events[0][0], rif.y_events[0][0][1]
    print(f"  riferimento (RK45, tolleranze strette): impatto t = {t_rif:.3f} s, v = {v_rif:.3f} m/s")
    for h in (0.1, 0.09, 0.07, 0.05, 0.03):
        T, S = integra(f, 0.0, [0, 0], h, 20000, metodo="rk2", stop=lambda t, s: s[0] < 0 and t > 1)
        # interpolazione lineare dell'istante di impatto
        t_imp = T[-2] + (T[-1] - T[-2]) * S[-2, 0] / (S[-2, 0] - S[-1, 0])
        allineato = "si" if abs(round(4 / h) - 4 / h) < 1e-6 else "no"
        print(f"  h = {h:4.2f} (nodo a 4 s: {allineato:2s}): impatto t = {t_imp:.4f} s, "
              f"errore = {t_imp - t_rif:+.4f} s, z_max = {S[:,0].max():.2f} m")


# ====================================================================== 15
def es_15():
    titolo(15, "vortex blob (regolarizzazione di Krasny)")
    import prog1_7_scia_vorticosa as sv
    m, dT = sv.m, sv.dT

    def velocita_blob(delta):
        def vel(T, s):
            X, Y = s[:m], s[m:]
            DX = X[:, None] - X[None, :]; DY = Y[:, None] - Y[None, :]
            R2 = DX ** 2 + DY ** 2 + delta ** 2
            np.fill_diagonal(R2, np.inf)
            return np.concatenate([(sv.G[None, :] * DY / R2).sum(axis=1), -(sv.G[None, :] * DX / R2).sum(axis=1)])
        return vel

    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    for ax, delta, lab in zip(axs, (0.0, 0.05), ("$\\delta = 0$ (puntiformi)", "$\\delta = 0{,}05$ (blob)")):
        s = np.concatenate([sv.X0, np.zeros(m)]); T = 0.0
        while T < 0.35 - 1e-12:
            s = sv.simmetrizza(passo_rk4(velocita_blob(delta), T, s, dT)); T += dT
        ax.plot(s[m//2:m], s[m + m//2:], ".-", color=BLU, ms=4, lw=0.5)
        ax.set_title(lab, fontsize=8); ax.set_xlabel("$X$"); ax.set_aspect("equal")
        print(f"  delta = {delta}: Y centro = {s[m + m//2 - 1]:+.4f}, estremita' (X,Y) = ({s[2*m-1 - m]:.3f}, {s[2*m-1]:+.3f})")
    axs[0].set_ylabel("$Y$")
    salva(fig, "es_15")


# ====================================================================== 16
def es_16():
    titolo(16, "convergenza in m a T = 0.1")
    radice = lambda x: np.sqrt(np.clip(1 - x ** 2, 0, None))
    print(f"{'m':>4} {'dT':>8} {'Y centro':>10} {'X estr.':>9} {'Y estr.':>9} {'Y medio':>9}")
    for m in (20, 40, 80, 160):
        dX = 1.0 / m
        X0 = -1 + dX + 2 * dX * np.arange(m)
        Gm = radice(X0 + dX) - radice(X0 - dX)
        dT = 1.0 / (25 * m)

        def vel(T, s):
            X, Y = s[:m], s[m:]
            DX = X[:, None] - X[None, :]; DY = Y[:, None] - Y[None, :]
            R2 = DX ** 2 + DY ** 2; np.fill_diagonal(R2, np.inf)
            return np.concatenate([(Gm[None, :] * DY / R2).sum(axis=1), -(Gm[None, :] * DX / R2).sum(axis=1)])

        s = np.concatenate([X0, np.zeros(m)]); T = 0.0
        n = int(round(0.1 / dT))
        for _ in range(n):
            s = passo_rk4(vel, T, s, dT); T += dT
        # baricentro della meta' destra (pesato con |G|): invariante del moto
        Xr, Yr, Gr = s[m//2:m], s[m + m//2:], Gm[m//2:]
        print(f"{m:4d} {dT:8.5f} {s[m + m//2 - 1]:+10.4f} {s[m-1]:9.4f} {s[2*m-1]:+9.4f} {np.sum(Gr*Yr)/np.sum(Gr):+9.4f}")


# ====================================================================== 17
def es_17():
    titolo(17, "vortici di uguale intensita'")
    import prog1_7_scia_vorticosa as sv
    m, dT = 40, 1.0 / (25 * 40)
    # meta' sinistra: 20 vortici con G = 1/20 ciascuno, in X_k tali che Gamma(X)/Gamma_0 = 1 - (k - 1/2)/20
    k = np.arange(1, m // 2 + 1)
    Gam = 1 - (k - 0.5) / (m / 2)                    # circolazione cumulata dall'estremita'... sqrt(1-X^2)
    Xl = -np.sqrt(1 - Gam ** 2)                       # X_k < 0, ordinati dall'estremita' verso il centro
    Xl = np.sort(Xl)
    X0 = np.concatenate([Xl, -Xl[::-1]])
    Gq = np.concatenate([np.full(m // 2, 1 / (m / 2)), np.full(m // 2, -1 / (m / 2))])
    print("  ascisse iniziali (meta' sinistra):", np.round(Xl, 4))

    def vel(T, s):
        X, Y = s[:m], s[m:]
        DX = X[:, None] - X[None, :]; DY = Y[:, None] - Y[None, :]
        R2 = DX ** 2 + DY ** 2; np.fill_diagonal(R2, np.inf)
        return np.concatenate([(Gq[None, :] * DY / R2).sum(axis=1), -(Gq[None, :] * DX / R2).sum(axis=1)])

    istanti = (0.05, 0.1, 0.2)
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.6))
    s_eq = np.concatenate([X0, np.zeros(m)]); s_un = np.concatenate([sv.X0, np.zeros(m)]); T = 0.0; k = 0
    while k < len(istanti):
        s_eq = sv.simmetrizza(passo_rk4(vel, T, s_eq, dT)); s_un = sv.simmetrizza(passo_rk4(sv.velocita, T, s_un, dT)); T += dT
        if T >= istanti[k] - 1e-12:
            ax = axs[k]
            ax.plot(s_un[m//2:m], s_un[m + m//2:], ".", color=BLU, ms=3, label="equispaziati")
            ax.plot(s_eq[m//2:m], s_eq[m + m//2:], "x", color=ARANCIO, ms=3, label="uguale intensit\\`a")
            ax.set_title(f"$T={istanti[k]}$", fontsize=8); ax.set_aspect("equal"); ax.set_xlabel("$X$")
            print(f"  T = {istanti[k]}: Y centro equisp. = {s_un[m + m//2 - 1]:+.4f}, ug. int. = {s_eq[m + m//2 - 1]:+.4f}")
            k += 1
    axs[0].set_ylabel("$Y$"); axs[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    salva(fig, "es_17")


if __name__ == "__main__":
    for fn in (es_01, es_02, es_03, es_04, es_05, es_06, es_07, es_08, es_09, es_10,
               es_11, es_12, es_13, es_14, es_15, es_16, es_17):
        fn()
