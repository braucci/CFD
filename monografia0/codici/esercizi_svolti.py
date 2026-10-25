"""
esercizi_svolti.py -- codice dei venti esercizi svolti del Modulo 0.

Ogni esercizio e' la funzione es_NN(); senza argomenti il programma li esegue
tutti, con  python3 esercizi_svolti.py 3 12  esegue solo il terzo e il dodicesimo.
Le figure sono salvate in ../figure con il nome es_NN.
"""
import os
import sys
import math
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO, PALETTE

QUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(QUI, "letture"))          # i programmi delle monografie usati nelle letture guidate


# =============================================================================
def es_01():
    """Conversioni di unita' e tabella formattata: velocita' in nodi, km/h, m/s; quote in piedi e metri."""
    print("Es. 1 -- conversioni e formattazione")
    NODO, PIEDE = 1852.0 / 3600.0, 0.3048                    # 1 kt = 1852 m/h; 1 ft = 0,3048 m
    print(f"   {'kt':>6} {'km/h':>8} {'m/s':>8}      {'ft':>7} {'m':>8}")
    for kt, ft in zip((100, 250, 450, 500), (1000, 10000, 35000, 41000)):
        v = kt * NODO
        print(f"   {kt:6d} {v*3.6:8.1f} {v:8.2f}      {ft:7d} {ft*PIEDE:8.1f}")
    v = 250 * NODO
    print(f"   velocita' di {250} kt in notazione italiana: {f'{v:.2f}'.replace('.', ',')} m/s; in esponenziale {v:.3e}")


# =============================================================================
def es_02():
    """Numero di Mach e pressione dinamica in quota con la funzione isa() del Programma 0.1."""
    print("Es. 2 -- Mach e pressione dinamica in quota")
    from prog0_1_primi_passi import isa                      # riuso di una funzione di un altro file

    def mach_e_q(v, z):
        T, p, rho, a = isa(z)
        return v / a, 0.5 * rho * v**2

    print(f"   {'z [m]':>7} {'a [m/s]':>8} {'Mach':>6} {'q [Pa]':>9}   (v = 250 m/s)")
    for z in (0, 3000, 6000, 9000, 11000):
        M, q = mach_e_q(250.0, z)
        print(f"   {z:7d} {isa(z)[3]:8.2f} {M:6.3f} {q:9.0f}")
    print("   a parita' di velocita' vera, in quota il Mach cresce (a diminuisce) e la pressione dinamica diminuisce (rho diminuisce)")


# =============================================================================
def es_03():
    """Virgola mobile: somma compensata di Kahan e formula stabile per le radici di un'equazione di secondo grado."""
    print("Es. 3 -- virgola mobile")
    valori = [1e-8] * 10**6
    ingenua = 0.0
    for v in valori:
        ingenua += v
    somma, c = 0.0, 0.0                                       # Kahan: 'c' accumula l'errore commesso
    for v in valori:
        y = v - c
        t = somma + y
        c = (t - somma) - y
        somma = t
    print(f"   un milione di 1e-8: ingenua {ingenua:.17f}, Kahan {somma:.17f}, esatta 0.01 (math.fsum: {math.fsum(valori):.17f})")
    a, b, cc = 1.0, 1e8, 1.0                                  # x^2 + 1e8 x + 1 = 0: radici -1e8 e -1e-8
    disc = math.sqrt(b * b - 4 * a * cc)
    x1, x2 = (-b + disc) / (2 * a), (-b - disc) / (2 * a)    # formula classica: x1 soffre di cancellazione
    q = -0.5 * (b + math.copysign(disc, b))                   # formula stabile: x1 = q/a, x2 = c/q
    print(f"   radici classiche: {x1:.6e}, {x2:.6e};  stabili: {cc / q:.6e}, {q / a:.6e}  (esatte -1e-8, -1e8)")


# =============================================================================
def es_04():
    """Cicli con criterio di arresto: serie di Taylor di sin(x) fino a precisione prefissata."""
    print("Es. 4 -- serie di Taylor con un ciclo while")
    for x in (0.5, 2.0, 10.0):
        termine, somma, k = x, x, 0
        while abs(termine) > 1e-15 * abs(somma):
            k += 1
            termine *= -x * x / ((2 * k) * (2 * k + 1))       # termine successivo per ricorrenza: niente fattoriali enormi
            somma += termine
        print(f"   x = {x:4.1f}: {k + 1} termini, somma {somma:.15f}, math.sin {math.sin(x):.15f}, errore {abs(somma - math.sin(x)):.1e}")
    print("   per x = 10 l'errore cresce: i termini intermedi (~10^8) si cancellano fra loro (cancellazione catastrofica)")


# =============================================================================
def es_05():
    """Funzioni: schema di Horner per un polinomio e la sua derivata; Newton sul polinomio dello spessore NACA."""
    print("Es. 5 -- Horner e Newton")

    def horner(coeff, x):
        """Valore e derivata di p(x) = c0 + c1 x + ... + cn x^n con lo schema di Horner."""
        p, dp = 0.0, 0.0
        for c in reversed(coeff):
            dp = dp * x + p
            p = p * x + c
        return p, dp

    coeff = [-6.0, 11.0, -6.0, 1.0]                           # x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3), potenze crescenti
    print(f"   p(2,5) = {horner(coeff, 2.5)[0]:.4f}, p'(2,5) = {horner(coeff, 2.5)[1]:.4f} (esatti -0,375 e p'(2,5) = 3x^2 - 12x + 11 = {3*6.25 - 30 + 11})")
    # spessore NACA con bordo d'uscita aperto (-0,1015): dove si annulla y_t oltre x = 1? (Newton in sqrt(x) = s)
    a = [0.2969, -0.1260, -0.3516, 0.2843, -0.1015]
    ys = [0.0, a[0], a[1], 0.0, a[2], 0.0, a[3], 0.0, a[4]]   # polinomio in s = sqrt(x): a0 s - a1 s^2 - ...
    s = 1.0
    for k in range(20):
        p, dp = horner(ys, s)
        s_nuovo = s - p / dp
        if abs(s_nuovo - s) < 1e-14:
            break
        s = s_nuovo
    print(f"   bordo d'uscita aperto: y_t(1) = {horner(ys, 1.0)[0] * 5 * 0.12:.5f} c (NACA 0012); y_t = 0 in x = s^2 = {s**2:.6f} ({k} iterazioni di Newton)")


# =============================================================================
def es_06():
    """Liste e dizionari: elaborare una polare (efficienza massima, angolo di portanza nulla, ordinamenti)."""
    print("Es. 6 -- la polare come dizionario")
    polare = {-4: (-0.20, 0.0090), -2: (0.00, 0.0075), 0: (0.25, 0.0070), 2: (0.45, 0.0080), 4: (0.67, 0.0100),
              6: (0.88, 0.0130), 8: (1.08, 0.0170), 10: (1.25, 0.0230), 12: (1.35, 0.0320), 14: (1.30, 0.0500)}
    eff = {alfa: cl / cd for alfa, (cl, cd) in polare.items()}
    a_max = max(eff, key=eff.get)
    print(f"   efficienza massima E = {eff[a_max]:.1f} a alfa = {a_max} gradi (C_L = {polare[a_max][0]}, C_D = {polare[a_max][1]})")
    stallo = max(polare, key=lambda alfa: polare[alfa][0])
    print(f"   C_L massimo {polare[stallo][0]} a {stallo} gradi; C_L0 = {polare[0][0]}; pendenza media fra -2 e 8 gradi: "
          f"{(polare[8][0] - polare[-2][0]) / 10:.4f} per grado")
    sotto = [alfa for alfa, (cl, cd) in polare.items() if cd < 0.012]
    print(f"   angoli con C_D < 0,012: {sotto}; polare ordinata per C_D decrescente: {sorted(polare, key=lambda a: -polare[a][1])[:4]} ...")


# =============================================================================
def es_07():
    """Stringhe e file: scrivere un file di coordinate del profilo e rileggerlo analizzando le righe."""
    print("Es. 7 -- file di coordinate")
    from prog0_3_strutture_dati import profilo
    dorso, ventre = profilo("2412", n=20)
    percorso = os.path.join(QUI, "..", "figure", "es_07_naca2412.dat")
    with open(percorso, "w", encoding="utf-8") as fp:
        fp.write("# NACA 2412, coordinate x/c y/c (dorso dal bordo d'uscita, poi ventre)\n")
        for x, y in list(reversed(dorso)) + ventre[1:]:
            fp.write(f"{x:10.6f} {y:10.6f}\n")
    punti = []
    with open(percorso, encoding="utf-8") as fp:
        for riga in fp:
            riga = riga.strip()
            if not riga or riga.startswith("#"):          # salta commenti e righe vuote
                continue
            campi = riga.split()
            punti.append((float(campi[0]), float(campi[1])))
    print(f"   scritte e rilette {len(punti)} coppie; prima {punti[0]}, ultima {punti[-1]}")
    print(f"   lo stesso file letto con NumPy: forma {np.loadtxt(percorso).shape}; ordinata massima {max(y for x, y in punti):.4f} c")


# =============================================================================
def es_08():
    """Vettorializzare: media mobile e derivata di un segnale con cicli e con lo slicing; tempi e uguaglianza."""
    print("Es. 8 -- vettorializzazione")
    n = 200000
    x = np.linspace(0, 10, n)
    f = np.sin(x) + 0.05 * np.random.default_rng(1).standard_normal(n)
    t0 = time.perf_counter()
    media1 = np.empty(n - 2)
    for i in range(1, n - 1):
        media1[i - 1] = (f[i - 1] + f[i] + f[i + 1]) / 3
    t1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    media2 = (f[:-2] + f[1:-1] + f[2:]) / 3
    t2 = time.perf_counter() - t0
    print(f"   media mobile a tre punti su {n} valori: ciclo {t1*1e3:.0f} ms, slicing {t2*1e3:.2f} ms, uguali? {np.allclose(media1, media2)}")
    t0 = time.perf_counter(); d = np.gradient(f, x); t3 = time.perf_counter() - t0
    print(f"   np.gradient (differenze centrate, unilatere agli estremi): {t3*1e3:.2f} ms; np.convolve per la media: "
          f"{np.allclose(np.convolve(f, np.ones(3) / 3, mode='valid'), media2)}")


# =============================================================================
def es_09():
    """Broadcasting: tabella bidimensionale della velocita' limite per diametri e densita' diversi."""
    print("Es. 9 -- broadcasting")
    d = np.array([0.001, 0.005, 0.01, 0.05])[:, None]           # (4, 1)
    rho = np.array([1000.0, 2700.0, 7800.0])[None, :]            # (1, 3)
    m = rho * np.pi * d**3 / 6                                   # (4, 3)
    vt = np.sqrt(2 * m * 9.80665 / (1.225 * np.pi * d**2 / 4 * 0.5))  # c_d = 0,5 costante
    print("   v_t [m/s] (righe: d = 1, 5, 10, 50 mm; colonne: legno/acqua 1000, alluminio 2700, acciaio 7800 kg/m^3)")
    for riga in vt:
        print("   " + " ".join(f"{v:8.2f}" for v in riga))
    print(f"   forma di vt: {vt.shape}; v_t cresce come sqrt(rho d): rapporto fra d = 50 e 5 mm = {vt[3, 0] / vt[1, 0]:.3f} (sqrt(10) = {np.sqrt(10):.3f})")


# =============================================================================
def es_10():
    """Slicing 2D: laplaciano con condizioni di Neumann imposte con i nodi fittizi (array esteso)."""
    print("Es. 10 -- nodi fittizi")
    N = 40; h = 1.0 / N
    x = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = np.cos(np.pi * X) * np.cos(np.pi * Y)                   # du/dn = 0 su tutto il contorno
    ue = np.zeros((N + 3, N + 3))
    ue[1:-1, 1:-1] = u
    ue[0, :] = ue[2, :]; ue[-1, :] = ue[-3, :]; ue[:, 0] = ue[:, 2]; ue[:, -1] = ue[:, -3]   # riflessione: derivata normale nulla
    lap = (ue[2:, 1:-1] + ue[:-2, 1:-1] + ue[1:-1, 2:] + ue[1:-1, :-2] - 4 * ue[1:-1, 1:-1]) / h**2
    esatto = -2 * np.pi**2 * u
    err = np.max(np.abs(lap - esatto))
    print(f"   laplaciano con nodi fittizi su tutta la griglia (bordi compresi), N = {N}: errore massimo {err:.2e},"
          f" sul bordo {np.max(np.abs((lap - esatto)[0, :])):.2e}; stima teorica h^2 pi^4/6 = {h**2 * np.pi**4 / 6:.2e}")
    N2 = 2 * N; h2 = 1.0 / N2; x2 = np.linspace(0, 1, N2 + 1); X2, Y2 = np.meshgrid(x2, x2, indexing="ij")
    u2 = np.cos(np.pi * X2) * np.cos(np.pi * Y2); ue2 = np.zeros((N2 + 3, N2 + 3)); ue2[1:-1, 1:-1] = u2
    ue2[0, :] = ue2[2, :]; ue2[-1, :] = ue2[-3, :]; ue2[:, 0] = ue2[:, 2]; ue2[:, -1] = ue2[:, -3]
    lap2 = (ue2[2:, 1:-1] + ue2[:-2, 1:-1] + ue2[1:-1, 2:] + ue2[1:-1, :-2] - 4 * ue2[1:-1, 1:-1]) / h2**2
    print(f"   con N = {N2}: errore {np.max(np.abs(lap2 + 2 * np.pi**2 * u2)):.2e}, rapporto {err / np.max(np.abs(lap2 + 2 * np.pi**2 * u2)):.2f} (atteso 4)")


# =============================================================================
def es_11():
    """Matplotlib: polare e curva C_L-alfa in due pannelli con annotazioni, nello stile della casa."""
    print("Es. 11 -- figura a due pannelli")
    alfa = np.array([-4, -2, 0, 2, 4, 6, 8, 10, 12, 14])
    cl = np.array([-0.20, 0.00, 0.25, 0.45, 0.67, 0.88, 1.08, 1.25, 1.35, 1.30])
    cd = np.array([0.0090, 0.0075, 0.0070, 0.0080, 0.0100, 0.0130, 0.0170, 0.0230, 0.0320, 0.0500])
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax[0].plot(alfa, cl, "o-", color=BLU, ms=3)
    ax[0].set_xlabel("$\\alpha$ [gradi]"); ax[0].set_ylabel("$C_L$"); ax[0].axhline(0, color=GRIGIO, lw=0.5); ax[0].axvline(0, color=GRIGIO, lw=0.5)
    k = np.argmax(cl)
    ax[0].annotate("stallo", xy=(alfa[k], cl[k]), xytext=(alfa[k] - 6, cl[k] + 0.05), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax[1].plot(cd, cl, "o-", color=ARANCIO, ms=3)
    e = cl / cd; j = np.argmax(e)
    ax[1].plot([0, cd[j]], [0, cl[j]], ":", color=GRIGIO)
    ax[1].annotate(f"$E_{{\\max}}={e[j]:.0f}$", xy=(cd[j], cl[j]), xytext=(cd[j] + 0.008, cl[j] - 0.3), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax[1].set_xlabel("$C_D$"); ax[1].set_ylabel("$C_L$"); ax[1].set_title("polare", fontsize=9)
    fig.tight_layout()
    salva(fig, "es_11")


# =============================================================================
def es_12():
    """SciPy fsolve: il sistema ridotto di Galerkin per la sfera (Modulo 4) a Re = 100; confronto con la tabella."""
    print("Es. 12 -- fsolve sul sistema di Galerkin della sfera")
    from scipy.optimize import fsolve
    Re = 100.0
    g = lambda A1, B1: 9.931035 / Re * A1 + 0.011713 * B1 - 0.002546 * A1 * B1 + 44.689656 / Re
    h = lambda A1, B1: 0.050148 * A1 + 2.201058 / Re * B1 + 0.002421 * A1**2 - 0.000376 * B1**2 + 0.379018
    sol, info, ier, msg = fsolve(lambda p: [g(*p), h(*p)], [-4.0, -1.0], full_output=True)
    print(f"   A1 = {sol[0]:.4f}, B1 = {sol[1]:.3f} (tabella: -2,6083, -10,236); residui {g(*sol):.1e}, {h(*sol):.1e}; {info['nfev']} valutazioni")
    A2 = -15 / 29 * (8 + 5 * sol[0]); A3 = 9 / 29 * (17 + 7 * sol[0]); A4 = -(95 + 34 * sol[0]) / 58
    cd = 32 / (3 * Re) * (8 * A2 + 25 * A3 + 54 * A4) + 64 / (3 * Re) * (2 * A2 + 5 * A3 + 9 * A4)
    print(f"   C_D = {cd:.4f} (tabella: 1,0436)")


# =============================================================================
def es_13():
    """solve_ivp: periodo del pendolo non lineare in funzione dell'ampiezza, confrontato con l'integrale ellittico."""
    print("Es. 13 -- pendolo con solve_ivp ed eventi")
    from scipy.integrate import solve_ivp
    from scipy.special import ellipk
    print(f"   {'theta0 [gradi]':>14} {'T/T0 numerico':>14} {'T/T0 esatto':>12}")
    for th0 in (10, 45, 90, 135, 170):
        th0r = np.radians(th0)
        passaggio = lambda t, y: y[0]; passaggio.direction = -1; passaggio.terminal = True
        s = solve_ivp(lambda t, y: [y[1], -np.sin(y[0])], (0, 50), [th0r, 0.0], events=passaggio, rtol=1e-11, atol=1e-12)
        T_num = 4 * s.t_events[0][0]                               # un quarto di periodo fino al primo passaggio per zero
        T_es = 4 * ellipk(np.sin(th0r / 2)**2)                     # T = 4 K(k), k = sin(theta0/2), in unita' di sqrt(l/g)
        print(f"   {th0:14d} {T_num / (2 * np.pi):14.6f} {T_es / (2 * np.pi):12.6f}")


# =============================================================================
def es_14():
    """Schema implicito per la diffusione con l'algoritmo di Thomas: nessun limite su R, confronto con FTCS."""
    print("Es. 14 -- BTCS con Thomas")
    from metodi_numerici import thomas
    from prog0_8_anatomia import esatta, integra
    n = 40; x = np.linspace(0, 1, n + 1); h = x[1] - x[0]
    for R in (0.4, 2.0, 10.0):
        dt = R * h**2; n_passi = int(round(0.1 / dt))
        u = np.sin(np.pi * x)
        a = np.full(n - 1, -R); b = np.full(n - 1, 1 + 2 * R); c = np.full(n - 1, -R)
        for _ in range(n_passi):
            u[1:-1] = thomas(a, b, c, u[1:-1])                     # u = 0 agli estremi: nessun termine noto aggiuntivo
        err = np.max(np.abs(u - esatta(x, n_passi * dt)))
        print(f"   R = {R:5.1f}: {n_passi:4d} passi, errore {err:.2e}" + ("" if R > 0.5 else f" (FTCS: {np.max(np.abs(integra(n, R)[1] - esatta(x, 0.1))):.2e})"))
    print("   l'implicito e' stabile per ogni R ma l'errore cresce con dt (primo ordine nel tempo)")


# =============================================================================
def es_15():
    """SymPy: formula centrata a cinque punti per la derivata prima con il metodo dei coefficienti indeterminati."""
    print("Es. 15 -- formula a cinque punti con SymPy")
    import sympy as sp
    h = sp.symbols("h", positive=True)
    f = sp.symbols("f0:7")                                         # f0 = f(x), f1 = f'(x), ...
    taylor = lambda k: sum((k * h)**j / sp.factorial(j) * f[j] for j in range(7))
    c = sp.symbols("c_m2 c_m1 c_0 c_1 c_2")
    approx = sp.expand(sum(ck * taylor(k) for ck, k in zip(c, (-2, -1, 0, 1, 2))) / h)
    eqs = [approx.coeff(f[0]), approx.coeff(f[1]) - 1] + [approx.coeff(f[j]) for j in (2, 3, 4)]
    sol = sp.solve(eqs, c)
    print(f"   coefficienti: {[sol[ck] for ck in c]}  -> f' = (f_{{-2}} - 8 f_{{-1}} + 8 f_1 - f_2)/(12 h)")
    errore = sp.expand(approx.subs(sol) - f[1])
    print(f"   errore di troncamento: {errore.coeff(f[5]) * f[5]}  (quarto ordine)")


# =============================================================================
def es_16():
    """Lettura guidata 1: la caduta della sfera (Programma 1.1) riscritta con l'integratore generico ode_solvers.integra."""
    print("Es. 16 -- Programma 1.1 con ode_solvers")
    from ode_solvers import integra, cd_sfera
    from prog1_1_caduta_sfera import caduta, costanti, NU
    d = 0.01
    A, B, C = costanti(d)
    F = lambda t, y: np.array([y[1], (B - C * y[1] * abs(y[1]) * cd_sfera(y[1] * d / NU)) / A])
    T, Y = integra(F, 0.0, [0.0, 0.0], 0.1, 100, metodo="rk4")
    T1, Z1, V1, _ = caduta(d, h=0.1, tmax=10.0)
    print(f"   a t = 10 s: z = {Y[-1, 0]:.4f} m, v = {Y[-1, 1]:.4f} m/s (Programma 1.1: {Z1[-1]:.4f}, {V1[-1]:.4f});"
          f" differenza {abs(Y[-1, 1] - V1[-1]):.1e}")
    print("   stesso risultato: il Programma 1.1 scrive RK4 'a mano' per due equazioni, ode_solvers lo fa per un vettore di stato qualsiasi")


# =============================================================================
def es_17():
    """Lettura guidata 2: il problema di Rayleigh (Programma 3.4) con lo schema implicito al posto dell'esplicito."""
    print("Es. 17 -- Programma 3.4 in forma implicita")
    from metodi_numerici import thomas
    from scipy.special import erfc
    M, D, NU, U0 = 21, 1.0, 1.0e-6, 1.0
    H = D / (M - 1)
    y = np.linspace(0, D, M)

    def esatta(y, t, nmax=200):
        u = y / D
        for n in range(1, nmax + 1):
            u = u + 2 * (-1)**n / (n * np.pi) * np.sin(n * np.pi * y / D) * np.exp(-(n * np.pi)**2 * NU * t / D**2)
        return U0 * u

    for R in (0.25, 2.0, 8.0):
        tau = R * H**2 / NU
        u = np.zeros(M); u[-1] = U0
        a = np.full(M - 2, -R); b = np.full(M - 2, 1 + 2 * R); c = np.full(M - 2, -R)
        t, n = 0.0, 0
        while n * tau < 2.5e5 - 1e-9:
            rhs = u[1:-1].copy(); rhs[-1] += R * U0                 # il nodo superiore (u = U0) entra nel termine noto
            u[1:-1] = thomas(a, b, c, rhs); n += 1
        t = n * tau
        print(f"   R = {R:4.2f}: tau = {tau:7.0f} s, {n:4d} passi fino a t = {t:.3e} s, errore massimo {np.max(np.abs(u - esatta(y, t))):.2e} m/s")
    print("   il Programma 3.4 (esplicito, R = 1/4) usa 400 passi; l'implicito con R = 8 ne usa 13, con un errore 20 volte maggiore")


# =============================================================================
def es_18():
    """Lettura guidata 3: usare il Programma 4.5 (cavita' MAC) come libreria: energia cinetica e vorticita' di parete."""
    print("Es. 18 -- Programma 4.5 come libreria")
    from prog4_5_cavita_mac import cavita
    r = cavita(100, 32, 20.0, stampa=False)
    u, v, h, N = r["u"], r["v"], r["h"], r["N"]
    uc = 0.5 * (u[1:, 1:-1] + u[:-1, 1:-1]); vc = 0.5 * (v[1:-1, 1:] + v[1:-1, :-1])   # velocita' nei centri delle celle
    ek = 0.5 * np.sum(uc**2 + vc**2) * h * h
    om_cop = -(2.0 - 2 * u[:, N]) / h                                                  # vorticita' sul coperchio: -du/dy con il nodo fittizio
    print(f"   Re = 100, griglia {N+1}x{N+1}: psi_min = {r['psi'].min():.4f}, energia cinetica {ek:.5f}, vorticita' media sul coperchio {om_cop[1:-1].mean():.2f}")
    print(f"   divergenza massima {np.max(np.abs((u[1:, 1:-1] - u[:-1, 1:-1]) / h + (v[1:-1, 1:] - v[1:-1, :-1]) / h)):.1e}: la funzione restituisce un dizionario con tutti i campi")


# =============================================================================
def es_19():
    """Programma completo da zero: convezione lineare con upwind e Lax-Wendroff, ordine di convergenza, figura."""
    print("Es. 19 -- programma completo: convezione lineare")

    def integra(schema, nx, C, t_fin, L=1.0):
        x = np.linspace(0, L, nx, endpoint=False); dx = x[1] - x[0]
        u = np.exp(-200 * (x - 0.3)**2)
        n = int(round(t_fin / (C * dx)))
        for _ in range(n):
            um, up = np.roll(u, 1), np.roll(u, -1)
            if schema == "upwind":
                u = u - C * (u - um)
            else:
                u = u - 0.5 * C * (up - um) + 0.5 * C**2 * (up - 2 * u + um)
        return x, u, np.exp(-200 * ((x - 0.3 - n * C * dx + L / 2) % L - L / 2)**2)

    print(f"   {'nx':>5} {'upwind':>10} {'Lax-W':>10}")
    errs = {"upwind": [], "laxw": []}
    for nx in (100, 200, 400, 800):
        riga = f"   {nx:5d}"
        for schema in ("upwind", "laxw"):
            x, u, ue = integra(schema, nx, 0.5, 0.5)
            errs[schema].append(np.max(np.abs(u - ue))); riga += f" {errs[schema][-1]:10.2e}"
        print(riga)
    for schema in errs:
        print(f"   ordine {schema}: {-np.polyfit(np.log([100, 200, 400, 800]), np.log(errs[schema]), 1)[0]:.2f}")
    print("   l'upwind tende all'ordine 1 solo su griglie fini (la gaussiana stretta e' mal risolta a 100 nodi); Lax-Wendroff e' gia' al secondo")
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    x, u1, ue = integra("upwind", 200, 0.5, 0.5); _, u2, _ = integra("laxw", 200, 0.5, 0.5)
    ax.plot(x, ue, color=GRIGIO, label="esatta"); ax.plot(x, u1, color=BLU, label="upwind"); ax.plot(x, u2, color=ARANCIO, label="Lax--Wendroff")
    ax.set_xlabel("$x$"); ax.set_ylabel("$u$"); ax.legend(fontsize=7); fig.tight_layout()
    salva(fig, "es_19")


# =============================================================================
def es_20():
    """Caccia agli errori: una funzione con cinque errori (tre silenziosi) e la sua versione corretta."""
    print("Es. 20 -- caccia agli errori")

    def media_profilo_sbagliata(u, h):
        # calcola la media di u sul dominio [0, 1] con la regola dei trapezi: contiene errori
        n = len(u)
        somma = 0
        for i in range(1, n):                     # (1) salta il nodo 0 e pesa male gli estremi
            somma += u[i]
        h = 1 / n                                  # (2) h = 1/(n-1), non 1/n
        return somma * h / 2                       # (3) il fattore 1/2 dei trapezi va solo agli estremi

    def media_profilo(u, h):
        return (0.5 * u[0] + np.sum(u[1:-1]) + 0.5 * u[-1]) * h

    x = np.linspace(0, 1, 11); u = x**2
    print(f"   integrale di x^2 su [0, 1]: sbagliato {media_profilo_sbagliata(u, 0.1):.4f}, corretto {media_profilo(u, 0.1):.4f}, esatto 0,3333"
          f" (trapezi: {np.trapezoid(u, x):.4f})")
    # errori che il calcolatore segnala, da leggere nel traceback
    for nome, azione in (("(4) indice", lambda: u[11]), ("(5) forme", lambda: u + np.zeros(10))):
        try:
            azione()
        except (IndexError, ValueError) as e:
            print(f"   {nome}: {type(e).__name__} -- {e}")
    print("   gli errori 1-3 non producono messaggi: solo il confronto con un valore noto li rivela")


# =============================================================================
if __name__ == "__main__":
    numeri = [int(a) for a in sys.argv[1:]] or range(1, 21)
    for k in numeri:
        globals()[f"es_{k:02d}"]()
        print()
