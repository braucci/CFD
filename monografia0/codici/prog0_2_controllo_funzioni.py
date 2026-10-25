"""
prog0_2_controllo_funzioni.py -- Programma 0.2: controllo del flusso e funzioni.

Esempio conduttore: la velocita' limite di una sfera in caduta in aria,
radice dell'equazione  f(v) = m g - (1/2) rho_a v^2 (pi d^2/4) c_d(Re) = 0,
con c_d(Re) a tratti (Re = v d/nu); e il moto della stessa sfera integrato
con il metodo di Eulero e con Runge-Kutta 4 fino alla velocita' limite.
Il programma mostra: if/elif/else, for e while, break/continue, range,
enumerate e zip, definizione di funzioni con argomenti predefiniti e
docstring, funzioni come argomenti (bisezione e Newton ricevono f), lambda,
ricorsione (fattoriale e Fibonacci), variabili locali e globali, moduli e
il blocco  if __name__ == "__main__".
Verifica: Newton e bisezione convergono alla stessa radice; il moto tende
a v_t; RK4 e' del quarto ordine (l'errore si divide per 16 dimezzando h).
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

RHO_A, NU, G = 1.225, 1.46e-5, 9.80665          # aria al livello del mare


# ------------------------------------------------------------------ funzioni con if/elif/else
def cd_sfera(Re):
    """Coefficiente di resistenza di una sfera liscia: spezzata a tratti in Re."""
    if Re < 1e-9:
        return 1e9                       # evita la divisione per zero: resistenza di Stokes enorme
    elif Re < 1.0:
        return 24.0 / Re
    elif Re < 400.0:
        return 24.0 / Re**0.646
    elif Re < 3e5:
        return 0.5
    elif Re < 2e6:
        return 3.66e-4 * Re**0.4275
    else:
        return 0.18


def cd_liscio(Re):
    """Correlazione di Clift-Gauvin, regolare per ogni Re > 0 (valida fino a Re ~ 3e5)."""
    return 24.0 / Re * (1 + 0.15 * Re**0.687) + 0.42 / (1 + 42500.0 * Re**-1.16)


def forza_netta(v, d, m, cd=cd_sfera):
    """f(v) = peso - resistenza, per unita' di massa; cd e' la legge di resistenza (funzione di Re)."""
    Re = max(v * d / NU, 1e-9)
    return G - 0.5 * RHO_A * v * abs(v) * math.pi * d**2 / 4 * cd(Re) / m


# ------------------------------------------------------------------ funzioni che ricevono funzioni
def bisezione(f, a, b, tol=1e-10, it_max=200):
    """Radice di f in [a, b] (f(a) f(b) < 0) con il metodo di bisezione. Restituisce (x, iterazioni)."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) e f(b) hanno lo stesso segno")
    for k in range(1, it_max + 1):
        c = 0.5 * (a + b)
        fc = f(c)
        if fc == 0.0 or 0.5 * (b - a) < tol:
            return c, k
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5 * (a + b), it_max


def newton(f, x0, tol=1e-12, it_max=50, h=1e-6):
    """Radice di f con il metodo di Newton e derivata numerica centrata. Restituisce (x, iterazioni, storia)."""
    x = x0
    storia = [x]
    for k in range(1, it_max + 1):
        fp = (f(x + h) - f(x - h)) / (2 * h)
        x_nuovo = x - f(x) / fp
        storia.append(x_nuovo)
        if abs(x_nuovo - x) < tol:
            return x_nuovo, k, storia
        x = x_nuovo
    return x, it_max, storia


# ------------------------------------------------------------------ integrazione nel tempo
def integra(F, v0, h, t_fin, metodo="rk4"):
    """Integra dv/dt = F(v) da 0 a t_fin; restituisce le liste t e v.
    Il numero di passi e' fissato in anticipo (n = t_fin/h) e il tempo e' calcolato come k h:
    accumulare t += h in un ciclo while produce errori di arrotondamento che possono
    aggiungere o togliere un passo (vedi Programma 0.9)."""
    n = round(t_fin / h)
    v = v0
    T, V = [0.0], [v]
    for k in range(1, n + 1):
        if metodo == "eulero":
            v = v + h * F(v)
        elif metodo == "rk4":
            k1 = h * F(v)
            k2 = h * F(v + k1 / 2)
            k3 = h * F(v + k2 / 2)
            k4 = h * F(v + k3)
            v = v + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        else:
            raise ValueError(f"metodo sconosciuto: {metodo}")
        T.append(k * h); V.append(v)
    return T, V


# ------------------------------------------------------------------ ricorsione
def fattoriale(n):
    """n! per ricorsione: n! = n (n-1)!, 0! = 1."""
    return 1 if n == 0 else n * fattoriale(n - 1)


def fibonacci(n, memo={}):
    """Numero di Fibonacci con memorizzazione (senza, il costo cresce come 1,6^n)."""
    if n < 2:
        return n
    if n not in memo:
        memo[n] = fibonacci(n - 1) + fibonacci(n - 2)
    return memo[n]


contatore = 0                                    # variabile globale


def incrementa():
    global contatore                             # senza 'global' l'assegnazione creerebbe una variabile locale
    contatore += 1


if __name__ == "__main__":
    d, rho_s = 0.01, 7800.0                      # sfera d'acciaio da 1 cm
    m = rho_s * math.pi * d**3 / 6
    f = lambda v: forza_netta(v, d, m)           # funzione anonima di una sola variabile
    print(f"Sfera d'acciaio, d = {d} m, m = {m*1000:.2f} g")

    # --- cicli for con range, enumerate, zip, break e continue
    print("\n1) Ricerca di un intervallo con cambio di segno (for con break):")
    velocita = [10, 20, 30, 40, 50, 60]
    segni = [f(v) for v in velocita]
    for i, (v, s) in enumerate(zip(velocita, segni)):
        print(f"   v = {v:3d} m/s: f = {s:+.3f}", end="")
        if i > 0 and segni[i - 1] * s < 0:
            print("   <- cambio di segno")
            a, b = velocita[i - 1], v
            break
        print()
    print("\n2) Bisezione e Newton:")
    vb, kb = bisezione(f, a, b)
    vn, kn, storia = newton(f, 10.0)
    print(f"   bisezione: v_t = {vb:.8f} m/s in {kb} iterazioni (ogni iterazione guadagna un bit)")
    print(f"   Newton:    v_t = {vn:.8f} m/s in {kn} iterazioni; errori successivi: "
          + ", ".join(f"{abs(x - vn):.1e}" for x in storia[:-1]))
    Re_t = vn * d / NU
    print(f"   Re = {Re_t:.0f}, c_d = {cd_sfera(Re_t)}; verifica dell'equilibrio: f(v_t) = {f(vn):.1e}")

    # --- while e integrazione nel tempo; ordine di convergenza
    print("\n3) Moto della sfera: v(t) con Eulero e RK4, confronto con v_t")
    t_fin = 8.0
    for metodo in ("eulero", "rk4"):
        T, V = integra(f, 0.0, 0.5, t_fin, metodo)
        print(f"   {metodo:6s} h = 0.5: " + "  ".join(f"v({t:.0f})={v:6.3f}" for t, v in zip(T[::4], V[::4])))
    T, V = integra(f, 0.0, 0.01, 30.0, "rk4")
    k = 0
    while V[k] < 0.99 * vn:                      # ciclo while: avanza finche' la condizione e' vera
        k += 1
    print(f"   il 99% della velocita' limite e' raggiunto a t = {T[k]:.2f} s (ciclo while sulla soluzione RK4)")
    print("   ordine di convergenza (errore a t = 8 rispetto a h = 1e-4), con la legge c_d a tratti e con quella regolare:")
    for nome, cd in (("c_d a tratti", cd_sfera), ("c_d regolare", cd_liscio)):
        fl = lambda v: forza_netta(v, d, m, cd)
        _, Vrif = integra(fl, 0.0, 1e-4, t_fin, "rk4")
        for metodo in ("eulero", "rk4"):
            errori = []
            for h in (0.4, 0.2, 0.1):
                _, V = integra(fl, 0.0, h, t_fin, metodo)
                errori.append(abs(V[-1] - Vrif[-1]))
            rapporti = [errori[i] / errori[i + 1] for i in range(len(errori) - 1)]
            print(f"   {nome}, {metodo:6s}: errori {', '.join(f'{e:.2e}' for e in errori)}; rapporti "
                  f"{', '.join(f'{r:.1f}' for r in rapporti)} (attesi {2 if metodo == 'eulero' else 16})")
    print("   con la legge a tratti (derivata discontinua in Re = 400) RK4 perde l'ordine: il quarto ordine")
    print("   richiede una funzione F regolare; con la legge regolare il rapporto 16 e' rispettato.")

    # --- ricorsione e variabili globali
    print("\n4) Ricorsione e ambito delle variabili")
    print(f"   10! = {fattoriale(10)}, F(50) = {fibonacci(50)}, F(90) = {fibonacci(90)}")
    for _ in range(3):
        incrementa()
    print(f"   contatore globale dopo tre chiamate: {contatore}")
    print(f"   il modulo si chiama {__name__!r}: questo blocco gira solo se il file e' eseguito direttamente")

    # --- figura
    T, V = integra(f, 0.0, 0.05, t_fin, "rk4")
    Te, Ve = integra(f, 0.0, 0.5, t_fin, "eulero")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
    ax[0].plot(T, V, color=BLU, label="RK4, $h=0{,}05$")
    ax[0].plot(Te, Ve, "o--", color=ARANCIO, ms=3, label="Eulero, $h=0{,}5$")
    ax[0].axhline(vn, color=GRIGIO, ls=":", label="$v_t$")
    ax[0].set_xlabel("$t$ [s]"); ax[0].set_ylabel("$v$ [m/s]"); ax[0].legend(fontsize=7)
    err = [abs(x - vn) for x in storia[:-1]]
    ax[1].semilogy(range(len(err)), err, "o-", color=BLU, label="Newton")
    ax[1].semilogy(range(kb), [abs(b - a) / 2**(k + 1) for k in range(kb)], "-", color=ARANCIO, label="bisezione (ampiezza)")
    ax[1].set_xlabel("iterazione"); ax[1].set_ylabel("errore [m/s]"); ax[1].legend(fontsize=7)
    fig.tight_layout()
    salva(fig, "prog0_2_funzioni")
