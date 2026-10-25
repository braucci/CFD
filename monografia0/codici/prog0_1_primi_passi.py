"""
prog0_1_primi_passi.py -- Programma 0.1: variabili, numeri, stringhe e stampa.

Esempio conduttore: l'atmosfera standard internazionale (ISA) fino a 20 km,
    troposfera (z <= 11 km):  T = T0 - L z,   p = p0 (T/T0)^(g/(R L)),
    stratosfera (11-20 km):   T = T11 = 216,65 K,  p = p11 exp[-g (z - z11)/(R T11)],
    rho = p/(R T),  a = sqrt(gamma R T).
Il programma mostra i tipi numerici (int, float, complex, bool), gli operatori,
la precisione finita dei numeri in virgola mobile (epsilon di macchina,
cancellazione, non associativita' della somma), le stringhe formattate (f-string)
e la stampa di una tabella; infine traccia T, p, rho in funzione della quota.
Verifica: a 11 000 m la tabella ISA da' p = 22 632 Pa e rho = 0,3639 kg/m^3.
"""
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

# ------------------------------------------------------------------ costanti e funzione dell'atmosfera standard
G0 = 9.80665                       # accelerazione di gravita' standard [m/s^2]
T0, P0, L, R, GAMMA = 288.15, 101325.0, 0.0065, 287.05287, 1.4
Z11, T11 = 11000.0, T0 - 0.0065 * 11000.0
P11 = P0 * (T11 / T0) ** (G0 / (R * L))


def isa(z):
    """Temperatura [K], pressione [Pa], densita' [kg/m^3] e velocita' del suono [m/s] alla quota z [m]."""
    if z <= Z11:
        T = T0 - L * z
        p = P0 * (T / T0) ** (G0 / (R * L))
    else:
        T = T11
        p = P11 * math.exp(-G0 * (z - Z11) / (R * T11))
    rho = p / (R * T)
    return T, p, rho, math.sqrt(GAMMA * R * T)


if __name__ == "__main__":
    # ------------------------------------------------------------------ 1) numeri e tipi
    print("1) Tipi numerici")
    n_passi = 400                      # int: intero di lunghezza arbitraria
    g = 9.80665                        # float: virgola mobile a 64 bit (IEEE 754 "double")
    z = 3 + 4j                         # complex: parte reale e immaginaria (j = unita' immaginaria)
    aperto = True                      # bool: True/False (sono anche gli interi 1 e 0)
    for v in (n_passi, g, z, aperto):
        print(f"   {v!r:>12}  e' di tipo {type(v).__name__}")
    print(f"   7 / 2 = {7 / 2}   (divisione vera)      7 // 2 = {7 // 2}   (quoziente intero)"
          f"      7 % 2 = {7 % 2}   (resto)      2 ** 10 = {2 ** 10}   (potenza)")
    print(f"   2 ** 100 = {2 ** 100}   (gli interi non vanno in overflow)")
    print(f"   |3+4j| = {abs(z)},  parte reale {z.real},  coniugato {z.conjugate()}")
    print(f"   math.sqrt(2) = {math.sqrt(2)},  math.pi = {math.pi},  math.e = {math.e}")

    # ------------------------------------------------------------------ 2) virgola mobile
    print("\n2) Virgola mobile: precisione finita")
    eps = sys.float_info.epsilon
    print(f"   epsilon di macchina: {eps} = 2^-52  (1 + eps e' il primo numero > 1 rappresentabile)")
    print(f"   1 + eps/2 == 1 ? {1 + eps / 2 == 1}      0.1 + 0.2 == 0.3 ? {0.1 + 0.2 == 0.3}   (0.1 + 0.2 = {0.1 + 0.2:.17f})")
    print(f"   confronto corretto: math.isclose(0.1 + 0.2, 0.3) = {math.isclose(0.1 + 0.2, 0.3)}")
    print(f"   massimo float: {sys.float_info.max:.3e}; 1e308 * 10 = {1e308 * 10};  1e-320 = {1e-320} (denormalizzato)")
    # cancellazione: (1 - cos x)/x^2 -> 1/2 per x -> 0
    for x in (1e-2, 1e-4, 1e-6, 1e-8):
        ingenuo = (1 - math.cos(x)) / x**2
        stabile = 2 * math.sin(x / 2)**2 / x**2      # identita' 1 - cos x = 2 sin^2(x/2): nessuna sottrazione
        print(f"   x = {x:.0e}: (1 - cos x)/x^2 = {ingenuo:.10f} (ingenuo)   {stabile:.10f} (stabile)   esatto -> 0,5")
    # non associativita': somma di 1/k^2 in avanti e all'indietro
    N = 10**6
    avanti = 0.0
    for k in range(1, N + 1):
        avanti += 1.0 / k**2
    indietro = 0.0
    for k in range(N, 0, -1):
        indietro += 1.0 / k**2
    esatto = math.pi**2 / 6 - 1.0 / (N + 0.5)          # coda stimata con l'integrale
    print(f"   somma di 1/k^2, k = 1..{N}: in avanti {avanti:.15f}, all'indietro {indietro:.15f}")
    print(f"   differenza {abs(avanti - indietro):.1e}; la somma all'indietro (dai termini piccoli) e' la piu' accurata"
          f" (riferimento {esatto:.15f})")

    # ------------------------------------------------------------------ 3) stringhe
    print("\n3) Stringhe e formattazione")
    nome = "Costruzioni Aeronautiche"
    print(f"   lunghezza {len(nome)}, maiuscolo {nome.upper()!r}, prime 12 lettere {nome[:12]!r}, ultime 12 {nome[-12:]!r}")
    print(f"   {'a capo e tabulazioni si scrivono con \\n e \\t'}; concatenazione: " + "ITIS " + "Majorana")
    v, Mach = 251.3, 0.7382
    print(f"   f-string: v = {v:.1f} m/s, Mach = {Mach:.3f}, in notazione esponenziale {v:.3e}, in colonna [{v:10.2f}] [{v:<10.2f}]")
    print(f"   percentuale {0.0523:.1%}, separatore delle migliaia {123456789:,}, intero in binario {37:b} e in esadecimale {255:x}")
    print(f"   virgola decimale italiana: {v:.1f}".replace(".", ","))

    # ------------------------------------------------------------------ 4) atmosfera standard
    print("\n4) Atmosfera standard (ISA)")
    print(f"   {'z [m]':>8} {'T [K]':>8} {'p [Pa]':>10} {'rho [kg/m3]':>12} {'a [m/s]':>8} {'p/p0':>7}")
    for z in range(0, 20001, 2000):
        T, p, rho, a = isa(z)
        print(f"   {z:8d} {T:8.2f} {p:10.1f} {rho:12.4f} {a:8.2f} {p / P0:7.4f}")
    T, p, rho, a = isa(11000.0)
    print(f"   verifica alla tropopausa: p = {p:.0f} Pa (tabella ISA: 22632), rho = {rho:.4f} (tabella: 0,3639), T = {T:.2f} K")
    assert abs(p - 22632) < 1 and abs(rho - 0.3639) < 5e-4, "la tabella ISA non torna"

    # ------------------------------------------------------------------ 5) figura
    zz = np.linspace(0, 20000, 401)
    dati = np.array([isa(z) for z in zz])           # colonne: T, p, rho, a
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.8))
    ax[0].plot(dati[:, 0], zz / 1000, color=BLU); ax[0].set_xlabel("$T$ [K]"); ax[0].set_ylabel("$z$ [km]")
    ax[1].plot(dati[:, 1] / 1000, zz / 1000, color=ARANCIO); ax[1].set_xlabel("$p$ [kPa]")
    ax[2].semilogx(dati[:, 2], zz / 1000, color=VERDE); ax[2].set_xlabel("$\\rho$ [kg/m$^3$]")
    for a_ in ax:
        a_.axhline(11, color=GRIGIO, lw=0.6, ls=":")
    fig.tight_layout()
    salva(fig, "prog0_1_isa")
