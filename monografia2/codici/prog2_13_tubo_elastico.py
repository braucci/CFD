"""
prog2_13_tubo_elastico.py -- flusso pulsante in un tubo elastico
(modello monodimensionale di un'arteria).

   S_t + (S V)_z = 0
   V_t + (V^2/2)_z = -(1/rho) p_z + f
   S = S(p, z) = S0 e^{-k z} exp[(p - p0)/(rho c(p,z) c(p0,z))]
   c(p, z) = (97 + 2,03 p)(1 + 0,02 z)  [cm/s, p in mmHg]
   f = -8 pi (mu/rho) V/S  (attrito laminare di Poiseuille)
Unita' CGS (cm, s, g; pressioni in dyn/cm^2 nelle equazioni, in mmHg nei dati).
Schema di MacCormack con alternanza delle differenze (avanti/indietro e
indietro/avanti a passi alterni) e dissipazione artificiale del quarto ordine
sull'equazione della velocita'. Condizioni al contorno: pressione prossimale
pulsante p_L(t), pressione distale fissa 15 mmHg, velocita' estrapolate.
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, ROSSO, GRIGIO, VERDE, VIOLA

MMHG = 1333.22                     # 1 mmHg in dyn/cm^2
RHO, MU = 1.06, 0.049              # sangue: g/cm^3, poise
L, NZ = 25.0, 101                  # lunghezza [cm] e numero di nodi (dz = 0,25 cm)
S0, K_RASTR = 7.07, 0.25           # area di riferimento [cm^2], rastremazione [1/cm]
P0, PR = 80.0, 15.0                # pressione di riferimento e distale [mmHg]
PERIODO = 0.7                      # periodo cardiaco [s] (circa 86 battiti/min)


def _forma(t, periodo):
    """Forma d'onda generica (4 armoniche) della pressione prossimale."""
    w = 2 * np.pi / periodo
    return (10 * np.cos(w * t - 1.8) + 5 * np.cos(2 * w * t - 2.8)
            + 2.5 * np.cos(3 * w * t - 3.6) + 1.2 * np.cos(4 * w * t - 4.5))


def pressione_prossimale(t, periodo=PERIODO, p_min=55.0, p_max=85.0):
    tg = np.linspace(0, periodo, 2001)
    s = _forma(tg, periodo)
    return p_min + (p_max - p_min) * (_forma(t, periodo) - s.min()) / (s.max() - s.min())


def c_onda(p, z):
    return (97.0 + 2.03 * p) * (1.0 + 0.02 * z)


def area(p, z):
    return S0 * np.exp(-K_RASTR * z) * np.exp((p - P0) * MMHG / (RHO * c_onda(p, z) * c_onda(P0, z)))


def pressione(S, z, p):
    """Inverte l'equazione di stato S(p, z) con il metodo di Newton."""
    lnr = np.log(S / (S0 * np.exp(-K_RASTR * z)))
    k = MMHG / (RHO * c_onda(P0, z))
    for _ in range(30):
        cp = c_onda(p, z)
        g = lnr - k * (p - P0) / cp
        dg = -k * (1.0 / cp - (p - P0) * 2.03 * (1 + 0.02 * z) / cp**2)
        dp = -g / dg
        p = p + dp
        if np.max(np.abs(dp)) < 1e-10:
            break
    return p


def attrito(V, S, tipo="laminare"):
    if tipo == "laminare":
        return -8.0 * np.pi * MU / RHO * V / S
    return -0.1360 * (MU / RHO)**0.25 * np.abs(V)**1.75 / S**0.625 * np.sign(V)


def simula(t_fin=2.1, periodo=PERIODO, p_min=55.0, p_max=85.0, mu_e=0.04, cfl=0.8,
           tipo_attrito="laminare", z_reg=(15.0, 20.0, 22.0)):
    z = np.linspace(0.0, L, NZ); dz = z[1] - z[0]
    pL = lambda t: pressione_prossimale(t, periodo, p_min, p_max)
    VL, VR = 3.0, 0.1                                      # condizioni iniziali
    V = VL * np.exp(z / L * np.log(VR / VL))
    p = pL(0.0) * np.exp(z / L * np.log(PR / pL(0.0)))
    S = area(p, z)
    idx = [int(round(zz / dz)) for zz in z_reg]
    t, n, storia, CV = 0.0, 0, [], 0.0

    def derivate(S, V, p, verso):
        F1, F2 = S * V, 0.5 * V**2
        d1, d2, dp = np.zeros(NZ), np.zeros(NZ), np.zeros(NZ)
        if verso == "avanti":
            d1[:-1] = np.diff(F1); d2[:-1] = np.diff(F2); dp[:-1] = np.diff(p)
        else:
            d1[1:] = np.diff(F1); d2[1:] = np.diff(F2); dp[1:] = np.diff(p)
        return (-d1 / dz,
                -d2 / dz - dp * MMHG / (RHO * dz) + attrito(V, S, tipo_attrito))

    def contorno(S, V, p, tt):
        p[0], p[-1] = pL(tt), PR
        S[0], S[-1] = area(p[0], z[0]), area(p[-1], z[-1])
        V[0], V[-1] = V[1], V[-2]                          # estrapolazione di ordine zero

    while t < t_fin:
        dt = cfl * dz / np.max(np.abs(V) + c_onda(p, z))  # stabilita': |V| + c
        CV = max(CV, np.max(np.abs(V)) * dt / dz)
        v1, v2 = ("avanti", "indietro") if n % 2 == 0 else ("indietro", "avanti")
        r1, r2 = derivate(S, V, p, v1)                    # predittore
        Ss, Vs = S + dt * r1, V + dt * r2
        ps = pressione(Ss, z, p.copy())
        contorno(Ss, Vs, ps, t + dt)
        q1, q2 = derivate(Ss, Vs, ps, v2)                  # correttore
        Sn, Vn = 0.5 * (S + Ss + dt * q1), 0.5 * (V + Vs + dt * q2)
        D = np.zeros(NZ)
        D[2:-2] = -mu_e / 8 * (V[4:] - 4 * V[3:-1] + 6 * V[2:-2] - 4 * V[1:-3] + V[:-4])
        Vn += D
        pn = pressione(Sn, z, p.copy())
        t, n = t + dt, n + 1
        contorno(Sn, Vn, pn, t)
        S, V, p = Sn, Vn, pn
        storia.append((t, *[p[i] for i in idx], *[V[i] for i in idx],
                       *[S[i] * V[i] for i in idx]))
        if not np.all(np.isfinite(V)):
            print("soluzione divergente a t =", t)
            break
    # colonne di 'storia': t | p nelle sezioni z_reg | V | portata S V
    return z, np.array(storia), n, CV, (S, V, p)


if __name__ == "__main__":
    z_reg = (5.0, 12.5, 15.0, 20.0, 22.0)
    nr = len(z_reg)
    z, H, n, CV, (S, V, p) = simula(t_fin=4.2, z_reg=z_reg)     # 6 periodi
    t = H[:, 0]
    print(f"passi = {n}, massimo numero di Courant basato su V: {CV:.3f}")
    print(f"area: {S0:.2f} cm^2 a z = 0, {S0*np.exp(-K_RASTR*L):.4f} cm^2 a z = L (p = p0)")
    ult = t >= t[-1] - PERIODO                       # ultimo periodo
    for k, zz in enumerate(z_reg):
        P, Vz = H[ult, 1 + k], H[ult, 1 + nr + k]
        q = np.trapezoid(H[ult, 1 + 2 * nr + k], t[ult]) / PERIODO
        print(f"z = {zz:4.1f} cm: p in [{P.min():6.1f}, {P.max():6.1f}] mmHg, "
              f"V in [{Vz.min():6.1f}, {Vz.max():6.1f}] cm/s, portata media {q:.3f} cm^3/s")

    # condizione al contorno prossimale
    tt = np.linspace(0, PERIODO, 400)
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.plot(tt, pressione_prossimale(tt), color=ROSSO)
    ax.set_xlabel("$t$ [s]"); ax.set_ylabel("$p_L$ [mmHg]")
    salva(fig, "prog2_13_pressione_prossimale")

    # pressione e velocita' in tre sezioni sull'ultimo periodo
    fig, axs = plt.subplots(2, 3, figsize=(6.8, 4.0), sharex=True)
    for k, zz in enumerate(z_reg[2:]):
        axs[0, k].plot(t[ult], H[ult, 3 + k], color=ROSSO)
        axs[1, k].plot(t[ult], H[ult, 3 + nr + k], color=BLU)
        axs[1, k].axhline(0, color=GRIGIO, lw=0.5)
        axs[0, k].set_title(f"$z = {zz:g}$ cm", fontsize=8)
        axs[1, k].set_xlabel("$t$ [s]")
        axs[0, k].set_ylim(30, 110); axs[1, k].set_ylim(-30, 110)
    axs[0, 0].set_ylabel("$p$ [mmHg]"); axs[1, 0].set_ylabel("$V$ [cm/s]")
    salva(fig, "prog2_13_segnali")
