"""
prog2_4_joukowski.py -- PROGRAMMA 2.4
Dal flusso attorno a un cilindro circolare al flusso attorno a un profilo di
Joukowski mediante la trasformazione conforme  z = zeta + b^2/zeta.

Piano zeta (del cerchio): cerchio di raggio a centrato in P = (x_P, y_P), con
x_P = b - sqrt(a^2 - y_P^2) cosi' che il cerchio passi per Q = (b, 0).
Potenziale complesso (corrente U inclinata di alfa, condizione di Kutta in Q):
   w = U [ zeta' e^{-i alfa} + a^2 e^{i alfa}/zeta' ] + i Gamma/(2 pi) ln(zeta'/a),
   zeta' = zeta - zeta_P,   Gamma = 4 pi a U sin(alfa + beta),  sin beta = y_P/a.
(Per alfa = 0 si ritrova w = U [zeta' + a^2/zeta' + i 2 y_P ln(zeta'/a)].)
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, ROSSO, GRIGIO, VERDE
from metodi_numerici import cerca_x

U, A, B, YP = 1.0, 1.0, 0.8, 0.199
XP = B - np.sqrt(A**2 - YP**2)
ZP = XP + 1j * YP
BETA = np.arcsin(YP / A)


def joukowski(zeta):
    return zeta + B**2 / zeta


def circolazione(alfa):
    """Circolazione (oraria positiva) che soddisfa la condizione di Kutta."""
    return 4 * np.pi * A * U * np.sin(alfa + BETA)


def w_complesso(zeta, alfa=0.0):
    zl = zeta - ZP
    gam = circolazione(alfa)
    return (U * (zl * np.exp(-1j * alfa) + A**2 * np.exp(1j * alfa) / zl)
            + 1j * gam / (2 * np.pi) * np.log(zl / A))


def _velocita_cerchio(fi, alfa):
    """|dw/dz| nel punto del cerchio di anomalia fi (fi misurata da P)."""
    zl = A * np.exp(1j * fi)                      # zeta' sul cerchio
    zeta = zl + ZP
    gam = circolazione(alfa)
    dw_dzeta = U * (np.exp(-1j * alfa) - A**2 * np.exp(1j * alfa) / zl**2) \
        + 1j * gam / (2 * np.pi * zl)
    dz_dzeta = 1 - (B / zeta)**2
    return np.abs(dw_dzeta / dz_dzeta)


def superficie(npunti=200, alfa=0.0):
    """Punti esatti del profilo e velocita' (formula 2.6.25 generalizzata).
    Il primo punto e' il bordo d'uscita (immagine di Q, fi = -beta): li'
    numeratore e denominatore si annullano e la velocita' e' il limite,
    calcolato come media dei valori a fi = -beta +- 1e-5."""
    fi = -BETA + np.linspace(0.0, 2 * np.pi, npunti, endpoint=False)
    z = joukowski(A * np.exp(1j * fi) + ZP)
    with np.errstate(divide="ignore", invalid="ignore"):
        V = _velocita_cerchio(fi, alfa)
    V[0] = 0.5 * (_velocita_cerchio(fi[0] - 1e-5, alfa) +
                  _velocita_cerchio(fi[0] + 1e-5, alfa))
    return fi, z, V


if __name__ == "__main__":
    print(f"centro del cerchio P = ({XP:.4f}, {YP:.4f}),  beta = {np.degrees(BETA):.3f} gradi")

    # ---------------- geometria del profilo --------------------------------
    fi, z, V = superficie(2000)
    x, y = z.real, z.imag
    i_le, i_te = np.argmin(x), np.argmax(x)
    corda = x[i_te] - x[i_le]
    # dorso e ventre: si interpola su una stessa ascissa
    xs = np.linspace(x[i_le], x[i_te], 400)
    up = np.interp(xs, *zip(*sorted(zip(x[fi < fi[i_le]], y[fi < fi[i_le]]))))
    lo = np.interp(xs, *zip(*sorted(zip(x[fi > fi[i_le]], y[fi > fi[i_le]]))))
    spess = up - lo
    curv = 0.5 * (up + lo)
    print(f"bordo d'attacco x = {x[i_le]:.4f}, bordo d'uscita x = {x[i_te]:.4f}")
    print(f"corda c = {corda:.4f} (4b = {4*B:.4f})")
    print(f"spessore massimo = {spess.max():.4f} = {100*spess.max()/corda:.2f}% c "
          f"a x/c = {(xs[np.argmax(spess)]-x[i_le])/corda:.3f}")
    print(f"freccia massima  = {curv.max():.4f} = {100*curv.max()/corda:.2f}% c "
          f"a x/c = {(xs[np.argmax(curv)]-x[i_le])/corda:.3f}")

    # ---------------- linee di corrente: ricerca nel piano zeta + mappa ----
    DX1 = 0.05
    x1 = np.arange(-3.0, 3.0 + 1e-9, DX1)
    y1 = np.arange(-3.0, 3.0 + 1e-9, DX1)
    X1, Y1 = np.meshgrid(x1, y1, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        PSI = w_complesso(X1 + 1j * Y1).imag
    PSI[np.abs(X1 + 1j * Y1 - ZP) < A] = np.nan    # interno del cerchio
    COST = -2.0 + 0.5 * np.arange(10)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for k, c in enumerate(COST):
        xx, yy = cerca_x(x1, y1, PSI, c)
        zm = joukowski(xx + 1j * yy)
        tieni = (np.abs(zm.real) <= 3) & (np.abs(zm.imag) <= 3)
        ax.plot(zm.real[tieni], zm.imag[tieni], ".", ms=2.2,
                color=ROSSO if c == 0 else BLU)
    ax.fill(x, y, color=GRIGIO, alpha=0.4)
    ax.plot(x, y, color="k", lw=0.8)
    ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-2.6, 2.6)
    ax.set_xlabel("$x$ [m]"); ax.set_ylabel("$y$ [m]")
    salva(fig, "prog2_4_linee")

    # ---------------- distribuzione di pressione ----------------------------
    fi, z, V = superficie(200)
    cp = 1 - (V / U)**2
    gam = circolazione(0.0)
    # portanza dall'integrale della pressione (contorno chiuso, normale esterna)
    fid, zd, Vd = superficie(20000)
    cpd = 1 - (Vd / U)**2
    zc = 0.5 * (zd + np.roll(zd, -1))
    cpc = 0.5 * (cpd + np.roll(cpd, -1))
    cpc[~np.isfinite(cpc)] = 0.0
    dz = np.roll(zd, -1) - zd                   # contorno percorso in senso antiorario
    # forza = -oint p n ds; per contorno antiorario n ds = -i dz  ->  F = i oint p dz
    F = 1j * np.sum(cpc * dz)                   # in unita' di (1/2) rho U^2
    CL_p, CD_p = F.imag / corda, F.real / corda
    print(f"\nGamma = 4 pi y_P U = {gam:.5f} m^2/s")
    print(f"C_L (Kutta-Joukowski) = 2 Gamma/(U c) = {2*gam/(U*corda):.5f}")
    print(f"C_L (integrale di C_p) = {CL_p:.5f},   C_D (integrale) = {CD_p:+.2e}")
    print(f"C_p al bordo d'uscita = {cp[0]:.4f};  C_p minimo = {cp.min():.4f}")

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(z.real, cp, ".", color=BLU, ms=4)
    ax.invert_yaxis()
    ax.set_xlabel("$x$ [m]"); ax.set_ylabel("$C_p$")
    ax.annotate("ventre", (0.45, 0.75), color=GRIGIO)
    ax.annotate("dorso", (0.45, -1.2), color=GRIGIO)
    salva(fig, "prog2_4_cp")

    # ---------------- estensione: incidenza variabile -----------------------
    print("\n alfa [gradi]   C_L esatto   8 pi a (alfa+beta)/c  (seno linearizzato)")
    alfas = np.radians(np.arange(-4, 13, 2))
    CL = [2 * circolazione(al) / (U * corda) for al in alfas]
    for al, cl in zip(alfas, CL):
        print(f"   {np.degrees(al):5.1f}       {cl:7.4f}      {8*np.pi*A*(al+BETA)/corda:7.4f}")
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for al, col in zip(np.radians([0, 4, 8]), [BLU, ARANCIO, ROSSO]):
        fi, z, V = superficie(400, al)
        ax[0].plot(z.real, 1 - V**2, color=col, lw=1.0,
                   label=fr"$\alpha={np.degrees(al):.0f}^\circ$")
    ax[0].set_ylim(1.1, -4.5); ax[0].set_xlabel("$x$ [m]"); ax[0].set_ylabel("$C_p$")
    ax[0].legend()
    ax[1].plot(np.degrees(alfas), CL, "o-", color=BLU, ms=3.5)
    ax[1].axvline(-np.degrees(BETA), color=GRIGIO, ls=":", lw=0.8)
    ax[1].annotate(r"$\alpha_{0L}=-\beta$", (-np.degrees(BETA) + 0.5, 2.6), color=GRIGIO)
    ax[1].set_xlabel(r"$\alpha$ [gradi]"); ax[1].set_ylabel("$C_L$")
    salva(fig, "prog2_4_incidenza")
