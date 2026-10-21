"""
prog1_7_scia_vorticosa.py  --  Programma 1.7
Avvolgimento della scia vorticosa dietro un'ala a carico ellittico:
m vortici puntiformi che si muovono con la velocita' indotta dagli altri
(Biot-Savart bidimensionale), integrati nel tempo con RK4.

    dX_i/dT =  sum_{j != i} G_j (Y_i - Y_j) / R_ij^2
    dY_i/dT = -sum_{j != i} G_j (X_i - X_j) / R_ij^2
"""
import numpy as np
import matplotlib.pyplot as plt
from ode_solvers import passo_rk4
from stile_grafici import salva, BLU, ARANCIO

m = 40                         # numero di vortici (pari)
dT = 1.0 / (25 * m)            # passo: soddisfa dT << 8 pi/m^2 (Moore, 1974)
istanti = [0.005, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.35]

# posizioni iniziali e intensita' (eq. 1.7.9)
dX = 1.0 / m
X0 = -1.0 + dX + 2.0 * dX * np.arange(m)
radice = lambda x: np.sqrt(np.clip(1.0 - x ** 2, 0.0, None))   # evita 1 - X^2 < 0 per arrotondamento
G = radice(X0 + dX) - radice(X0 - dX)
print(f"Verifica: somma delle G sul semipiano sinistro = {G[:m//2].sum():.6f} (atteso 1)")
print(f"dT = {dT:.4f}; limite di Moore 8 pi/m^2 = {8*np.pi/m**2:.4f}")


def velocita(T, s):
    """s = [X_1..X_m, Y_1..Y_m]: velocita' indotta su ciascun vortice da tutti gli altri."""
    X, Y = s[:m], s[m:]
    DX = X[:, None] - X[None, :]
    DY = Y[:, None] - Y[None, :]
    R2 = DX ** 2 + DY ** 2
    np.fill_diagonal(R2, np.inf)                     # esclude l'autoinduzione
    U = (G[None, :] * DY / R2).sum(axis=1)
    V = -(G[None, :] * DX / R2).sum(axis=1)
    return np.concatenate([U, V])


def simmetrizza(s):
    """Impone la simmetria rispetto all'asse Y: il vortice k = m+1-i e' l'immagine dell'i-esimo."""
    X, Y = s[:m].copy(), s[m:].copy()
    X[m // 2:] = -X[: m // 2][::-1]
    Y[m // 2:] = Y[: m // 2][::-1]
    return np.concatenate([X, Y])


if __name__ == "__main__":
    s = np.concatenate([X0, np.zeros(m)])
    T, k = 0.0, 0
    fig, axs = plt.subplots(4, 2, figsize=(6.2, 7.0))
    axs = axs.T.ravel()
    y_mid = []
    while k < len(istanti):
        s = simmetrizza(passo_rk4(velocita, T, s, dT))
        T += dT
        y_mid.append((T, s[m + m // 2 - 1]))
        if T >= istanti[k] - 1e-12:
            ax = axs[k]
            ax.plot(s[m // 2:m], s[m + m // 2:], ".", color=BLU, ms=3)
            ax.set_title(f"$T = {istanti[k]}$", fontsize=8, loc="right")
            ax.set_xlim(0, 1.05); ax.set_aspect("equal")
            ax.set_ylim(min(-0.12, s[m:].min() - 0.03), 0.12)
            ax.set_xlabel("$X$"); ax.set_ylabel("$Y$")
            print(f"T = {istanti[k]:5.3f}: Y vortice centrale = {s[m + m//2 - 1]:+.4f}, "
                  f"Y vortice di estremita' = {s[2*m - 1]:+.4f}")
            k += 1
    fig.tight_layout()
    salva(fig, "prog1_7_scia")

    # velocita' di discesa del punto medio del foglio
    y_mid = np.array(y_mid)
    vel = np.gradient(y_mid[:, 1], y_mid[:, 0])
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    ax.plot(y_mid[:, 0], -vel, color=BLU)
    # in forma dimensionale dy/dt = Gamma_0/(2 pi a) dY/dT: il valore Gamma_0/(2a) corrisponde a dY/dT = pi
    ax.axhline(np.pi, color=ARANCIO, ls="--", lw=0.8, label="$\\pi$  (cioe' $\\Gamma_0/2a$)")
    ax.axhline(np.pi / 2, color="gray", ls=":", lw=0.8, label="$\\pi/2$")
    ax.set_xlabel("$T$"); ax.set_ylabel("$-\\mathrm{d}Y/\\mathrm{d}T$ del punto medio")
    ax.set_ylim(0, 3.5)
    ax.legend(fontsize=7)
    salva(fig, "prog1_7_discesa")
    print(f"Velocita' di discesa adimensionale del centro: iniziale {-vel[0]:.3f} (teorico pi = {np.pi:.3f}), "
          f"a T = 0.35: {-vel[-1]:.3f}")

    # applicazione: Boeing 747
    a, W, Uinf, Gmed = 30.0, 4.0e6, 67.0, 815.0
    G0 = 4 * Gmed / np.pi
    t_car = 2 * np.pi * a ** 2 / G0
    print(f"Boeing 747: Gamma_0 = {G0:.0f} m^2/s, tempo caratteristico = {t_car:.2f} s, "
          f"T = 0.35 -> t = {0.35*t_car:.2f} s -> {0.35*t_car*Uinf:.0f} m dietro l'aereo")
