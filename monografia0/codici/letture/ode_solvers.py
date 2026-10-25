"""
ode_solvers.py -- integratori a un passo per problemi ai valori iniziali
                  dy/dt = f(t, y),  y(t0) = y0

Lo stato y puo' essere uno scalare o un array NumPy: le stesse formule
valgono per una singola equazione e per un sistema di equazioni.
"""
import numpy as np


# ----------------------------------------------------------------------
# Passi elementari: da (t, y) a y(t + h)
# ----------------------------------------------------------------------
def passo_eulero(f, t, y, h):
    """Metodo di Eulero esplicito (ordine 1)."""
    return y + h * f(t, y)


def passo_rk2(f, t, y, h):
    """Runge-Kutta del secondo ordine (metodo del punto medio)."""
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
    return y + k2


def passo_rk4(f, t, y, h):
    """Runge-Kutta classico del quarto ordine."""
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def passo_heun(f, t, y, h):
    """Runge-Kutta del secondo ordine, variante di Heun (a1 = a2 = 1/2, p = q = 1)."""
    k1 = h * f(t, y)
    k2 = h * f(t + h, y + k1)
    return y + 0.5 * (k1 + k2)


def passo_ralston(f, t, y, h):
    """Runge-Kutta del secondo ordine, variante di Ralston (a2 = 2/3, p = q = 3/4)."""
    k1 = h * f(t, y)
    k2 = h * f(t + 0.75 * h, y + 0.75 * k1)
    return y + (k1 + 2.0 * k2) / 3.0


METODI = {"eulero": passo_eulero, "rk2": passo_rk2, "rk4": passo_rk4,
          "heun": passo_heun, "ralston": passo_ralston}


# ----------------------------------------------------------------------
# Marcia nel tempo a passo costante
# ----------------------------------------------------------------------
def integra(f, t0, y0, h, n_passi, metodo="rk4", stop=None):
    """
    Integra dy/dt = f(t, y) da t0 con passo h per n_passi passi.

    f       : funzione f(t, y) -> dy/dt (stessa forma di y)
    stop    : funzione opzionale stop(t, y) -> bool; se vera la marcia
              si arresta (es. proiettile che tocca il suolo)
    Restituisce gli array t e y (y ha forma (n+1, len(y0))).
    """
    passo = METODI[metodo]
    y = np.atleast_1d(np.asarray(y0, dtype=float))
    t = float(t0)
    T, Y = [t], [y.copy()]
    for _ in range(n_passi):
        y = passo(f, t, y, h)
        t = t + h
        T.append(t)
        Y.append(y.copy())
        if stop is not None and stop(t, y):
            break
    return np.array(T), np.array(Y)


# ----------------------------------------------------------------------
# Coefficiente di resistenza di una sfera liscia (approssimazione a spezzata)
# ----------------------------------------------------------------------
def cd_sfera(Re):
    """
    Approssimazione a tratti della curva sperimentale c_d(Re)
    per sfere lisce nel piano log-log (punti di raccordo a, b, c-d, e).
    """
    Re = abs(Re)
    if Re == 0.0:
        return 0.0
    if Re <= 1.0:
        return 24.0 / Re                       # regime di Stokes
    if Re <= 400.0:
        return 24.0 / Re ** 0.646              # retta a-b nel piano log-log
    if Re <= 3.0e5:
        return 0.5                             # plateau laminare
    if Re <= 2.0e6:
        return 0.000366 * Re ** 0.4275         # risalita post-crisi
    return 0.18                                # plateau turbolento
