"""
metodi_numerici.py -- strumenti comuni della monografia 3 (flussi viscosi).

Contiene:
  * trid(C)              eliminazione di Gauss per sistemi tridiagonali (forma a 4 colonne)
  * thomas(a, b, c, d)   algoritmo di Thomas in memoria compatta
  * rk4_passo(f, x, y, h) un passo di Runge-Kutta del quarto ordine per y' = f(x, y)
  * rk4_integra(...)     integrazione a passo costante con memorizzazione della soluzione
  * liebmann(F, Q, h, errmax)  Gauss-Seidel per lap(F) = Q su griglia quadrata
  * sor_poisson(...)     sovrarilassamento successivo per lap(F) = Q
  * omega_ottimo(m, n)   fattore di sovrarilassamento ottimo per un rettangolo
  * cheb(N)              matrice di differenziazione di Chebyshev e nodi di Gauss-Lobatto
  * numeri_onda(N, L)    numeri d'onda della trasformata discreta di Fourier
"""
import numpy as np


# ---------------------------------------------------------------------------
# 1. Sistemi tridiagonali
# ---------------------------------------------------------------------------
def trid(C):
    """Risolve il sistema tridiagonale scritto riga per riga come
         C[i,0]*f[i-1] + C[i,1]*f[i] + C[i,2]*f[i+1] = C[i,3],  i = 0..n-1
    (C[0,0] e C[n-1,2] devono essere gia' stati azzerati dalle condizioni al
    contorno). Eliminazione in avanti per combinazione di righe, poi
    sostituzione all'indietro. Restituisce il vettore f (C non viene alterata).
    """
    C = np.array(C, dtype=float)          # copia di lavoro
    n = C.shape[0]
    for i in range(1, n - 1):             # eliminazione di C[i,0], i = 1..n-2
        c2 = C[i, 1] * C[i - 1, 1] - C[i, 0] * C[i - 1, 2]
        c3 = C[i, 2] * C[i - 1, 1]
        c4 = C[i, 3] * C[i - 1, 1] - C[i, 0] * C[i - 1, 3]
        C[i, 1], C[i, 2], C[i, 3] = c2, c3, c4
    f = np.zeros(n)
    # ultime due equazioni risolte insieme
    f[n - 1] = ((C[n - 1, 3] * C[n - 2, 1] - C[n - 1, 0] * C[n - 2, 3]) /
                (C[n - 1, 1] * C[n - 2, 1] - C[n - 1, 0] * C[n - 2, 2]))
    for j in range(n - 2, -1, -1):        # sostituzione all'indietro
        f[j] = (C[j, 3] - C[j, 2] * f[j + 1]) / C[j, 1]
    return f


def thomas(a, b, c, d):
    """Algoritmo di Thomas per  a[i] x[i-1] + b[i] x[i] + c[i] x[i+1] = d[i].
    a[0] e c[-1] non vengono usati. Funziona anche con coefficienti complessi."""
    n = len(d)
    tipo = np.result_type(a, b, c, d, float)
    cp = np.zeros(n, dtype=tipo)
    dp = np.zeros(n, dtype=tipo)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):                 # eliminazione in avanti
        den = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / den if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / den
    x = np.zeros(n, dtype=tipo)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):        # sostituzione all'indietro
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# ---------------------------------------------------------------------------
# 2. Runge-Kutta del quarto ordine per sistemi  y' = f(x, y)
# ---------------------------------------------------------------------------
def rk4_passo(f, x, y, h):
    """Un passo RK4: restituisce y(x + h)."""
    k1 = h * f(x, y)
    k2 = h * f(x + h / 2, y + k1 / 2)
    k3 = h * f(x + h / 2, y + k2 / 2)
    k4 = h * f(x + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def rk4_integra(f, y0, x_fin, h, x0=0.0):
    """Integra y' = f(x, y) da x0 a x_fin con passo h costante.
    Restituisce i vettori x (n+1) e la matrice Y (n+1, dim)."""
    n = int(round((x_fin - x0) / h))
    x = x0 + h * np.arange(n + 1)
    Y = np.zeros((n + 1, len(y0)))
    Y[0] = y0
    for i in range(n):
        Y[i + 1] = rk4_passo(f, x[i], Y[i], h)
    return x, Y


# ---------------------------------------------------------------------------
# 3. Equazione di Poisson  F_xx + F_yy = Q  su griglia di passo h.
#    F[i, j] = valore nel nodo (x_i, y_j); i valori sul contorno restano fissi.
# ---------------------------------------------------------------------------
def liebmann(F, Q, h, errmax, azzera=True):
    """Metodo di Liebmann (Gauss-Seidel) sui nodi interni di un rettangolo;
    si ferma quando la somma dei |variazioni| di un'iterazione e' <= errmax.
    Se azzera=True i nodi interni partono da zero (come la subroutine LIEBMN).
    Restituisce (F, numero di iterazioni)."""
    M, N = F.shape
    if azzera:
        F[1:-1, 1:-1] = 0.0
    it = 0
    while True:
        it += 1
        err = 0.0
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                nuovo = 0.25 * (F[i - 1, j] + F[i + 1, j] + F[i, j - 1] +
                                F[i, j + 1] - h * h * Q[i, j])
                err += abs(nuovo - F[i, j])
                F[i, j] = nuovo
        if err <= errmax:
            return F, it


def sor_poisson(F, Q, h, omega, errmax, nodi, prima_del_nodo=None, it_max=100000):
    """Sovrarilassamento successivo sui nodi 'nodi' (lista di coppie (i, j),
    visitati nell'ordine dato). prima_del_nodo(F, i, j), se fornita, viene
    chiamata prima di aggiornare ogni nodo (serve ad aggiornare i nodi fittizi
    delle condizioni di derivata). Restituisce (F, iterazioni)."""
    for it in range(1, it_max + 1):
        err = 0.0
        for i, j in nodi:
            if prima_del_nodo is not None:
                prima_del_nodo(F, i, j)
            gs = 0.25 * (F[i - 1, j] + F[i + 1, j] + F[i, j - 1] + F[i, j + 1] - h * h * Q[i, j])
            nuovo = (1.0 - omega) * F[i, j] + omega * gs
            err += abs(nuovo - F[i, j])
            F[i, j] = nuovo
        if err <= errmax:
            return F, it
    raise RuntimeError("SOR: convergenza non raggiunta")


def omega_ottimo(m, n):
    """Fattore di sovrarilassamento ottimo per un rettangolo con m x n linee di
    griglia (come nei programmi originali):
    omega = (8 - 4 sqrt(4 - a^2)) / a^2,  a = cos(pi/m) + cos(pi/n)."""
    a = np.cos(np.pi / m) + np.cos(np.pi / n)
    return (8.0 - 4.0 * np.sqrt(4.0 - a * a)) / (a * a)


# ---------------------------------------------------------------------------
# 4. Metodi spettrali
# ---------------------------------------------------------------------------
def cheb(N):
    """Nodi di Chebyshev-Gauss-Lobatto y_j = cos(j pi / N), j = 0..N (da +1 a -1)
    e matrice di differenziazione D (N+1 x N+1): (D u)_j approssima u'(y_j).
    Elementi fuori diagonale: D_jk = c_j (-1)^(j+k) / (c_k (y_j - y_k)), c_0 = c_N = 2;
    la diagonale si ottiene imponendo che la derivata di una costante sia nulla
    (somma per righe nulla), che e' anche la scelta numericamente piu' stabile."""
    if N == 0:
        return np.zeros((1, 1)), np.array([1.0])
    j = np.arange(N + 1)
    y = np.cos(np.pi * j / N)
    c = np.ones(N + 1)
    c[0] = c[-1] = 2.0
    c = c * (-1.0) ** j
    Y = np.tile(y, (N + 1, 1)).T
    dY = Y - Y.T
    D = np.outer(c, 1.0 / c) / (dY + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return D, y


def numeri_onda(N, L):
    """Numeri d'onda k_n = 2 pi n / L nell'ordine usato da numpy.fft
    (n = 0, 1, ..., N/2-1, -N/2, ..., -1)."""
    return 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
