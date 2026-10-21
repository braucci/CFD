"""
metodi_numerici.py -- strumenti comuni della monografia 2 (flussi non viscosi).

Contiene:
  * trid(C)            eliminazione di Gauss per sistemi tridiagonali (forma a 4 colonne)
  * thomas(a, b, c, d) algoritmo di Thomas in memoria compatta
  * cerca_x(...)       ricerca dei punti di una linea di corrente lungo le verticali
  * cerca_xy(...)      ricerca lungo le verticali e lungo le orizzontali
  * jacobi_passo, gauss_seidel_passo, sor_passo: un'iterazione dei metodi
                       iterativi per l'equazione di Poisson su griglia quadrata
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
    a[0] e c[-1] non vengono usati. Costo: 8n operazioni circa."""
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):                 # eliminazione in avanti
        den = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / den if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / den
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):        # sostituzione all'indietro
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# ---------------------------------------------------------------------------
# 2. Ricerca dei punti di una linea psi = psia su una griglia rettangolare
#    psi[i, j] = valore nel nodo (x[i], y[j]); i nodi esterni al campo di moto
#    possono essere marcati con NaN e vengono ignorati.
# ---------------------------------------------------------------------------
def _attraversamenti(s, v, va):
    """Punti dell'intervallo [s_k, s_k+1] in cui v - va cambia segno:
    interpolazione lineare  s = s_k+1 - ds*|Q|/(|P|+|Q|)."""
    P = v[:-1] - va
    Q = v[1:] - va
    ok = np.isfinite(P) & np.isfinite(Q) & (P * Q <= 0.0) & ((P != 0) | (Q != 0))
    k = np.nonzero(ok)[0]
    ds = s[k + 1] - s[k]
    return s[k + 1] - ds * np.abs(Q[k]) / (np.abs(P[k]) + np.abs(Q[k]))


def cerca_x(x, y, psi, psia):
    """Ricerca lungo le sole linee verticali (subroutine SEARCH)."""
    xx, yy = [], []
    for i in range(len(x)):
        ys = _attraversamenti(y, psi[i, :], psia)
        xx.extend([x[i]] * len(ys))
        yy.extend(ys)
    return np.array(xx), np.array(yy)


def cerca_xy(x, y, psi, psia):
    """Ricerca lungo le verticali e lungo le orizzontali (SEARCH2):
    utile quando le linee di corrente sono quasi parallele all'asse y."""
    xx, yy = cerca_x(x, y, psi, psia)
    x2, y2 = [], []
    for j in range(len(y)):
        xs = _attraversamenti(x, psi[:, j], psia)
        x2.extend(xs)
        y2.extend([y[j]] * len(xs))
    return np.concatenate([xx, x2]), np.concatenate([yy, y2])


# ---------------------------------------------------------------------------
# 3. Metodi iterativi per  psi_xx + psi_yy = -zeta  su griglia quadrata di
#    passo h. 'interni' e' una lista di coppie (i, j) da aggiornare, nell'ordine
#    in cui vengono visitate (colonna per colonna, dal basso verso l'alto).
#    Ogni funzione esegue UNA iterazione e restituisce la somma dei |delta|.
# ---------------------------------------------------------------------------
def jacobi_passo(psi, zeta, h, interni):
    vecchio = psi.copy()                  # si usano SOLO i valori precedenti
    err = 0.0
    for i, j in interni:
        nuovo = 0.25 * (vecchio[i - 1, j] + vecchio[i + 1, j] +
                        vecchio[i, j - 1] + vecchio[i, j + 1] + h * h * zeta[i, j])
        err += abs(nuovo - vecchio[i, j])
        psi[i, j] = nuovo
    return err


def gauss_seidel_passo(psi, zeta, h, interni):
    err = 0.0                             # metodo di Liebmann (Gauss-Seidel)
    for i, j in interni:
        nuovo = 0.25 * (psi[i - 1, j] + psi[i + 1, j] +
                        psi[i, j - 1] + psi[i, j + 1] + h * h * zeta[i, j])
        err += abs(nuovo - psi[i, j])
        psi[i, j] = nuovo
    return err


def sor_passo(psi, zeta, h, interni, omega):
    err = 0.0                             # sovrarilassamento successivo
    for i, j in interni:
        gs = 0.25 * (psi[i - 1, j] + psi[i + 1, j] +
                     psi[i, j - 1] + psi[i, j + 1] + h * h * zeta[i, j])
        nuovo = (1.0 - omega) * psi[i, j] + omega * gs
        err += abs(nuovo - psi[i, j])
        psi[i, j] = nuovo
    return err


def omega_ottimo(m, n):
    """Fattore di sovrarilassamento ottimo per un rettangolo diviso in m x n
    intervalli:  omega = (8 - 4 sqrt(4 - a^2)) / a^2,  a = cos(pi/m) + cos(pi/n)."""
    a = np.cos(np.pi / m) + np.cos(np.pi / n)
    return (8.0 - 4.0 * np.sqrt(4.0 - a * a)) / (a * a)


def nodi_interni_rettangolo(M, N):
    """Coppie (i, j) dei nodi interni di una griglia M x N, colonna per colonna."""
    return [(i, j) for i in range(1, M - 1) for j in range(1, N - 1)]
