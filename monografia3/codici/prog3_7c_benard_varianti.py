"""
prog3_7c_benard_varianti.py -- complementi sulla convezione di Rayleigh-Benard
con la collocazione di Chebyshev su z = (1 + y)/2.

1) Tassi di crescita: problema agli autovalori in sigma (non solo marginale)
       [sigma/Pr - L] L w = -k^2 Ra theta,   [sigma - L] theta = w,   L = D^2 - k^2,
   per verificare il principio di scambio delle stabilita' (sigma reale) e
   misurare sigma_max in funzione di Ra oltre la soglia.
2) Numero di Rayleigh critico per le tre combinazioni di condizioni al contorno:
   rigido-rigido, rigido-libero, libero-libero.
3) Stima variazionale: Ra(k) = int |L w|^2 dz / ( k^2 int (|D theta|^2 + k^2 |theta|^2) dz ),
   con L theta = -w, per la funzione di prova w = z^2 (1 - z)^2.
"""
import numpy as np
from scipy.linalg import eig, solve
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from metodi_numerici import cheb
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

N = 40
D, y = cheb(N)
z = (1 + y) / 2                 # z_0 = 1 (parete superiore), z_N = 0 (inferiore)
Dz = 2 * D
D2 = Dz @ Dz
I = np.eye(N + 1)
n1 = N + 1


def condizioni(A, B, sup, inf):
    """Impone theta = 0, w = 0 e (rigido) Dw = 0 oppure (libero) D^2 w = 0 alle due pareti."""
    for riga in (0, N):                                   # theta = 0
        A[riga, :] = 0; A[riga, riga] = 1; B[riga, :] = 0
    for riga in (0, N):                                   # w = 0
        A[n1 + riga, :] = 0; A[n1 + riga, n1 + riga] = 1; B[n1 + riga, :] = 0
    for riga, tipo, nodo in ((1, sup, 0), (N - 1, inf, N)):
        A[n1 + riga, :] = 0; B[n1 + riga, :] = 0
        A[n1 + riga, n1:] = Dz[nodo] if tipo == "rigido" else D2[nodo]
    return A, B


def ra_marginale(k, sup="rigido", inf="rigido"):
    L = D2 - k**2 * I
    Z = np.zeros((n1, n1))
    A = np.block([[L, I], [Z, L @ L]])
    B = np.block([[Z, Z], [k**2 * I, Z]])
    A, B = condizioni(A, B, sup, inf)
    lam = eig(A, B, right=False)
    ok = np.isfinite(lam) & (np.abs(lam.imag) < 1e-6 * np.abs(lam)) & (lam.real > 0)
    return lam[ok].real.min()


def sigma_max(k, Ra, Pr=1.0, sup="rigido", inf="rigido"):
    L = D2 - k**2 * I
    Z = np.zeros((n1, n1))
    A = np.block([[L, I], [-k**2 * Ra * I, L @ L]])
    B = np.block([[I, Z], [Z, L / Pr]])
    A, B = condizioni(A, B, sup, inf)
    lam = eig(A, B, right=False)
    lam = lam[np.isfinite(lam) & (np.abs(lam) < 1e6)]      # scarta gli autovalori spuri (infiniti)
    # parte immaginaria massima fra i modi ben risolti (|sigma| < 1000; i modi piu' smorzati
    # non sono rappresentati dalla griglia e possono risultare complessi per errore numerico)
    return lam[np.argmax(lam.real)], np.max(np.abs(lam.imag[np.abs(lam) < 1e3]))


# ---------------------------------------------------------------- 1) scambio delle stabilita'
print("Pareti rigide, k = 3,1163, Pr = 1: autovalore sigma meno stabile")
print("      Ra      sigma_max           max |Im sigma| (modi con |sigma| < 1000)")
for Ra in (1000, 1500, 1707.76, 2000, 3000, 5000):
    s, im = sigma_max(3.1163, Ra)
    print(f"   {Ra:8.2f}   {s.real:+9.4f} {s.imag:+.1e}i      {im:.1e}")
Ras = np.linspace(1500, 3000, 31)
sig = np.array([sigma_max(3.1163, Ra)[0].real for Ra in Ras])
pend = np.polyfit(Ras[:8], sig[:8], 1)
print(f"   vicino alla soglia: sigma ~ {pend[0]:.3e} (Ra - {-pend[1]/pend[0]:.1f});"
      f" per Pr = 7 (acqua): sigma_max(Ra = 2000) = {sigma_max(3.1163, 2000, Pr=7)[0].real:.4f},"
      f" per Pr = 0,71: {sigma_max(3.1163, 2000, Pr=0.71)[0].real:.4f}")

# ---------------------------------------------------------------- 2) condizioni al contorno
print("\nNumero di Rayleigh critico per diverse condizioni al contorno (Chebyshev, N = 40):")
critici = {}
for sup, inf in (("rigido", "rigido"), ("rigido", "libero"), ("libero", "libero")):
    ris = minimize_scalar(lambda k: ra_marginale(k, sup, inf), bounds=(1.5, 4.5), method="bounded",
                          options={"xatol": 1e-5})
    critici[(sup, inf)] = (ris.x, ris.fun)
    print(f"   {sup:7s} - {inf:7s}: k_c = {ris.x:.4f}, Ra_c = {ris.fun:.3f}")
print(f"   libero-libero esatto: k_c = pi/sqrt(2) = {np.pi/np.sqrt(2):.4f}, Ra_c = 27 pi^4/4 = {27*np.pi**4/4:.3f}")

# ---------------------------------------------------------------- 3) stima variazionale
def clenshaw_curtis(N):
    """Pesi di Clenshaw-Curtis sui nodi di Chebyshev-Gauss-Lobatto di [-1, 1]
    (integrazione esatta del polinomio interpolante)."""
    theta = np.pi * np.arange(N + 1) / N
    w = np.zeros(N + 1)
    ii = np.arange(1, N)
    v = np.ones(N - 1)
    if N % 2 == 0:
        w[0] = w[N] = 1.0 / (N * N - 1)
        for k in range(1, N // 2):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k * k - 1)
        v -= np.cos(N * theta[ii]) / (N * N - 1)
    else:
        w[0] = w[N] = 1.0 / N**2
        for k in range(1, (N - 1) // 2 + 1):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k * k - 1)
    w[ii] = 2 * v / N
    return w


def ra_variazionale(k, w):
    """Quoziente di Rayleigh con theta ottenuta da L theta = -w, theta = 0 alle pareti."""
    L = D2 - k**2 * I
    M = L.copy(); M[0, :] = 0; M[0, 0] = 1; M[N, :] = 0; M[N, N] = 1
    rhs = -w.copy(); rhs[0] = rhs[N] = 0
    th = solve(M, rhs)
    pesi = clenshaw_curtis(N) / 2                          # su [0, 1]
    num = np.sum(pesi * (L @ w)**2)
    den = k**2 * np.sum(pesi * ((Dz @ th)**2 + k**2 * th**2))
    return num / den


w_prova = z**2 * (1 - z)**2
print(f"\nVerifica dei pesi di quadratura: int_0^1 z^2 (1-z)^2 dz = {np.sum(clenshaw_curtis(N) / 2 * w_prova):.10f} (esatto {1/30:.10f})")
ris = minimize_scalar(lambda k: ra_variazionale(k, w_prova), bounds=(2, 4.5), method="bounded")
print(f"\nStima variazionale con w = z^2 (1 - z)^2: minimo Ra = {ris.fun:.2f} per k = {ris.x:.4f}"
      f" (esatto 1707,76 per k = 3,1163: eccesso {100*(ris.fun/1707.762-1):.2f} %)")
w2 = np.sin(np.pi * z)**2          # altra funzione di prova (ammissibile: w = Dw = 0 alle pareti)
ris2 = minimize_scalar(lambda k: ra_variazionale(k, w2), bounds=(2, 4.5), method="bounded")
print(f"   con w = sin^2(pi z):          minimo Ra = {ris2.fun:.2f} per k = {ris2.x:.4f} (eccesso {100*(ris2.fun/1707.762-1):.2f} %)")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
ax[0].plot(Ras, sig, color=BLU)
ax[0].axhline(0, color="k", lw=0.6)
ax[0].axvline(1707.76, color=GRIGIO, ls=":", lw=1)
ax[0].set_xlabel("$Ra$")
ax[0].set_ylabel("$\\sigma_{\\max}$  ($k=3{,}116$, $Pr=1$)")
ax[0].set_title("tasso di crescita oltre la soglia")
ks = np.linspace(1.5, 6, 46)
for (sup, inf), col, et in zip(critici, (BLU, ARANCIO, VERDE), ("rigido--rigido", "rigido--libero", "libero--libero")):
    ax[1].plot(ks, [ra_marginale(k, sup, inf) for k in ks], color=col, label=et)
    ax[1].plot(critici[(sup, inf)][0], critici[(sup, inf)][1], "o", color=col, ms=4)
ax[1].plot(ks, [ra_variazionale(k, w_prova) for k in ks], "--", color=ROSSO, lw=1, label="variazionale, $z^2(1-z)^2$")
ax[1].set_ylim(0, 6000)
ax[1].set_xlabel("$k$")
ax[1].set_ylabel("$Ra$")
ax[1].legend(fontsize=7)
ax[1].set_title("curve neutre")
fig.tight_layout()
salva(fig, "prog3_7c_benard_varianti")
