"""
prog3_5c_adi.py -- equazione della diffusione in due dimensioni: metodo ADI
(Peaceman-Rachford) confrontato con lo schema esplicito.

Problema: avviamento della corrente in un condotto quadrato 0 <= x, y <= 1
per un gradiente di pressione applicato bruscamente. Con U = u/[-(dp/dx) L^2/mu]
e T = nu t/L^2, la differenza W = U_s - U fra il profilo di regime e quello
attuale soddisfa  W_T = W_xx + W_yy,  W = 0 sulle pareti,  W(x, y, 0) = U_s,
con U_s soluzione di  lap U_s = -1  (serie doppia di Fourier):
    U_s = sum_{m,n dispari} 16/(pi^4 m n (m^2+n^2)) sin(m pi x) sin(n pi y),
    W   = sum_{m,n dispari} 16/(pi^4 m n (m^2+n^2)) sin(m pi x) sin(n pi y) exp[-(m^2+n^2) pi^2 T].

Schema ADI: ogni passo e' diviso in due mezzi passi, implicito in x (poi in y)
ed esplicito nell'altra direzione; ogni mezzo passo richiede solo sistemi
tridiagonali lungo le righe (o le colonne). E' incondizionatamente stabile e
del secondo ordine nel tempo.
"""
import numpy as np
import matplotlib.pyplot as plt
from metodi_numerici import thomas
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO


def serie(x, y, T, nmax=99):
    X, Y = np.meshgrid(x, y, indexing="ij")
    W = np.zeros_like(X)
    for m in range(1, nmax + 1, 2):
        for n in range(1, nmax + 1, 2):
            W += 16 / (np.pi**4 * m * n * (m * m + n * n)) * np.sin(m * np.pi * X) * np.sin(n * np.pi * Y) \
                 * np.exp(-(m * m + n * n) * np.pi**2 * T)
    return W


def adi(W, h, tau, passi):
    """Peaceman-Rachford: (1 - r/2 d_xx) W* = (1 + r/2 d_yy) W^n,
                          (1 - r/2 d_yy) W^(n+1) = (1 + r/2 d_xx) W*."""
    n = W.shape[0] - 1
    r = tau / h**2
    a = np.full(n - 1, -r / 2)
    b = np.full(n - 1, 1 + r)
    for _ in range(passi):
        # primo mezzo passo: implicito in x, riga per riga (y fissato)
        Ws = np.zeros_like(W)
        rhs = W[1:-1, 1:-1] + r / 2 * (W[1:-1, 2:] - 2 * W[1:-1, 1:-1] + W[1:-1, :-2])
        for j in range(n - 1):
            Ws[1:-1, j + 1] = thomas(a, b, a, rhs[:, j])
        # secondo mezzo passo: implicito in y, colonna per colonna (x fissato)
        Wn = np.zeros_like(W)
        rhs = Ws[1:-1, 1:-1] + r / 2 * (Ws[2:, 1:-1] - 2 * Ws[1:-1, 1:-1] + Ws[:-2, 1:-1])
        for i in range(n - 1):
            Wn[i + 1, 1:-1] = thomas(a, b, a, rhs[i, :])
        W = Wn
    return W


def esplicito(W, h, tau, passi):
    r = tau / h**2
    for _ in range(passi):
        W = W.copy()
        W[1:-1, 1:-1] += r * (W[2:, 1:-1] + W[:-2, 1:-1] + W[1:-1, 2:] + W[1:-1, :-2] - 4 * W[1:-1, 1:-1])
    return W


def semidiscreta(W0, h, T):
    """Soluzione esatta del problema semidiscreto (differenze centrate in x e y,
    tempo continuo): W(T) = V [ (V^T W0 V) * exp((lam_i + lam_j) T) ] V^T,
    con V autovettori (seni discreti) e lam autovalori del laplaciano discreto 1D."""
    n = W0.shape[0] - 1
    i = np.arange(1, n)
    lam = -4 / h**2 * np.sin(i * np.pi * h / 2)**2
    V = np.sqrt(2 * h) * np.sin(np.outer(i, i) * np.pi * h)      # ortonormale
    C = V.T @ W0[1:-1, 1:-1] @ V
    W = np.zeros_like(W0)
    W[1:-1, 1:-1] = V @ (C * np.exp(np.add.outer(lam, lam) * T)) @ V.T
    return W


n = 40
h = 1.0 / n
x = np.linspace(0, 1, n + 1)
W0 = serie(x, x, 0.0)
T_fin = 0.1
print(f"Condotto quadrato, griglia {n+1}x{n+1}, T = {T_fin}; U_s al centro = {W0[n//2, n//2]:.6f} (esatto 0,073671)")
W_es = serie(x, x, T_fin)
W_sd = semidiscreta(W0, h, T_fin)
print(f"   errore di discretizzazione spaziale (semidiscreta - esatta): {np.max(np.abs(W_sd - W_es)):.3e}")
print("   schema       tau       r=tau/h^2   passi   err. temporale   ordine   err. totale")
err_prec = None
for tau in (0.02, 0.01, 0.005, 0.0025, 0.00125):
    passi = int(round(T_fin / tau))
    W = adi(W0, h, tau, passi)
    err = np.max(np.abs(W - W_sd))
    ordine = f"{np.log2(err_prec / err):.2f}" if err_prec else "--"
    print(f"   ADI        {tau:8.5f}   {tau/h**2:8.2f}  {passi:6d}   {err:.3e}        {ordine:>5s}    {np.max(np.abs(W - W_es)):.3e}")
    err_prec = err
for r in (0.25, 0.26):
    tau = r * h * h
    passi = int(round(T_fin / tau))
    W = esplicito(W0, h, tau, passi)
    err = np.max(np.abs(W - W_sd))
    print(f"   esplicito  {tau:8.5f}   {r:8.2f}  {passi:6d}   {err:.3e}" + ("   (instabile)" if err > 1 else f"                 {np.max(np.abs(W - W_es)):.3e}"))
# errore spaziale: raffinando la griglia con tau = h/4 (r = n/4, grande)
print("\nConvergenza spaziale con tau = h/4 (ADI, errore rispetto alla serie esatta):")
err_prec = None
for nn in (10, 20, 40, 80):
    hh = 1.0 / nn
    xx = np.linspace(0, 1, nn + 1)
    tau = hh / 4
    W = adi(serie(xx, xx, 0.0), hh, tau, int(round(T_fin / tau)))
    err = np.max(np.abs(W - serie(xx, xx, T_fin)))
    print(f"   n = {nn:3d}: r = {tau/hh**2:5.1f}, errore max {err:.3e}" + (f", ordine {np.log2(err_prec/err):.2f}" if err_prec else ""))
    err_prec = err

# velocita' al centro e portata in funzione del tempo
Ts = np.linspace(0, 0.6, 61)
Uc = [W0[n//2, n//2] - serie(np.array([0.5]), np.array([0.5]), T)[0, 0] for T in Ts]
Q0 = np.sum(W0) * h * h
Q = [Q0 - np.sum(serie(x, x, T)) * h * h for T in Ts]
Uc = np.array(Uc) / W0[n//2, n//2]
T90, T99 = np.interp(0.9, Uc, Ts), np.interp(0.99, Uc, Ts)
print(f"\nVelocita' al centro: 90% del regime per T = {T90:.3f}, 99% per T = {T99:.3f} (T = nu t/L^2, L = lato);")
print(f"   con la semi-larghezza L/2 come nel canale piano: {4*T90:.3f} e {4*T99:.3f}"
      f" (canale piano 0,947 e 1,881; tubo circolare 0,417 e 0,814)")

fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.8))
for T, col in zip((0.005, 0.02, 0.05, 0.1, 0.3), [GRIGIO, VERDE, ARANCIO, ROSSO, BLU]):
    U = W0 - adi(W0, h, 0.0025, int(round(T / 0.0025)))
    ax[0].plot(x, U[:, n // 2], color=col, label=f"$T={T:g}$".replace(".", "{,}"))
ax[0].plot(x, W0[:, n // 2], "k--", lw=1.0, label="regime")
ax[0].set_xlabel("$x$ (mezzeria $y=0{,}5$)")
ax[0].set_ylabel("$U$")
ax[0].legend(fontsize=6.5)
ax[0].set_title("avviamento (ADI)")
ax[1].plot(Ts, Uc, color=BLU, label="centro")
ax[1].plot(Ts, np.array(Q) / Q0, "--", color=ARANCIO, label="portata")
ax[1].set_xlabel("$T=\\nu t/L^2$")
ax[1].set_ylabel("frazione del regime")
ax[1].legend(fontsize=7)
taus = np.array([0.02, 0.01, 0.005, 0.0025, 0.00125])
errs = [np.max(np.abs(adi(W0, h, t, int(round(T_fin / t))) - W_sd)) for t in taus]
ax[2].loglog(taus, errs, "o-", color=BLU, label="ADI")
ax[2].loglog(taus, errs[0] * (taus / taus[0])**2, ":", color=GRIGIO, label="$\\tau^2$")
ax[2].set_xlabel("$\\tau$")
ax[2].set_ylabel("errore temporale")
ax[2].legend(fontsize=7)
ax[2].set_title("ordine nel tempo")
fig.tight_layout()
salva(fig, "prog3_5c_adi")
