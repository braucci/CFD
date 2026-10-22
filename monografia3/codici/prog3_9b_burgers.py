"""
prog3_9b_burgers.py -- equazione di Burgers con il metodo pseudo-spettrale di
Fourier: aliasing, regola dei 3/2 e trasformata rapida.

    u_t + u u_x = nu u_xx,   0 <= x < 2 pi periodico,   u(x, 0) = sin x.

Nello spazio dei numeri d'onda:  d u^_k/dt = -(u u_x)^_k - nu k^2 u^_k.
Il termine non lineare si calcola nello spazio fisico (prodotto u * u_x) e si
ritrasforma: il prodotto di due polinomi trigonometrici di grado N/2 ha grado N,
e i modi oltre quello di Nyquist si "ripiegano" (aliasing) sui modi bassi. Con
la regola dei 3/2 (Orszag) il prodotto si calcola su una griglia di M = 3N/2
nodi, riempiendo di zeri i coefficienti aggiunti, e si troncano poi i modi
oltre N/2: il risultato e' esente da aliasing.
Avanzamento nel tempo: Runge-Kutta del quarto ordine con fattore integrante
exp(-nu k^2 t) per il termine diffusivo (nessun vincolo diffusivo sul passo).
Soluzione esatta: trasformazione di Cole-Hopf, u = -2 nu phi_x/phi, con phi
soluzione dell'equazione del calore, calcolata per quadratura.
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import quad
from metodi_numerici import numeri_onda
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

NU = 0.05


def esatta(x, t, nu=NU):
    """Cole-Hopf: phi(x, t) = int phi0(y) G(x - y, t) dy, phi0 = exp(-(1 - cos y)/(2 nu))
    (dato iniziale u0 = sin x), G nucleo del calore periodico."""
    def phi_e_phix(xx):
        def nucleo(y):                     # somma delle immagini periodiche del nucleo gaussiano
            s = sum(np.exp(-(xx - y - 2 * np.pi * m)**2 / (4 * nu * t)) for m in range(-3, 4))
            return s
        def dnucleo(y):
            return sum(-(xx - y - 2 * np.pi * m) / (2 * nu * t) * np.exp(-(xx - y - 2 * np.pi * m)**2 / (4 * nu * t))
                       for m in range(-3, 4))
        f0 = lambda y: np.exp(-(1 - np.cos(y)) / (2 * nu))
        p = quad(lambda y: f0(y) * nucleo(y), 0, 2 * np.pi, limit=200)[0]
        px = quad(lambda y: f0(y) * dnucleo(y), 0, 2 * np.pi, limit=200)[0]
        return p, px
    u = np.empty_like(x)
    for i, xx in enumerate(x):
        p, px = phi_e_phix(xx)
        u[i] = -2 * nu * px / p
    return u


def burgers(N, t_fin, dt, dealias, nu=NU):
    x = 2 * np.pi * np.arange(N) / N
    k = numeri_onda(N, 2 * np.pi)
    uh = np.fft.fft(np.sin(x))
    M = 3 * N // 2
    kM = numeri_onda(M, 2 * np.pi)

    def nonlin(uh):
        """Trasformata di -(u u_x), con o senza de-aliasing (regola dei 3/2)."""
        if not dealias:
            u = np.fft.ifft(uh).real
            ux = np.fft.ifft(1j * k * uh).real
            return -np.fft.fft(u * ux)
        uhM = np.zeros(M, dtype=complex)                 # riempimento con zeri: N -> M modi
        uhM[:N // 2] = uh[:N // 2]
        uhM[M - N // 2:] = uh[N // 2:]
        uM = np.fft.ifft(uhM).real * M / N
        uxM = np.fft.ifft(1j * kM * uhM).real * M / N
        wM = np.fft.fft(uM * uxM) * N / M
        w = np.zeros(N, dtype=complex)                   # troncamento: M -> N modi
        w[:N // 2] = wM[:N // 2]
        w[N // 2:] = wM[M - N // 2:]
        return -w

    E = np.exp(-nu * k**2 * dt / 2)                      # fattore integrante su mezzo passo
    passi = int(round(t_fin / dt))
    with np.errstate(over="ignore", invalid="ignore"):   # un calcolo instabile produce inf/nan
        for _ in range(passi):
            a = dt * nonlin(uh)
            b = dt * nonlin(E * (uh + a / 2))
            c = dt * nonlin(E * uh + b / 2)
            d = dt * nonlin(E * E * uh + E * c)
            uh = E * E * uh + (E * E * a + 2 * E * (b + c) + d) / 6
            if not np.all(np.isfinite(uh)):
                return x, None
    return x, np.fft.ifft(uh).real


# ---------------------------------------------------------------- convergenza (nu = 0,05)
t_fin = 1.5
xr = np.linspace(0, 2 * np.pi, 65)[:-1]
u_rif = esatta(xr, t_fin)
print(f"Burgers, nu = {NU}, t = {t_fin}: pendenza massima della soluzione esatta "
      f"{np.max(np.abs(np.gradient(u_rif, xr))):.2f} (iniziale 1)")
print("     N     dt      errore senza de-aliasing   errore con regola 3/2")
Ns = (16, 32, 64, 128, 256)
ea, ed = [], []
for N in Ns:
    dt = 0.5 / N
    x, ua = burgers(N, t_fin, dt, dealias=False)
    x, ud = burgers(N, t_fin, dt, dealias=True)
    ue = esatta(x[::max(N // 64, 1)], t_fin)
    ea.append(np.max(np.abs(ua[::max(N // 64, 1)] - ue)))
    ed.append(np.max(np.abs(ud[::max(N // 64, 1)] - ue)))
    print(f"   {N:4d}   {dt:.4f}      {ea[-1]:11.3e}                {ed[-1]:11.3e}")
# spettro a t = 1,5
N = 128
x, u = burgers(N, t_fin, 0.5 / N, dealias=True)
uh = np.fft.fft(u) / N
print(f"   coefficienti |u^_k| a t = {t_fin} (N = 128): k = 1: {abs(uh[1]):.3e}, 8: {abs(uh[8]):.3e},"
      f" 16: {abs(uh[16]):.3e}, 32: {abs(uh[32]):.3e}, 48: {abs(uh[48]):.3e}, 63: {abs(uh[63]):.3e}")

# ---------------------------------------------------------------- aliasing a viscosita' piccola
print("\nViscosita' piccola (nu = 0,001), t = 2 (l'urto si forma per t = 1):")
print("     N     senza de-aliasing              regola dei 3/2")
for N in (64, 128, 256, 512):
    riga = f"   {N:4d}"
    for da in (False, True):
        x, u = burgers(N, 2.0, 0.25 / N, da, nu=0.001)
        riga += "     instabile                " if u is None else f"     max|u| = {np.max(np.abs(u)):.3f}, E = {np.sum(u**2)/N:.4f}"
    print(riga)
print("   (E = media di u^2; vale 0,5 a t = 0 e, per nu -> 0 con l'urto, decresce lentamente)")

# ---------------------------------------------------------------- trasformata rapida "a mano"
def fft_ricorsiva(f):
    """FFT radix-2 (Cooley-Tukey): T(N) = 2 T(N/2) + O(N)  =>  O(N log N)."""
    N = len(f)
    if N == 1:
        return f.astype(complex)
    pari, dispari = fft_ricorsiva(f[0::2]), fft_ricorsiva(f[1::2])
    w = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    return np.concatenate([pari + w * dispari, pari - w * dispari])


def dft_diretta(f):
    N = len(f)
    n = np.arange(N)
    return np.exp(-2j * np.pi * np.outer(n, n) / N) @ f


print("\nTrasformata rapida ricorsiva contro somma diretta:")
for N in (64, 256, 1024, 4096):
    f = np.random.default_rng(0).standard_normal(N)
    t0 = time.perf_counter(); F1 = dft_diretta(f); t1 = time.perf_counter() - t0
    t0 = time.perf_counter(); F2 = fft_ricorsiva(f); t2 = time.perf_counter() - t0
    t0 = time.perf_counter(); F3 = np.fft.fft(f); t3 = time.perf_counter() - t0
    print(f"   N = {N:5d}: scarto max {np.max(np.abs(F2 - F3)):.1e}; tempi: diretta {t1*1e3:7.2f} ms,"
          f" ricorsiva {t2*1e3:6.2f} ms, numpy {t3*1e6:6.1f} us;  N^2/(N log2 N) = {N/np.log2(N):.0f}")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.8))
xe = np.linspace(0, 2 * np.pi, 201)
for t, col in zip((0.0, 0.5, 1.0, 1.5, 3.0), (GRIGIO, VERDE, ARANCIO, ROSSO, BLU)):
    ax[0].plot(xe, esatta(xe, t) if t > 0 else np.sin(xe), color=col, label=f"$t={t:g}$".replace(".", "{,}"))
x, u = burgers(32, 1.5, 0.5 / 32, dealias=True)
ax[0].plot(x, u, "o", color=ROSSO, ms=2.5, mfc="none")
ax[0].set_xlabel("$x$")
ax[0].set_ylabel("$u$")
ax[0].legend(fontsize=6.5)
ax[0].set_title("soluzione esatta e $N=32$ (cerchi)")
ax[1].semilogy(Ns, np.maximum(ea, 1e-16), "s--", color=ROSSO, label="senza de-aliasing")
ax[1].semilogy(Ns, np.maximum(ed, 1e-16), "o-", color=BLU, label="regola dei 3/2")
ax[1].set_xlabel("$N$")
ax[1].set_ylabel("errore massimo a $t=1{,}5$")
ax[1].legend(fontsize=7)
ax[1].set_title("convergenza")
x2, u2 = burgers(256, 2.0, 0.25 / 256, True, nu=0.001)
x1, u1 = burgers(64, 2.0, 0.25 / 64, False, nu=0.001)
ax[2].plot(x2, u2, color=BLU, lw=1.0, label="$N=256$, 3/2")
ax[2].plot(x1, u1, "--", color=ROSSO, lw=1.0, label="$N=64$, con aliasing")
ax[2].set_xlabel("$x$")
ax[2].set_ylabel("$u$")
ax[2].legend(fontsize=7)
ax[2].set_title("$\\nu=0{,}001$, $t=2$")
fig.tight_layout()
salva(fig, "prog3_9b_burgers")
