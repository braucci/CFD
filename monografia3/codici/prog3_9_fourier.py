"""
prog3_9_fourier.py -- PROGRAMMA 3.9
Metodo di Fourier (trasformata discreta) per un problema ai limiti periodico:
    u'' - u' - 2 u = 2 + 6 sin(6x) - 38 cos(6x),   0 <= x <= 2 pi,
con soluzione esatta u = cos(6x) - 1. Nello spazio di Fourier la derivata
diventa una moltiplicazione per i k_n, quindi
    (-k_n^2 - i k_n - 2) u_hat(k_n) = b_hat(k_n)   =>   u_hat = b_hat / (-k_n^2 - i k_n - 2).
Il coefficiente del modo di Nyquist (n = -N/2) si annulla.
Si studia poi la convergenza in N per tre soluzioni di regolarita' diversa:
polinomio trigonometrico, funzione analitica periodica, funzione periodica
con derivata discontinua.
"""
import numpy as np
import matplotlib.pyplot as plt
from metodi_numerici import numeri_onda
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

L = 2 * np.pi


def risolvi(b, N):
    """Risolve u'' - u' - 2u = b(x) su N nodi equispaziati di [0, 2 pi)."""
    x = L * np.arange(N) / N
    k = numeri_onda(N, L)
    bh = np.fft.fft(b(x))
    bh[N // 2] = 0.0                       # modo di Nyquist
    uh = bh / (-k**2 - 1j * k - 2.0)
    return x, np.fft.ifft(uh).real


casi = {
    "cos 6x - 1": (lambda x: np.cos(6 * x) - 1,
                   lambda x: 2 + 6 * np.sin(6 * x) - 38 * np.cos(6 * x)),
    "exp(sin x)": (lambda x: np.exp(np.sin(x)),
                   lambda x: np.exp(np.sin(x)) * (np.cos(x)**2 - np.sin(x) - np.cos(x) - 2)),
    "|sin x|^3":  (lambda x: np.abs(np.sin(x))**3,
                   None),
}
# per |sin x|^3 il secondo membro si calcola analiticamente a tratti
def b_abs3(x):
    s, c = np.sin(x), np.cos(x)
    a = np.abs(s)
    u = a**3
    du = 3 * a * s * c                     # derivata di |s|^3 = 3 |s| s c
    d2u = 3 * a * (2 * c**2 - s**2)        # derivata seconda: 6|s|c^2 - 3|s|s^2
    return d2u - du - 2 * u
casi["|sin x|^3"] = (casi["|sin x|^3"][0], b_abs3)

x64, u64 = risolvi(casi["cos 6x - 1"][1], 64)
print(f"N = 64: max |u - u_es| = {np.max(np.abs(u64 - (np.cos(6*x64) - 1))):.2e}")
Nlista = [8, 12, 14, 16, 24, 32, 48, 64, 128]
errori = {}
print("\n   N   " + "   ".join(f"{nome:>12s}" for nome in casi))
for N in Nlista:
    riga = []
    for nome, (ue, b) in casi.items():
        x, u = risolvi(b, N)
        e = np.max(np.abs(u - ue(x)))
        errori.setdefault(nome, []).append(e)
        riga.append(e)
    print(f"  {N:4d} " + "   ".join(f"{e:12.3e}" for e in riga))
e3 = errori["|sin x|^3"]
print(f"\nordine osservato per |sin x|^3 (N = 64 -> 128): {np.log2(e3[-2]/e3[-1]):.2f}")

# ------------------------------------------------------------------ figura
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
xx = np.linspace(0, L, 400)
ax[0].plot(xx, np.cos(6 * xx) - 1, color=GRIGIO, lw=1.8, label="esatta")
ax[0].plot(np.append(x64, L), np.append(u64, u64[0]), "o", ms=3, color=BLU, mfc="none",
           label="Fourier, $N=64$")
ax[0].set_xlabel("$x$")
ax[0].set_ylabel("$u(x)$")
ax[0].legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
for (nome, e), col, mk in zip(errori.items(), [BLU, VERDE, ROSSO], ["o", "s", "d"]):
    ax[1].semilogy(Nlista, np.maximum(e, 1e-17), mk + "-", color=col, label=f"$u={nome}$".replace(" x", "\\,x"))
ax[1].set_xlabel("$N$")
ax[1].set_ylabel("errore massimo")
ax[1].legend(fontsize=7)
fig.tight_layout()
salva(fig, "prog3_9_fourier")
