"""
prog3_4b_accuratezza.py -- accuratezza e stabilita' dello schema esplicito
per l'equazione della diffusione u_t = u_yy (variabili adimensionali).

1) Fattore di amplificazione G = 1 - 4 R sin^2(k h/2) confrontato con quello
   esatto exp(-R (k h)^2) per diversi R: per R > 1/2 si ha |G| > 1.
2) Errore al variare di R a passo h fisso, con dato iniziale regolare
   u(y, 0) = sin(pi y) (soluzione esatta exp(-pi^2 t) sin(pi y)): l'errore
   ha un minimo marcato per R = 1/6, dove il termine principale dell'errore di
   troncamento, (h^2/12)(6R - 1) u_yyyy, si annulla.
3) Ordine di convergenza in h per R = 1/4 (ordine 2) e R = 1/6 (ordine 4).
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO


def ftcs(h, R, t_fin):
    n = int(round(1.0 / h))
    y = np.linspace(0.0, 1.0, n + 1)
    u = np.sin(np.pi * y)
    tau = R * h * h
    passi = int(round(t_fin / tau))
    for _ in range(passi):
        u[1:-1] = u[1:-1] + R * (u[:-2] - 2 * u[1:-1] + u[2:])
    t = passi * tau
    return np.max(np.abs(u - np.exp(-np.pi**2 * t) * np.sin(np.pi * y)))


# 1) fattore di amplificazione
kh = np.linspace(0, np.pi, 200)
# 2) errore in funzione di R
Rv = np.linspace(0.02, 0.5, 49)
h0 = 1.0 / 20
err_R = []
for R in Rv:
    # tempo finale multiplo del passo, vicino a 0,1
    err_R.append(ftcs(h0, R, 0.1))
err_R = np.array(err_R)
print(f"Errore a t = 0,1 con h = 1/20: minimo per R = {Rv[np.argmin(err_R)]:.3f}  (1/6 = {1/6:.3f})")
for R in [0.1, 1 / 6, 0.25, 0.4, 0.5]:
    print(f"   R = {R:.4f}   max|u - u_es| = {ftcs(h0, R, 0.1):.3e}")

# 3) ordine di convergenza (tempo finale fisso t = 0,1 raggiunto esattamente)
print("\nOrdine di convergenza in h (t = 0,1):")
passi = [1 / 10, 1 / 20, 1 / 40, 1 / 80]
risultati = {}
for R in [0.25, 1 / 6]:
    e = []
    for h in passi:
        # si sceglie R' vicino a R tale che t_fin / tau sia intero
        n_t = int(round(0.1 / (R * h * h)))
        e.append(ftcs(h, 0.1 / (n_t * h * h), 0.1))
    risultati[R] = e
    ordini = [np.log2(e[k - 1] / e[k]) for k in range(1, len(e))]
    print(f"   R = {R:.4f}: errori " + ", ".join(f"{x:.2e}" for x in e) +
          "   ordini " + ", ".join(f"{o:.2f}" for o in ordini))

fig, ax = plt.subplots(1, 3, figsize=(7.8, 2.7))
for R, col in zip([1 / 6, 0.25, 0.5, 0.6], [VERDE, BLU, ARANCIO, ROSSO]):
    ax[0].plot(kh, 1 - 4 * R * np.sin(kh / 2) ** 2, color=col, label=f"$R={R:.3g}$")
    ax[0].plot(kh, np.exp(-R * kh**2), ":", color=col)
ax[0].axhline(-1, color=GRIGIO, lw=0.8)
ax[0].set_xlabel("$kh$")
ax[0].set_ylabel("$G$")
ax[0].set_title("continue: schema; punteggiate: esatto")
ax[0].legend(fontsize=6.5)
ax[1].semilogy(Rv, err_R, color=BLU)
ax[1].axvline(1 / 6, color=VERDE, ls=":")
ax[1].set_xlabel("$R=\\nu\\tau/h^2$")
ax[1].set_ylabel("errore massimo a $t=0{,}1$")
ax[1].set_title("$h=1/20$")
hh = np.array(passi)
ax[2].loglog(hh, risultati[0.25], "o-", color=BLU, label="$R=1/4$")
ax[2].loglog(hh, risultati[1 / 6], "s-", color=VERDE, label="$R=1/6$")
ax[2].loglog(hh, risultati[0.25][0] * (hh / hh[0]) ** 2, ":", color="k", label="$h^2$")
ax[2].loglog(hh, risultati[1 / 6][0] * (hh / hh[0]) ** 4, "--", color="k", label="$h^4$")
ax[2].set_xlabel("$h$")
ax[2].set_ylabel("errore massimo")
ax[2].legend(fontsize=6.5)
fig.tight_layout()
salva(fig, "prog3_4b_accuratezza")
