"""
prog0_5_matplotlib.py -- Programma 0.5: le figure con Matplotlib.

Catalogo dei grafici usati nelle monografie, costruiti sulla corrente potenziale
intorno a un cilindro con circolazione (Modulo 2):
    psi = U (r - a^2/r) sin(theta) - Gamma/(2 pi) ln(r/a),   C_p = 1 - (V/U)^2.
1) figura e assi (plt.subplots), plot con stili, marcatori, etichette in LaTeX,
   legenda, griglia, limiti; 2) scale logaritmiche (loglog/semilogy) e rette di
   pendenza; 3) piu' pannelli e condivisione degli assi; 4) campi 2D: contour,
   contourf con barra dei colori, streamplot, quiver, aspect; 5) annotazioni,
   linee di riferimento, testo; 6) salvataggio in PDF con la funzione salva() del
   modulo stile_grafici (e che cosa fa quel modulo).
Verifica: il C_p minimo sul cilindro senza circolazione e' -3; con circolazione
Gamma = 2 pi U a i punti di ristagno sono in theta = -pi/6 e -5pi/6 (sin theta = -1/2).
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO, PALETTE

U, A = 1.0, 1.0


def campo(X, Y, gamma=0.0):
    """Componenti u, v e funzione di corrente della corrente intorno al cilindro con circolazione gamma."""
    r2 = np.maximum(X**2 + Y**2, 1e-12)          # evita la divisione per zero nell'origine (dentro il cilindro)
    u = U * (1 - A**2 * (X**2 - Y**2) / r2**2) + gamma / (2 * np.pi) * Y / r2
    v = -U * 2 * A**2 * X * Y / r2**2 - gamma / (2 * np.pi) * X / r2
    psi = U * Y * (1 - A**2 / r2) - gamma / (2 * np.pi) * np.log(np.sqrt(r2) / A)
    return u, v, psi


# ------------------------------------------------------------------ 1) grafico di base: C_p sul cilindro
theta = np.linspace(0, 2 * np.pi, 361)
fig, ax = plt.subplots(figsize=(5.0, 3.0))                # una figura con un solo sistema di assi
for gamma, col, stile in ((0.0, BLU, "-"), (np.pi, ARANCIO, "--"), (2 * np.pi, VERDE, ":")):
    Vt = 2 * U * np.sin(theta) + gamma / (2 * np.pi * A)   # velocita' tangenziale sulla superficie
    cp = 1 - (Vt / U)**2
    ax.plot(np.degrees(theta), cp, stile, color=col, lw=1.4, label=f"$\\Gamma = {gamma / np.pi:g}\\pi Ua$".replace("0\\pi", "0"))
ax.axhline(1, color=GRIGIO, lw=0.6)
ax.set_xlabel("$\\theta$ [gradi]"); ax.set_ylabel("$C_p$")
ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 90))
ax.legend(fontsize=8, loc="lower right")
ax.set_title("coefficiente di pressione sul cilindro", fontsize=9)
fig.tight_layout()
salva(fig, "prog0_5_cp")
print(f"C_p minimo senza circolazione: {np.min(1 - (2 * np.sin(theta))**2):.1f} (esatto -3)")
Vt = 2 * U * np.sin(theta) + 1.0
print(f"punti di ristagno per Gamma = 2 pi U a: theta = {np.degrees(theta[np.argsort(np.abs(Vt))[:2]])} gradi (esatti 210 e 330)")

# ------------------------------------------------------------------ 2) scale logaritmiche: ordine di convergenza
h = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])
err1, err2 = 0.5 * h, 0.3 * h**2
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
ax[0].loglog(h, err1, "o-", color=BLU, label="ordine 1")
ax[0].loglog(h, err2, "s-", color=ARANCIO, label="ordine 2")
ax[0].loglog(h, err2[0] * (h / h[0])**2, ":", color=GRIGIO, label="pendenza 2")
ax[0].set_xlabel("$h$"); ax[0].set_ylabel("errore"); ax[0].legend(fontsize=7)
ax[0].invert_xaxis()                                       # h decresce verso destra
t = np.linspace(0, 10, 200)
ax[1].semilogy(t, np.exp(-t), color=BLU, label="$e^{-t}$")
ax[1].semilogy(t, np.exp(-t / 3), color=ARANCIO, label="$e^{-t/3}$")
ax[1].set_xlabel("$t$"); ax[1].set_ylabel("residuo"); ax[1].legend(fontsize=7); ax[1].grid(True, which="both", alpha=0.3)
fig.tight_layout()
salva(fig, "prog0_5_log")

# ------------------------------------------------------------------ 3) e 4) campi 2D
n = 201
x = np.linspace(-3, 3, n); y = np.linspace(-2.2, 2.2, n)
X, Y = np.meshgrid(x, y, indexing="ij")
fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.0))
for ax, gamma, titolo in zip(axs, (0.0, 2 * np.pi), ("$\\Gamma=0$", "$\\Gamma=2\\pi Ua$")):
    u, v, psi = campo(X, Y, gamma)
    dentro = X**2 + Y**2 < A**2
    psi = np.where(dentro, np.nan, psi)                    # niente linee dentro il cilindro
    livelli = np.linspace(-3, 3, 25)
    ax.contour(X, Y, psi, levels=livelli, colors=BLU, linewidths=0.6, linestyles="solid")
    ax.contour(X, Y, psi, levels=[0.0], colors=ROSSO, linewidths=1.0)
    ax.add_patch(plt.Circle((0, 0), A, color=GRIGIO))
    ax.set_aspect("equal"); ax.set_title(titolo, fontsize=9); ax.set_xlabel("$x/a$")
axs[0].set_ylabel("$y/a$")
u, v, psi = campo(X, Y, 0.0)
cp = np.where(X**2 + Y**2 < A**2, np.nan, 1 - (u**2 + v**2) / U**2)
cf = axs[2].contourf(X, Y, cp, levels=np.linspace(-3, 1, 17), cmap="RdBu_r")
fig.colorbar(cf, ax=axs[2], label="$C_p$", shrink=0.9)
passo = 12
axs[2].quiver(X[::passo, ::passo], Y[::passo, ::passo], u[::passo, ::passo], v[::passo, ::passo], color="k", scale=25, width=0.003)
axs[2].add_patch(plt.Circle((0, 0), A, color="w"))
axs[2].set_aspect("equal"); axs[2].set_title("$C_p$ e vettori velocità", fontsize=9); axs[2].set_xlabel("$x/a$")
axs[0].annotate("punto di ristagno", xy=(-1, 0), xytext=(-2.9, 1.6), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.6))
fig.tight_layout()
salva(fig, "prog0_5_campi")

# streamplot (richiede griglia con indexing='xy': assi scambiati rispetto a 'ij')
fig, ax = plt.subplots(figsize=(4.6, 3.2))
Xs, Ys = np.meshgrid(x, y)                                 # indexing='xy': Xs[j, i] = x_i
us, vs, _ = campo(Xs, Ys, np.pi)
dentro = Xs**2 + Ys**2 < A**2
us = np.where(dentro, np.nan, us); vs = np.where(dentro, np.nan, vs)   # niente linee dentro il cilindro
vel = np.hypot(us, vs)
sp = ax.streamplot(x, y, us, vs, color=vel, cmap="viridis", density=1.2, linewidth=0.8, arrowsize=0.7)
fig.colorbar(sp.lines, ax=ax, label="$|V|/U$")
ax.add_patch(plt.Circle((0, 0), A, color=GRIGIO)); ax.set_aspect("equal")
ax.set_xlabel("$x/a$"); ax.set_ylabel("$y/a$"); ax.set_title("streamplot, $\\Gamma=\\pi Ua$", fontsize=9)
fig.tight_layout()
salva(fig, "prog0_5_streamplot")

# ------------------------------------------------------------------ 5) piu' pannelli con assi condivisi
fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.4), sharex=True)
u, v, psi = campo(X, Y, 0.0)
campi = ((u, "$u/U$"), (v, "$v/U$"), (psi, "$\\psi/(Ua)$"), (1 - (u**2 + v**2), "$C_p$"))
for ax, (f, nome), col in zip(axs.ravel(), campi, PALETTE):
    for off, ls in ((0.0, "-"), (0.5, "--"), (1.5, ":")):
        j = np.argmin(np.abs(y - off))
        ax.plot(x, np.where(np.abs(x) < np.sqrt(max(A**2 - off**2, 0)), np.nan, f[:, j]), ls, color=col, label=f"$y/a={off:g}$")
    ax.set_ylabel(nome); ax.axvline(-1, color=GRIGIO, lw=0.5); ax.axvline(1, color=GRIGIO, lw=0.5)
axs[0, 0].legend(fontsize=6)
for ax in axs[1]:
    ax.set_xlabel("$x/a$")
fig.suptitle("profili lungo rette orizzontali ($\\Gamma=0$)", fontsize=9)
fig.tight_layout()
salva(fig, "prog0_5_pannelli")
print("figure salvate: prog0_5_cp, prog0_5_log, prog0_5_campi, prog0_5_streamplot, prog0_5_pannelli")
