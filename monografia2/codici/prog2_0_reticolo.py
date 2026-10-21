"""
prog2_0_reticolo.py -- il reticolo di flusso (linee phi = cost e psi = cost)
di un semicorpo di Rankine (corrente uniforme + sorgente) e la verifica
numerica che phi soddisfa l'equazione di Laplace.

Uscite: figure prog2_0_reticolo.pdf e prog2_0_laplaciano.pdf, tabella a video.
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, ROSSO, GRIGIO, VERDE

U = 1.0                  # velocita' della corrente uniforme  [m/s]
SIGMA = 2.0 * np.pi      # intensita' della sorgente [m^2/s] (portata per unita' di apertura)


def phi(x, y):
    """Potenziale di velocita':  U x + sigma/(2 pi) ln r."""
    return U * x + SIGMA / (2 * np.pi) * np.log(np.hypot(x, y))


def psi(x, y):
    """Funzione di corrente:  U y + sigma/(2 pi) theta,  theta in (-pi, pi]."""
    return U * y + SIGMA / (2 * np.pi) * np.arctan2(y, x)


def velocita(x, y):
    r2 = x**2 + y**2
    u = U + SIGMA / (2 * np.pi) * x / r2
    v = SIGMA / (2 * np.pi) * y / r2
    return u, v


# --- grandezze caratteristiche del semicorpo ---------------------------------
xs = -SIGMA / (2 * np.pi * U)          # punto di ristagno
semilarghezza = SIGMA / (2 * U)        # semispessore a valle infinita
print(f"punto di ristagno       x_s = {xs:.4f} m")
print(f"semispessore asintotico h   = {semilarghezza:.4f} m  (spessore {2*semilarghezza:.4f} m)")

# --- reticolo di flusso --------------------------------------------------------
x = np.linspace(-4, 6, 501)
y = np.linspace(-4, 4, 401)
X, Y = np.meshgrid(x, y, indexing="ij")
with np.errstate(divide="ignore"):
    PHI, PSI = phi(X, Y), psi(X, Y)
# i punti interni al corpo (|psi| < sigma/2) sono mascherati: li' il campo non
# interessa e il taglio di arctan2 sul semiasse x<0 produrrebbe un salto di psi
interno = np.abs(PSI) < SIGMA / 2
PSIm = np.where(interno, np.nan, PSI)
PHIm = np.where(interno, np.nan, PHI)

fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.contour(X, Y, PSIm, levels=np.arange(-7.0, 7.01, 0.5), colors=BLU, linewidths=0.8,
           linestyles="solid")
ax.contour(X, Y, PHIm, levels=np.arange(-6.0, 9.01, 0.5), colors=ARANCIO,
           linewidths=0.6, linestyles="--")
# linea di corrente divisoria psi = +-sigma/2: contorno del semicorpo
# r(theta) = sigma (pi - |theta|) / (2 pi U |sin theta|)
t = np.linspace(-np.pi + 1e-3, np.pi - 1e-3, 800)      # theta del contorno
rc = SIGMA / (2 * U) * (np.pi - np.abs(t)) / (np.pi * np.abs(np.sin(t)) + 1e-15)
ax.fill(rc * np.cos(t), rc * np.sin(t), color=GRIGIO, alpha=0.25, lw=0)
ax.plot(rc * np.cos(t), rc * np.sin(t), color=ROSSO, lw=1.6)
ax.plot([xs], [0], "o", color=ROSSO, ms=4)
ax.plot([0], [0], "+", color="k", ms=7)
ax.set_xlim(-4, 6); ax.set_ylim(-4, 4); ax.set_aspect("equal")
ax.set_xlabel("$x$ [m]"); ax.set_ylabel("$y$ [m]")
salva(fig, "prog2_0_reticolo")

# --- verifica: il laplaciano discreto di phi tende a zero come h^2 -----------
x0, y0 = 1.0, 1.5                 # punto di prova nel campo di moto
print("\n      h        lapl. discreto       |errore|     rapporto")
hh = 0.4 / 2.0**np.arange(0, 7)
errori = []
for h in hh:
    lap = (phi(x0 + h, y0) + phi(x0 - h, y0) + phi(x0, y0 + h) + phi(x0, y0 - h)
           - 4 * phi(x0, y0)) / h**2
    errori.append(abs(lap))
for k, h in enumerate(hh):
    rapp = errori[k - 1] / errori[k] if k else np.nan
    print(f"  {h:8.5f}   {errori[k]:+.6e}   {errori[k]:.3e}   {rapp:6.2f}")

fig, ax = plt.subplots(figsize=(4.6, 3.3))
ax.loglog(hh, errori, "o-", color=BLU, label="laplaciano discreto di $\\phi$")
ax.loglog(hh, errori[0] * (hh / hh[0])**2, "--", color=GRIGIO, label="pendenza 2")
ax.set_xlabel("$h$ [m]"); ax.set_ylabel(r"$|\nabla_h^2\phi|$ [1/s]")
ax.legend()
salva(fig, "prog2_0_laplaciano")

# --- coefficiente di pressione sul contorno del corpo (Bernoulli) ------------
uc, vc = velocita(rc * np.cos(t), rc * np.sin(t))
cp = 1 - (uc**2 + vc**2) / U**2
k = np.argmin(cp)
print(f"\nC_p minimo sul corpo = {cp[k]:.4f}  in x = {rc[k]*np.cos(t[k]):.3f} m")
