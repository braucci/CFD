"""
prog3_2_termometro.py -- PROGRAMMA 3.2
Il "termometro a lamina piana": profili di temperatura di similitudine nello
strato limite su una lamina adiabatica, per acqua, aria e mercurio.

Equazione dell'energia ridotta (theta = (T - T1)/(U^2/2cp)):
    theta'' + (Pr/2) f theta' = -2 Pr (f'')^2,   theta'(0) = 0,  theta(eta_max) = 0
discretizzata con differenze centrate su n intervalli; la condizione di
derivata nulla alla parete si impone con un nodo fittizio (theta_0 = theta_2).
Il sistema tridiagonale si risolve con TRID. Le funzioni f e f'' vengono da
un'integrazione di Blasius con f''(0) = 0,3320572 (Programma 3.1).
"""
import numpy as np
import matplotlib.pyplot as plt
from metodi_numerici import trid, rk4_integra
from stile_grafici import salva, BLU, ARANCIO, VERDE

N = 100
ETAMAX = 10.0
H = ETAMAX / N
PR = {"acqua": 6.75, "aria": 0.714, "mercurio": 0.044}


def blasius(eta, z):
    f, p, q = z
    return np.array([p, q, -0.5 * f * q])


eta, Z = rk4_integra(blasius, np.array([0.0, 0.0, 0.3320572]), ETAMAX, H)
f, q = Z[:, 0], Z[:, 2]


def temperatura(Pr, f, q, H):
    """Risolve l'equazione dell'energia sui nodi eta_1..eta_n (indici 0..n-1);
    theta_(n+1) = 0 e' la condizione 'all'infinito'."""
    n = len(f) - 1
    C = np.zeros((n, 4))
    C[:, 0] = 1.0 - H / 4.0 * Pr * f[:n]       # coefficienti (2.2.11) con A = Pr f/2
    C[:, 1] = -2.0
    C[:, 2] = 1.0 + H / 4.0 * Pr * f[:n]
    C[:, 3] = -2.0 * H**2 * Pr * q[:n] ** 2
    # condizione di derivata alla parete: theta_0 = theta_2
    C[0, 2] = C[0, 0] + C[0, 2]
    C[0, 0] = 0.0
    # condizione all'estremo superiore: theta_(n+1) = 0
    theta_sup = 0.0
    C[n - 1, 3] -= C[n - 1, 2] * theta_sup
    C[n - 1, 2] = 0.0
    theta = np.zeros(n + 1)
    theta[:n] = trid(C)
    theta[n] = theta_sup
    return theta


risultati = {nome: temperatura(Pr, f, q, H) for nome, Pr in PR.items()}

print("     eta    theta(acqua)  theta(aria)  theta(mercurio)")
for i in range(0, N + 1, 5):
    print(f"   {eta[i]:5.2f}   {risultati['acqua'][i]:10.4e}  {risultati['aria'][i]:10.4e}"
          f"  {risultati['mercurio'][i]:10.4e}")
print("\nFattore di recupero r = theta(0) e confronto con sqrt(Pr):")
for nome, Pr in PR.items():
    r = risultati[nome][0]
    print(f"   {nome:9s} Pr = {Pr:6.3f}   r = {r:.5f}   sqrt(Pr) = {np.sqrt(Pr):.5f}")
print(f"   (Pr = 1: r = {temperatura(1.0, f, q, H)[0]:.5f})")

# esempio: aria a 100 m/s e T1 = 288 K
U, cp = 100.0, 1005.0
r = risultati["aria"][0]
print(f"\nAria, U = {U} m/s: aumento di temperatura della parete Tw - T1 = r U^2/(2 cp) = "
      f"{r*U**2/(2*cp):.2f} K (isentropico: {U**2/(2*cp):.2f} K)")

fig, ax = plt.subplots(figsize=(4.6, 3.3))
for (nome, Pr), col, mk in zip(PR.items(), [BLU, ARANCIO, VERDE], ["d", "x", "o"]):
    ax.plot(eta, risultati[nome], color=col, label=f"{nome} ($Pr={Pr}$)")
    ax.plot(eta[::5], risultati[nome][::5], mk, ms=3.5, color=col)
ax.set_xlabel(r"$\eta=y\sqrt{U/(\nu x)}$")
ax.set_ylabel(r"$\theta=(T-T_1)/(U^2/2c_p)$")
ax.set_xlim(0, 10)
ax.legend()
salva(fig, "prog3_2_termometro")
