"""
prog3_2b_prandtl.py -- influenza del numero di Prandtl sullo strato limite
termico della lamina piana: soluzioni "per quadratura".

Posto G(eta) = (1/2) int_0^eta f, la soluzione di Blasius da' f'' = q0 exp(-G).
1) Lamina adiabatica (termometro): l'equazione theta'' + (Pr/2) f theta' =
   -2 Pr (f'')^2 con theta'(0) = 0 e theta(inf) = 0 si integra esattamente:
       r = theta(0) = 2 Pr q0^2 int_0^inf int_0^eta exp[-Pr (G(eta)-G(s))] exp[-2 G(s)] ds deta
2) Lamina isoterma (problema di Pohlhausen): theta_T'' + (Pr/2) f theta_T' = 0,
   theta_T(0) = 1, theta_T(inf) = 0, da cui il numero di Nusselt locale
       Nu_x / sqrt(Re_x) = -theta_T'(0) = 1 / int_0^inf exp(-Pr G) deta .
Confronto con le differenze finite del Programma 3.2 e con le leggi
approssimate r ~ sqrt(Pr), Nu_x ~ 0,332 Pr^(1/3) Re_x^(1/2).
"""
import numpy as np
import matplotlib.pyplot as plt
from metodi_numerici import trid, rk4_integra
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

Q0 = 0.332057336215          # f''(0) (Programma 3.1b)
BETA = 1.7207876575          # f ~ eta - BETA per eta grande
H = 0.005


def blasius(eta, z):
    f, p, q = z
    return np.array([p, q, -0.5 * f * q])


_eta10, _Z = rk4_integra(blasius, np.array([0.0, 0.0, Q0]), 10.0, H)
_f10 = _Z[:, 0]


def griglia(etamax):
    """eta, f e G su [0, etamax]; oltre eta = 10 si usa f = eta - BETA."""
    eta = np.arange(0.0, etamax + H / 2, H)
    f = np.where(eta <= 10.0, np.interp(eta, _eta10, _f10), eta - BETA)
    G = np.concatenate([[0.0], np.cumsum(0.5 * H * (f[1:] + f[:-1]))]) * 0.5
    return eta, f, G


def recupero_quadratura(Pr):
    etamax = max(14.0, 14.0 / np.sqrt(Pr))
    eta, f, G = griglia(etamax)
    # J(eta) = int_0^eta exp[-Pr (G(eta)-G(s))] exp(-2 G(s)) ds, per ricorrenza
    J = np.zeros_like(eta)
    e2 = np.exp(-2.0 * G)
    for k in range(len(eta) - 1):
        fatt = np.exp(-Pr * (G[k + 1] - G[k]))
        J[k + 1] = fatt * J[k] + 0.5 * H * (fatt * e2[k] + e2[k + 1])
    return 2.0 * Pr * Q0**2 * np.trapezoid(J, eta)


def nusselt_quadratura(Pr):
    etamax = max(14.0, 14.0 / np.sqrt(Pr))
    eta, f, G = griglia(etamax)
    return 1.0 / np.trapezoid(np.exp(-Pr * G), eta)


def recupero_differenze(Pr, etamax, n):
    """Schema del Programma 3.2 su [0, etamax] con n intervalli."""
    eta = np.linspace(0.0, etamax, n + 1)
    h = eta[1]
    eta_f, f_f, G_f = griglia(etamax)
    f = np.interp(eta, eta_f, f_f)
    q = Q0 * np.exp(-np.interp(eta, eta_f, G_f))
    C = np.zeros((n, 4))
    C[:, 0] = 1.0 - h / 4.0 * Pr * f[:n]
    C[:, 1] = -2.0
    C[:, 2] = 1.0 + h / 4.0 * Pr * f[:n]
    C[:, 3] = -2.0 * h**2 * Pr * q[:n] ** 2
    C[0, 2] += C[0, 0]
    C[0, 0] = 0.0
    C[n - 1, 2] = 0.0
    return trid(C)[0]


print("1) Verifica per Pr = 1 (deve essere r = 1 e Nu/sqrt(Re) = f''(0)):")
print(f"   r = {recupero_quadratura(1.0):.6f},  Nu/sqrt(Re) = {nusselt_quadratura(1.0):.6f}")

print("\n2) Mercurio (Pr = 0,044): effetto dell'estremo eta_max sulle differenze finite (h = 0,1)")
r_merc = recupero_quadratura(0.044)
for emax in [10.0, 20.0, 40.0, 60.0]:
    r = recupero_differenze(0.044, emax, int(round(emax / 0.1)))
    print(f"   eta_max = {emax:5.1f}   r = {r:.5f}   (quadratura: {r_merc:.5f})")

print("\n3) Fattore di recupero e numero di Nusselt al variare di Pr:")
print("      Pr       r(quadr.)   sqrt(Pr)   Nu/sqrt(Re)   0,332 Pr^(1/3)")
Pr_tab = [0.01, 0.044, 0.1, 0.3, 0.714, 1.0, 2.0, 6.75, 10.0, 50.0, 100.0]
for Pr in Pr_tab:
    r = recupero_quadratura(Pr)
    nu = nusselt_quadratura(Pr)
    print(f"   {Pr:7.3f}   {r:9.5f}   {np.sqrt(Pr):8.5f}   {nu:10.5f}   {0.332*Pr**(1/3):10.5f}")

Pr = np.logspace(-2, 2, 33)
r = np.array([recupero_quadratura(p) for p in Pr])
nu = np.array([nusselt_quadratura(p) for p in Pr])
# esponenti locali alle due estremita'
print(f"\n   pendenza d(ln r)/d(ln Pr):   Pr = 0,01 -> {np.gradient(np.log(r), np.log(Pr))[0]:.3f},"
      f"  Pr = 100 -> {np.gradient(np.log(r), np.log(Pr))[-1]:.3f}")
print(f"   pendenza d(ln Nu)/d(ln Pr):  Pr = 0,01 -> {np.gradient(np.log(nu), np.log(Pr))[0]:.3f},"
      f"  Pr = 100 -> {np.gradient(np.log(nu), np.log(Pr))[-1]:.3f}")
print(f"   r/Pr^(1/3) a Pr = 100: {r[-1]/100**(1/3):.4f};  Nu/sqrt(Pr Re) a Pr = 0,01: {nu[0]/np.sqrt(0.01):.4f}"
      f"  (1/sqrt(pi) = {1/np.sqrt(np.pi):.4f})")

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax[0].loglog(Pr, r, color=BLU, label="quadratura esatta")
ax[0].loglog(Pr, np.sqrt(Pr), "--", color=ARANCIO, label=r"$\sqrt{Pr}$")
ax[0].loglog(Pr_tab[1::5], [recupero_differenze(p, max(10, 14/np.sqrt(p)), int(max(10, 14/np.sqrt(p))/0.05))
                            for p in Pr_tab[1::5]], "o", ms=4, color=VERDE, label="differenze finite")
ax[0].set_xlabel("$Pr$")
ax[0].set_ylabel(r"fattore di recupero $r=\theta(0)$")
ax[0].legend()
ax[1].loglog(Pr, nu, color=BLU, label="quadratura esatta")
ax[1].loglog(Pr, 0.332 * Pr ** (1 / 3), "--", color=ARANCIO, label=r"$0{,}332\,Pr^{1/3}$")
ax[1].loglog(Pr, Pr ** 0.5 / np.sqrt(np.pi), ":", color=ROSSO, label=r"$\sqrt{Pr/\pi}$")
ax[1].set_xlabel("$Pr$")
ax[1].set_ylabel(r"$Nu_x/\sqrt{Re_x}$ (parete isoterma)")
ax[1].legend()
fig.tight_layout()
salva(fig, "prog3_2b_prandtl")
