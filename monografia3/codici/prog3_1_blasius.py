"""
prog3_1_blasius.py -- PROGRAMMA 3.1
Strato limite laminare sulla lamina piana semi-infinita (problema di Blasius):
    f''' + (1/2) f f'' = 0,   f(0) = f'(0) = 0,   f'(inf) = 1
risolto come problema ai valori iniziali con Runge-Kutta del quarto ordine
(sistema f' = p, p' = q, q' = -f q / 2) e metodo del dimezzamento
dell'intervallo sul valore iniziale incognito q0 = f''(0).

Condizione all'infinito approssimata:  0 <= 1 - p(eta_max) <= eps.
Uscite: numero di iterazioni, f''(0), tabella f, f', f'' ogni 0,2,
spessori dello strato limite, figura dei profili u/U e v/sqrt(nu U/x).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from stile_grafici import salva, BLU, ARANCIO, GRIGIO

ETAMAX = 10.0
H = 0.1
EPSLON = 1.0e-6


def blasius(z):
    """Secondo membro del sistema: z = (f, p, q)."""
    f, p, q = z
    return np.array([p, q, -0.5 * f * q])


def integra(q0):
    """Integra da eta = 0 fino a superare ETAMAX (come il programma originale)."""
    eta = [0.0]
    Z = [np.array([0.0, 0.0, q0])]
    while eta[-1] <= ETAMAX:
        z = Z[-1]
        k1 = H * blasius(z)
        k2 = H * blasius(z + k1 / 2)
        k3 = H * blasius(z + k2 / 2)
        k4 = H * blasius(z + k3)
        Z.append(z + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0)
        eta.append(eta[-1] + H)
    return np.array(eta), np.array(Z)


# ---------------------------------------------- metodo del dimezzamento
q0, dq0 = 1.0, 0.2
iterazioni = 0
while True:
    iterazioni += 1
    q0 = q0 + dq0
    eta, Z = integra(q0)
    pmax = Z[-1, 1]
    errore = 1.0 - pmax
    if 0.0 <= errore <= EPSLON:
        break
    # si inverte il verso (e si dimezza il passo) quando si e' "scavalcato" 1
    if (pmax > 1.0 and dq0 > 0.0) or (pmax < 1.0 and dq0 < 0.0):
        dq0 = -dq0 / 2.0

f, p, q = Z[:, 0], Z[:, 1], Z[:, 2]
print(f"Numero di iterazioni = {iterazioni}")
print(f"f''(0) = q0 = {q0:.7f}    (1 - p_max = {errore:.2e})")
print("\n    eta        f          f'         f''")
for i in range(0, 101, 2):
    print(f"  {eta[i]:5.2f}  {f[i]:9.5f}  {p[i]:9.5f}  {q[i]:10.3e}")

# ---------------------------------------------- grandezze dello strato limite
v = 0.5 * (eta * p - f)                       # v / sqrt(nu U / x)
i99 = np.argmax(p >= 0.99)
eta99 = np.interp(0.99, p[i99 - 1:i99 + 1], eta[i99 - 1:i99 + 1])
i994 = np.argmax(p >= 0.994)
eta994 = np.interp(0.994, p[i994 - 1:i994 + 1], eta[i994 - 1:i994 + 1])
n10 = int(round(10.0 / H))
spost = eta[n10] - f[n10]                     # delta* sqrt(U/(nu x)) = lim (eta - f)
quant = simpson(p[:n10 + 1] * (1 - p[:n10 + 1]), x=eta[:n10 + 1])   # regola di Simpson
print(f"\nu/U = 0,99  per eta = {eta99:.3f};   u/U = 0,994 per eta = {eta994:.3f}")
print(f"spessore di spostamento   delta* sqrt(U/nu x) = {spost:.4f}")
print(f"spessore di quantita' di moto theta sqrt(U/nu x) = {quant:.4f}  (2 f''(0) = {2*q0:.4f})")
print(f"fattore di forma H = delta*/theta = {spost/quant:.4f}")
print(f"velocita' normale all'esterno v/sqrt(nu U/x) = {v[n10]:.4f}")
print(f"coefficiente d'attrito locale  C_f sqrt(Re_x) = 2 f''(0) = {2*q0:.4f}")
print(f"resistenza di una faccia       C_D sqrt(Re_L) = 4 f''(0) = {4*q0:.4f}")

# esempio numerico: aria a 30 m/s, x = 1 m
U, nu, x = 30.0, 1.49e-5, 1.0
Rex = U * x / nu
print(f"\nAria, U = {U} m/s, x = {x} m: Re_x = {Rex:.3e}")
print(f"  delta(0,994) = {eta994*np.sqrt(nu*x/U)*100:.3f} cm,  "
      f"delta* = {spost*np.sqrt(nu*x/U)*1000:.3f} mm,  v_inf = {v[n10]*np.sqrt(nu*U/x):.4f} m/s")
print(f"  tau_w = {q0*np.sqrt(U/(nu*x))*U*1.225*nu:.4f} Pa  (rho = 1,225 kg/m^3)")

# ---------------------------------------------- figura
fig, ax = plt.subplots(figsize=(4.4, 3.6))
ax.plot(p[:n10 + 1], eta[:n10 + 1], color=BLU, label=r"$u/U=f'$")
ax.plot(v[:n10 + 1], eta[:n10 + 1], color=ARANCIO, ls="--",
        label=r"$v/\sqrt{\nu U/x}=\frac{1}{2}(\eta f'-f)$")
ax.plot(p[:n10 + 1:4], eta[:n10 + 1:4], "o", ms=3, color=BLU)
ax.plot(v[:n10 + 1:4], eta[:n10 + 1:4], "+", ms=4, color=ARANCIO)
ax.axhline(eta994, color=GRIGIO, lw=0.8, ls=":")
ax.text(0.55, eta994 + 0.08, r"$u/U=0{,}994$", fontsize=7, color=GRIGIO)
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 6)
ax.set_xlabel("componenti adimensionali della velocita'")
ax.set_ylabel(r"$\eta = y\sqrt{U/(\nu x)}$")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.84))
salva(fig, "prog3_1_blasius")
