"""
prog0_4_numpy.py -- Programma 0.4: NumPy, gli array e le differenze finite con lo slicing.

1) Creazione di array (array, zeros, ones, linspace, arange, meshgrid), forma, tipo,
   dimensioni; operazioni elemento per elemento e funzioni universali.
2) Indici e slicing: viste e copie (un sottovettore e' una finestra sullo stesso
   array), indicizzazione booleana, np.where, np.roll, argmax/argmin.
3) Broadcasting: regole e usi (griglie 2D, tabelle, normalizzazione).
4) Differenze finite con lo slicing: derivate prima e seconda, errore in funzione
   di h e ordine di accuratezza misurato (1 per la differenza in avanti, 2 per
   quella centrata e per la derivata seconda).
5) Vettorizzazione: laplaciano discreto 2D con doppio ciclo e con lo slicing,
   tempi a confronto; trasporto upwind di una gaussiana (lo schema del Modulo 4).
6) Riduzioni e algebra lineare: somme, norme, prodotto matrice-vettore, soluzione
   di un sistema, autovalori.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

np.set_printoptions(precision=4, suppress=True, linewidth=110)

# ------------------------------------------------------------------ 1) creazione
print("1) Creazione di array")
a = np.array([1.0, 2.0, 3.0])
M = np.array([[1, 2, 3], [4, 5, 6]])
print(f"   a = {a}, forma {a.shape}, tipo {a.dtype}, dimensioni {a.ndim};  M =\n{M}\n   forma {M.shape}, tipo {M.dtype}, M.T =\n{M.T}")
x = np.linspace(0.0, 1.0, 5)                     # 5 punti equispaziati, estremi inclusi
k = np.arange(0, 10, 2)                          # come range, ma restituisce un array
print(f"   linspace(0, 1, 5) = {x};  arange(0, 10, 2) = {k};  zeros((2, 3)):\n{np.zeros((2, 3))}")
print(f"   a + 1 = {a + 1},  a * a = {a * a},  a @ a = {a @ a} (prodotto scalare),  np.exp(a) = {np.exp(a)}")
print(f"   attenzione ai tipi: np.array([1, 2]) ha dtype {np.array([1, 2]).dtype}; np.zeros(3, dtype=int) = {np.zeros(3, dtype=int)}")
u = np.zeros(5, dtype=int)
u[0] = 2.7
print(f"   assegnare 2,7 in un array di interi lo tronca: {u}  -> usare float per i campi di moto")

# ------------------------------------------------------------------ 2) slicing, viste e copie
print("\n2) Indici, slicing, viste")
v = np.arange(10.0)
print(f"   v = {v};  v[2:5] = {v[2:5]};  v[-3:] = {v[-3:]};  v[::3] = {v[::3]};  v[1:-1] = {v[1:-1]}")
w = v[2:5]                                       # VISTA: condivide la memoria con v
w[:] = -1
print(f"   dopo w = v[2:5]; w[:] = -1  ->  v = {v}   (lo slicing e' una vista, non una copia)")
w = v[2:5].copy(); w[:] = 7
print(f"   con w = v[2:5].copy(); w[:] = 7  ->  v = {v}   (v non cambia)")
v = np.arange(10.0)
maschera = v > 6
print(f"   v > 6 = {maschera};  v[v > 6] = {v[v > 6]};  np.where(v > 6, 1, 0) = {np.where(v > 6, 1, 0)}")
print(f"   np.roll(v, 1) = {np.roll(v, 1)}  (scorrimento ciclico: utile per le condizioni periodiche)")
print(f"   argmax di sin(v) = {np.argmax(np.sin(v))} (indice), valore {np.max(np.sin(v)):.4f}")
A = np.arange(12.0).reshape(3, 4)
print(f"   A =\n{A}\n   riga 1: {A[1]};  colonna 2: {A[:, 2]};  blocco A[1:, 1:3] =\n{A[1:, 1:3]}\n   A[1:-1, 1:-1] (nodi interni) = {A[1:-1, 1:-1]}")

# ------------------------------------------------------------------ 3) broadcasting
print("\n3) Broadcasting")
r = np.array([1.0, 2.0, 3.0])                    # forma (3,)
c = np.array([[10.0], [20.0]])                   # forma (2, 1)
print(f"   r (3,) + c (2,1) ->\n{r + c}   (forma {(r + c).shape}: ogni dimensione 1 si 'stira')")
xg = np.linspace(0, 1, 4)[:, None]               # (4, 1)
yg = np.linspace(0, 1, 3)[None, :]               # (1, 3)
print(f"   tabella f(x, y) = x + 10 y senza meshgrid:\n{xg + 10 * yg}")
X, Y = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 3), indexing="ij")
print(f"   con meshgrid(indexing='ij'): X[i, j] = x_i, Y[i, j] = y_j; X + 10 Y =\n{X + 10 * Y}")
T = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
print(f"   normalizzare le colonne: T / T.max(axis=0) =\n{T / T.max(axis=0)}   (media per riga: {T.mean(axis=1)})")

# ------------------------------------------------------------------ 4) differenze finite
print("\n4) Differenze finite con lo slicing: f = sin(x) su [0, pi], errore massimo nei nodi interni")
print(f"   {'h':>8} {'avanti':>10} {'centrata':>10} {'seconda':>10}")
hs, e1, e2, e3 = [], [], [], []
for n in (10, 20, 40, 80, 160, 320):
    x = np.linspace(0, np.pi, n + 1); h = x[1] - x[0]
    f = np.sin(x)
    d_avanti = (f[1:] - f[:-1]) / h                      # nei nodi 0..n-1
    d_centr = (f[2:] - f[:-2]) / (2 * h)                 # nei nodi 1..n-1
    d2 = (f[2:] - 2 * f[1:-1] + f[:-2]) / h**2
    ea = np.max(np.abs(d_avanti[:-1] - np.cos(x[:-2])))  # stessi nodi interni (0..n-2 -> confrontati in x[:-2])
    ec = np.max(np.abs(d_centr - np.cos(x[1:-1])))
    es = np.max(np.abs(d2 + np.sin(x[1:-1])))
    hs.append(h); e1.append(ea); e2.append(ec); e3.append(es)
    print(f"   {h:8.4f} {ea:10.2e} {ec:10.2e} {es:10.2e}")
ordini = [np.polyfit(np.log(hs), np.log(e), 1)[0] for e in (e1, e2, e3)]
print(f"   ordini misurati (pendenza nel piano log-log): {ordini[0]:.2f}, {ordini[1]:.2f}, {ordini[2]:.2f}")

# ------------------------------------------------------------------ 5) vettorizzazione
print("\n5) Vettorizzazione: laplaciano discreto su griglia 201 x 201")
N = 200
h = 1.0 / N
xx = np.linspace(0, 1, N + 1)
X, Y = np.meshgrid(xx, xx, indexing="ij")
u = np.sin(np.pi * X) * np.sin(2 * np.pi * Y)
esatto = -5 * np.pi**2 * u
t0 = time.perf_counter()
lap1 = np.zeros_like(u)
for i in range(1, N):
    for j in range(1, N):
        lap1[i, j] = (u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] - 4 * u[i, j]) / h**2
t_cicli = time.perf_counter() - t0
t0 = time.perf_counter()
lap2 = np.zeros_like(u)
lap2[1:-1, 1:-1] = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]) / h**2
t_slicing = time.perf_counter() - t0
print(f"   doppio ciclo for: {t_cicli*1000:7.1f} ms;  slicing: {t_slicing*1000:6.2f} ms;  rapporto {t_cicli / t_slicing:.0f}")
print(f"   stessi risultati? {np.allclose(lap1, lap2)};  errore rispetto al laplaciano esatto {np.max(np.abs(lap2 - esatto)[1:-1, 1:-1]):.2e} (O(h^2))")

print("\n   Trasporto upwind di una gaussiana (zeta_t + u zeta_x = 0, u = 1, C = 0,5, griglia periodica):")
L, nx = 20.0, 400
x = np.linspace(0, L, nx, endpoint=False); dx = x[1] - x[0]
z = np.exp(-(x - 4) ** 2 / (2 * 0.5**2))
C = 0.5
n_passi = int(8.0 / (C * dx))
for _ in range(n_passi):
    z = (1 - C) * z + C * np.roll(z, 1)          # z[i-1] con np.roll: periodicita' automatica
print(f"   dopo {n_passi} passi il massimo ({z.max():.3f}) e' in x = {x[np.argmax(z)]:.2f} (atteso 12,0): il profilo e' traslato"
      f" e attenuato dalla viscosita' artificiale")

# ------------------------------------------------------------------ 6) riduzioni e algebra lineare
print("\n6) Riduzioni e algebra lineare")
A = np.array([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 4.0]])
b = np.array([1.0, 2.0, 3.0])
sol = np.linalg.solve(A, b)
print(f"   A x = b: x = {sol}, residuo {np.linalg.norm(A @ sol - b):.1e}, cond(A) = {np.linalg.cond(A):.2f}")
lam, V = np.linalg.eigh(A)
print(f"   autovalori di A: {lam} (esatti 4 - 2cos(k pi/4): {4 - 2*np.cos(np.arange(1, 4)*np.pi/4)})")
print(f"   somma {u.sum():.3f}, media {u.mean():.3e}, massimo {u.max():.3f} nel nodo {tuple(int(i) for i in np.unravel_index(np.argmax(u), u.shape))},"
      f" norma {np.linalg.norm(u):.3f}, trapezi {np.trapezoid(np.sin(xx * np.pi), xx):.6f} (esatto 2/pi = {2/np.pi:.6f})")

# ------------------------------------------------------------------ figura
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
ax[0].loglog(hs, e1, "o-", color=BLU, label="in avanti (ordine 1)")
ax[0].loglog(hs, e2, "s-", color=ARANCIO, label="centrata (ordine 2)")
ax[0].loglog(hs, e3, "^-", color=VERDE, label="derivata seconda (ordine 2)")
ax[0].set_xlabel("$h$"); ax[0].set_ylabel("errore massimo"); ax[0].legend(fontsize=7)
ax[1].plot(x, np.exp(-(x - 4) ** 2 / (2 * 0.5**2)), color=GRIGIO, label="$t=0$")
ax[1].plot(x, np.exp(-(x - 12) ** 2 / (2 * 0.5**2)), ":", color=GRIGIO, label="esatta, $t=8$")
ax[1].plot(x, z, color=BLU, label="upwind, $C=0{,}5$")
ax[1].set_xlabel("$x$"); ax[1].set_ylabel("$\\zeta$"); ax[1].legend(fontsize=7)
fig.tight_layout()
salva(fig, "prog0_4_numpy")
