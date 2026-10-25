"""
prog0_9_verifiche.py -- Programma 0.9: errori frequenti, eccezioni, verifiche e file.

1) Errori che il calcolatore segnala (eccezioni): come leggere un traceback,
   try/except, raise; gli errori piu' comuni nei codici di calcolo (indici fuori
   intervallo, forme incompatibili, divisione per zero, nomi non definiti).
2) Errori che il calcolatore NON segnala: lo scostamento di un passo (off by one),
   l'accumulo del tempo in un ciclo while, l'uguaglianza fra float, le viste
   NumPy modificate per sbaglio, gli argomenti predefiniti mutabili, la divisione
   intera, il ciclo che non copia (u_vecchio = u), l'overflow silenzioso (inf, nan).
3) Verifiche automatiche con assert: i metodi del modulo metodi_numerici.py
   (Thomas contro numpy.linalg.solve, RK4 del quarto ordine, SOR contro la
   soluzione esatta) messi alla prova in poche righe.
4) File: salvare e rileggere array (np.savetxt/np.loadtxt, np.save), parametri
   (json), risultati in testo; i percorsi con os.path.
5) Misurare il tempo (time.perf_counter) e trovare il collo di bottiglia (cProfile).
"""
import os
import json
import time
import cProfile
import pstats
import io
import numpy as np
from metodi_numerici import thomas, rk4_integra, sor_poisson, omega_ottimo

QUI = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ 1) eccezioni
print("1) Eccezioni: errori segnalati")
prove = [
    ("indice fuori intervallo", lambda: np.zeros(5)[5]),
    ("forme incompatibili", lambda: np.zeros(3) + np.zeros(4)),
    ("divisione per zero (float Python)", lambda: 1.0 / 0.0),
    ("nome non definito", lambda: velocita_inesistente + 1),
    ("tipo sbagliato", lambda: "2" + 2),
    ("chiave assente", lambda: {"aria": 1.2}["elio"]),
    ("conversione impossibile", lambda: float("dodici")),
]
for nome, azione in prove:
    try:
        azione()
    except Exception as e:                       # 'Exception' cattura tutto; in pratica si cattura il tipo specifico
        print(f"   {nome:34s} -> {type(e).__name__}: {e}")
with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
    print(f"   NumPy invece NON si ferma: 1/0 = {np.float64(1.0) / 0.0}, 0/0 = {np.float64(0.0) / 0.0}, exp(1000) = {np.exp(1000.0)}"
          f"  (da controllare con np.isfinite)")


def velocita_limite(m, rho, A, cd):
    """Esempio di controllo esplicito degli argomenti con raise."""
    if m <= 0 or rho <= 0 or A <= 0 or cd <= 0:
        raise ValueError(f"tutti i parametri devono essere positivi: m={m}, rho={rho}, A={A}, cd={cd}")
    return np.sqrt(2 * m * 9.80665 / (rho * A * cd))


try:
    velocita_limite(1.0, 1.225, -0.01, 0.5)
except ValueError as e:
    print(f"   raise in una funzione propria            -> ValueError: {e}")

# ------------------------------------------------------------------ 2) errori silenziosi
print("\n2) Errori silenziosi (nessun messaggio, risultato sbagliato)")
# a) off by one
x = np.linspace(0, 1, 11)                        # 11 nodi, 10 intervalli: h = 0,1
h_sbagliato = 1.0 / len(x)                       # 1/11 !
print(f"   a) passo h: corretto {x[1] - x[0]:.4f} = 1/(len(x)-1), sbagliato 1/len(x) = {h_sbagliato:.4f}")
# b) accumulo del tempo
t, n, t10 = 0.0, 0, None
while t < 1.0:
    t += 0.1; n += 1
    if n == 10:
        t10 = t
print(f"   b) while t < 1 con t += 0.1: {n} passi (attesi 10) perche' dopo 10 passi t = {t10!r} < 1")
# c) uguaglianza fra float
print(f"   c) 0.1*3 == 0.3 e' {0.1 * 3 == 0.3}; usare abs(a - b) < tol oppure np.isclose: {np.isclose(0.1 * 3, 0.3)}")
# d) viste
u = np.zeros(5); u_vecchio = u                    # NON e' una copia
u[0] = 1.0
print(f"   d) u_vecchio = u, poi u[0] = 1: u_vecchio = {u_vecchio} (stesso oggetto); serve u.copy()")
# e) argomento predefinito mutabile
def aggiungi(valore, lista=[]):                   # la lista e' creata UNA volta sola
    lista.append(valore); return lista
print(f"   e) argomento predefinito mutabile: aggiungi(1) -> {aggiungi(1)}, aggiungi(2) -> {aggiungi(2)} (la lista persiste!)")
def aggiungi_bene(valore, lista=None):
    lista = [] if lista is None else lista
    lista.append(valore); return lista
print(f"      forma corretta: aggiungi_bene(1) -> {aggiungi_bene(1)}, aggiungi_bene(2) -> {aggiungi_bene(2)}")
# f) interi e float
N = 100
print(f"   f) indice del nodo centrale: N/2 = {N / 2} (float, non valido come indice), N//2 = {N // 2}")
# g) il ciclo che aggiorna 'sul posto' quando non dovrebbe (Gauss-Seidel invece di Jacobi)
u = np.array([1.0, 0.0, 0.0, 0.0, 0.0])                                   # contorno sinistro = 1
uj = u.copy(); uj[1:-1] = 0.5 * (u[2:] + u[:-2])                       # Jacobi: usa i valori vecchi
ugs = u.copy()
for i in range(1, 4):
    ugs[i] = 0.5 * (ugs[i + 1] + ugs[i - 1])                            # sul posto: usa gia' i valori nuovi
print(f"   g) stessa formula, risultati diversi: vettorializzata (Jacobi) {uj}, sul posto (Gauss-Seidel) {ugs}")
# h) overflow silenzioso in un ciclo instabile
R = 0.6; v = np.sin(np.pi * np.linspace(0, 1, 21))
with np.errstate(over="ignore", invalid="ignore"):
    for k in range(3000):
        v[1:-1] = v[1:-1] + R * (v[2:] - 2 * v[1:-1] + v[:-2])
print(f"   h) schema instabile dopo 3000 passi: max|v| = {np.max(np.abs(v))} -> controllare np.isfinite(v).all() = {np.isfinite(v).all()}")

# ------------------------------------------------------------------ 3) verifiche con assert
print("\n3) Verifiche automatiche del modulo metodi_numerici.py")
n = 50
a = -np.ones(n); b = 2.5 * np.ones(n); c = -np.ones(n); f = np.random.default_rng(0).standard_normal(n)
A = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
x_th = thomas(a, b, c, f)
assert np.allclose(x_th, np.linalg.solve(A, f)), "Thomas non coincide con numpy.linalg.solve"
print(f"   Thomas contro solve: differenza {np.max(np.abs(x_th - np.linalg.solve(A, f))):.1e}  [ok]")
fy = lambda t, y: -2 * y
errs = [abs(rk4_integra(fy, np.array([1.0]), 1.0, h)[1][-1, 0] - np.exp(-2)) for h in (0.1, 0.05, 0.025)]
rapp = errs[0] / errs[1], errs[1] / errs[2]
assert all(12 < r < 20 for r in rapp), f"RK4 non e' del quarto ordine: rapporti {rapp}"
print(f"   RK4: errori {errs[0]:.2e}, {errs[1]:.2e}, {errs[2]:.2e}, rapporti {rapp[0]:.1f}, {rapp[1]:.1f}  [ok]")
m = 33; xg = np.linspace(0, 1, m); X, Y = np.meshgrid(xg, xg, indexing="ij")
u_es = np.sin(np.pi * X) * np.sin(np.pi * Y)
nodi = [(i, j) for i in range(1, m - 1) for j in range(1, m - 1)]
u_sor, it = sor_poisson(np.zeros((m, m)), -2 * np.pi**2 * u_es, xg[1] - xg[0], omega_ottimo(m, m), 1e-8, nodi)
assert np.max(np.abs(u_sor - u_es)) < 5e-3, "SOR: errore troppo grande"
print(f"   SOR su Poisson: {it} iterazioni, errore {np.max(np.abs(u_sor - u_es)):.2e} (O(h^2))  [ok]")
print("   (un assert falso interrompe il programma con AssertionError e il messaggio: e' il test piu' semplice)")

# ------------------------------------------------------------------ 4) file
print("\n4) File")
cartella = os.path.join(QUI, "..", "figure")
tab = np.column_stack([xg, u_es[:, m // 2]])
percorso = os.path.join(cartella, "prog0_9_profilo.txt")
np.savetxt(percorso, tab, fmt="%.6f", header="x  u(x, 0.5)", delimiter="  ")
letto = np.loadtxt(percorso)
print(f"   np.savetxt/np.loadtxt: {os.path.basename(percorso)}, forma {letto.shape}, identico? {np.allclose(letto, tab, atol=1e-6)}")
np.save(os.path.join(cartella, "prog0_9_campo.npy"), u_es)
print(f"   np.save/np.load (binario, esatto): {np.array_equal(np.load(os.path.join(cartella, 'prog0_9_campo.npy')), u_es)}")
parametri = {"Re": 100, "griglia": [65, 65], "schema": "FTCS", "tolleranza": 1e-6}
with open(os.path.join(cartella, "prog0_9_parametri.json"), "w", encoding="utf-8") as fp:
    json.dump(parametri, fp, indent=2)
with open(os.path.join(cartella, "prog0_9_parametri.json"), encoding="utf-8") as fp:
    riletti = json.load(fp)
print(f"   json: {riletti} (riletti == parametri: {riletti == parametri})")
with open(os.path.join(cartella, "prog0_9_rapporto.txt"), "w", encoding="utf-8") as fp:
    fp.write("Rapporto di calcolo\n")
    for k, v in riletti.items():
        fp.write(f"{k:12s} = {v}\n")
print("   testo: scritto prog0_9_rapporto.txt con 'with open(...) as fp' (il file e' chiuso automaticamente)")

# ------------------------------------------------------------------ 5) tempi e profilo
print("\n5) Tempi e profilo")


def laplaciano_lento(u, h):
    out = np.zeros_like(u)
    for i in range(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            out[i, j] = (u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] - 4 * u[i, j]) / h**2
    return out


def laplaciano_veloce(u, h):
    out = np.zeros_like(u)
    out[1:-1, 1:-1] = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]) / h**2
    return out


def simulazione(n_passi=20):
    for _ in range(n_passi):
        laplaciano_lento(u_es, 1 / 32); laplaciano_veloce(u_es, 1 / 32)


t0 = time.perf_counter(); simulazione(); print(f"   simulazione di prova: {(time.perf_counter() - t0)*1e3:.1f} ms")
pr = cProfile.Profile(); pr.enable(); simulazione(); pr.disable()
buf = io.StringIO(); pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(4)
righe = [r for r in buf.getvalue().splitlines() if "laplaciano" in r]
print("   cProfile (tempo cumulativo per funzione):")
for r in righe:
    print("   " + r.strip().replace(QUI + os.sep, ""))
print("   il profilo dice dove va il tempo: qui quasi tutto nel laplaciano con i cicli, da vettorializzare")
