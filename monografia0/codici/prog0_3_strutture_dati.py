"""
prog0_3_strutture_dati.py -- Programma 0.3: liste, tuple, dizionari, insiemi e stringhe.

Esempio conduttore: la geometria dei profili NACA a quattro cifre. La sigla
"2412" e' una stringa da cui si estraggono curvatura massima m = 2 %, posizione
p = 40 % e spessore t = 12 %; la linea media e la distribuzione di spessore sono
calcolate con liste (senza NumPy, che arriva nel Programma 0.4), i punti del
profilo sono tuple (x, y), un dizionario raccoglie i profili per sigla.
Il programma mostra: indici e slicing, metodi delle liste, list comprehension,
spacchettamento delle tuple, dizionari e insiemi, ordinamento con chiave, la
differenza fra copia e riferimento (la trappola degli alias), le stringhe.
Verifica: lo spessore massimo del NACA 0012 e' 0,12 c in x = 0,30 c (tabulato:
0,3 c); l'area della sezione del NACA 0012 e' 0,0817 c^2 (0,0822 con il bordo d'uscita aperto).
"""
import math
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO

def parametri_naca(sigla):
    """Da '2412' restituisce (m, p, t) = (0,02, 0,4, 0,12)."""
    if len(sigla) != 4 or not sigla.isdigit():
        raise ValueError(f"sigla non valida: {sigla!r}")
    return int(sigla[0]) / 100, int(sigla[1]) / 10, int(sigla[2:]) / 100


def spessore(x, t):
    """Semispessore NACA a quattro cifre (bordo d'uscita chiuso con l'ultimo coefficiente -0,1036)."""
    return 5 * t * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)


def linea_media(x, m, p):
    """Ordinata e pendenza della linea media NACA a quattro cifre."""
    if m == 0:
        return 0.0, 0.0
    if x < p:
        return m / p**2 * (2 * p * x - x**2), 2 * m / p**2 * (p - x)
    return m / (1 - p)**2 * (1 - 2 * p + 2 * p * x - x**2), 2 * m / (1 - p)**2 * (p - x)


def profilo(sigla, n=100):
    """Punti (x, y) del dorso e del ventre, come liste di tuple; ascisse addensate al bordo d'attacco."""
    m, p, t = parametri_naca(sigla)
    xs = [0.5 * (1 - math.cos(math.pi * i / n)) for i in range(n + 1)]     # spaziatura coseno
    dorso, ventre = [], []
    for x in xs:
        yc, dyc = linea_media(x, m, p)
        yt = spessore(x, t)
        th = math.atan(dyc)
        dorso.append((x - yt * math.sin(th), yc + yt * math.cos(th)))
        ventre.append((x + yt * math.sin(th), yc - yt * math.cos(th)))
    return dorso, ventre



if __name__ == "__main__":
    # ------------------------------------------------------------------ 1) liste
    print("1) Liste")
    quote = [0, 2000, 4000, 6000, 8000]              # lista di interi
    quote.append(10000)                              # aggiunge in coda
    quote.insert(0, -500)                            # inserisce in testa
    print(f"   {quote}, lunghezza {len(quote)}, primo {quote[0]}, ultimo {quote[-1]}, secondo-terzo {quote[1:3]}")
    print(f"   ogni due {quote[::2]}, rovesciata {quote[::-1]}, 4000 in lista? {4000 in quote}, indice di 4000: {quote.index(4000)}")
    quote.remove(-500)
    ultimo = quote.pop()                             # toglie e restituisce l'ultimo
    print(f"   dopo remove e pop: {quote} (tolto {ultimo}); somma {sum(quote)}, massimo {max(quote)}, ordinata decrescente {sorted(quote, reverse=True)}")
    mista = [1, "due", 3.0, [4, 5]]                  # una lista puo' contenere tipi diversi (ma e' raro che convenga)
    print(f"   lista eterogenea: {mista}, elemento annidato {mista[3][1]}")

    # ------------------------------------------------------------------ 2) copie e riferimenti
    print("\n2) Copie e riferimenti: la trappola degli alias")
    a = [1, 2, 3]
    b = a                                            # b e' un altro NOME dello stesso oggetto
    b[0] = 99
    print(f"   a = {a} dopo b[0] = 99 con b = a      (stesso oggetto: id(a) == id(b) e' {a is b})")
    c = a.copy()                                     # copia (anche a[:] o list(a))
    c[0] = -1
    print(f"   a = {a}, c = {c} dopo c = a.copy(); c[0] = -1")
    m = [[0] * 3] * 2                                # TRAPPOLA: due riferimenti alla stessa riga
    m[0][0] = 7
    print(f"   [[0]*3]*2 dopo m[0][0] = 7: {m}   (le righe sono lo stesso oggetto!)")
    m = [[0] * 3 for _ in range(2)]                  # corretto: una riga nuova per ogni iterazione
    m[0][0] = 7
    print(f"   [[0]*3 for _ in range(2)] dopo m[0][0] = 7: {m}")

    # ------------------------------------------------------------------ 3) tuple e stringhe: la sigla NACA
    print("\n3) Tuple, stringhe e profili NACA a quattro cifre")
    punto = (0.25, 0.05)                             # tupla: immutabile, si spacchetta
    x, y = punto
    print(f"   punto = {punto}, x = {x}, y = {y}; (3,) e' una tupla di un elemento, (3) e' il numero 3")


    for sigla in ("0012", "2412", "4415"):
        m, p, t = parametri_naca(sigla)
        print(f"   NACA {sigla}: curvatura {m:.0%} in x = {p:.0%} c, spessore {t:.0%} c")
    try:
        parametri_naca("24a2")
    except ValueError as errore:
        print(f"   errore intercettato: {errore}")

    # ------------------------------------------------------------------ 4) list comprehension e funzioni su liste
    print("\n4) Comprehension, zip, ordinamento con chiave")
    dorso, ventre = profilo("0012", n=400)
    ys = [y for x, y in dorso]                       # list comprehension con spacchettamento
    k = ys.index(max(ys))
    print(f"   NACA 0012: spessore massimo {2 * max(ys):.4f} c in x = {dorso[k][0]:.3f} c (tabulato: 0,12 c in 0,30 c)")
    area = 0.0
    for (x1, y1), (x2, y2) in zip(dorso[:-1], dorso[1:]):          # regola dei trapezi sul dorso, x2 per simmetria
        area += (y1 + y2) * (x2 - x1)
    print(f"   area della sezione = {area:.4f} c^2 (valore noto: 0,0822 c^2 con il bordo d'uscita aperto, coefficiente -0,1015;")
    print("   con il bordo chiuso, -0,1036, l'area e' 0,0817 c^2)")
    assert abs(area - 0.0817) < 2e-4
    pari = [x for x in range(20) if x % 2 == 0]
    quadrati = {x: x**2 for x in range(5)}           # dict comprehension
    print(f"   pari < 20: {pari}; quadrati: {quadrati}")
    profili = {"0012": 0.12, "2412": 0.12, "4415": 0.15, "0009": 0.09, "6409": 0.09}
    per_spessore = sorted(profili, key=lambda s: (profili[s], s))  # chiave di ordinamento: (spessore, sigla)
    print(f"   sigle ordinate per spessore: {per_spessore}")

    # ------------------------------------------------------------------ 5) dizionari e insiemi
    print("\n5) Dizionari e insiemi")
    fluidi = {"aria": {"rho": 1.225, "nu": 1.46e-5}, "acqua": {"rho": 998.0, "nu": 1.0e-6}}
    fluidi["olio"] = {"rho": 870.0, "nu": 1.0e-4}
    for nome, prop in fluidi.items():
        print(f"   {nome:6s}: rho = {prop['rho']:7.1f} kg/m^3, nu = {prop['nu']:.2e} m^2/s, mu = {prop['rho'] * prop['nu']:.2e} Pa s")
    print(f"   chiavi: {list(fluidi)}; 'elio' presente? {'elio' in fluidi}; valore con default: {fluidi.get('elio', 'assente')}")
    spessori = {t for t in profili.values()}         # insieme: senza duplicati, senza ordine
    print(f"   spessori distinti: {sorted(spessori)}; intersezione {{9, 12, 15}} & {{12, 18}} = {set([9, 12, 15]) & set([12, 18])}")
    parole = "la portanza nasce dalla differenza di pressione fra ventre e dorso".split()
    print(f"   {len(parole)} parole, la piu' lunga {max(parole, key=len)!r}, unite con '-': {'-'.join(parole[:4])!r}")

    # ------------------------------------------------------------------ figura
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    for sigla, col in zip(("0012", "2412", "4415"), (BLU, ARANCIO, VERDE)):
        dorso, ventre = profilo(sigla)
        xs = [x for x, y in dorso] + [x for x, y in ventre[::-1]]
        ys = [y for x, y in dorso] + [y for x, y in ventre[::-1]]
        ax.plot(xs, ys, color=col, label=f"NACA {sigla}")
    ax.set_aspect("equal"); ax.set_xlabel("$x/c$"); ax.set_ylabel("$y/c$"); ax.legend(fontsize=7)
    fig.tight_layout()
    salva(fig, "prog0_3_naca")
