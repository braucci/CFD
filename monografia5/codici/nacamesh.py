"""
nacamesh.py -- genera il blockMeshDict di una mesh a C intorno a un profilo NACA a
quattro cifre (sei blocchi, bordi 'spline' sul profilo e 'arc' sul contorno anteriore),
con l'altezza della prima cella a parete e i rapporti di espansione calcolati dai
parametri richiesti.

Uso:  python3 nacamesh.py [--sigla 0012] [--caso casi/c15_naca0012] [--y1 2e-5]
                          [--nNorm 80] [--nRear 60] [--nFront 40] [--nWake 60] [--R 20] [--L 20]
Scrive <caso>/system/blockMeshDict. Topologia (meta' superiore; la inferiore e' speculare):

            o2 (0,R) ------------------------ t (1,R) ------- o1 (L,R)
          .    |   blocco A-up               |   blocco W-up      |
        .      |                             |                    |
      o3(-R,0) |  F-up                       |                    |
        .      LE ----- UM ---- (profilo) -- TE ---------------- w1 (L,0)
"""
import argparse
import os
import numpy as np
from scipy.optimize import brentq


# ----------------------------------------------------------------------------- profilo NACA
def naca4(sigla, n=200):
    """Coordinate (x, y) del dorso e del ventre di un NACA a quattro cifre, bordo d'uscita chiuso.
    Spaziatura a coseno (fitta ai bordi). Restituisce due array (n, 2) ordinati dal bordo d'attacco al bordo d'uscita."""
    m, p, t = int(sigla[0]) / 100, int(sigla[1]) / 10, int(sigla[2:]) / 100
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    if m == 0:
        yc = np.zeros_like(x); dyc = np.zeros_like(x)
    else:
        yc = np.where(x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p)**2 * (1 - 2 * p + 2 * p * x - x**2))
        dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p)**2 * (p - x))
    th = np.arctan(dyc)
    dorso = np.column_stack([x - yt * np.sin(th), yc + yt * np.cos(th)])
    ventre = np.column_stack([x + yt * np.sin(th), yc - yt * np.cos(th)])
    return dorso, ventre


# ----------------------------------------------------------------------------- gradazioni
def rapporto_da_prima_cella(H, n, y1):
    """Rapporto di espansione geometrico r tale che n celle con la prima alta y1 coprano H;
    restituisce il 'grading' di blockMesh (ultima/prima) = r^(n-1)."""
    if n * y1 >= H:
        return 1.0
    f = lambda r: y1 * (r**n - 1) / (r - 1) - H
    r = brentq(f, 1.0 + 1e-9, 10.0)
    return r ** (n - 1)


def punto(xy, z):
    return f"({xy[0]:.7g} {xy[1]:.7g} {z})"


# ----------------------------------------------------------------------------- generazione
def scrivi_blockmesh(caso, sigla="0012", R=20.0, L=20.0, xm=0.3, nNorm=80, nRear=60, nFront=40, nWake=60,
                     y1=2e-5, gTE=0.15, gLE=0.08, gWake=None, gUscita=None, n_spline=200):
    dorso, ventre = naca4(sigla, n_spline)
    # punti notevoli
    LE = (0.0, 0.0); TE = (1.0, 0.0)
    iUM = np.argmin(np.abs(dorso[:, 0] - xm)); UM = tuple(dorso[iUM])
    iLM = np.argmin(np.abs(ventre[:, 0] - xm)); LM = tuple(ventre[iLM])
    c45 = np.sqrt(0.5)
    V = {  # nome -> (x, y)
        "TE": TE, "UM": UM, "LE": LE, "LM": LM,
        "w1": (L, 0.0), "o1": (L, R), "t": (1.0, R), "o2": (0.0, R), "o3": (-R, 0.0),
        "o4": (0.0, -R), "b": (1.0, -R), "o5": (L, -R),
    }
    nomi = list(V)                       # indice dei vertici: z = 0 -> 0..11, z = 1 -> 12..23
    idx = {n: i for i, n in enumerate(nomi)}
    def v(n, z=0):
        return idx[n] + 12 * z
    # gradazione normale al profilo (dalla parete verso l'esterno): H = R per i blocchi anteriori, R per i posteriori
    gN = rapporto_da_prima_cella(R, nNorm, y1)
    if gWake is None:
        gWake = 8.0                      # celle che si allargano verso l'uscita nella scia
    righe = []
    a = righe.append
    a("/*--------------------------------*- C++ -*----------------------------------*\\")
    a(f"  blockMeshDict generato da nacamesh.py: NACA {sigla}, mesh a C, R = {R} c, L = {L} c,")
    a(f"  {nNorm} celle normali (prima cella {y1:g} c, rapporto {gN:.1f}), {nFront}+{nRear} celle per lato, {nWake} nella scia.")
    a("\\*" + "-"*75 + "*/")
    a("FoamFile\n{\n    format      ascii;\n    class       dictionary;\n    object      blockMeshDict;\n}")
    a("\nconvertToMeters 1;\n")
    a("vertices\n(")
    for z in (0, 1):
        for n in nomi:
            a(f"    {punto(V[n], z)}    // {v(n, z)} {n}")
    a(");\n")
    a("blocks\n(")
    # (b0 b1 b2 b3 b4 b5 b6 b7): b0->b1 e' la direzione 1 (lungo il profilo), b1->b2 la direzione 2 (normale)
    def hex_(p0, p1, p2, p3, n1, n2, g1, g2, nome, g2b=None):
        """Blocco con gradazione semplice; se g2b e' dato, la gradazione normale e' g2 sullo spigolo p1-p2 e g2b
        sullo spigolo opposto p0-p3 (edgeGrading: le celle si allargano lungo la scia)."""
        if g2b is None:
            grad = f"simpleGrading ({g1} {g2} 1)"
        else:
            grad = f"edgeGrading ({g1} {g1} {g1} {g1}  {g2b} {g2} {g2} {g2b}  1 1 1 1)"
        return (f"    hex ({v(p0)} {v(p1)} {v(p2)} {v(p3)} {v(p0,1)} {v(p1,1)} {v(p2,1)} {v(p3,1)})"
                f" ({n1} {n2} 1) {grad}    // {nome}")
    # gradazioni lungo il profilo: verso TE (gTE) nel blocco posteriore, verso LE (gLE) in quello anteriore
    # orientamento: direzione 1 x direzione 2 deve puntare verso +z (volumi positivi):
    # sul dorso si percorre il profilo da LE a TE, sul ventre da TE a LE
    a(hex_("UM", "TE", "t", "o2", nRear, nNorm, f"{gTE:g}", f"{gN:.4g}", "A-up: dorso posteriore (celle fitte verso TE)"))
    a(hex_("LE", "UM", "o2", "o3", nFront, nNorm, f"{1/gLE:g}", f"{gN:.4g}", "F-up: dorso anteriore (fitte verso LE)"))
    a(hex_("LM", "LE", "o3", "o4", nFront, nNorm, f"{gLE:g}", f"{gN:.4g}", "F-low: ventre anteriore"))
    a(hex_("TE", "LM", "o4", "b", nRear, nNorm, f"{1/gTE:g}", f"{gN:.4g}", "A-low: ventre posteriore"))
    # scia: sullo spigolo al TE la gradazione normale e' quella del profilo (gN), sullo spigolo d'uscita e' piu' dolce (gUscita):
    # le celle sottili dello strato limite si allargano lungo la scia invece di restare alte y1 fino all'uscita
    if gUscita is None:          # celle della scia alte y1 fino all'uscita (ortogonali ma con rapporto di forma altissimo)
        a(hex_("TE", "w1", "o1", "t", nWake, nNorm, f"{gWake:g}", f"{gN:.4g}", "W-up: scia superiore (celle crescenti verso l'uscita)"))
        a(hex_("w1", "TE", "b", "o5", nWake, nNorm, f"{1/gWake:g}", f"{gN:.4g}", "W-low: scia inferiore"))
    else:
        a(hex_("TE", "w1", "o1", "t", nWake, nNorm, f"{gWake:g}", f"{gUscita:.4g}", "W-up: scia superiore (celle crescenti verso l'uscita)", g2b=f"{gN:.4g}"))
        a(hex_("w1", "TE", "b", "o5", nWake, nNorm, f"{1/gWake:g}", f"{gN:.4g}", "W-low: scia inferiore", g2b=f"{gUscita:.4g}"))
    a(");\n")
    a("edges\n(")
    def spline(p0, p1, pts, z):
        a(f"    spline {v(p0, z)} {v(p1, z)}\n    (")
        for xy in pts:
            a(f"        {punto(xy, z)}")
        a("    )")
    for z in (0, 1):
        spline("TE", "UM", dorso[iUM + 1:-1][::-1], z)          # dal TE verso UM (punti interni)
        spline("UM", "LE", dorso[1:iUM][::-1], z)
        spline("LE", "LM", ventre[1:iLM], z)
        spline("LM", "TE", ventre[iLM + 1:-1], z)
        a(f"    arc {v('o2', z)} {v('o3', z)} ({-R*c45:.7g} {R*c45:.7g} {z})")
        a(f"    arc {v('o3', z)} {v('o4', z)} ({-R*c45:.7g} {-R*c45:.7g} {z})")
    a(");\n")
    a("boundary\n(")
    a("    profilo\n    {\n        type wall;\n        faces\n        (")
    for p0, p1 in (("TE", "UM"), ("UM", "LE"), ("LE", "LM"), ("LM", "TE")):
        a(f"            ({v(p0)} {v(p1)} {v(p1,1)} {v(p0,1)})")
    a("        );\n    }")
    a("    esterno\n    {\n        type patch;\n        faces\n        (")
    for p0, p1 in (("t", "o2"), ("o2", "o3"), ("o3", "o4"), ("o4", "b"), ("o1", "t"), ("b", "o5"), ("w1", "o1"), ("o5", "w1")):
        a(f"            ({v(p0)} {v(p1)} {v(p1,1)} {v(p0,1)})")
    a("        );\n    }")
    a("    frontEretro\n    {\n        type empty;\n        faces\n        (")
    for blk in (("UM", "TE", "t", "o2"), ("LE", "UM", "o2", "o3"), ("LM", "LE", "o3", "o4"), ("TE", "LM", "o4", "b"),
                ("TE", "w1", "o1", "t"), ("w1", "TE", "b", "o5")):
        a(f"            ({v(blk[0])} {v(blk[3])} {v(blk[2])} {v(blk[1])})")
        a(f"            ({v(blk[0],1)} {v(blk[1],1)} {v(blk[2],1)} {v(blk[3],1)})")
    a("        );\n    }")
    a(");\n\nmergePatchPairs\n(\n);\n\n// ************************************************************************* //")
    os.makedirs(os.path.join(caso, "system"), exist_ok=True)
    percorso = os.path.join(caso, "system", "blockMeshDict")
    with open(percorso, "w", encoding="utf-8") as fp:
        fp.write("\n".join(righe) + "\n")
    n_celle = (2 * nRear + 2 * nFront + 2 * nWake) * nNorm
    print(f"scritto {percorso}: {n_celle} celle, rapporto normale {gN:.1f}, prima cella {y1:g} c")
    return percorso


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigla", default="0012"); ap.add_argument("--caso", default=".")
    ap.add_argument("--R", type=float, default=20.0); ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--y1", type=float, default=2e-5); ap.add_argument("--nNorm", type=int, default=80)
    ap.add_argument("--nRear", type=int, default=60); ap.add_argument("--nFront", type=int, default=40)
    ap.add_argument("--nWake", type=int, default=60)
    a = ap.parse_args()
    scrivi_blockmesh(a.caso, a.sigla, a.R, a.L, nNorm=a.nNorm, nRear=a.nRear, nFront=a.nFront, nWake=a.nWake, y1=a.y1)
