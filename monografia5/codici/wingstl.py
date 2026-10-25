"""
wingstl.py -- scrive la superficie STL (triangoli, ASCII) di un'ala rettangolare NACA a quattro cifre per
snappyHexMesh: corda unitaria, semiapertura b2, radice prolungata di 'eccesso' oltre il piano di simmetria z = 0
(cosi' il piano di simmetria taglia l'ala senza che la superficie coincida con il contorno) e tip piano chiuso.
Uso: python3 wingstl.py <file.stl> [sigla] [b2] [n]      (es. python3 wingstl.py ala.stl 0012 2.0 120)
"""
import os
import sys
import numpy as np
from nacamesh import naca4


def ala(sigla="0012", b2=2.0, n=120, eccesso=0.05, nz=40):
    dorso, ventre = naca4(sigla, n)
    # contorno chiuso della sezione, da TE (dorso) a LE a TE (ventre), senza ripetere LE e TE
    sez = np.vstack([dorso[::-1], ventre[1:-1]])          # (2n-2, 2), orientata in senso antiorario visto da +z
    z = np.linspace(-eccesso, b2, nz)
    tri = []
    # superficie laterale: strisce fra due sezioni consecutive
    m = len(sez)
    for k in range(nz - 1):
        for i in range(m):
            j = (i + 1) % m
            a = (*sez[i], z[k]); b = (*sez[j], z[k]); c = (*sez[j], z[k + 1]); d = (*sez[i], z[k + 1])
            tri.append((a, b, c)); tri.append((a, c, d))
    # tappi (radice e tip): ventaglio di triangoli dal centro della sezione
    for zz, verso in ((z[0], -1), (z[-1], 1)):
        c0 = (0.5, 0.0, zz)
        for i in range(m):
            j = (i + 1) % m
            a = (*sez[i], zz); b = (*sez[j], zz)
            tri.append((c0, a, b) if verso > 0 else (c0, b, a))
    return np.array(tri)


def scrivi_stl(percorso, tri, nome="ala"):
    os.makedirs(os.path.dirname(os.path.abspath(percorso)), exist_ok=True)
    with open(percorso, "w") as fp:
        fp.write(f"solid {nome}\n")
        for a, b, c in tri:
            nrm = np.cross(np.subtract(b, a), np.subtract(c, a)); nn = np.linalg.norm(nrm)
            nrm = nrm / nn if nn > 0 else nrm
            fp.write(f"  facet normal {nrm[0]:.6e} {nrm[1]:.6e} {nrm[2]:.6e}\n    outer loop\n")
            for v in (a, b, c):
                fp.write(f"      vertex {v[0]:.7e} {v[1]:.7e} {v[2]:.7e}\n")
            fp.write("    endloop\n  endfacet\n")
        fp.write(f"endsolid {nome}\n")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "ala.stl"
    sigla = sys.argv[2] if len(sys.argv) > 2 else "0012"
    b2 = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 120
    t = ala(sigla, b2, n)
    scrivi_stl(out, t)
    print(f"scritto {out}: {len(t)} triangoli, semiapertura {b2}, {n} punti per lato")
