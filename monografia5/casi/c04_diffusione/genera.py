"""
genera.py -- caso c04_diffusione: conduzione stazionaria su mesh ortogonali e distorte.

Per ogni mesh (quadrato S = 0, parallelogrammi S = 0,5 e 1, quadrato con i punti interni spostati a caso
di +-0,3 h: non ortogonale E skew) e ogni N (10, 20, 40, 80):
  1. scrive system/blockMeshDict dal modello e lancia blockMesh (poi, se richiesto, perturba i punti);
  2. legge i centroidi delle facce del patch 'contorno' con foamcase.py e scrive 0/T con il
     valore ESATTO T = sin(pi x) sinh(pi y)/sinh(pi) su ogni faccia (nonuniform List<scalar>);
  3. per ogni schema del laplaciano (orthogonal, uncorrected, corrected, limited 0.5) sostituisce la
     voce in fvSchemes, lancia laplacianFoam e misura l'errore massimo nelle celle;
  4. per il gradiente: assegna T esatto nelle celle e calcola grad(T) con foamPostProcess con tre
     gradSchemes, misurando l'errore rispetto al gradiente analitico.
Scrive risultati.json, letto da codice/fig04_diffusione.py.   Uso: python3 genera.py
"""
import os
import re
import sys
import json
import shutil
import subprocess
import numpy as np

QUI = os.path.dirname(os.path.abspath(__file__))
for _d in ("codice", "codici"):                                   # gli strumenti Python: codice/ nel progetto, codici/ nell'archivio pubblicato
    sys.path.insert(0, os.path.join(QUI, "..", "..", _d))
from foamcase import Mesh, Caso                                   # noqa: E402


def esatta(x, y):
    return np.sin(np.pi * x) * np.sinh(np.pi * y) / np.sinh(np.pi)


def grad_esatto(x, y):
    gx = np.pi * np.cos(np.pi * x) * np.sinh(np.pi * y) / np.sinh(np.pi)
    gy = np.pi * np.sin(np.pi * x) * np.cosh(np.pi * y) / np.sinh(np.pi)
    return np.column_stack([gx, gy, np.zeros_like(gx)])


def esegui(cmd, log):
    with open(os.path.join(QUI, log), "w") as fp:
        r = subprocess.run(cmd, cwd=QUI, stdout=fp, stderr=subprocess.STDOUT, shell=True)
    if r.returncode != 0:
        raise RuntimeError(f"comando fallito: {cmd} (vedi {log})")


def scrivi_mesh(S, N):
    t = open(os.path.join(QUI, "system", "blockMeshDict.template")).read()
    t = t.replace("__X2__", f"{1 + S:g}").replace("__X3__", f"{S:g}").replace("__N__", str(N))
    open(os.path.join(QUI, "system", "blockMeshDict"), "w").write(t)
    esegui("blockMesh", "log.blockMesh")


def perturba(ampiezza=0.3, seme=1):
    """Sposta a caso i punti interni della mesh (quadrato unitario) di +-ampiezza*h in x e y, uguale sui due
    piani z: la mesh resta valida ma diventa non ortogonale e skew in modo irregolare, come una mesh reale."""
    m = Mesh(QUI)
    P = m.points.copy()
    h = 1.0 / round(np.sqrt(m.nCells))
    interni = (P[:, 0] > 1e-9) & (P[:, 0] < 1 - 1e-9) & (P[:, 1] > 1e-9) & (P[:, 1] < 1 - 1e-9)
    rng = np.random.default_rng(seme)
    # stesso spostamento per i punti sovrapposti dei due piani: si indicizza per (x, y)
    chiavi = {}
    for i in np.where(interni)[0]:
        k = (round(P[i, 0], 9), round(P[i, 1], 9))
        if k not in chiavi:
            chiavi[k] = rng.uniform(-ampiezza * h, ampiezza * h, 2)
        P[i, :2] += chiavi[k]
    testo = open(os.path.join(QUI, "constant", "polyMesh", "points")).read()
    i0 = testo.index("(\n(")
    nuovo = testo[:i0] + "(\n" + "\n".join(f"({x:.12g} {y:.12g} {z:.12g})" for x, y, z in P) + "\n)\n"
    open(os.path.join(QUI, "constant", "polyMesh", "points"), "w").write(nuovo)


def scrivi_T(mesh, interno=None):
    """0/T con il valore esatto sulle facce del contorno e, se richiesto, nelle celle."""
    idx = mesh.facce_patch("contorno")
    Tb = esatta(mesh.Cf[idx, 0], mesh.Cf[idx, 1])
    if interno is None:
        campo = "uniform 0"
    else:
        campo = f"nonuniform List<scalar> {len(interno)}\n(\n" + "\n".join(f"{v:.12g}" for v in interno) + "\n)"
    testo = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      T;
}}
dimensions      [0 0 0 1 0 0 0];

internalField   {campo};

boundaryField
{{
    contorno
    {{
        type            fixedValue;
        value           nonuniform List<scalar> {len(Tb)}
(
{chr(10).join(f"{v:.12g}" for v in Tb)}
);
    }}
    frontEretro
    {{
        type            empty;
    }}
}}
"""
    open(os.path.join(QUI, "0", "T"), "w").write(testo)


def imposta_schema(voce, valore, blocco=None):
    """Sostituisce il valore di una voce di fvSchemes (se 'blocco' e' dato, solo dentro quel sotto-dizionario)."""
    p = os.path.join(QUI, "system", "fvSchemes")
    s = open(p).read()
    if blocco:
        i = s.index(blocco); j = s.index("}", i)
        s = s[:i] + re.sub(rf"({voce}\s+)[^;]*;", rf"\g<1>{valore};", s[i:j]) + s[j:]
    else:
        s = re.sub(rf"({voce}\s+)[^;]*;", rf"\g<1>{valore};", s)
    open(p, "w").write(s)


def pulisci():
    for d in os.listdir(QUI):
        if re.fullmatch(r"[1-9][0-9]*(\.[0-9]+)?", d) or d == "0.orig":
            shutil.rmtree(os.path.join(QUI, d), ignore_errors=True)
    shutil.rmtree(os.path.join(QUI, "postProcessing"), ignore_errors=True)


MESH = {"quadrato": (0.0, False), "S=0.5": (0.5, False), "S=1": (1.0, False), "perturbata": (0.0, True)}


def main():
    risultati = {"diffusione": [], "gradiente": []}
    schemi = {"orthogonal": "Gauss linear orthogonal", "uncorrected": "Gauss linear uncorrected",
              "corrected": "Gauss linear corrected", "limited 0.5": "Gauss linear limited 0.5"}
    for nome_mesh, (S, pert) in MESH.items():
        for N in (10, 20, 40, 80):
            pulisci()
            scrivi_mesh(S, N)
            if pert:
                perturba()
            mesh = Mesh(QUI)
            no = mesh.non_ortogonalita().max(); sk = mesh.skewness().max()
            scrivi_T(mesh)
            for nome, valore in schemi.items():
                imposta_schema("laplacian\\(DT,T\\)", valore, "laplacianSchemes")
                pulisci()
                esegui("laplacianFoam", f"log.laplacianFoam")
                c = Caso(QUI)
                T = c.campo("T")
                err = np.max(np.abs(T - esatta(mesh.C[:, 0], mesh.C[:, 1])))
                risultati["diffusione"].append({"mesh": nome_mesh, "S": S, "N": N, "h": 1.0 / N, "nonorto": float(no), "skew": float(sk),
                                                "schema": nome, "errore": float(err)})
                print(f"{nome_mesh:10s} N = {N:3d}  theta = {no:5.1f}  skew = {sk:5.3f}  {nome:12s}  errore max {err:.3e}")
            # --- gradiente di un campo esatto ---
            if N in (20, 40, 80) and nome_mesh != "S=0.5":
                pulisci()
                scrivi_T(mesh, esatta(mesh.C[:, 0], mesh.C[:, 1]))
                for gs in ("Gauss linear", "leastSquares", "Gauss pointLinear"):
                    imposta_schema("default", gs, "gradSchemes")
                    esegui('foamPostProcess -func "grad(T)" -time 0', "log.foamPostProcess")
                    g = Caso(QUI).campo("grad(T)", tempo="0")
                    ge = grad_esatto(mesh.C[:, 0], mesh.C[:, 1])
                    interni = np.ones(mesh.nCells, bool)
                    # si escludono le celle adiacenti al contorno, dove il gradiente usa il valore di faccia
                    interni[mesh.owner[mesh.facce_patch("contorno")]] = False
                    err = np.max(np.linalg.norm(g - ge, axis=1)[interni])
                    risultati["gradiente"].append({"mesh": nome_mesh, "S": S, "N": N, "h": 1.0 / N, "schema": gs, "errore": float(err)})
                    print(f"      gradiente {gs:18s} errore max (celle interne) {err:.3e}")
                imposta_schema("default", "Gauss linear", "gradSchemes")
    imposta_schema("laplacian\\(DT,T\\)", "Gauss linear corrected", "laplacianSchemes")
    json.dump(risultati, open(os.path.join(QUI, "risultati.json"), "w"), indent=1)
    # si lascia nel caso la mesh perturbata con N = 40, risolta con 'corrected', per la figura
    pulisci(); scrivi_mesh(0.0, 40); perturba(); scrivi_T(Mesh(QUI)); esegui("laplacianFoam", "log.laplacianFoam")


if __name__ == "__main__":
    main()
