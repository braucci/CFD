"""
genera.py -- caso c05_convezione: gradino trasportato da un campo di velocita' uniforme a 45 gradi.

Per N = 40 (e 80 per gli schemi principali) e per ogni schema convettivo della lista:
  1. scrive la voce div(phi,s) in fvSchemes e lancia foamRun (solutore 'functions');
  2. legge s con foamcase.py e misura: il profilo lungo la verticale x = 0,5, lo spessore del
     gradino (distanza fra s = 0,1 e s = 0,9), il minimo e il massimo (limitatezza),
     l'errore medio |s - s_esatta| e il numero di iterazioni fino al residuo 1e-6.
Scrive risultati.json e profili.npz, letti da codice/fig05_convezione.py.
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
from foamcase import Caso                                        # noqa: E402
from foamlog import leggi_log                                    # noqa: E402

SCHEMI = {
    "upwind": "Gauss upwind",
    "linear": "Gauss linear",
    "linearUpwind": "Gauss linearUpwind grad(s)",
    "QUICK": "Gauss QUICK",
    "vanLeer": "Gauss vanLeer",
    "limitedLinear 1": "Gauss limitedLinear 1",
    "Minmod": "Gauss Minmod",
    "SuperBee": "Gauss SuperBee",
    "MUSCL": "Gauss MUSCL",
    "limitedLinear 0.2": "Gauss limitedLinear 0.2",
}


def esegui(cmd, log):
    with open(os.path.join(QUI, log), "w") as fp:
        r = subprocess.run(cmd, cwd=QUI, stdout=fp, stderr=subprocess.STDOUT, shell=True)
    return r.returncode


def imposta(file, chiave, valore):
    p = os.path.join(QUI, "system", file)
    s = open(p).read()
    s = re.sub(rf"(\n\s*{re.escape(chiave)}\s+)[^;]*;", rf"\g<1>{valore};", s, count=1)
    open(p, "w").write(s)


def pulisci():
    for d in os.listdir(QUI):
        if re.fullmatch(r"[1-9][0-9]*(\.[0-9]+)?", d):
            shutil.rmtree(os.path.join(QUI, d), ignore_errors=True)
    shutil.rmtree(os.path.join(QUI, "postProcessing"), ignore_errors=True)


def main():
    risultati = []
    profili = {}
    for N in (40, 80):
        imposta("blockMeshDict", "N", str(N))
        pulisci()
        esegui("blockMesh", "log.blockMesh")
        lista = SCHEMI if N == 40 else {k: SCHEMI[k] for k in ("upwind", "linear", "linearUpwind", "vanLeer", "limitedLinear 1")}
        for nome, voce in lista.items():
            imposta("fvSchemes", "div(phi,s)", voce)
            pulisci()
            ok = esegui("foamRun", "log.foamRun")
            c = Caso(QUI)
            m = c.mesh
            if ok != 0 or not c.tempi()[1:]:
                print(f"N = {N:3d}  {nome:18s}  NON CONVERGE / fallito")
                risultati.append({"N": N, "schema": nome, "fallito": True})
                continue
            s = c.campo("s")
            x, y = m.C[:, 0], m.C[:, 1]
            esatta = (y > x).astype(float)
            # profilo lungo x = 0,5: celle della colonna centrale
            col = np.isclose(x, x[np.argmin(np.abs(x - 0.5))])
            yy, ss = y[col], s[col]
            o = np.argsort(yy); yy, ss = yy[o], ss[o]
            # spessore del gradino: fra s = 0,1 e s = 0,9 (interpolazione lineare)
            try:
                y10 = np.interp(0.1, ss, yy); y90 = np.interp(0.9, ss, yy); spess = y90 - y10
            except Exception:
                spess = float("nan")
            d = leggi_log(os.path.join(QUI, "log.foamRun"))
            res = d["campi"]["s"]["res0"]
            it = int(np.argmax(res < 1e-6)) + 1 if (res < 1e-6).any() else len(res)
            r = {"N": N, "schema": nome, "min": float(s.min()), "max": float(s.max()),
                 "spessore": float(spess), "errore_medio": float(np.mean(np.abs(s - esatta))),
                 "iterazioni": it, "residuo_finale": float(res[-1])}
            risultati.append(r)
            profili[f"{nome}|{N}"] = np.column_stack([yy, ss])
            print(f"N = {N:3d}  {nome:18s}  min {s.min():+.4f} max {s.max():.4f}  spessore {spess:.4f}"
                  f"  errore medio {r['errore_medio']:.4f}  iterazioni {it}  residuo {res[-1]:.1e}")
    json.dump(risultati, open(os.path.join(QUI, "risultati.json"), "w"), indent=1)
    np.savez(os.path.join(QUI, "profili.npz"), **profili)
    # si lascia il caso con N = 80 e vanLeer, risolto, per la figura del campo
    imposta("fvSchemes", "div(phi,s)", SCHEMI["vanLeer"]); pulisci(); esegui("foamRun", "log.foamRun")
    imposta("blockMeshDict", "N", "40")


if __name__ == "__main__":
    main()
