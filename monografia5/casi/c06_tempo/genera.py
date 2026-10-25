"""
genera.py -- caso c06_tempo: ordine di convergenza nel tempo e stabilita' a Courant alto.

A) diffusione/: dT/dt = D d2T/dx2 su [0,1], T(x,0) = sin(pi x), T = 0 agli estremi; laplacianFoam su 400
   celle fino a t = 0,1 con dt = 0,02 / 0,01 / 0,005 / 0,0025 e ddtSchemes Euler, backward, CrankNicolson 1,
   CrankNicolson 0.9; errore massimo rispetto a sin(pi x) exp(-pi^2 D t).
B) convezione/: gaussiana trasportata a velocita' 1 su [0,10] (400 celle) fino a t = 5 con scalarTransport,
   schema spaziale linear, Co = 0,5 / 1 / 2 / 4 e i tre schemi temporali; errore e profilo finale.
Scrive risultati.json e profili.npz (letti da codice/fig06_tempo.py).
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
from foamcase import Caso, Mesh                                  # noqa: E402

SCHEMI = {"Euler": "Euler", "backward": "backward", "CrankNicolson 1": "CrankNicolson 1", "CrankNicolson 0.9": "CrankNicolson 0.9"}


def esegui(cmd, cartella, log):
    with open(os.path.join(cartella, log), "w") as fp:
        r = subprocess.run(cmd, cwd=cartella, stdout=fp, stderr=subprocess.STDOUT, shell=True)
    if r.returncode != 0:
        raise RuntimeError(f"comando fallito in {cartella}: {cmd} (vedi {log})")


def imposta(percorso, chiave, valore):
    s = open(percorso).read()
    s = re.sub(rf"(\n\s*{re.escape(chiave)}\s+)[^;]*;", rf"\g<1>{valore};", s, count=1)
    open(percorso, "w").write(s)


def pulisci(cartella):
    for d in os.listdir(cartella):
        if re.fullmatch(r"[0-9.e+-]+", d) and d != "0" and os.path.isdir(os.path.join(cartella, d)):
            shutil.rmtree(os.path.join(cartella, d))
    shutil.rmtree(os.path.join(cartella, "postProcessing"), ignore_errors=True)


def campo_scalare(nome, dim, interno, bordi):
    corpo = "\n".join(f"{v:.12g}" for v in interno)
    b = "\n".join(f"    {k} {{ {v} }}" for k, v in bordi.items())
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      {nome};
}}
dimensions      {dim};

internalField   nonuniform List<scalar> {len(interno)}
(
{corpo}
);

boundaryField
{{
{b}
}}
"""


def diffusione(ris):
    c = os.path.join(QUI, "diffusione")
    pulisci(c)
    esegui("blockMesh", c, "log.blockMesh")
    m = Mesh(c)
    x = m.C[:, 0]
    T0 = np.sin(np.pi * x)
    open(os.path.join(c, "0", "T"), "w").write(campo_scalare("T", "[0 0 0 1 0 0 0]", T0,
        {"estremi": "type fixedValue; value uniform 0;", "lati": "type empty;"}))
    for nome, voce in SCHEMI.items():
        imposta(os.path.join(c, "system", "fvSchemes"), "default", voce)   # la prima voce 'default' e' ddtSchemes
        for dt in (0.02, 0.01, 0.005, 0.0025):
            imposta(os.path.join(c, "system", "controlDict"), "deltaT", f"{dt:g}")
            pulisci(c)
            esegui("laplacianFoam", c, "log.laplacianFoam")
            T = Caso(c).campo("T")
            err = np.max(np.abs(T - T0 * np.exp(-np.pi**2 * 0.1)))
            ris["diffusione"].append({"schema": nome, "dt": dt, "errore": float(err)})
            print(f"diffusione  {nome:18s} dt = {dt:7.4f}  errore max {err:.3e}")


def convezione(ris, profili):
    c = os.path.join(QUI, "convezione")
    pulisci(c)
    esegui("blockMesh", c, "log.blockMesh")
    m = Mesh(c)
    x = m.C[:, 0]
    dx = 10.0 / 400
    s0 = np.exp(-(x - 2.0)**2 / 0.18)
    open(os.path.join(c, "0", "s"), "w").write(campo_scalare("s", "[0 0 0 0 0 0 0]", s0,
        {"ingresso": "type fixedValue; value uniform 0;", "uscita": "type zeroGradient;", "lati": "type empty;"}))
    esatta = np.exp(-(x - 7.0)**2 / 0.18)
    profili["x"] = x; profili["esatta"] = esatta
    for nome in ("Euler", "backward", "CrankNicolson 1"):
        imposta(os.path.join(c, "system", "fvSchemes"), "default", SCHEMI[nome])
        for Co in (0.5, 1.0, 2.0, 4.0):
            imposta(os.path.join(c, "system", "controlDict"), "deltaT", f"{Co * dx:g}")
            pulisci(c)
            esegui("foamRun", c, "log.foamRun")
            s = Caso(c).campo("s")
            err = np.max(np.abs(s - esatta))
            ris["convezione"].append({"schema": nome, "Co": Co, "errore": float(err), "max": float(s.max()), "min": float(s.min())})
            profili[f"{nome}|{Co:g}"] = s
            print(f"convezione  {nome:18s} Co = {Co:3.1f}  errore max {err:.3e}  max {s.max():.4f}  min {s.min():+.4f}")


if __name__ == "__main__":
    ris = {"diffusione": [], "convezione": []}
    profili = {}
    diffusione(ris)
    convezione(ris, profili)
    json.dump(ris, open(os.path.join(QUI, "risultati.json"), "w"), indent=1)
    np.savez(os.path.join(QUI, "profili.npz"), **profili)
