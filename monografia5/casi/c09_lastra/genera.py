"""
genera.py -- caso c09_lastra: strato limite su lastra piana, laminare e turbolento.

Varianti (ciascuna: mesh con la prima cella y1 richiesta, viscosita', modello, funzione di parete):
  laminare      nu = 1e-5 (Re_L = 1e5), laminar,  y1 = 2e-4
  SST_risolta   nu = 1e-7 (Re_L = 1e7), kOmegaSST, y1 = 2e-6 (y+ ~ 1),  nutLowReWallFunction
  SST_wf        nu = 1e-7,              kOmegaSST, y1 = 1e-4 (y+ ~ 50), nutkWallFunction
  kEpsilon_wf   nu = 1e-7,              kEpsilon,  y1 = 1e-4,           nutkWallFunction
Per ogni variante: C_f(x) con foamcase (sforzo_parete) e con la function object wallShearStress, y+,
profilo u+(y+) a x = 0,5 m; riferimento: Blasius (laminare), Schultz-Grunow e White (turbolento). Scrive risultati.json e profili.npz.
"""
import os
import re
import sys
import json
import shutil
import subprocess
import numpy as np
from scipy.optimize import brentq

QUI = os.path.dirname(os.path.abspath(__file__))
for _d in ("codice", "codici"):                                   # gli strumenti Python: codice/ nel progetto, codici/ nell'archivio pubblicato
    sys.path.insert(0, os.path.join(QUI, "..", "..", _d))
from foamcase import Caso                                        # noqa: E402
from foamlog import leggi_log                                    # noqa: E402

NY = 80


def grading(H, n, y1):
    f = lambda r: y1 * (r**n - 1) / (r - 1) - H
    r = brentq(f, 1 + 1e-9, 10)
    return r ** (n - 1)


def sh(cmd, log):
    with open(os.path.join(QUI, log), "w") as fp:
        return subprocess.run(cmd, cwd=QUI, stdout=fp, stderr=subprocess.STDOUT, shell=True).returncode


def imposta(file, voce, valore):
    r = subprocess.run(f'foamDictionary -entry "{voce}" -set "{valore}" {file}', cwd=QUI, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)


def pulisci():
    for d in os.listdir(QUI):
        if d[0].isdigit() and d != "0" and os.path.isdir(os.path.join(QUI, d)):
            shutil.rmtree(os.path.join(QUI, d))
    shutil.rmtree(os.path.join(QUI, "postProcessing"), ignore_errors=True)


def mesh(y1):
    t = open(os.path.join(QUI, "system", "blockMeshDict.template")).read()
    nx = 150
    t = t.replace("__NY__", str(NY)).replace("__GY__", f"{grading(0.5, NY, y1):.4g}").replace("__NX__", str(nx)).replace("__GX__", "20")
    open(os.path.join(QUI, "system", "blockMeshDict"), "w").write(t)
    sh("blockMesh", "log.blockMesh")


VARIANTI = {
    "laminare": dict(nu=1e-5, y1=2e-4, sim="laminar", modello="kOmegaSST", nutw="nutkWallFunction"),
    "SST_risolta": dict(nu=1e-7, y1=2e-6, sim="RAS", modello="kOmegaSST", nutw="nutLowReWallFunction"),
    "SST_wf": dict(nu=1e-7, y1=1e-4, sim="RAS", modello="kOmegaSST", nutw="nutkWallFunction"),
    "kEpsilon_wf": dict(nu=1e-7, y1=1e-4, sim="RAS", modello="kEpsilon", nutw="nutkWallFunction"),
}


def cf_schultz_grunow(Rex):
    """C_f locale turbolento dalla correlazione di Schultz-Grunow: C_f = 0,370 (log10 Re_x)^-2,584."""
    return 0.370 * np.log10(Rex) ** (-2.584)


def cf_white(Rex):
    """C_f locale turbolento dalla correlazione di White: C_f = 0,455 / ln^2(0,06 Re_x)."""
    return 0.455 / np.log(0.06 * Rex) ** 2


def main():
    ris = {}
    prof = {}
    for nome, v in VARIANTI.items():
        pulisci()
        mesh(v["y1"])
        imposta("constant/physicalProperties", "nu", f"{v['nu']:g} [m^2/s]")
        imposta("constant/momentumTransport", "simulationType", v["sim"])
        imposta("constant/momentumTransport", "RAS/model", v["modello"])
        imposta("0/nut", "boundaryField/lastra/type", v["nutw"])
        ok = sh("foamRun", f"log.foamRun_{nome}")
        d = leggi_log(os.path.join(QUI, f"log.foamRun_{nome}"))
        c = Caso(QUI)
        nu = v["nu"]
        xf, tau = c.sforzo_parete("lastra", nu)
        o = np.argsort(xf[:, 0]); xf, tau = xf[o], tau[o]
        cf = tau[:, 0] / (0.5 * 1.0**2)
        Rex = xf[:, 0] * 1.0 / nu
        # confronto con la function object wallShearStress (campo scritto all'ultimo istante)
        try:
            wss = c.valore_patch("wallShearStress", "lastra")[o]
            scarto = float(np.max(np.abs(np.abs(wss[:, 0]) - np.abs(tau[:, 0]))) / np.max(np.abs(tau[:, 0])))
        except Exception as e:
            scarto = float("nan")
        yp = c.yplus("lastra", nu)[o]
        # riferimento
        if nome == "laminare":
            cf_rif = 0.664 / np.sqrt(Rex)
        else:
            cf_rif = cf_schultz_grunow(Rex)
        sel = (xf[:, 0] > 0.1) & (xf[:, 0] < 0.95)
        err = float(np.mean(np.abs(cf[sel] - cf_rif[sel]) / cf_rif[sel]))
        # profilo a x = 0,5 in variabili di parete
        i5 = np.argmin(np.abs(xf[:, 0] - 0.5))
        utau = np.sqrt(abs(tau[i5, 0]))
        y, U = c.linea("U", (0.5, 1e-9, 0.05), (0.5, 0.05, 0.05), 400, metodo="cella")
        prof[f"{nome}|yplus"] = y * utau / nu
        prof[f"{nome}|uplus"] = U[:, 0] / utau
        prof[f"{nome}|Rex"] = Rex; prof[f"{nome}|cf"] = cf; prof[f"{nome}|cf_rif"] = cf_rif
        ris[nome] = {"iterazioni": len(d["campi"]["p"]["t"]), "tempo": float(d["exec"][-1, 1]), "ok": ok == 0,
                     "yplus_medio": float(np.mean(yp[sel])), "yplus_max": float(yp.max()),
                     "errore_cf_medio": err, "cf_x05": float(cf[i5]), "cf_rif_x05": float(cf_rif[i5]),
                     "scarto_FO_vs_python": scarto, "Cf_medio": float(np.trapezoid(cf[xf[:, 0] > 0], xf[xf[:, 0] > 0, 0]))}
        print(f"{nome:12s} it {ris[nome]['iterazioni']:5d}  y+ medio {ris[nome]['yplus_medio']:6.2f}  Cf(0,5) {cf[i5]:.5f} (rif {cf_rif[i5]:.5f})"
              f"  errore medio {err*100:.1f}%  FO vs python {scarto:.1e}")
    json.dump(ris, open(os.path.join(QUI, "risultati.json"), "w"), indent=1)
    np.savez(os.path.join(QUI, "profili.npz"), **prof)


if __name__ == "__main__":
    main()
