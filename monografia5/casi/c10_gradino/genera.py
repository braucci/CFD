"""
genera.py -- caso c10_gradino: gradino a valle di Driver & Seegmiller con tre modelli RANS.
Per kEpsilon, kOmegaSST, SpalartAllmaras: esegue il calcolo (stessa mesh, funzioni di parete), estrae il
coefficiente d'attrito sul fondo con foamcase (e lo confronta con la function object wallShearStress),
trova il punto di riattacco (cambio di segno di tau_w) e salva i profili di velocita' a x/h = 4, 6, 8, 10.
"""
import os
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

H = 0.0127; U0 = 44.2; NU = 1.5e-5


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


def main():
    ris = {}; prof = {}
    pulisci(); sh("blockMesh", "log.blockMesh"); sh("checkMesh", "log.checkMesh")
    for modello in ("kEpsilon", "SpalartAllmaras", "kOmegaSST"):   # SST per ultimo: i suoi campi restano nel caso per la figura
        imposta("constant/momentumTransport", "RAS/model", modello)
        pulisci()
        ok = sh("foamRun", f"log.foamRun_{modello}")
        d = leggi_log(os.path.join(QUI, f"log.foamRun_{modello}"))
        c = Caso(QUI)
        def riattacco(tempo=None):
            xf, tau = c.sforzo_parete("fondo", NU, tempo=tempo)
            o = np.argsort(xf[:, 0]); xf, tau = xf[o], tau[o]
            cf = tau[:, 0] / (0.5 * U0**2)
            x_h = xf[:, 0] / H
            # riattacco: primo cambio di segno di tau_x da negativo (ricircolo) a positivo, oltre x/h = 1
            s = np.sign(cf); i = np.where((s[:-1] < 0) & (s[1:] >= 0) & (x_h[:-1] > 1))[0]
            xr = float(np.interp(0.0, [cf[i[0]], cf[i[0] + 1]], [x_h[i[0]], x_h[i[0] + 1]])) if len(i) else float("nan")
            return x_h, cf, xr, o
        x_h, cf, xr, o = riattacco()
        storia_xr = [(float(t), riattacco(t)[2]) for t in c.tempi()[1:]]    # il riattacco agli istanti di scrittura: e' stabile?
        yp = c.yplus("fondo", NU)
        try:
            wss = c.valore_patch("wallShearStress", "fondo")[o]
            scarto = float(np.max(np.abs(np.abs(wss[:, 0]) - np.abs(tau[:, 0]))) / np.max(np.abs(tau[:, 0])))
        except Exception:
            scarto = float("nan")
        prof[f"{modello}|x_h"] = x_h; prof[f"{modello}|cf"] = cf
        for xh in (4, 6, 8, 10):
            y, U = c.linea("U", (xh * H, -H + 1e-6, 0.5 * H), (xh * H, 3 * H, 0.5 * H), 200)
            prof[f"{modello}|u_x{xh}"] = U[:, 0] / U0; prof[f"{modello}|y_x{xh}"] = y / H      # y e' la distanza dal fondo (punto iniziale della linea)
        ris[modello] = {"x_r_h": xr, "storia_xr": storia_xr, "iterazioni": len(d["campi"]["p"]["t"]), "tempo": float(d["exec"][-1, 1]), "ok": ok == 0,
                        "yplus_fondo_medio": float(np.mean(yp[x_h > 8])), "cf_min": float(cf.min()), "scarto_FO_vs_python": scarto,
                        "convergenza": "SIMPLE solution converged" in open(os.path.join(QUI, f"log.foamRun_{modello}")).read()}
        print(f"{modello:16s} x_r/h = {xr:5.2f}  (esp. 6,26)  iterazioni {ris[modello]['iterazioni']}  y+ medio sul fondo {ris[modello]['yplus_fondo_medio']:.1f}  FO vs python {scarto:.1e}")
    json.dump(ris, open(os.path.join(QUI, "risultati.json"), "w"), indent=1)
    np.savez(os.path.join(QUI, "profili.npz"), **prof)
    imposta("constant/momentumTransport", "RAS/model", "kOmegaSST")


if __name__ == "__main__":
    main()
