"""
genera.py -- caso c08_cavita_re100: esperimenti sull'accoppiamento pressione-velocita' e sui solutori lineari.

Tutte le modifiche ai dizionari sono fatte con l'utility foamDictionary, cosi' come si farebbe a mano:
    foamDictionary -entry relaxationFactors/fields/p -set 0.3 system/fvSolution

  python3 genera.py base        SIMPLE 0,7/0,3 su 128x128 fino a residui 1e-6; profili confrontati con Ghia
  python3 genera.py rilassamento  numero di iterazioni per diverse coppie (alpha_U, alpha_p) e per SIMPLEC
  python3 genera.py schemi      upwind / linear / linearUpwind su 32x32: errore rispetto a Ghia
  python3 genera.py piso        transitorio PISO (Euler, Co = 0,5) fino a t = 20: stesso stato stazionario?
  python3 genera.py solutori    (Cap. 7) 100 iterazioni SIMPLE con diversi solutori lineari per p: tempo e iterazioni
  python3 genera.py tutto
I risultati sono accumulati in risultati.json; i profili in profili.npz.
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

# Ghia, Ghia & Shin (1982), Re = 100: u lungo la verticale x = 0,5 e v lungo l'orizzontale y = 0,5
GHIA_Y = [1.0, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0]
GHIA_U = [1.0, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.0]
GHIA_X = [1.0, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0]
GHIA_V = [0.0, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091, 0.09233, 0.0]


def sh(cmd, log=None):
    """Esegue un comando nella cartella del caso; con 'log' salva l'uscita su file."""
    if log:
        with open(os.path.join(QUI, log), "w") as fp:
            return subprocess.run(cmd, cwd=QUI, stdout=fp, stderr=subprocess.STDOUT, shell=True).returncode
    return subprocess.run(cmd, cwd=QUI, shell=True, capture_output=True, text=True)


def imposta(file, voce, valore):
    """foamDictionary -entry <voce> -set <valore> <file>"""
    r = sh(f'foamDictionary -entry "{voce}" -set "{valore}" {file}')
    if r.returncode != 0:
        raise RuntimeError(r.stderr)


def pulisci():
    for d in os.listdir(QUI):
        if d not in ("0", "constant", "system") and os.path.isdir(os.path.join(QUI, d)) and d[0].isdigit():
            shutil.rmtree(os.path.join(QUI, d))
    shutil.rmtree(os.path.join(QUI, "postProcessing"), ignore_errors=True)


def carica():
    p = os.path.join(QUI, "risultati.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def salva(ris):
    json.dump(ris, open(os.path.join(QUI, "risultati.json"), "w"), indent=1)


def profili(c, tempo=None):
    """u(y) su x = 0,5 e v(x) su y = 0,5: media delle due colonne (righe) di celle adiacenti alla mezzeria,
    cioe' interpolazione lineare esatta sulla mezzeria, con i valori di parete aggiunti agli estremi."""
    m = c.mesh
    U = c.campo("U", tempo=tempo)
    x, y = m.C[:, 0], m.C[:, 1]
    xs = np.unique(np.round(x, 9)); ys = np.unique(np.round(y, 9))
    ix = np.argsort(np.abs(xs - 0.5))[:2]; iy = np.argsort(np.abs(ys - 0.5))[:2]
    col = np.isclose(x, xs[ix[0]]) | np.isclose(x, xs[ix[1]])
    riga = np.isclose(y, ys[iy[0]]) | np.isclose(y, ys[iy[1]])
    # media per quota (le due colonne hanno le stesse y)
    yy = np.unique(np.round(y[col], 9)); uu = np.array([U[col & np.isclose(y, q), 0].mean() for q in yy])
    xx = np.unique(np.round(x[riga], 9)); vv = np.array([U[riga & np.isclose(x, q), 1].mean() for q in xx])
    yy = np.concatenate([[0.0], yy, [1.0]]); uu = np.concatenate([[0.0], uu, [1.0]])     # pareti: u = 0 in basso, 1 sul coperchio
    xx = np.concatenate([[0.0], xx, [1.0]]); vv = np.concatenate([[0.0], vv, [0.0]])
    return yy, uu, xx, vv


def errore_ghia(y, u, x, v):
    eu = np.max(np.abs(np.interp(GHIA_Y[::-1], y, u) - np.array(GHIA_U[::-1])))
    ev = np.max(np.abs(np.interp(GHIA_X[::-1], x, v) - np.array(GHIA_V[::-1])))
    return float(eu), float(ev)


def stato_simple(log):
    d = leggi_log(os.path.join(QUI, log))
    t = d["exec"][-1, 1] if d["exec"] is not None else float("nan")
    n = len(d["campi"]["p"]["t"])
    conv = "SIMPLE solution converged" in open(os.path.join(QUI, log)).read()
    return n, float(t), conv, d


def ripristina():
    imposta("system/fvSolution", "relaxationFactors/fields/p", 0.3)
    imposta("system/fvSolution", "relaxationFactors/equations/U", 0.7)
    imposta("system/fvSolution", "SIMPLE/consistent", "no")
    imposta("system/fvSolution", "solvers/p/solver", "GAMG")
    imposta("system/fvSolution", "solvers/p/relTol", 0.1)
    imposta("system/fvSchemes", "divSchemes/div(phi,U)", "bounded Gauss linearUpwind grad(U)")
    imposta("system/fvSchemes", "ddtSchemes/default", "steadyState")
    imposta("system/controlDict", "endTime", 6000)
    imposta("system/controlDict", "deltaT", 1)
    imposta("system/controlDict", "writeInterval", 1000)
    imposta("system/blockMeshDict", "blocks", "( hex (0 1 2 3 4 5 6 7) (128 128 1) simpleGrading (1 1 1) )")


def base(ris):
    ripristina(); pulisci()
    sh("blockMesh", "log.blockMesh"); sh("foamRun", "log.foamRun")
    n, t, conv, d = stato_simple("log.foamRun")
    c = Caso(QUI)
    y, u, x, v = profili(c)
    eu, ev = errore_ghia(y, u, x, v)
    ris["base"] = {"iterazioni": n, "tempo": t, "convergenza": conv, "err_u": eu, "err_v": ev,
                   "psi_min_info": "vedi fig08", "res_finali": {k: float(c_["res0"][-1]) for k, c_ in d["campi"].items()}}
    np.savez(os.path.join(QUI, "profili_base.npz"), y=y, u=u, x=x, v=v)
    print(f"base: {n} iterazioni in {t:.1f} s, convergenza {conv}; errore max vs Ghia: u {eu:.4f}, v {ev:.4f}")


def rilassamento(ris):
    ripristina()
    prove = [(0.7, 0.3, "no"), (0.5, 0.2, "no"), (0.9, 0.1, "no"), (0.7, 0.5, "no"), (0.8, 0.4, "no"), (0.9, 1.0, "yes"), (0.95, 1.0, "yes")]
    out = []
    for aU, ap, cons in prove:
        imposta("system/fvSolution", "relaxationFactors/equations/U", aU)
        imposta("system/fvSolution", "relaxationFactors/fields/p", ap)
        imposta("system/fvSolution", "SIMPLE/consistent", cons)
        log = f"log.foamRun_rilassamento_{aU}_{ap}_{cons}"
        pulisci(); sh("foamRun", log)
        n, t, conv, d = stato_simple(log)
        out.append({"alpha_U": aU, "alpha_p": ap, "SIMPLEC": cons == "yes", "iterazioni": n, "tempo": t, "convergenza": conv,
                    "res_p_finale": float(d["campi"]["p"]["res0"][-1])})
        print(f"alpha_U = {aU}, alpha_p = {ap}, SIMPLEC = {cons}: {n} iterazioni, {t:.1f} s, convergenza {conv}")
    ris["rilassamento"] = out
    ripristina()


def schemi(ris):
    ripristina()
    imposta("system/blockMeshDict", "blocks", "( hex (0 1 2 3 4 5 6 7) (32 32 1) simpleGrading (1 1 1) )")
    pulisci(); sh("blockMesh", "log.blockMesh_32")
    out = {}
    for nome, voce in (("upwind", "bounded Gauss upwind"), ("linear", "bounded Gauss linear"),
                       ("linearUpwind", "bounded Gauss linearUpwind grad(U)")):
        imposta("system/fvSchemes", "divSchemes/div(phi,U)", voce)
        pulisci(); sh("foamRun", f"log.foamRun_32_{nome}")
        n, t, conv, d = stato_simple(f"log.foamRun_32_{nome}")
        y, u, x, v = profili(Caso(QUI))
        eu, ev = errore_ghia(y, u, x, v)
        out[nome] = {"iterazioni": n, "err_u": eu, "err_v": ev, "convergenza": conv}
        np.savez(os.path.join(QUI, f"profili_32_{nome}.npz"), y=y, u=u, x=x, v=v)
        print(f"32x32 {nome:12s}: {n} iterazioni, errore vs Ghia u {eu:.4f} v {ev:.4f}")
    ris["schemi_32"] = out
    ripristina()


def piso(ris):
    ripristina()
    imposta("system/fvSchemes", "ddtSchemes/default", "Euler")
    imposta("system/fvSchemes", "divSchemes/div(phi,U)", "Gauss linearUpwind grad(U)")
    imposta("system/controlDict", "deltaT", 0.004)          # Co = U dt / h = 0,5
    imposta("system/controlDict", "endTime", 20)
    imposta("system/controlDict", "writeInterval", 1250)    # ogni 5 s
    shutil.copy(os.path.join(QUI, "system", "fvSolution"), os.path.join(QUI, "system", "fvSolution.simple"))
    shutil.copy(os.path.join(QUI, "system", "fvSolution.piso"), os.path.join(QUI, "system", "fvSolution"))
    pulisci(); sh("blockMesh", "log.blockMesh"); sh("foamRun", "log.foamRun_piso")
    d = leggi_log(os.path.join(QUI, "log.foamRun_piso"))
    c = Caso(QUI)
    out = {"tempo_calcolo": float(d["exec"][-1, 1]), "passi": len(d["campi"]["p"]["t"]), "Co_max": float(d["Co"][:, 2].max()), "storia": []}
    base = np.load(os.path.join(QUI, "profili_base.npz"))
    for tt in c.tempi()[1:]:
        y, u, x, v = profili(c, tempo=tt)
        eu, ev = errore_ghia(y, u, x, v)
        scarto = float(np.max(np.abs(u - base["u"])))
        out["storia"].append({"t": float(tt), "err_u": eu, "err_v": ev, "scarto_da_simple": scarto})
        print(f"PISO t = {tt}: errore vs Ghia u {eu:.4f} v {ev:.4f}; scarto dal profilo SIMPLE {scarto:.2e}")
    np.savez(os.path.join(QUI, "profili_piso.npz"), y=y, u=u, x=x, v=v)
    ris["piso"] = out
    shutil.copy(os.path.join(QUI, "system", "fvSolution.simple"), os.path.join(QUI, "system", "fvSolution"))
    ripristina()


def solutori(ris):
    """Cap. 7: 100 iterazioni SIMPLE con diversi solutori per p; tempo totale e iterazioni del solutore lineare."""
    ripristina()
    imposta("system/controlDict", "endTime", 100)
    imposta("system/fvSolution", "SIMPLE/residualControl/p", 1e-12)   # nessun arresto anticipato
    prove = [
        ("GAMG GaussSeidel relTol 0.1", {"solver": "GAMG", "smoother": "GaussSeidel", "relTol": 0.1}),
        ("GAMG GaussSeidel relTol 0.01", {"solver": "GAMG", "smoother": "GaussSeidel", "relTol": 0.01}),
        ("GAMG DICGaussSeidel relTol 0.1", {"solver": "GAMG", "smoother": "DICGaussSeidel", "relTol": 0.1}),
        ("PCG DIC relTol 0.1", {"solver": "PCG", "preconditioner": "DIC", "relTol": 0.1}),
        ("PCG DIC relTol 0.01", {"solver": "PCG", "preconditioner": "DIC", "relTol": 0.01}),
        ("PCG GAMG relTol 0.1", {"solver": "PCG", "preconditioner": "{ preconditioner GAMG; smoother GaussSeidel; nCellsInCoarsestLevel 10; }", "relTol": 0.1}),
        ("smoothSolver GaussSeidel relTol 0.1", {"solver": "smoothSolver", "smoother": "GaussSeidel", "relTol": 0.1, "nSweeps": 2}),
        ("PBiCGStab DIC relTol 0.1", {"solver": "PBiCGStab", "preconditioner": "DIC", "relTol": 0.1}),
    ]
    out = []
    shutil.copy(os.path.join(QUI, "system", "fvSolution"), os.path.join(QUI, "system", "fvSolution.simple"))
    pulisci(); sh("blockMesh", "log.blockMesh")
    for k, (nome, voci) in enumerate(prove):
        # si riscrive il sotto-dizionario p per intero
        corpo = " ".join(f"{k} {v};" for k, v in voci.items()) + " tolerance 1e-08;"
        imposta("system/fvSolution", "solvers/p", "{ " + corpo + " }")
        pulisci()
        log = f"log.foamRun_solutori_{k}"
        ok = sh("foamRun", log)
        d = leggi_log(os.path.join(QUI, log))
        it = d["campi"]["p"]["iter"]
        out.append({"solutore": nome, "tempo": float(d["exec"][-1, 1]), "iter_p_medie": float(it.mean()), "iter_p_totali": int(it.sum()),
                    "res_p_100": float(d["campi"]["p"]["res0"][-1]), "ok": ok == 0})
        print(f"{nome:36s} tempo {out[-1]['tempo']:6.1f} s  iterazioni di p: medie {it.mean():5.1f}, totali {it.sum():5d}; residuo p a 100: {out[-1]['res_p_100']:.2e}")
    ris["solutori"] = out
    shutil.copy(os.path.join(QUI, "system", "fvSolution.simple"), os.path.join(QUI, "system", "fvSolution"))
    ripristina()


def finale(ris):
    """Rilancia il caso base e lo lascia risolto nella cartella (per le figure del campo)."""
    ripristina(); pulisci()
    sh("blockMesh", "log.blockMesh"); sh("foamRun", "log.foamRun")


if __name__ == "__main__":
    ris = carica()
    cosa = sys.argv[1] if len(sys.argv) > 1 else "tutto"
    passi = {"base": base, "rilassamento": rilassamento, "schemi": schemi, "piso": piso, "solutori": solutori, "finale": finale}
    for nome in (passi if cosa == "tutto" else [cosa]):
        passi[nome](ris)
        salva(ris)
