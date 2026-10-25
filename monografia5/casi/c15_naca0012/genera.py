"""
genera.py -- caso c15_naca0012: profilo NACA 0012 a Re = 6e6 con Spalart-Allmaras.

  python3 genera.py mesh [livello]   genera la mesh a C con nacamesh.py (livelli: grossa, media, fine)
  python3 genera.py polare           incidenze 0, 2, 4, 6, 8, 10, 12, 15 gradi sulla mesh media: C_L, C_D, C_m, Cp
  python3 genera.py gci              alfa = 10 gradi su tre mesh: indice di convergenza di griglia (Roache)
  python3 genera.py dominio          alfa = 10 gradi con il contorno esterno a 20 e 40 corde: sensibilita' al dominio
  python3 genera.py tutto
L'incidenza e' imposta ruotando la velocita' asintotica (la mesh non cambia); liftDir e dragDir della
function object forceCoeffs ruotano con essa. Risultati in risultati.json; Cp e storia in dati/.
"""
import os
import sys
import json
import shutil
import subprocess
import numpy as np

QUI = os.path.dirname(os.path.abspath(__file__))
CODICE = next(d for d in (os.path.join(QUI, "..", "..", n) for n in ("codice", "codici")) if os.path.isdir(d))   # codici/ nell'archivio pubblicato
sys.path.insert(0, CODICE)
from foamcase import Caso                                        # noqa: E402
from foamlog import leggi_log                                    # noqa: E402
from nacamesh import scrivi_blockmesh                            # noqa: E402

LIVELLI = {  # nNorm, nRear, nFront, nWake, y1 (rapporto sqrt(2) fra livelli); gWake: celle della scia crescenti verso l'uscita
    "grossa": dict(nNorm=57, nRear=42, nFront=28, nWake=42, y1=2.83e-5, gWake=50),
    "media": dict(nNorm=80, nRear=60, nFront=40, nWake=60, y1=2e-5, gWake=50),
    "fine": dict(nNorm=113, nRear=85, nFront=57, nWake=85, y1=1.41e-5, gWake=50),
}
ITER_UPWIND = 300      # iterazioni della prima fase (upwind, rilassamento forte) dopo l'inizializzazione potenziale
ITER_MAX = 4000        # iterazioni della seconda fase (linearUpwind): le forze si assestano entro 2500-3500 (vedi storia)
RAGGIO = 20.0          # raggio dell'arco a monte e lunghezza della scia, in corde
PHI = """FoamFile { format ascii; class volScalarField; object Phi; }
// Potenziale di velocita' per potentialFoam: Neumann su tutto il contorno, cosi' il flusso al contorno e' quello di U.
// Senza questo file potentialFoam impone Phi = 0 sul patch 'esterno' (freestreamPressure e' una condizione mista,
// che 'fissa il valore') e restituisce un campo quasi in quiete: vedi testo.
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    esterno     { type zeroGradient; }
    profilo     { type zeroGradient; }
    frontEretro { type empty; }
}
"""
ALFA = [0, 4, 8, 10, 12, 15]


def sh(cmd, log=None):
    if log:
        with open(os.path.join(QUI, log), "w") as fp:
            return subprocess.run(cmd, cwd=QUI, stdout=fp, stderr=subprocess.STDOUT, shell=True).returncode
    r = subprocess.run(cmd, cwd=QUI, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return r.stdout


def imposta(file, voce, valore):
    sh(f'foamDictionary -entry "{voce}" -set "{valore}" {file}')


def pulisci():
    """Rimuove i risultati e ripristina 0/ da 0.orig (potentialFoam e incidenza() lo modificano)."""
    for d in os.listdir(QUI):
        if d[0].isdigit() and os.path.isdir(os.path.join(QUI, d)) and d != "0.orig":
            shutil.rmtree(os.path.join(QUI, d))
    shutil.copytree(os.path.join(QUI, "0.orig"), os.path.join(QUI, "0"))
    shutil.rmtree(os.path.join(QUI, "postProcessing"), ignore_errors=True)


def mesh(livello="media", R=RAGGIO, etichetta=None):
    scrivi_blockmesh(QUI, "0012", R=R, L=R, **LIVELLI[livello])
    etichetta = etichetta or livello
    sh("blockMesh", f"log.blockMesh_{etichetta}")
    sh("checkMesh", f"log.checkMesh_{etichetta}")
    return Caso(QUI).mesh.nCells


def incidenza(alfa):
    a = np.radians(alfa)
    U = f"uniform ({np.cos(a):.8f} {np.sin(a):.8f} 0)"
    imposta("0/U", "internalField", U)
    imposta("0/U", "boundaryField/esterno/value", U)      # valore iniziale del patch: e' il flusso al contorno di potentialFoam
    with open(os.path.join(QUI, "0", "Phi"), "w") as fp:
        fp.write(PHI)
    imposta("system/forzeDict", "coefficienti/liftDir", f"({-np.sin(a):.8f} {np.cos(a):.8f} 0)")
    imposta("system/forzeDict", "coefficienti/dragDir", f"({np.cos(a):.8f} {np.sin(a):.8f} 0)")


def coefficienti():
    """Legge postProcessing/coefficienti/*/coefficient.dat (una cartella per ogni avvio) e media le ultime 50 righe."""
    cart = os.path.join(QUI, "postProcessing", "coefficienti")
    tempi = sorted(os.listdir(cart), key=float)
    blocchi = []
    for t in tempi:
        f = [x for x in os.listdir(os.path.join(cart, t)) if x.endswith(".dat")][0]
        blocchi.append(np.loadtxt(os.path.join(cart, t, f), comments="#"))
        with open(os.path.join(cart, t, f)) as fp:
            intest = [l for l in fp if l.startswith("#")][-1].strip("# \n").split()
    d = np.vstack(blocchi)
    col = {n: i for i, n in enumerate(intest)}
    ult = d[-50:]
    return {k: float(ult[:, col[k]].mean()) for k in ("Cd", "Cl", "Cm") if k in col}, d, col


def esegui_alfa(alfa, etichetta):
    """Esecuzione in tre passi (vedi testo): potentialFoam per un campo iniziale solenoidale (con il file 0/Phi di
    Neumann); ITER_UPWIND iterazioni con upwind; poi linearUpwind fino a ITER_MAX. Il rilassamento resta 0,2/0,5 in
    entrambe le fasi: 0,3/0,7 e' stabile a R = 20 ma diverge a R = 40. I due registri di foamRun sono concatenati in
    log.foamRun_<etichetta> (foamlog li legge come un'unica storia)."""
    pulisci()
    incidenza(alfa)
    log = f"log.foamRun_{etichetta}"
    sh("potentialFoam", f"log.potentialFoam_{etichetta}")
    imposta("system/controlDict", "startFrom", "startTime")
    imposta("system/controlDict", "endTime", ITER_UPWIND)
    imposta("system/controlDict", "writeInterval", ITER_UPWIND)
    imposta("system/fvSchemes", "divSchemes/div(phi,U)", "bounded Gauss upwind")
    imposta("system/fvSchemes", "divSchemes/div(phi,nuTilda)", "bounded Gauss upwind")
    imposta("system/fvSolution", "relaxationFactors/fields/p", 0.2)
    imposta("system/fvSolution", "relaxationFactors/equations/U", 0.5)
    ok = sh("foamRun", log)
    imposta("system/controlDict", "startFrom", "latestTime")
    imposta("system/controlDict", "endTime", ITER_MAX)
    imposta("system/controlDict", "writeInterval", ITER_MAX)
    imposta("system/fvSchemes", "divSchemes/div(phi,U)", "bounded Gauss linearUpwind grad(U)")
    imposta("system/fvSchemes", "divSchemes/div(phi,nuTilda)", "bounded Gauss linearUpwind grad(nuTilda)")
    imposta("system/fvSolution", "relaxationFactors/fields/p", 0.2)
    imposta("system/fvSolution", "relaxationFactors/equations/U", 0.5)
    with open(os.path.join(QUI, log), "a") as fp:
        ok = ok or subprocess.run("foamRun", cwd=QUI, stdout=fp, stderr=subprocess.STDOUT, shell=True).returncode
    imposta("system/controlDict", "startFrom", "startTime")
    log = leggi_log(os.path.join(QUI, f"log.foamRun_{etichetta}"))
    c = Caso(QUI)
    co, storia, col = coefficienti()
    # Cp sul profilo: p cinematica / (0,5 U^2) con U = 1
    m = c.mesh
    idx = m.facce_patch("profilo")
    pf = c.valore_patch("p", "profilo")
    xy = m.Cf[idx, :2]
    yp = c.yplus("profilo", 1.6667e-7)
    nu = 1.6667e-7
    Fp, Fv = c.forze("profilo", nu)
    a = np.radians(alfa)
    L_py = -Fp[0] * np.sin(a) + Fp[1] * np.cos(a) - Fv[0] * np.sin(a) + Fv[1] * np.cos(a)
    D_py = (Fp[0] + Fv[0]) * np.cos(a) + (Fp[1] + Fv[1]) * np.sin(a)
    spess = m.points[:, 2].max() - m.points[:, 2].min()
    out = dict(alfa=alfa, Cl=co["Cl"], Cd=co["Cd"], Cm=co.get("Cm", float("nan")),
               Cl_python=float(2 * L_py / spess), Cd_python=float(2 * D_py / spess),
               Cd_pressione=float(2 * (Fp[0] * np.cos(a) + Fp[1] * np.sin(a)) / spess),
               Cd_attrito=float(2 * (Fv[0] * np.cos(a) + Fv[1] * np.sin(a)) / spess),
               iterazioni=len(log["campi"]["p"]["t"]), tempo=float(log["exec"][-1, 1]), ok=ok == 0,
               yplus_medio=float(yp.mean()), yplus_max=float(yp.max()),
               convergenza="SIMPLE solution converged" in open(os.path.join(QUI, f"log.foamRun_{etichetta}")).read())
    os.makedirs(os.path.join(QUI, "dati"), exist_ok=True)
    np.savez(os.path.join(QUI, "dati", f"{etichetta}.npz"), x=xy[:, 0], y=xy[:, 1], Cp=2 * pf,
             storia=storia, colonne=np.array(list(col)), res_t=log["campi"]["p"]["t"], res_p=log["campi"]["p"]["res0"],
             res_Ux=log["campi"]["Ux"]["res0"])
    print(f"{etichetta:12s} alfa {alfa:4.1f}: Cl {co['Cl']:.4f} Cd {co['Cd']:.5f} (python {out['Cl_python']:.4f} {out['Cd_python']:.5f};"
          f" pressione {out['Cd_pressione']:.5f} attrito {out['Cd_attrito']:.5f})  it {out['iterazioni']}  y+ medio {out['yplus_medio']:.2f}")
    return out


def carica():
    p = os.path.join(QUI, "risultati.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def main(cosa):
    ris = carica()
    if cosa in ("mesh",):
        n = mesh(sys.argv[2] if len(sys.argv) > 2 else "media"); print("celle:", n)
    if cosa in ("polare", "tutto"):
        n = mesh("media"); ris["celle_media"] = n
        ris["polare"] = [esegui_alfa(a, f"a{a:02d}") for a in ALFA]
        json.dump(ris, open(os.path.join(QUI, "risultati.json"), "w"), indent=1)
    if cosa in ("dominio", "tutto"):
        ris["dominio"] = {}
        for R in (40.0,):                            # il confronto con R = 20 e' il punto a 10 gradi della polare
            n = mesh("media", R=R, etichetta=f"R{R:.0f}")
            r = esegui_alfa(10, f"R{R:.0f}"); r["celle"] = n; r["R"] = R
            ris["dominio"][f"R{R:.0f}"] = r
            json.dump(ris, open(os.path.join(QUI, "risultati.json"), "w"), indent=1)
    if cosa in ("gci", "tutto"):
        ris["gci"] = {}
        for liv in ("grossa", "fine", "media"):      # la media per ultima: la mesh e i campi a 10 gradi restano nel caso
            n = mesh(liv)
            r = esegui_alfa(10, f"gci_{liv}"); r["celle"] = n
            ris["gci"][liv] = r
            json.dump(ris, open(os.path.join(QUI, "risultati.json"), "w"), indent=1)
    json.dump(ris, open(os.path.join(QUI, "risultati.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tutto")
