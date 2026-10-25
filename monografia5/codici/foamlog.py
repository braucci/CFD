"""
foamlog.py -- lettura del log di un solutore OpenFOAM: residui iniziali e finali di
ogni equazione, numero di iterazioni del solutore lineare, errori di continuita',
numero di Courant, tempo di calcolo; grafico dei residui in scala semilogaritmica.

Uso:   python3 foamlog.py casi/c01_cavita/log.foamRun [nome_figura]
       from foamlog import leggi_log;  d = leggi_log("log.foamRun");  d["p"]["t"], d["p"]["res0"]

Il log e' l'unico "strumento" sempre disponibile per giudicare una simulazione:
questo modulo fa in Python cio' che fa foamLog (lo script di OpenFOAM) e la
function object 'residuals', e si presta a essere esteso.
"""
import re
import sys
import numpy as np

RE_TEMPO = re.compile(r"^Time = ([-+0-9.eE]+)")
RE_RES = re.compile(r"(\w+):\s+Solving for (\w+), Initial residual = ([-+0-9.eE]+), Final residual = ([-+0-9.eE]+), No Iterations (\d+)")
RE_CONT = re.compile(r"time step continuity errors : sum local = ([-+0-9.eE]+), global = ([-+0-9.eE]+), cumulative = ([-+0-9.eE]+)")
RE_CO = re.compile(r"Courant Number mean: ([-+0-9.eE]+) max: ([-+0-9.eE]+)")
RE_EXEC = re.compile(r"ExecutionTime = ([-+0-9.eE]+) s")
RE_DT = re.compile(r"^deltaT = ([-+0-9.eE]+)")


def leggi_log(percorso):
    """Restituisce un dizionario: per ogni campo risolto {'t','res0','resF','iter','solver'} (array);
    inoltre 'continuita' (t, locale, globale, cumulativo), 'Co' (t, medio, max), 'exec' (t, s), 'deltaT'."""
    campi = {}
    cont, co, ex, dts = [], [], [], []
    t = None
    visti = set()
    with open(percorso, encoding="utf-8", errors="replace") as fp:
        for riga in fp:
            m = RE_TEMPO.match(riga)
            if m:
                t = float(m.group(1)); visti = set(); continue
            if t is None:
                continue
            m = RE_RES.search(riga)
            if m:
                solv, nome, r0, rf, it = m.groups()
                # in un passo con piu' correttori si tiene il PRIMO residuo iniziale (il piu' significativo)
                if nome in visti:
                    continue
                visti.add(nome)
                d = campi.setdefault(nome, {"t": [], "res0": [], "resF": [], "iter": [], "solver": solv})
                d["t"].append(t); d["res0"].append(float(r0)); d["resF"].append(float(rf)); d["iter"].append(int(it))
                continue
            m = RE_CONT.search(riga)
            if m:
                cont.append((t,) + tuple(float(x) for x in m.groups())); continue
            m = RE_CO.search(riga)
            if m:
                co.append((t, float(m.group(1)), float(m.group(2)))); continue
            m = RE_EXEC.search(riga)
            if m:
                ex.append((t, float(m.group(1)))); continue
            m = RE_DT.match(riga)
            if m:
                dts.append((t, float(m.group(1))))
    for d in campi.values():
        for k in ("t", "res0", "resF", "iter"):
            d[k] = np.array(d[k])
    out = {"campi": campi}
    out["continuita"] = np.array(cont) if cont else None
    out["Co"] = np.array(co) if co else None
    out["exec"] = np.array(ex) if ex else None
    out["deltaT"] = np.array(dts) if dts else None
    return out


def riassunto(d):
    """Stringa con l'ultimo residuo iniziale di ogni campo, il tempo di calcolo e il numero di passi."""
    righe = []
    for nome, c in d["campi"].items():
        righe.append(f"   {nome:10s} {c['solver']:14s} passi {len(c['t']):6d}  residuo iniziale finale {c['res0'][-1]:.2e}"
                     f"  (max {c['res0'].max():.2e})  iterazioni medie {c['iter'].mean():.1f}")
    if d["exec"] is not None:
        righe.append(f"   tempo di calcolo {d['exec'][-1, 1]:.1f} s per {len(d['exec'])} passi")
    if d["continuita"] is not None:
        righe.append(f"   errore di continuita' cumulativo {d['continuita'][-1, 3]:.2e}")
    if d["Co"] is not None:
        righe.append(f"   numero di Courant massimo {d['Co'][:, 2].max():.3f}")
    return "\n".join(righe)


def grafico_residui(d, ax, campi=None, iniziali=True, **kw):
    """Residui iniziali (o finali) in funzione del tempo/iterazione, in scala semilogaritmica."""
    for nome, c in d["campi"].items():
        if campi and nome not in campi:
            continue
        ax.semilogy(c["t"], c["res0"] if iniziali else c["resF"], label=nome, **kw)
    ax.set_xlabel("tempo / iterazione"); ax.set_ylabel("residuo iniziale" if iniziali else "residuo finale")
    ax.legend(fontsize=7)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from stile_grafici import salva
    d = leggi_log(sys.argv[1])
    print(riassunto(d))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    grafico_residui(d, ax, lw=1.0)
    fig.tight_layout()
    salva(fig, sys.argv[2] if len(sys.argv) > 2 else "residui")
