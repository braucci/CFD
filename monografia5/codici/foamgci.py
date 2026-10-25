"""
foamgci.py -- indice di convergenza di griglia (GCI) di Roache secondo la procedura ASME V&V 20 / Celik et al. (2008).

Date tre mesh di dimensione rappresentativa h1 < h2 < h3 (fine, media, grossa) e i tre valori f1, f2, f3 di una
grandezza integrale (C_L, C_D, una caduta di pressione...), si calcola:
  - l'ordine apparente p (iterazione a punto fisso, perche' i rapporti di raffinamento possono differire),
  - il valore estrapolato alla mesh infinitamente fine f_ext (Richardson),
  - l'errore relativo approssimato e_a e quello estrapolato e_ext,
  - il GCI della mesh fine, GCI = Fs |e_a| / (r^p - 1) con fattore di sicurezza Fs = 1,25.
Uso da riga di comando:  python3 foamgci.py h1 h2 h3 f1 f2 f3
"""
import sys
import numpy as np


def gci(h, f, Fs=1.25, iterazioni=50):
    """h = (h1,h2,h3) con h1<h2<h3, f = (f1,f2,f3) valori corrispondenti. Restituisce un dizionario."""
    h1, h2, h3 = h; f1, f2, f3 = f
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0:
        raise ValueError("due soluzioni coincidono: il GCI non e' definito")
    s = np.sign(e32 / e21)
    if s < 0:
        convergenza = "oscillatoria"
    else:
        convergenza = "monotona"
    # ordine apparente: p = |ln|e32/e21| + q(p)| / ln r21,  q(p) = ln((r21^p - s)/(r32^p - s))
    p = abs(np.log(abs(e32 / e21))) / np.log(r21)
    for _ in range(iterazioni):
        q = np.log((r21**p - s) / (r32**p - s))
        p_nuovo = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if abs(p_nuovo - p) < 1e-10:
            break
        p = p_nuovo
    f_ext = (r21**p * f1 - f2) / (r21**p - 1)
    e_a = abs((f1 - f2) / f1)
    e_ext = abs((f_ext - f1) / f_ext)
    GCI_fine = Fs * e_a / (r21**p - 1)
    GCI_media = Fs * abs((f2 - f3) / f2) / (r32**p - 1)
    # il rapporto GCI_media / (r^p GCI_fine) vale 1 nell'intervallo asintotico
    asintotico = GCI_media / (r21**p * GCI_fine)
    return dict(p=float(p), f_ext=float(f_ext), e_a=float(e_a), e_ext=float(e_ext), GCI_fine=float(GCI_fine), GCI_media=float(GCI_media),
                r21=float(r21), r32=float(r32), convergenza=convergenza, asintotico=float(asintotico))


def h_da_celle(N, dimensioni=2):
    """Dimensione rappresentativa h = (1/N)^(1/d) per una mesh di N celle in d dimensioni (dominio di misura unitaria)."""
    return (1.0 / np.asarray(N, float)) ** (1.0 / dimensioni)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(__doc__); sys.exit(1)
    v = [float(a) for a in sys.argv[1:]]
    r = gci(v[:3], v[3:])
    for k, val in r.items():
        print(f"{k:12s} {val}")
