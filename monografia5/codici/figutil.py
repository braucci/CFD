"""
figutil.py -- funzioni comuni agli script fig*.py: disegno di campi 2D letti con foamcase
(triangolazione dei centri di cella), scrittura dei riquadri di risultati in LaTeX,
formattazione dei numeri alla italiana.
"""
import os
import numpy as np
import matplotlib.tri as mtri

QUI = os.path.dirname(os.path.abspath(__file__))
CASI = os.path.join(QUI, "..", "casi")
CAPITOLI = os.path.join(QUI, "..", "capitoli")


def caso(nome):
    return os.path.join(CASI, nome)


def it(x, cifre=3, esp=False):
    """Numero con la virgola decimale italiana (per i riquadri LaTeX)."""
    if esp:
        s = f"{x:.{cifre}e}"
        m, e = s.split("e")
        return f"${m.replace('.', '{,}')}\\times10^{{{int(e)}}}$"
    s = f"{x:.{cifre}f}" if isinstance(cifre, int) else f"{x:{cifre}}"
    return s.replace(".", "{,}")


def scrivi_tex(nome, testo):
    """Scrive capitoli/gen_<nome>.tex (incluso dai capitoli con \\input)."""
    p = os.path.join(CAPITOLI, f"gen_{nome}.tex")
    with open(p, "w", encoding="utf-8") as fp:
        fp.write("% file generato automaticamente da codice/fig*.py: non modificare a mano\n" + testo.strip() + "\n")
    print(f"  -> scritto {os.path.relpath(p, QUI)}")


def triangolazione(mesh, maschera_lato=None):
    """Triangolazione di Delaunay dei centri di cella (mesh 2D nel piano xy)."""
    x, y = mesh.C[:, 0], mesh.C[:, 1]
    tri = mtri.Triangulation(x, y)
    if maschera_lato is not None:
        # elimina i triangoli con lati piu' lunghi di una soglia (attraversano i corpi)
        p = tri.triangles
        l = np.max([np.hypot(x[p[:, i]] - x[p[:, j]], y[p[:, i]] - y[p[:, j]]) for i, j in ((0, 1), (1, 2), (2, 0))], axis=0)
        tri.set_mask(l > maschera_lato)
    return tri


def campo2d(ax, mesh, campo, livelli=21, cmap="viridis", maschera_lato=None, linee=False, colore_linee="k", lw=0.4, **kw):
    """Riempimento a colori (e isolinee opzionali) di un campo scalare di cella."""
    tri = triangolazione(mesh, maschera_lato)
    cf = ax.tricontourf(tri, campo, levels=livelli, cmap=cmap, **kw)
    if linee:
        ax.tricontour(tri, campo, levels=livelli if np.ndim(livelli) else 10, colors=colore_linee, linewidths=lw)
    ax.set_aspect("equal")
    return cf


def linee_corrente(ax, mesh, U, n=200, densita=1.2, colore="k", lw=0.5, estensione=None):
    """Linee di corrente di un campo 2D interpolato su una griglia regolare."""
    from scipy.interpolate import griddata
    x, y = mesh.C[:, 0], mesh.C[:, 1]
    x0, x1, y0, y1 = estensione if estensione else (x.min(), x.max(), y.min(), y.max())
    gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    u = griddata((x, y), U[:, 0], (gx, gy), method="linear")
    v = griddata((x, y), U[:, 1], (gx, gy), method="linear")
    ax.streamplot(gx, gy, u, v, density=densita, color=colore, linewidth=lw, arrowsize=0.6)
    ax.set_aspect("equal")
