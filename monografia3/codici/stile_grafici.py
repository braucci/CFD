"""
stile_grafici.py -- impostazioni grafiche comuni (palette "della casa").
Importato da tutti gli script della monografia.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Palette coerente con raucci-aero.sty
BLU      = "#1E2761"
ARANCIO  = "#DC7633"
ROSSO    = "#C0392B"
VERDE    = "#1E8449"
GIALLO   = "#B7950B"
VIOLA    = "#6C3483"
GRIGIO   = "#566573"
PALETTE  = [BLU, ARANCIO, VERDE, VIOLA, ROSSO, GIALLO, GRIGIO]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "lines.linewidth": 1.4,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

# Le figure vengono salvate nella cartella "figure" accanto a quella degli script
# (creata se manca): cosi' ogni programma funziona anche se scaricato da solo.
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figure")


def salva(fig, nome):
    """Salva la figura in PDF nella cartella delle figure del progetto LaTeX."""
    os.makedirs(FIG_DIR, exist_ok=True)
    from matplotlib.ticker import NullFormatter
    for ax in fig.get_axes():              # niente etichette secondarie sugli assi logaritmici
        if ax.get_xscale() == "log":
            ax.xaxis.set_minor_formatter(NullFormatter())
        if ax.get_yscale() == "log":
            ax.yaxis.set_minor_formatter(NullFormatter())
    fig.savefig(os.path.join(FIG_DIR, f"{nome}.pdf"))
    plt.close(fig)
    print(f"  -> figura salvata: {nome}.pdf")
