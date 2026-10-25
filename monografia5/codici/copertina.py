"""copertina.py -- immagine di copertina: vorticita' della scia del cilindro (caso c16) con la mesh a O in trasparenza."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foamcase import Caso
from figutil import caso
c = Caso(caso("c16_cilindro")); m = c.mesh
om = np.clip(c.campo("vorticity")[:, 2], -4, 4)
fig = plt.figure(figsize=(12, 8), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("white")
x, y = m.C[:, 0], m.C[:, 1]
tri = mtri.Triangulation(x, y); cx = x[tri.triangles].mean(axis=1); cy = y[tri.triangles].mean(axis=1)
tri.set_mask(np.hypot(cx, cy) < 0.52)
ax.tricontourf(tri, om, levels=np.linspace(-4, 4, 41), cmap="RdBu_r", alpha=0.95)
m.disegna(ax, lw=0.15, colore="#1E2761")
ax.add_patch(plt.Circle((0, 0), 0.53, color="#1E2761"))
ax.set_xlim(-2.5, 9.5); ax.set_ylim(-4, 4); ax.set_aspect("equal"); ax.axis("off")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "immagini", "copertina.png")
fig.savefig(out, dpi=100, facecolor="white"); print("scritto", out)
