"""
prog2_7_cilindro_canale.py -- PROGRAMMA 2.7
Flusso potenziale in un canale con un cilindro circolare: contorno curvo che
non passa per i nodi della griglia (formula a cinque punti con bracci
disuguali) e condizione di derivata all'uscita (nodi fittizi).

Dominio (meta' superiore, per simmetria): 0 <= x <= 6, 0 <= y <= 2, cilindro di
raggio 1 centrato in (3, 0). Condizioni: psi = 0 sul cilindro e sull'asse,
psi = 1 sulla parete superiore e sulla parete d'ingresso (ingresso: piccola
apertura sull'asse, fra y = 0 e y = h), dpsi/dx = 0 all'uscita.
Metodo iterativo: Liebmann (Gauss-Seidel).
"""
import numpy as np
import matplotlib.pyplot as plt
from stile_grafici import salva, BLU, ARANCIO, ROSSO, GRIGIO, VERDE

XC, R, LX, LY = 3.0, 1.0, 6.0, 2.0


def dentro(x, y):
    """Vero per i nodi interni al cilindro o sulla sua superficie (psi = 0)."""
    return (x - XC)**2 + y**2 <= R**2 + 1e-12


def prepara(h):
    M, N = int(round(LX / h)) + 1, int(round(LY / h)) + 1
    x, y = np.arange(M) * h, np.arange(N) * h
    psi = np.zeros((M, N))
    psi[:, N - 1] = 1.0                    # parete superiore
    psi[0, 1:] = 1.0                       # parete d'ingresso (apertura fra y=0 e y=h)
    tipo = np.zeros((M, N), dtype=int)     # 0 regolare, 1 vicino al cilindro, -1 uscita, 9 contorno/solido
    tipo[0, :] = 9; tipo[:, 0] = 9; tipo[:, N - 1] = 9
    bracci = np.full((M, N, 4), h)         # a (sinistra), b (destra), c (sotto), d (sopra)
    for i in range(1, M):
        for j in range(1, N - 1):
            if dentro(x[i], y[j]):
                tipo[i, j] = 9             # nodo interno al cilindro: psi = 0
                continue
            if i == M - 1:
                tipo[i, j] = -1
                continue
            # distanza dal cilindro lungo le quattro direzioni, se il vicino e' interno
            if dentro(x[i + 1], y[j]):     # b: verso destra (lato sinistro del cilindro)
                bracci[i, j, 1] = (XC - np.sqrt(R**2 - y[j]**2)) - x[i]
            if dentro(x[i - 1], y[j]):     # a: verso sinistra (lato destro del cilindro)
                bracci[i, j, 0] = x[i] - (XC + np.sqrt(R**2 - y[j]**2))
            if dentro(x[i], y[j - 1]):     # c: verso il basso
                bracci[i, j, 2] = y[j] - np.sqrt(R**2 - (x[i] - XC)**2)
            # formula a bracci disuguali solo se almeno un braccio e' piu' corto di h
            if np.min(bracci[i, j]) < h - 1e-12:
                tipo[i, j] = 1
    interni = [(i, j) for i in range(1, M) for j in range(1, N - 1) if tipo[i, j] != 9]
    # valori di tentativo
    for i, j in interni:
        psi[i, j] = 0.5
    return x, y, psi, tipo, bracci, interni


def liebmann(h, tol=0.0005, kmax=20000):
    x, y, psi, tipo, br, interni = prepara(h)
    it, err = 0, 1.0
    while err > tol and it < kmax:
        it += 1
        err = 0.0
        for i, j in interni:
            vecchio = psi[i, j]
            if tipo[i, j] == 0:
                nuovo = 0.25 * (psi[i - 1, j] + psi[i + 1, j] + psi[i, j - 1] + psi[i, j + 1])
            elif tipo[i, j] == -1:         # uscita: psi_{M+1,j} = psi_{M-1,j}
                nuovo = 0.25 * (2.0 * psi[i - 1, j] + psi[i, j - 1] + psi[i, j + 1])
            else:                          # bracci disuguali: i nodi interni al cilindro
                a, b, c, d = br[i, j]      # valgono 0, come il punto di contorno
                nuovo = (psi[i - 1, j] / (a * (a + b)) + psi[i + 1, j] / (b * (a + b))
                         + psi[i, j - 1] / (c * (c + d)) + psi[i, j + 1] / (d * (c + d))) \
                    / (1.0 / (a * b) + 1.0 / (c * d))
            psi[i, j] = nuovo
            err += abs(nuovo - vecchio)
    return x, y, psi, tipo, br, it


def velocita_parete(x, y, psi, tipo, br, h):
    """Stima della velocita' sul cilindro nei nodi di tipo 1 piu' vicini alla
    superficie: V ~ psi / (distanza normale dalla parete)."""
    pts = []
    for i in range(len(x)):
        for j in range(len(y)):
            if tipo[i, j] == 1:
                dist = np.hypot(x[i] - XC, y[j]) - R
                th = np.degrees(np.arctan2(y[j], x[i] - XC))
                pts.append((th, psi[i, j] / dist))
    return np.array(sorted(pts))


if __name__ == "__main__":
    h = 0.25
    x, y, psi, tipo, br, it = liebmann(h)
    print(f"h = {h}: {it} iterazioni di Liebmann")
    print("nodi con formula a bracci disuguali (i, j da 1 come nel testo) e bracci a, b, c:")
    for i in range(len(x)):
        for j in range(len(y)):
            if tipo[i, j] == 1:
                a, b, c, d = br[i, j]
                print(f"   ({i+1:2d},{j+1}) x={x[i]:.2f} y={y[j]:.2f}:  a={a:.4f} b={b:.4f} c={c:.4f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    livelli = np.round(np.arange(0.1, 1.0, 0.1), 2)
    # i nodi solidi valgono 0: le curve vicine alla parete, che entrerebbero
    # leggermente nel cilindro, vengono coperte dal disegno del corpo
    ax.contour(x, y, psi.T, levels=livelli, colors=BLU, linewidths=0.9, linestyles="solid")
    ax.contour(x, -y[::-1], psi[:, ::-1].T, levels=livelli, colors=BLU, linewidths=0.9,
               linestyles="solid")
    t = np.linspace(0, 2 * np.pi, 200)
    ax.fill(XC + R * np.cos(t), R * np.sin(t), color="#d5d8dc", zorder=3)
    ax.plot(XC + R * np.cos(t), R * np.sin(t), "k", lw=1, zorder=4)
    ii, jj = np.nonzero(tipo == 1)
    ax.plot(x[ii], y[jj], "o", mfc="none", color=ROSSO, ms=4, zorder=5)
    ax.plot([0, 6], [2, 2], "k", lw=2); ax.plot([0, 6], [-2, -2], "k", lw=2)
    ax.plot([0, 0], [h, 2], "k", lw=2); ax.plot([0, 0], [-h, -2], "k", lw=2)
    ax.plot([6, 6], [-2, 2], ":", color=GRIGIO)
    ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.grid(False)
    salva(fig, "prog2_7_cilindro")

    # asimmetria a monte/valle e convergenza con il passo
    print("\n   h     iterazioni   psi(3, 1.5)   V media lato monte / lato valle (nodi vicini)")
    for hh in (0.25, 0.125, 0.0625):
        xx, yy, ps, tp, bb, k = liebmann(hh, tol=0.0005 * (0.25 / hh)**2)
        i, j = int(round(3.0 / hh)), int(round(1.5 / hh))
        vp = velocita_parete(xx, yy, ps, tp, bb, hh)
        monte = vp[vp[:, 0] > 90, 1].mean(); valle = vp[vp[:, 0] < 90, 1].mean()
        print(f"  {hh:6.4f}    {k:5d}       {ps[i, j]:.4f}        {monte:.3f} / {valle:.3f}")
