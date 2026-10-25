"""
fig11_sod.py -- figura e risultati del caso c11_tubo_urto (Cap. 10): soluzione esatta del problema di Riemann di
Sod (gas perfetto, gamma = 1,4) confrontata con shockFluid; varianti con tre ricostruzioni (vanLeer, vanAlbada,
Minmod) e due risoluzioni (1000 e 4000 celle) eseguite qui in copie del caso.
Uso: python3 fig11_sod.py   (dopo casi/c11_tubo_urto/Allrun)
"""
import os
import re
import sys
import json
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stile_grafici import salva, BLU, ARANCIO, VERDE, ROSSO, GRIGIO
from foamcase import Caso
from figutil import caso, it, scrivi_tex

C = caso("c11_tubo_urto")
GAMMA = 1.4; RGAS = 8314.47 / 28.96; T_FIN = 0.007
pL, TL, pR, TR = 1e5, 348.432, 1e4, 278.746
rhoL, rhoR = pL / (RGAS * TL), pR / (RGAS * TR)


def riemann_esatto(x, t):
    """Soluzione esatta del problema di Riemann per gas perfetto (stati in quiete): p, rho, u in x al tempo t."""
    g = GAMMA; aL = np.sqrt(g * pL / rhoL); aR = np.sqrt(g * pR / rhoR)
    def f(p):   # u_star da sinistra (ventaglio) meno u_star da destra (urto)
        uL = 2 * aL / (g - 1) * (1 - (p / pL) ** ((g - 1) / (2 * g)))
        uR = (p - pR) * np.sqrt(2 / ((g + 1) * rhoR) / (p + (g - 1) / (g + 1) * pR))
        return uL - uR
    ps = brentq(f, pR, pL)
    us = 2 * aL / (g - 1) * (1 - (ps / pL) ** ((g - 1) / (2 * g)))
    rho_sL = rhoL * (ps / pL) ** (1 / g)                                   # dietro il ventaglio
    rho_sR = rhoR * (ps / pR + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * ps / pR + 1)   # dietro l'urto
    S = us * rho_sR / (rho_sR - rhoR)                                      # velocità dell'urto
    a_sL = np.sqrt(g * ps / rho_sL)
    xi = x / t
    p = np.empty_like(x); rho = np.empty_like(x); u = np.empty_like(x)
    m = xi < -aL;                       p[m], rho[m], u[m] = pL, rhoL, 0.0
    m = (xi >= -aL) & (xi < us - a_sL)  # ventaglio di espansione
    uu = 2 / (g + 1) * (aL + xi[m]); aa = aL - (g - 1) / 2 * uu
    p[m] = pL * (aa / aL) ** (2 * g / (g - 1)); rho[m] = rhoL * (aa / aL) ** (2 / (g - 1)); u[m] = uu
    m = (xi >= us - a_sL) & (xi < us);  p[m], rho[m], u[m] = ps, rho_sL, us
    m = (xi >= us) & (xi < S);          p[m], rho[m], u[m] = ps, rho_sR, us
    m = xi >= S;                        p[m], rho[m], u[m] = pR, rhoR, 0.0
    return p, rho, u, dict(p_star=ps, u_star=us, rho_sL=rho_sL, rho_sR=rho_sR, S=S, x_urto=S * t, x_contatto=us * t,
                           x_testa=-aL * t, x_coda=(us - a_sL) * t)


def esegui_variante(ricostruzione, N):
    dest = C + f"_{ricostruzione}_{N}"
    if not os.path.isdir(dest):
        shutil.copytree(C, dest, ignore=shutil.ignore_patterns("0.0*", "[1-9]*", "log.*", "postProcessing"))
        p = os.path.join(dest, "system", "fvSchemes"); s = open(p).read()
        s = re.sub(r"reconstruct\(rho\)\s+\w+;", f"reconstruct(rho) {ricostruzione};", s)
        s = re.sub(r"reconstruct\(U\)\s+\w+;", f"reconstruct(U)  {ricostruzione}V;", s)
        s = re.sub(r"reconstruct\(T\)\s+\w+;", f"reconstruct(T)  {ricostruzione};", s)
        open(p, "w").write(s)
        p = os.path.join(dest, "system", "blockMeshDict"); s = open(p).read()
        open(p, "w").write(re.sub(r"\(1000 1 1\)", f"({N} 1 1)", s))
        if N != 1000:
            p = os.path.join(dest, "system", "controlDict"); s = open(p).read()
            open(p, "w").write(re.sub(r"(deltaT\s+)[^;]*;", rf"\g<1>{5e-6 * 1000 / N:g};", s))
        subprocess.run("./Allrun > /dev/null", cwd=dest, shell=True, check=True)
    return Caso(dest)


def leggi(c):
    x = c.mesh.C[:, 0]; o = np.argsort(x)
    p = c.campo("p")[o]; T = c.campo("T")[o]; U = c.campo("U")[o, 0]
    return x[o], p, p / (RGAS * T), U


ris = {}
dati = {}
for N in (1000, 4000):
    for rc in ("vanLeer", "vanAlbada", "Minmod"):
        c = Caso(C) if (rc == "vanAlbada" and N == 1000) else esegui_variante(rc, N)
        x, p, rho, u = leggi(c)
        pe, rhoe, ue, info = riemann_esatto(x, T_FIN)
        # errore L1 della densità (normalizzato) e larghezza numerica dell'urto (celle fra il 5% e il 95% del salto)
        err = float(np.mean(np.abs(rho - rhoe)) / (rhoL - rhoR))
        salto = info["rho_sR"] - rhoR
        vicino = (x > info["x_urto"] - 0.5) & (x < info["x_urto"] + 0.5)
        xs, rs = x[vicino], rho[vicino]
        try:
            x95 = np.interp(rhoR + 0.95 * salto, rs[::-1], xs[::-1]); x05 = np.interp(rhoR + 0.05 * salto, rs[::-1], xs[::-1])
            larg = float((x05 - x95) / (10.0 / N))
        except Exception:
            larg = float("nan")
        # posizione dell'urto numerica (dove rho attraversa il valore medio) e overshoot dietro l'urto
        x_urto_num = float(np.interp(rhoR + 0.5 * salto, rs[::-1], xs[::-1]))
        plateau = (x > info["x_contatto"] + 0.3) & (x < info["x_urto"] - 0.3)
        over = float((rho[plateau].max() - info["rho_sR"]) / info["rho_sR"])
        ris[f"{rc}|{N}"] = dict(errore_L1_rho=err, larghezza_urto_celle=larg, x_urto=x_urto_num, x_urto_esatto=info["x_urto"], overshoot=over)
        dati[f"{rc}|{N}"] = (x, p, rho, u)
        print(f"{rc:10s} N = {N}: errore L1 rho {err:.4e}  larghezza urto {larg:.1f} celle  urto in {x_urto_num:.4f} (esatto {info['x_urto']:.4f})  overshoot {over:+.2e}")

x, p, rho, u = dati["vanAlbada|1000"]
xe = np.linspace(-5, 5, 4001); pe, rhoe, ue, info = riemann_esatto(xe, T_FIN)
fig, ax = plt.subplots(1, 4, figsize=(10.5, 3.0), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.1]))
for k, (q, qe, et) in enumerate(((p / 1e5, pe / 1e5, "$p/p_L$"), (rho, rhoe, "$\\rho$ [kg/m$^3$]"), (u, ue, "$u$ [m/s]"))):
    ax[k].plot(xe, qe, color=GRIGIO, lw=1.6, alpha=0.7, label="esatta")
    ax[k].plot(x, q, "-", color=BLU, lw=0.9, label="shockFluid, vanAlbada")
    ax[k].set_xlabel("$x$ [m]"); ax[k].set_ylabel(et)
ax[0].legend(fontsize=7)
for rc, col in (("vanLeer", ARANCIO), ("vanAlbada", BLU), ("Minmod", VERDE)):
    xx, _, rr, _ = dati[f"{rc}|1000"]
    ax[3].plot(xx, rr, "o-", color=col, ms=2.2, lw=0.8, label=rc)
ax[3].plot(xe, rhoe, color=GRIGIO, lw=1.6, alpha=0.7)
ax[3].set_xlim(info["x_contatto"] - 0.5, info["x_urto"] + 0.4); ax[3].set_ylim(0.05, 0.5); ax[3].set_xlabel("$x$ [m]"); ax[3].set_ylabel("$\\rho$ [kg/m$^3$]")
ax[3].legend(fontsize=7); ax[3].set_title("contatto e urto, 1000 celle", fontsize=9)
fig.tight_layout(); salva(fig, "c11_sod")

json.dump(ris, open(os.path.join(C, "risultati.json"), "w"), indent=1)
r = ris
righe = ["\\begin{itemize}"]
righe.append(f"\\item Soluzione esatta a $t=7$~ms: $p^*={it(info['p_star']/1e5,4)}\\,p_L$, $u^*={it(info['u_star'],1)}$~m/s, densità ${it(info['rho_sL'],4)}$ (dietro il ventaglio) e ${it(info['rho_sR'],4)}$~kg/m$^3$ (dietro l'urto); "
             f"urto in $x={it(info['x_urto'],3)}$~m (velocità ${it(info['S'],1)}$~m/s), contatto in $x={it(info['x_contatto'],3)}$~m, ventaglio fra ${it(info['x_testa'],3)}$ e ${it(info['x_coda'],3)}$~m.")
righe.append(f"\\item Con 1000 celle e \\kw{{vanAlbada}} l'urto è in $x={it(r['vanAlbada|1000']['x_urto'],3)}$~m ed è risolto in {it(r['vanAlbada|1000']['larghezza_urto_celle'],1)} celle; "
             f"il plateau dietro l'urto ha uno scostamento di {it(100*r['vanAlbada|1000']['overshoot'],2)}\\% dal valore esatto: le posizioni e i salti sono quelli giusti perché lo schema è conservativo.")
righe.append("\\item Errore $L_1$ sulla densità (normalizzato con $\\rho_L-\\rho_R$):\\\\[2pt]\\begin{tabular}{lccc} celle & \\kw{vanLeer} & \\kw{vanAlbada} & \\kw{Minmod}\\\\\\hline "
             + " ".join(f"{N} & " + " & ".join(it(r[f'{rc}|{N}']['errore_L1_rho'], 2, True) for rc in ('vanLeer', 'vanAlbada', 'Minmod')) + "\\\\" for N in (1000, 4000))
             + "\\end{tabular}\\\\[2pt]" + f"Passando da 1000 a 4000 celle l'errore si riduce di {it(r['vanAlbada|1000']['errore_L1_rho']/r['vanAlbada|4000']['errore_L1_rho'],2)} volte: "
             f"ordine osservato {it(np.log(r['vanAlbada|1000']['errore_L1_rho']/r['vanAlbada|4000']['errore_L1_rho'])/np.log(4),2)}, cioè fra $1/2$ e $1$ come ci si aspetta quando la soluzione ha discontinuità "
             "(l'ordine formale dello schema vale solo nelle regioni lisce; l'urto e il contatto occupano un numero fisso di celle, quindi il loro contributo all'errore scala come $h$, "
             "e il contatto, che non si autocomprime come l'urto, si allarga nel tempo).")
righe.append(f"\\item \\kw{{Minmod}} è il più diffusivo (urto risolto in {it(r['Minmod|1000']['larghezza_urto_celle'],0)} celle contro {it(r['vanLeer|1000']['larghezza_urto_celle'],0)}), \\kw{{vanLeer}} il più netto; la differenza fra i tre è del "
             f"{100*(max(r[f'{rc}|1000']['errore_L1_rho'] for rc in ('vanLeer','vanAlbada','Minmod'))/min(r[f'{rc}|1000']['errore_L1_rho'] for rc in ('vanLeer','vanAlbada','Minmod'))-1):.0f}\\% sull'errore $L_1$.")
righe.append("\\end{itemize}")
scrivi_tex("c11_risultati", "\n".join(righe))
