"""
foamcase.py -- lettura di un caso OpenFOAM in Python puro (NumPy), nello spirito
di uFVM: la mesh poliedrica (points, faces, owner, neighbour, boundary), i campi
(volScalarField, volVectorField, in formato ASCII, anche compressi .gz) e le
grandezze geometriche che il metodo dei volumi finiti richiede (centri e aree
delle facce, centri e volumi delle celle, vettore d_CF, non ortogonalita').

Uso tipico:
    from foamcase import Caso
    c = Caso("casi/c01_cavita")
    m = c.mesh                      # oggetto Mesh: m.C (centri celle), m.V, m.Cf, m.Sf, m.owner, ...
    p = c.campo("p")                # ultimo istante; c.campo("U", tempo=0.5)
    y, u = c.linea("U", (0.5, 0, 0.005), (0.5, 1, 0.005), 100)     # campionamento lungo un segmento

Il modulo e' volutamente semplice (nessuna dipendenza oltre NumPy e, per il
campionamento, SciPy): serve a CAPIRE che cosa contengono i file di un caso,
non a sostituire ParaView o le function object di OpenFOAM.
"""
import os
import re
import gzip
import numpy as np


# =============================================================================
# lettura dei file di OpenFOAM (ASCII)
# =============================================================================
def _apri(percorso):
    """Apre un file di OpenFOAM, anche nella versione compressa .gz."""
    if os.path.exists(percorso):
        return open(percorso, encoding="utf-8", errors="replace").read()
    if os.path.exists(percorso + ".gz"):
        return gzip.open(percorso + ".gz", "rt", encoding="utf-8", errors="replace").read()
    raise FileNotFoundError(percorso)


def _senza_commenti(testo):
    testo = re.sub(r"/\*.*?\*/", "", testo, flags=re.S)
    return re.sub(r"//[^\n]*", "", testo)


def _intestazione(testo):
    """Il dizionario FoamFile { ... } in testa a ogni file."""
    m = re.search(r"FoamFile\s*\{(.*?)\}", testo, flags=re.S)
    voci = {}
    if m:
        for riga in m.group(1).split(";"):
            parti = riga.split()
            if len(parti) >= 2:
                voci[parti[0]] = parti[1]
    return voci


def _lista_numeri(testo, inizio):
    """Legge 'N ( ... )' a partire da testo[inizio:]; restituisce (array, posizione dopo la parentesi)."""
    m = re.compile(r"\s*(\d+)\s*\(").match(testo, inizio)
    n = int(m.group(1))
    i = m.end()
    j = testo.index(")", i) if n == 0 else _chiudi(testo, i - 1)
    corpo = testo[i:j].replace("(", " ").replace(")", " ")
    return np.array(corpo.split(), dtype=float), j + 1, n


def _chiudi(testo, i):
    """Indice della parentesi che chiude quella aperta in testo[i]."""
    livello = 0
    for k in range(i, len(testo)):
        if testo[k] == "(":
            livello += 1
        elif testo[k] == ")":
            livello -= 1
            if livello == 0:
                return k
    raise ValueError("parentesi non chiusa")


def leggi_lista_etichette(percorso):
    """owner, neighbour: lista di interi."""
    t = _senza_commenti(_apri(percorso))
    t = t[t.index("}", t.index("FoamFile")) + 1:]           # dopo l'intestazione
    v, _, _ = _lista_numeri(t, 0)
    return v.astype(int)


def leggi_punti(percorso):
    t = _senza_commenti(_apri(percorso))
    t = t[t.index("}", t.index("FoamFile")) + 1:]
    v, _, n = _lista_numeri(t, 0)
    return v.reshape(n, 3)


def leggi_facce(percorso):
    """faces: lista di liste di etichette di punti (facce con numero di vertici variabile)."""
    t = _senza_commenti(_apri(percorso))
    t = t[t.index("}", t.index("FoamFile")) + 1:]
    m = re.compile(r"\s*(\d+)\s*\(").match(t, 0)
    n = int(m.group(1))
    corpo = t[m.end():_chiudi(t, m.end() - 1)]
    facce = [np.array(f.split(), dtype=int) for f in re.findall(r"\d+\s*\(([^)]*)\)", corpo)]
    assert len(facce) == n, "numero di facce incoerente"
    return facce


def leggi_boundary(percorso):
    """boundary: lista di dizionari (nome, type, nFaces, startFace, ...) nell'ordine del file."""
    t = _senza_commenti(_apri(percorso))
    t = t[t.index("}", t.index("FoamFile")) + 1:]
    patches = []
    for m in re.finditer(r"(\w+)\s*\{([^{}]*)\}", t):
        voci = {"nome": m.group(1)}
        for riga in m.group(2).split(";"):
            parti = riga.split()
            if len(parti) >= 2:
                voci[parti[0]] = parti[1] if len(parti) == 2 else " ".join(parti[1:])
        voci["nFaces"] = int(voci["nFaces"]); voci["startFace"] = int(voci["startFace"])
        patches.append(voci)
    return patches


def leggi_campo(percorso, n_celle=None):
    """Campo vol*Field ASCII: restituisce (interno, contorno, dimensioni).
    interno: array (n,) o (n,3); contorno: dizionario patch -> {"type": ..., "value": array o None}."""
    t = _senza_commenti(_apri(percorso))
    intest = _intestazione(t)
    classe = intest.get("class", "volScalarField")
    ncomp = 3 if "Vector" in classe else (6 if "SymmTensor" in classe else (9 if "Tensor" in classe else 1))
    dim = re.search(r"dimensions\s*\[([^\]]*)\]", t)
    dimensioni = [float(x) for x in dim.group(1).split()] if dim else None
    interno = _valore(t, t.index("internalField"), ncomp, n_celle)
    contorno = {}
    ib = t.find("boundaryField")
    if ib >= 0:
        i = t.index("{", ib) + 1
        corpo = t[i:_chiudi_graffa(t, i - 1)]
        for m in re.finditer(r"([\w\"|().*]+)\s*\{", corpo):
            nome = m.group(1).strip('"')
            j = m.end()
            fine = _chiudi_graffa(corpo, j - 1)
            blocco = corpo[j:fine]
            tipo = re.search(r"type\s+(\w+)", blocco)
            voce = {"type": tipo.group(1) if tipo else None, "value": None}
            mv = re.search(r"(?<![\w])value\s+(uniform|nonuniform)", blocco)   # la voce 'value' (non inletValue, freestreamValue...)
            if mv:
                voce["value"] = _valore(blocco, mv.start(), ncomp, None)
            contorno[nome] = voce
    return interno, contorno, dimensioni


def _chiudi_graffa(testo, i):
    livello = 0
    for k in range(i, len(testo)):
        if testo[k] == "{":
            livello += 1
        elif testo[k] == "}":
            livello -= 1
            if livello == 0:
                return k
    raise ValueError("graffa non chiusa")


def _valore(testo, i, ncomp, n):
    """Legge 'uniform x' / 'uniform (x y z)' / 'nonuniform List<...> N (...)' a partire da testo[i:]."""
    m = re.compile(r"\w+\s+(uniform|nonuniform)").match(testo, i)
    if m.group(1) == "uniform":
        j = m.end()
        if ncomp == 1:
            v = float(re.compile(r"\s*([-+0-9.eE]+)").match(testo, j).group(1))
            return np.full(n, v) if n else v
        mm = re.compile(r"\s*\(([^)]*)\)").match(testo, j)
        v = np.array(mm.group(1).split(), dtype=float)
        return np.tile(v, (n, 1)) if n else v
    j = testo.index(">", m.end()) + 1
    v, _, nn = _lista_numeri(testo, j)
    return v.reshape(nn, ncomp) if ncomp > 1 else v


# =============================================================================
# la mesh e le sue grandezze geometriche
# =============================================================================
class Mesh:
    """Mesh poliedrica di OpenFOAM con le grandezze geometriche calcolate come fa il codice:
    centro e area di ogni faccia per decomposizione in triangoli intorno al centro geometrico,
    centro e volume di ogni cella per decomposizione in piramidi con vertice nel centro stimato."""

    def __init__(self, cartella):
        pm = os.path.join(cartella, "constant", "polyMesh")
        self.points = leggi_punti(os.path.join(pm, "points"))
        self.faces = leggi_facce(os.path.join(pm, "faces"))
        self.owner = leggi_lista_etichette(os.path.join(pm, "owner"))
        self.neighbour = leggi_lista_etichette(os.path.join(pm, "neighbour"))
        self.boundary = leggi_boundary(os.path.join(pm, "boundary"))
        self.nFaces = len(self.faces)
        self.nInternalFaces = len(self.neighbour)
        self.nCells = int(self.owner.max()) + 1
        self.nPoints = len(self.points)
        self._geometria()

    # --- facce ---------------------------------------------------------------
    def _geometria(self):
        P = self.points
        Cf = np.zeros((self.nFaces, 3)); Sf = np.zeros((self.nFaces, 3))
        for i, f in enumerate(self.faces):
            pts = P[f]
            if len(f) == 3:
                Cf[i] = pts.mean(axis=0)
                Sf[i] = 0.5 * np.cross(pts[1] - pts[0], pts[2] - pts[0])
                continue
            cg = pts.mean(axis=0)                  # stima del centro
            sumN = np.zeros(3); sumA = 0.0; sumAc = np.zeros(3)
            for k in range(len(f)):
                a, b = pts[k], pts[(k + 1) % len(f)]
                n = 0.5 * np.cross(b - a, cg - a)  # normale del triangolo (a, b, cg)
                area = np.linalg.norm(n)
                sumN += n; sumA += area; sumAc += area * (a + b + cg) / 3
            Cf[i] = sumAc / sumA if sumA > 1e-300 else cg
            Sf[i] = sumN
        self.Cf, self.Sf = Cf, Sf
        self.magSf = np.linalg.norm(Sf, axis=1)
        # --- celle: piramidi faccia-centro ---
        nC = self.nCells
        cEst = np.zeros((nC, 3)); nF = np.zeros(nC)
        for i in range(self.nFaces):
            cEst[self.owner[i]] += Cf[i]; nF[self.owner[i]] += 1
            if i < self.nInternalFaces:
                cEst[self.neighbour[i]] += Cf[i]; nF[self.neighbour[i]] += 1
        cEst /= nF[:, None]
        V = np.zeros(nC); C = np.zeros((nC, 3))
        for i in range(self.nFaces):
            o = self.owner[i]
            pyr = np.dot(Sf[i], Cf[i] - cEst[o])   # 3 x volume della piramide (con segno)
            V[o] += pyr; C[o] += pyr * (0.75 * Cf[i] + 0.25 * cEst[o])
            if i < self.nInternalFaces:
                nb = self.neighbour[i]
                pyr = -np.dot(Sf[i], Cf[i] - cEst[nb])
                V[nb] += pyr; C[nb] += pyr * (0.75 * Cf[i] + 0.25 * cEst[nb])
        self.C = C / V[:, None]
        self.V = V / 3.0

    # --- grandezze derivate ---------------------------------------------------
    def dCF(self):
        """Vettore dal centro della cella owner a quello della neighbour (facce interne)."""
        n = self.nInternalFaces
        return self.C[self.neighbour] - self.C[self.owner[:n]]

    def non_ortogonalita(self):
        """Angolo (gradi) fra S_f e d_CF per ogni faccia interna. checkMesh ne riporta il massimo e la
        'media' calcolata come arccos della media dei coseni (metodo media_non_ortogonalita)."""
        d = self.dCF(); S = self.Sf[:self.nInternalFaces]
        cosang = np.einsum("ij,ij->i", d, S) / (np.linalg.norm(d, axis=1) * np.linalg.norm(S, axis=1))
        self._cos_no = np.clip(cosang, -1, 1)
        return np.degrees(np.arccos(self._cos_no))

    def media_non_ortogonalita(self):
        self.non_ortogonalita()
        return np.degrees(np.arccos(self._cos_no.mean()))

    def skewness(self, contorno=True):
        """Skewness con la definizione di checkMesh. Facce interne: distanza fra il centro della faccia e
        il punto in cui la retta C-F la attraversa, normalizzata con la semi-estensione della faccia nella
        direzione dello scostamento (non meno di 0,2 |d_CF|). Facce di contorno: scostamento fra il centro
        della faccia e il piede della normale dal centro della cella (normalizzazione non meno di 0,4 |d|)."""
        n = self.nInternalFaces
        C_o = self.C[self.owner[:n]]; d = self.dCF(); S = self.Sf[:n]; Cf = self.Cf[:n]
        Cpf = Cf - C_o
        lam = np.einsum("ij,ij->i", S, Cpf) / (np.einsum("ij,ij->i", S, d) + 1e-300)
        sv = Cpf - lam[:, None] * d                       # vettore di scostamento (giace sulla faccia)
        msv = np.linalg.norm(sv, axis=1)
        svHat = sv / (msv[:, None] + 1e-300)
        fd = 0.2 * np.linalg.norm(d, axis=1)
        for i in range(n):
            if msv[i] > 0:
                fd[i] = max(fd[i], np.abs((self.points[self.faces[i]] - Cf[i]) @ svHat[i]).max())
        sk = msv / fd
        if not contorno:
            return sk
        idx = np.arange(n, self.nFaces)
        Cpf = self.Cf[idx] - self.C[self.owner[idx]]
        nrm = self.Sf[idx] / (self.magSf[idx][:, None] + 1e-300)
        dd = np.einsum("ij,ij->i", nrm, Cpf)[:, None] * nrm
        sv = Cpf - dd; msv = np.linalg.norm(sv, axis=1); svHat = sv / (msv[:, None] + 1e-300)
        fd = 0.4 * np.linalg.norm(dd, axis=1)
        for k, i in enumerate(idx):
            if msv[k] > 0:
                fd[k] = max(fd[k], np.abs((self.points[self.faces[i]] - self.Cf[i]) @ svHat[k]).max())
        return np.concatenate([sk, msv / fd])

    def patch(self, nome):
        for b in self.boundary:
            if b["nome"] == nome:
                return b
        raise KeyError(nome)

    def facce_patch(self, nome):
        b = self.patch(nome)
        return np.arange(b["startFace"], b["startFace"] + b["nFaces"])

    def piano_xy(self, z=None):
        """Per una mesh 2D (una cella di spessore): indici delle facce 'empty' su un piano, per disegnarla."""
        import collections
        facce = []
        for b in self.boundary:
            if b.get("type") == "empty":
                idx = self.facce_patch(b["nome"])
                zf = self.Cf[idx, 2]
                facce.append(idx[np.isclose(zf, zf.min())] if z is None else idx[np.isclose(zf, z)])
        return np.concatenate(facce) if facce else np.array([], dtype=int)

    def disegna(self, ax, colore="#566573", lw=0.3):
        """Disegna il reticolo di una mesh 2D (facce empty del piano inferiore) con Matplotlib."""
        from matplotlib.collections import LineCollection
        segs = []
        for i in self.piano_xy():
            f = self.faces[i]; pts = self.points[f][:, :2]
            segs.extend([(pts[k], pts[(k + 1) % len(f)]) for k in range(len(f))])
        ax.add_collection(LineCollection(segs, colors=colore, linewidths=lw))
        ax.autoscale()


# =============================================================================
# il caso
# =============================================================================
class Caso:
    def __init__(self, cartella):
        self.cartella = cartella
        self._mesh = None

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = Mesh(self.cartella)
        return self._mesh

    def tempi(self):
        """Le cartelle temporali presenti, ordinate."""
        t = []
        for d in os.listdir(self.cartella):
            try:
                v = float(d)
            except ValueError:
                continue
            if os.path.isdir(os.path.join(self.cartella, d)):
                t.append((v, d))
        return [d for v, d in sorted(t)]

    def campo(self, nome, tempo=None, contorno=False):
        """Campo di cella all'istante indicato (predefinito: l'ultimo)."""
        if tempo is None:
            tempo = self.tempi()[-1]
        elif not isinstance(tempo, str):
            tempo = self._cartella_tempo(tempo)
        interno, bordo, dim = leggi_campo(os.path.join(self.cartella, tempo, nome), self.mesh.nCells)
        return (interno, bordo) if contorno else interno

    def _cartella_tempo(self, t):
        for d in self.tempi():
            if np.isclose(float(d), t):
                return d
        raise FileNotFoundError(f"istante {t} assente in {self.cartella}")

    def valore_patch(self, nome_campo, patch, tempo=None):
        """Valori del campo sulle facce di un patch: il 'value' scritto nel file se c'e',
        altrimenti (zeroGradient e simili) il valore della cella adiacente."""
        interno, bordo = self.campo(nome_campo, tempo, contorno=True)
        voce = bordo.get(patch)
        idx = self.mesh.facce_patch(patch)
        if voce is not None and voce["value"] is not None and np.ndim(voce["value"]) > 0 \
                and len(np.atleast_1d(voce["value"])) == len(idx):
            return np.asarray(voce["value"])
        if voce is not None and voce["value"] is not None:
            v = np.asarray(voce["value"])
            return np.tile(v, (len(idx), 1)) if v.ndim == 1 and interno.ndim == 2 else np.full((len(idx),) + interno.shape[1:], v)
        if voce is not None and voce["type"] in ("noSlip", "empty", "wedge", "symmetryPlane", "symmetry"):
            # noSlip non scrive 'value': la velocita' di parete e' nulla; per i vincoli geometrici il valore e' quello della cella
            if voce["type"] == "noSlip":
                return np.zeros((len(idx),) + interno.shape[1:])
        return interno[self.mesh.owner[idx]]

    def linea(self, nome_campo, p0, p1, n=100, tempo=None, metodo="cella"):
        """Campiona un campo lungo il segmento p0-p1: 'cella' = valore della cella piu' vicina
        (come il sample 'cell' di OpenFOAM), 'idw' = media pesata con l'inverso della distanza delle 4 celle piu' vicine."""
        from scipy.spatial import cKDTree
        v = self.campo(nome_campo, tempo)
        t = np.linspace(0, 1, n)[:, None]
        pts = np.asarray(p0, float) + t * (np.asarray(p1, float) - np.asarray(p0, float))
        albero = cKDTree(self.mesh.C)
        if metodo == "cella":
            _, idx = albero.query(pts)
            # ascissa: proiezione del centro della cella scelta sulla retta (cosi' i valori sono quelli esatti delle celle)
            e = (np.asarray(p1, float) - np.asarray(p0, float)); e /= np.linalg.norm(e)
            return (self.mesh.C[idx] - np.asarray(p0, float)) @ e, v[idx]
        d, idx = albero.query(pts, k=4)
        w = 1.0 / np.maximum(d, 1e-12)
        val = (v[idx] * w[..., None]).sum(axis=1) / w.sum(axis=1)[:, None] if v.ndim == 2 else (v[idx] * w).sum(axis=1) / w.sum(axis=1)
        return np.linalg.norm(pts - pts[0], axis=1), val

    # --- forze su un patch (come la function object 'forces', scritta per esteso) ----------
    def forze(self, patch, nu, rho=1.0, tempo=None, nut="nut", p_cinematica=True):
        """Forza di pressione e viscosa [N] esercitate dal fluido su un patch, per un caso incomprimibile
        (come la function object 'forces'):  F_p = sum rho p_f S_f ;  F_v = sum rho (nu + nu_t,f) (dU/dn)_f |S_f|,
        con (dU/dn)_f = (U_P - U_f)/d_n la derivata della velocita' tangenziale nella direzione che va dalla
        parete verso il fluido. Restituisce (F_pressione, F_viscosa) come vettori."""
        m = self.mesh
        idx = m.facce_patch(patch)
        pf = self.valore_patch("p", patch, tempo)
        Uf = self.valore_patch("U", patch, tempo)
        UP = self.campo("U", tempo)[m.owner[idx]]
        try:
            nutf = self.valore_patch(nut, patch, tempo)
        except (FileNotFoundError, KeyError):
            nutf = np.zeros(len(idx))
        Sf = m.Sf[idx]; nf = Sf / m.magSf[idx][:, None]
        dn = np.einsum("ij,ij->i", m.Cf[idx] - m.C[m.owner[idx]], nf)     # distanza normale centro cella-faccia
        gradUn = (UP - Uf) / dn[:, None]                                   # derivata normale di U (verso l'esterno)
        gradUn -= np.einsum("ij,ij->i", gradUn, nf)[:, None] * nf          # componente tangenziale
        fatt = rho if p_cinematica else 1.0
        Fp = (fatt * pf[:, None] * Sf).sum(axis=0)
        Fv = (rho * (nu + nutf)[:, None] * gradUn * m.magSf[idx][:, None]).sum(axis=0)   # forza DEL fluido SULLA parete
        return Fp, Fv

    def sforzo_parete(self, patch, nu, rho=1.0, tempo=None, nut="nut"):
        """Sforzo tangenziale [Pa] su ogni faccia di un patch di parete: tau = rho (nu + nu_t,f) (dU/dn)_tang,
        con la derivata normale del primo ordine (U_P - U_f)/d_n, come in OpenFOAM (wallShearStress).
        Restituisce (centri delle facce, vettore sforzo per faccia, orientato come la velocita' della cella)."""
        m = self.mesh
        idx = m.facce_patch(patch)
        Uf = self.valore_patch("U", patch, tempo); UP = self.campo("U", tempo)[m.owner[idx]]
        try:
            nutf = self.valore_patch(nut, patch, tempo)
        except (FileNotFoundError, KeyError):
            nutf = np.zeros(len(idx))
        nf = m.Sf[idx] / m.magSf[idx][:, None]
        dn = np.abs(np.einsum("ij,ij->i", m.Cf[idx] - m.C[m.owner[idx]], nf))
        dU = UP - Uf; dU -= np.einsum("ij,ij->i", dU, nf)[:, None] * nf
        tau = rho * (nu + nutf)[:, None] * dU / dn[:, None]
        return m.Cf[idx], tau

    def yplus(self, patch, nu, tempo=None, nut="nut"):
        """y+ delle celle adiacenti a una parete: y+ = y_P u_tau / nu, u_tau = sqrt((nu + nu_t) |dU/dn|)."""
        m = self.mesh
        idx = m.facce_patch(patch)
        Uf = self.valore_patch("U", patch, tempo); UP = self.campo("U", tempo)[m.owner[idx]]
        try:
            nutf = self.valore_patch(nut, patch, tempo)
        except (FileNotFoundError, KeyError):
            nutf = np.zeros(len(idx))
        nf = m.Sf[idx] / m.magSf[idx][:, None]
        dn = np.abs(np.einsum("ij,ij->i", m.Cf[idx] - m.C[m.owner[idx]], nf))
        dU = UP - Uf; dU -= np.einsum("ij,ij->i", dU, nf)[:, None] * nf
        utau = np.sqrt((nu + nutf) * np.linalg.norm(dU, axis=1) / dn)
        return dn * utau / nu


if __name__ == "__main__":
    import sys
    c = Caso(sys.argv[1] if len(sys.argv) > 1 else ".")
    m = c.mesh
    print(f"caso {c.cartella}: {m.nPoints} punti, {m.nFaces} facce ({m.nInternalFaces} interne), {m.nCells} celle")
    print(f"   volume totale {m.V.sum():.6e}; patch: " + ", ".join(f"{b['nome']} ({b['nFaces']}, {b.get('type')})" for b in m.boundary))
    no = m.non_ortogonalita()
    print(f"   non ortogonalita' massima {no.max():.2f} gradi, media {m.media_non_ortogonalita():.2f}; skewness massima {m.skewness().max():.3f}")
    print(f"   istanti: {c.tempi()}")
