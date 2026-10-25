#!/bin/bash
# Esegue tutti i programmi del Modulo 0 e rigenera le figure in ../figure
cd "$(dirname "$0")"
for f in prog0_1_primi_passi.py prog0_2_controllo_funzioni.py prog0_3_strutture_dati.py prog0_4_numpy.py \
         prog0_5_matplotlib.py prog0_6_scipy.py prog0_7_sympy.py prog0_8_anatomia.py prog0_9_verifiche.py \
         esercizi_svolti.py; do
    echo "=================================================================== $f"
    python3 "$f" || { echo "ERRORE in $f"; exit 1; }
done
