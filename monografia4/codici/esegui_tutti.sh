#!/bin/bash
# Esegue tutti i programmi del Modulo 4 e rigenera le figure in ../figure
cd "$(dirname "$0")"
for f in prog4_1_sfera.py prog4_1b_galerkin_simbolico.py prog4_1c_jeffery_hamel.py prog4_2_upwind.py \
         prog4_3_benard.py prog4_3b_taylor.py prog4_3c_tazza.py prog4_3d_benard_varianti.py \
         prog4_4_burgers_abcn.py prog4_5_cavita_mac.py prog4_5b_cavita_vento.py prog4_5c_cavita_abcn.py \
         prog4_6_cavita_omega_psi.py prog4_6b_cilindro.py esercizi_svolti.py; do
    echo "=================================================================== $f"
    python3 "$f" || { echo "ERRORE in $f"; exit 1; }
done
