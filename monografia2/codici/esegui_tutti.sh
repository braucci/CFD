#!/bin/bash
# Rigenera tutte le figure della monografia 2 (circa 2 minuti).
# Uso: dalla cartella codice/   bash esegui_tutti.sh
set -e
for p in prog2_0_reticolo prog2_1_flusso_radiale prog2_2_doppiette prog2_2b_flussi_elementari \
         prog2_3_karman_sfera prog2_3b_ellissoide prog2_4_joukowski prog2_4b_mappe \
         prog2_4c_pannelli prog2_5_vortice_rettangolo prog2_5b_convergenza prog2_6_camera \
         prog2_7_cilindro_canale prog2_8_onda_tubo prog2_8b_courant prog2_9_onda_finita \
         prog2_10_maccormack prog2_11_beam_warming prog2_12_burgers prog2_13_tubo_elastico \
         esercizi_svolti; do
    echo "=== $p"
    python3 "$p.py"
done
