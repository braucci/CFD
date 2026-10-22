#!/bin/bash
# Rigenera tutte le figure della monografia 3 (circa 5 minuti).
# Uso: dalla cartella codice/   bash esegui_tutti.sh
set -e
for p in prog3_0_soluzioni_esatte prog3_1_blasius prog3_1b_tiro prog3_1c_karman prog3_2_termometro prog3_2b_prandtl \
         prog3_3_canale prog3_3b_condotti prog3_4_rayleigh prog3_4b_accuratezza \
         prog3_5_canale_impulsivo prog3_5b_crank_nicolson prog3_5c_adi prog3_6_cavita_stokes prog3_6b_cavita_fine prog3_6c_lubrificazione \
         prog3_7a_benard prog3_7c_benard_varianti prog3_7_chebyshev prog3_7b_runge prog3_8_orr_sommerfeld prog3_8b_curva_neutra \
         prog3_9_fourier prog3_9b_burgers prog3_10_convezione_diffusione esercizi_svolti; do
    echo "=== $p"
    python3 "$p.py"
done
