#!/usr/bin/env bash
# Esegue tutti gli script della monografia e rigenera le figure in ../figure
set -e
cd "$(dirname "$0")"
mkdir -p ../figure
for s in es01_ordine_stabilita prog1_1_caduta_sfera prog1_1b_problemi prog1_2_pendolo \
         prog1_3_ala_molla prog1_4_proiettile prog1_5_gittata_max prog1_6_aliante \
         prog1_7_scia_vorticosa prog1_8_cannone prog1_9_razzo esercizi_svolti; do
  echo "================ $s ================"
  python3 "$s.py"
done
