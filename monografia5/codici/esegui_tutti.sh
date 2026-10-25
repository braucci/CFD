#!/bin/bash
# Rigenera tutte le figure e i riquadri di risultati della monografia dagli output dei casi OpenFOAM
# (i casi vanno eseguiti prima: ./Allrun o python3 genera.py nella cartella di ciascuno).
cd "${0%/*}" || exit 1
for s in fig01_cavita fig02_mesh fig04_diffusione fig05_convezione fig06_tempo fig07_solutori fig08_cavita \
         fig09_lastra fig10_gradino fig11_sod fig12_cuneo fig13_poiseuille fig14_codice fig15_naca fig16_cilindro fig17_ala; do
    echo "=== $s"; python3 "$s.py" || echo "   (errore in $s: il caso corrispondente e' stato eseguito?)"
done
