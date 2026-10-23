#!/usr/bin/env bash
# Reanuda las etapas finales de la cadena.
cd "$(dirname "$0")/.."
P=/c/Users/User/miniconda3/envs/ml-final/python.exe
export LOKY_MAX_CPU_COUNT=10 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
LOG=results/log_cadena.txt
for paso in "robustez 10" "computational" "estadistica" "exportar_dashboard"; do
  echo "== $(date '+%H:%M:%S') INICIO $paso" >> $LOG
  $P -m scripts.$paso >> results/log_$(echo $paso | tr ' ' '_').txt 2>&1
  echo "== $(date '+%H:%M:%S') FIN $paso (código $?)" >> $LOG
done
echo "== CADENA COMPLETA" >> $LOG
