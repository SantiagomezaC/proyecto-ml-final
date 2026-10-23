#!/usr/bin/env bash
# Ejecuta en orden las etapas posteriores a los benchmarks. Uso: bash scripts/cadena_post.sh
set -u
P=/c/Users/User/miniconda3/envs/ml-final/python.exe
export LOKY_MAX_CPU_COUNT=10 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
LOG=results/log_cadena.txt
until [ -f results/maestra_reg.parquet ]; do sleep 30; done
for paso in "run_original 10" "final_models clf" "final_models reg" "final_models original" \
            "run_extra 10" "robustez 10" "computational" "estadistica" "exportar_dashboard"; do
  echo "== $(date '+%H:%M:%S') INICIO $paso" >> $LOG
  $P -m scripts.$paso >> results/log_$(echo $paso | tr ' ' '_').txt 2>&1
  echo "== $(date '+%H:%M:%S') FIN $paso (código $?)" >> $LOG
done
echo "== CADENA COMPLETA" >> $LOG
