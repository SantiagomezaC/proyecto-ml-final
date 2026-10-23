#!/usr/bin/env bash
# Corrida completa: benchmarks de clasificación y regresión y etapas posteriores.
P=/c/Users/User/miniconda3/envs/ml-final/python.exe
export LOKY_MAX_CPU_COUNT=10 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
echo "== $(date '+%H:%M:%S') INICIO clf" >> results/log_cadena.txt
$P -m scripts.run_benchmarks clf 16 10 > results/log_clf.txt 2>&1
echo "== $(date '+%H:%M:%S') INICIO reg" >> results/log_cadena.txt
$P -m scripts.run_benchmarks reg 16 10 > results/log_reg.txt 2>&1
bash scripts/cadena_post.sh
