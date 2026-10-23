cd "$(dirname "$0")/.."
export LOKY_MAX_CPU_COUNT=10 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
/c/Users/User/miniconda3/envs/ml-final/python.exe -m scripts.computational > results/log_computational.txt 2>&1
echo FIN >> results/log_computational.txt
