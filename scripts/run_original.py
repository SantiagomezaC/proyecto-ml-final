"""Validación anidada del modelo original y de las referencias distribucionales (Entregable 3).

Uso:  python -m scripts.run_original [n_jobs]
Mismas particiones externas, validación interna, semilla y presupuesto (16 evaluaciones)
que los benchmarks del Entregable 2. El modelo original se optimiza con TPE y con el
algoritmo genético; NGBoost y el enfoque de dos etapas, con TPE.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import fijar_semillas
from src.experiment_original import ejecutar

if __name__ == "__main__":
    fijar_semillas()
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    ejecutar(["blgb"], ["bayes", "genetico"], n_jobs=n_jobs)
    ejecutar(["dos_etapas", "ngboost"], ["bayes"], n_jobs=n_jobs)
