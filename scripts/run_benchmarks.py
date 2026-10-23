"""Ejecuta la validación anidada de los modelos benchmark (Entregable 2).

Uso:  python -m scripts.run_benchmarks clf|reg [presupuesto] [n_jobs]
Clasificación: 7 modelos × 4 balanceos × 4 optimizadores = 112 combinaciones.
Regresión:     7 modelos × 4 optimizadores = 28 combinaciones.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTADOS, fijar_semillas
from src.experiment import consolidar, ejecutar_rejilla
from src.models import BALANCEOS, MODELOS_CLF, MODELOS_REG, OPTIMIZADORES

if __name__ == "__main__":
    fijar_semillas()
    tarea = sys.argv[1]
    presupuesto = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    n_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    if tarea == "clf":
        ejecutar_rejilla("clf", MODELOS_CLF, BALANCEOS, OPTIMIZADORES, presupuesto, n_jobs=n_jobs)
    else:
        ejecutar_rejilla("reg", MODELOS_REG, ["ninguno"], OPTIMIZADORES, presupuesto, n_jobs=n_jobs)
    largo, maestra = consolidar(RESULTADOS / f"{tarea}_calamar")
    largo.to_parquet(RESULTADOS / f"pliegues_{tarea}.parquet", index=False)
    maestra.to_parquet(RESULTADOS / f"maestra_{tarea}.parquet", index=False)
    print(f"{len(maestra)} combinaciones consolidadas")
