"""Evaluación del modelo original en tres tareas adicionales (Entregable 3, evaluación amplia).

La misma tarea (banda de descarga efectiva a 15 días) en El Banco, Coyongal y Tres Cruces,
con su propia descarga efectiva, sus estaciones aguas arriba y sus particiones externas.
Todos los modelos reciben el mismo optimizador (TPE) y el mismo presupuesto (8
evaluaciones); cada benchmark usa la técnica de balanceo que resultó mejor en Calamar.

Uso:  python -m scripts.run_extra [n_jobs]
"""
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTADOS, fijar_semillas
from src.experiment import cargar_datos, ejecutar_tarea
from src.experiment_original import ejecutar_pliegue

ESTACIONES = ["el_banco", "coyongal", "tres_cruces"]
PRESUPUESTO = 8

if __name__ == "__main__":
    fijar_semillas()
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    mejores = pd.read_csv(RESULTADOS / "mejores_clf.csv")
    tareas = []
    for est in ESTACIONES:
        datos = cargar_datos(est)
        destino = RESULTADOS / f"extra_{est}"
        for p in datos.pliegues:
            for _, f in mejores.iterrows():
                tareas.append(("bench", datos, est, f.modelo, f.balanceo, p, destino))
            tareas.append(("orig", datos, est, "blgb", "ninguno", p, destino))
    costo = {"random_forest": 10, "blgb": 8, "svm": 4, "xgboost": 3}
    tareas.sort(key=lambda t: (-costo.get(t[3], 1), -len(t[5]["train"])))

    def correr(tipo, datos, est, modelo, balanceo, p, destino):
        if tipo == "bench":
            return ejecutar_tarea(datos, "clf", modelo, balanceo, "bayes", p, PRESUPUESTO, destino)
        return ejecutar_pliegue(datos, est, "blgb", "bayes", p, PRESUPUESTO, destino)

    Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(delayed(correr)(*t) for t in tareas)
