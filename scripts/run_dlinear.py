"""Evaluación de los modelos LTSF-Linear (Zeng et al., 2023) con el mismo protocolo que los demás.

Uso:  python -m scripts.run_dlinear [n_jobs] [variantes]

Las variantes son ``dlinear``, ``nlinear`` y ``linear``, separadas por comas (por defecto, las
tres).

1. Validación anidada: las 12 particiones externas, la validación interna cronológica, la
   semilla y el presupuesto de 16 evaluaciones (TPE) de los métodos distribucionales. Los
   resultados se guardan en ``results/original_calamar`` junto a los de BLGB, NGBoost y el
   enfoque de dos etapas, de modo que las pruebas pareadas los comparan bloque a bloque.
2. Modelo final: optimización sobre todo el entrenamiento (1947-2009), reajuste y una única
   evaluación en la prueba reservada (2010-2026), en clasificación (PR-AUC) y en regresión
   (RMSE del caudal a quince días). Se guarda en ``results/final_<variante>.pkl``.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTADOS, SEED, fijar_semillas
from src.dlinear import datos_con_ventanas
from src.evaluation import ece
from src.experiment import cargar_datos, metricas_clf, metricas_reg
from src.experiment_original import ESPACIOS, LTSF, ModeloDistribucional, _objetivo, ejecutar, parametros_banda
from src.optim import optimizar


def final_ltsf(variante: str = "dlinear", presupuesto: int = 16) -> dict:
    datos = datos_con_ventanas(cargar_datos())
    banda = parametros_banda("calamar")
    t0 = time.perf_counter()
    reg = optimizar("bayes", _objetivo(variante, banda, datos.X, datos.y_clf, datos.y_reg, SEED),
                    ESPACIOS[variante], presupuesto, SEED)
    params, interno = reg.mejor()
    t_opt = time.perf_counter() - t0
    modelo = ModeloDistribucional(variante, params, banda, SEED).fit(datos.X, datos.y_clf, datos.y_reg.values)
    p = modelo.predict_proba(datos.X_test)[:, 1]
    q_ref = np.sqrt(datos.X_test["calamar_cuad"].values)
    q_pred = q_ref * np.exp(modelo.m_.predict(datos.X_test))
    return {"modelo": variante, "optimizador": "bayes", "params": params, "puntaje_interno": interno,
            "t_opt": t_opt, "sigma": modelo.sigma_, "pred_test": p, "q_pred_test": q_pred,
            "metricas_test": metricas_clf(datos.y_clf_test, p), "ece_test": ece(datos.y_clf_test, p),
            "metricas_reg_test": metricas_reg(datos.y_reg_test.values, q_pred)}


if __name__ == "__main__":
    fijar_semillas()
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    variantes = sys.argv[2].split(",") if len(sys.argv) > 2 else list(LTSF)
    ejecutar(variantes, ["bayes"], n_jobs=n_jobs)
    for v in variantes:
        res = final_ltsf(v)
        with open(RESULTADOS / f"final_{v}.pkl", "wb") as f:
            pickle.dump(res, f)
        print(v, res["params"], {k: round(v, 4) for k, v in res["metricas_test"].items()},
              {k: round(v, 2) for k, v in res["metricas_reg_test"].items()})
