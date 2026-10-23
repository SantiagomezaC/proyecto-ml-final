"""Selección de las mejores combinaciones, reentrenamiento final y serialización (Bloque B).

Para cada modelo base se elige la combinación (balanceo, optimizador) de mayor PR-AUC media
en el bucle externo. Con esa combinación se repite la optimización sobre todo el
entrenamiento (1976-2009), se reentrena el pipeline completo, se evalúa una única vez en la
prueba reservada (2010-2026) y se serializa con su metadata.

Uso:  python -m scripts.final_models clf|reg|original
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.artifacts import guardar_artefacto
from src.config import ARTEFACTOS, RESULTADOS, SEED, fijar_semillas
from src.evaluation import Recalibrador, ece
from src.experiment import cargar_datos, metricas_clf, metricas_reg
from src.experiment_original import ESPACIOS, ModeloDistribucional, _objetivo, parametros_banda
from src.models import construir_pipeline, espacio, puntaje_continuo
from src.optim import evaluar_config, optimizar
from src.splits import cv_interna


def mejores_combinaciones(maestra: pd.DataFrame, metrica: str, mayor: bool = True) -> pd.DataFrame:
    """Mejor combinación por modelo según la media externa (desempate: menor desviación)."""
    orden = maestra.sort_values([f"{metrica}_mean", f"{metrica}_std"], ascending=[not mayor, True])
    return orden.groupby("modelo", sort=False).head(1).reset_index(drop=True)


def oof(carpeta: Path, modelo: str, balanceo: str, optimizador: str) -> pd.DataFrame:
    """Predicciones fuera de muestra del bucle externo de una combinación."""
    filas = []
    for f in sorted(carpeta.glob(f"{modelo}__{balanceo}__{optimizador}__f*.pkl")):
        r = pickle.load(open(f, "rb"))
        filas.append(pd.DataFrame({"idx": r["idx_test"], "y": r["y_test"], "pred": r["pred"],
                                   "pliegue": r["pliegue"]}))
    return pd.concat(filas).sort_values("idx")


def final_benchmark(tarea: str, fila: pd.Series, presupuesto: int) -> dict:
    datos = cargar_datos()
    m, b, o = fila.modelo, fila.balanceo, fila.optimizador
    y = datos.y_clf if tarea == "clf" else datos.y_reg
    yte = datos.y_clf_test if tarea == "clf" else datos.y_reg_test
    prev = datos.prevalencia

    def fabrica(params):
        return construir_pipeline(tarea, m, params, b, prev, SEED)

    predecir = puntaje_continuo if tarea == "clf" else (lambda est, X: est.predict(X))
    t0 = time.perf_counter()
    reg = optimizar(o, lambda p: evaluar_config(p, fabrica, datos.X, y, cv_interna(), tarea, predecir),
                    espacio(tarea, m), presupuesto, SEED)
    params, interno = reg.mejor()
    t_opt = time.perf_counter() - t0
    t1 = time.perf_counter()
    est = fabrica(params).fit(datos.X, y)
    t_fit = time.perf_counter() - t1
    t2 = time.perf_counter()
    pred = predecir(est, datos.X_test)
    t_inf = (time.perf_counter() - t2) / len(datos.X_test) * 1000
    salida = {"modelo": m, "balanceo": b, "optimizador": o, "params": params,
              "puntaje_interno": interno, "t_opt": t_opt, "t_fit": t_fit, "t_inf_ms_fila": t_inf,
              "pred_test": pred}
    if tarea == "clf":
        oo = oof(RESULTADOS / "clf_calamar", m, b, o)
        # Decisión de recalibrar sin mirar la prueba: Platt ajustado en los bloques externos
        # 1986-1997 y evaluado en 1998-2009 (mitades cronológicas de las predicciones fuera de
        # muestra); se recalibra solo si el ECE mejora en la segunda mitad.
        mitad = (oo.idx < oo.idx.median()).values
        cal_val = Recalibrador("platt").fit(oo.pred[mitad].values, oo.y[mitad].values)
        ece_crudo_oof = ece(oo.y[~mitad].values, oo.pred[~mitad].values)
        ece_cal_oof = ece(oo.y[~mitad].values, cal_val.transform(oo.pred[~mitad].values))
        cal = Recalibrador("platt").fit(oo.pred.values, oo.y.values)
        pred_cal = cal.transform(pred)
        salida.update({"metricas_test": metricas_clf(yte, pred),
                       "metricas_test_recalibrado": metricas_clf(yte, pred_cal),
                       "ece_test": ece(yte, pred), "ece_test_recalibrado": ece(yte, pred_cal),
                       "pred_test_recalibrado": pred_cal, "ece_oof_crudo": ece_crudo_oof,
                       "ece_oof_recalibrado": ece_cal_oof,
                       "recalibrar_por_oof": bool(ece_cal_oof < ece_crudo_oof)})
        meta = {"tarea": "clasificacion", "balanceo": b, "optimizador": o,
                "hiperparametros": params,
                "metricas_bucle_externo": {k.replace("_mean", ""): float(fila[k])
                                           for k in fila.index if k.endswith("_mean")},
                "metricas_bucle_externo_std": {k.replace("_std", ""): float(fila[k])
                                               for k in fila.index if k.endswith("_std")},
                "metricas_prueba": salida["metricas_test"],
                "metricas_prueba_recalibrado": salida["metricas_test_recalibrado"],
                "t_inferencia_ms_fila": t_inf}
        joblib_cal = ARTEFACTOS / f"{m}.recalibrador.joblib"
        ARTEFACTOS.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(cal, joblib_cal)
        guardar_artefacto(m, est, datos.X, datos.X_test.iloc[:500], predecir, meta)
    else:
        salida["metricas_test"] = metricas_reg(yte.values, pred)
        salida["pred_train"] = est.predict(datos.X)
        salida["metricas_train"] = metricas_reg(y.values, salida["pred_train"])
    return salida


def final_original(presupuesto: int, metodo: str = "bayes") -> dict:
    datos = cargar_datos()
    banda = parametros_banda("calamar")
    t0 = time.perf_counter()
    reg = optimizar(metodo, _objetivo("blgb", banda, datos.X, datos.y_clf, datos.y_reg, SEED),
                    ESPACIOS["blgb"], presupuesto, SEED)
    params, interno = reg.mejor()
    t_opt = time.perf_counter() - t0
    t1 = time.perf_counter()
    env = ModeloDistribucional("blgb", params, banda, SEED).fit(datos.X, datos.y_clf, datos.y_reg.values)
    est = env.m_
    t_fit = time.perf_counter() - t1
    t2 = time.perf_counter()
    pred = est.predict_proba(datos.X_test)[:, 1]
    t_inf = (time.perf_counter() - t2) / len(datos.X_test) * 1000
    dist = est.predict_distribucion(datos.X_test)
    met = metricas_clf(datos.y_clf_test, pred)
    meta = {"tarea": "clasificacion", "balanceo": "ninguno", "optimizador": metodo,
            "hiperparametros": params, "metricas_prueba": met, "t_inferencia_ms_fila": t_inf,
            "n_iter": est.n_iter_, "banda": [banda["L"], banda["U"]]}
    guardar_artefacto("original", est, datos.X, datos.X_test.iloc[:500],
                      lambda e, X: e.predict_proba(X)[:, 1], meta)
    return {"modelo": "original", "balanceo": "ninguno", "optimizador": metodo, "params": params,
            "puntaje_interno": interno, "t_opt": t_opt, "t_fit": t_fit, "t_inf_ms_fila": t_inf,
            "pred_test": pred, "metricas_test": met, "ece_test": ece(datos.y_clf_test, pred),
            "q_mediana_test": dist["mediana"], "q_inf_test": dist["inferior"],
            "q_sup_test": dist["superior"], "historial_boosting": est.historial_}


def main():
    fijar_semillas()
    tarea = sys.argv[1]
    presupuesto = 16
    if tarea == "original":
        res = [final_original(presupuesto)]
        sal = RESULTADOS / "final_original.pkl"
    else:
        maestra = pd.read_parquet(RESULTADOS / f"maestra_{tarea}.parquet")
        metrica, mayor = ("pr_auc", True) if tarea == "clf" else ("rmse", False)
        mejores = mejores_combinaciones(maestra, metrica, mayor)
        mejores.to_csv(RESULTADOS / f"mejores_{tarea}.csv", index=False)
        res = Parallel(n_jobs=7, backend="loky")(delayed(final_benchmark)(tarea, f, presupuesto)
                                                 for _, f in mejores.iterrows())
        sal = RESULTADOS / f"final_{tarea}.pkl"
    with open(sal, "wb") as f:
        pickle.dump(res, f)
    for r in res:
        print(r["modelo"], r.get("balanceo"), r.get("optimizador"),
              {k: round(v, 4) for k, v in r["metricas_test"].items()})


if __name__ == "__main__":
    main()
