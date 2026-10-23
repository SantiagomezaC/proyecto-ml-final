"""Motor de experimentos: validación anidada del pipeline combinatorio.

Cada tarea elemental es la tupla (modelo, balanceo, optimizador, pliegue externo). Dentro
de ella el optimizador busca hiperparámetros con validación interna sobre el
entrenamiento del pliegue; luego el pipeline se reajusta con esos hiperparámetros y se
evalúa en el bloque externo, que el optimizador nunca vio. Los resultados se guardan uno
por archivo para que la ejecución pueda reanudarse.
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)

from .config import DATOS_PROC, RESULTADOS, SEED
from .models import construir_pipeline, espacio, puntaje_continuo
from .optim import evaluar_config, optimizar
from .splits import cargar_pliegues, cv_interna, separar_prueba

# Costo relativo aproximado: las tareas caras se despachan primero (balanceo de carga)
COSTO = {"random_forest": 10, "svm": 4, "svr": 4, "xgboost": 3, "lasso": 2, "arbol": 1,
         "knn": 1, "logistica": 1, "ridge": 0.5, "naive_bayes": 0.2}


@dataclass
class Datos:
    """Conjunto de entrenamiento de una estación y sus particiones externas."""
    X: pd.DataFrame
    y_clf: pd.Series
    y_reg: pd.Series
    fechas: pd.Series
    pliegues: list
    X_test: pd.DataFrame
    y_clf_test: pd.Series
    y_reg_test: pd.Series
    fechas_test: pd.Series
    columnas: list
    prevalencia: float


def cargar_datos(estacion: str = "calamar") -> Datos:
    """Carga la tabla de modelado, separa la prueba reservada y lee las particiones."""
    par = json.loads((DATOS_PROC / "parametros.json").read_text(encoding="utf-8"))[estacion]
    d = pd.read_parquet(DATOS_PROC / f"tarea_{estacion}.parquet")
    cols = par["BASE_COLS"] + par["DOM_COLS"]
    tr, te = separar_prueba(d.fecha)
    a, b = d.iloc[tr].reset_index(drop=True), d.iloc[te].reset_index(drop=True)
    return Datos(X=a[cols], y_clf=a.y.astype(int), y_reg=a.q_futuro, fechas=a.fecha,
                 pliegues=cargar_pliegues(DATOS_PROC / f"pliegues_{estacion}.json"),
                 X_test=b[cols], y_clf_test=b.y.astype(int), y_reg_test=b.q_futuro,
                 fechas_test=b.fecha, columnas=cols, prevalencia=float(a.y.mean()))


def metricas_clf(y, s, umbral: float = 0.5) -> dict:
    """Métricas de clasificación a partir del puntaje continuo de la clase positiva."""
    p = (s >= umbral).astype(int)
    return {"pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s),
            "f1": f1_score(y, p, zero_division=0), "precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p, zero_division=0), "accuracy": accuracy_score(y, p),
            "brier": brier_score_loss(y, np.clip(s, 0, 1))}


def metricas_reg(y, yhat) -> dict:
    """Métricas de regresión (caudal en m³/s)."""
    return {"rmse": float(np.sqrt(mean_squared_error(y, yhat))), "mae": mean_absolute_error(y, yhat),
            "r2": r2_score(y, yhat),
            "mape": float(np.mean(np.abs((y - yhat) / np.maximum(np.abs(y), 1.0))) * 100)}


def _predictor(tarea: str):
    if tarea == "clf":
        return puntaje_continuo
    return lambda est, X: est.predict(X)


def ejecutar_tarea(datos: Datos, tarea: str, modelo: str, balanceo: str, optimizador: str,
                   pliegue: dict, presupuesto: int, destino: Path, seed: int = SEED) -> Path:
    """Ejecuta una tarea elemental de la validación anidada y guarda su resultado.

    Returns
    -------
    Path
        Archivo ``.pkl`` con métricas externas, predicciones, hiperparámetros e historial.
    """
    archivo = destino / f"{modelo}__{balanceo}__{optimizador}__f{pliegue['pliegue']:02d}.pkl"
    if archivo.exists():
        return archivo
    y = datos.y_clf if tarea == "clf" else datos.y_reg
    Xtr, ytr = datos.X.iloc[pliegue["train"]], y.iloc[pliegue["train"]]
    Xte, yte = datos.X.iloc[pliegue["test"]], y.iloc[pliegue["test"]]
    prev = float(datos.y_clf.iloc[pliegue["train"]].mean())

    def fabrica(params):
        return construir_pipeline(tarea, modelo, params, balanceo, prev, seed)

    predecir = _predictor(tarea)

    def objetivo(params):
        return evaluar_config(params, fabrica, Xtr, ytr, cv_interna(), tarea, predecir)

    t0 = time.perf_counter()
    reg = optimizar(optimizador, objetivo, espacio(tarea, modelo), presupuesto, seed)
    t_opt = time.perf_counter() - t0
    params, puntaje_interno = reg.mejor()
    t1 = time.perf_counter()
    final = fabrica(params).fit(Xtr, ytr)
    t_fit = time.perf_counter() - t1
    t2 = time.perf_counter()
    pred = predecir(final, Xte)
    t_inf = time.perf_counter() - t2
    met = metricas_clf(yte, pred) if tarea == "clf" else metricas_reg(yte.values, pred)
    res = {"tarea": tarea, "modelo": modelo, "balanceo": balanceo, "optimizador": optimizador,
           "pliegue": pliegue["pliegue"], "bloque": pliegue["bloque"], "params": params,
           "puntaje_interno": puntaje_interno, "metricas": met, "t_opt": t_opt, "t_fit": t_fit,
           "t_inferencia": t_inf, "n_eval": len(reg.filas), "historial": reg.filas,
           "idx_test": pliegue["test"], "y_test": yte.values, "pred": pred, "seed": seed}
    destino.mkdir(parents=True, exist_ok=True)
    with open(archivo, "wb") as f:
        pickle.dump(res, f)
    return archivo


def ejecutar_rejilla(tarea: str, modelos: list, balanceos: list, optimizadores: list,
                     presupuesto: int = 16, estacion: str = "calamar", n_jobs: int = 10,
                     subcarpeta: str | None = None) -> None:
    """Lanza en paralelo todas las tareas elementales del producto cartesiano."""
    datos = cargar_datos(estacion)
    destino = RESULTADOS / (subcarpeta or f"{tarea}_{estacion}")
    tareas = [(m, b, o, p) for m in modelos for b in balanceos for o in optimizadores
              for p in datos.pliegues]
    # Primero las más caras y, dentro de cada modelo, los pliegues con más entrenamiento
    tareas.sort(key=lambda t: (-COSTO.get(t[0], 1), -len(t[3]["train"])))
    Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(ejecutar_tarea)(datos, tarea, m, b, o, p, presupuesto, destino)
        for m, b, o, p in tareas)


def consolidar(carpeta: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reúne los resultados por pliegue en una tabla larga y en la tabla maestra.

    Returns
    -------
    largo : DataFrame
        Una fila por combinación y pliegue externo.
    maestra : DataFrame
        Una fila por combinación, con media y desviación de cada métrica externa.
    """
    filas = []
    for f in sorted(Path(carpeta).glob("*.pkl")):
        r = pickle.load(open(f, "rb"))
        filas.append({"tarea": r["tarea"], "modelo": r["modelo"], "balanceo": r["balanceo"],
                      "optimizador": r["optimizador"], "pliegue": r["pliegue"], "bloque": r["bloque"],
                      **r["metricas"], "puntaje_interno": r["puntaje_interno"], "t_opt": r["t_opt"],
                      "t_fit": r["t_fit"], "t_inferencia": r["t_inferencia"], "n_eval": r["n_eval"],
                      "params": json.dumps(r["params"]), "seed": r["seed"]})
    largo = pd.DataFrame(filas)
    claves = ["tarea", "modelo", "balanceo", "optimizador"]
    met = [c for c in largo.columns if c not in claves + ["pliegue", "bloque", "params", "seed",
                                                          "puntaje_interno"]]
    agg = largo.groupby(claves)[met].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_pliegues"] = largo.groupby(claves).size()
    agg["t_total"] = largo.groupby(claves)["t_opt"].sum() + largo.groupby(claves)["t_fit"].sum()
    agg["params_ultimo_pliegue"] = largo.sort_values("pliegue").groupby(claves)["params"].last()
    agg["seed"] = SEED
    return largo, agg.reset_index()
