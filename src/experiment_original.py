"""Validación anidada del modelo original y de los métodos distribucionales de referencia.

Usa exactamente las mismas particiones externas, la misma validación interna, la misma
semilla y el mismo presupuesto de evaluaciones que los modelos de referencia.

Variantes evaluadas
-------------------
* ``blgb``: Band-Likelihood Gradient Boosting (modelo original).
* ``ngboost``: NGBoost con distribución normal sobre z = log(Q_fut/Q_ref) y probabilidad de
  banda por integración (Duan et al., 2020); referencia reciente del estado del arte.
* ``dos_etapas``: XGBoost de regresión sobre z y residuo gaussiano homocedástico; es el
  enfoque "regresión y umbral" sin la pérdida de la tarea.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.metrics import average_precision_score

from .config import DATOS_PROC, RESULTADOS, SEED
from .experiment import Datos, cargar_datos, metricas_clf
from .optim import optimizar
from .original_model import BandLikelihoodBoosting
from .spaces import Param
from .splits import cv_interna

ESPACIO_BLGB = [
    Param("lambda_hibrido", "float", 0.05, 10.0, log=True),
    Param("tasa", "float", 0.03, 0.3, log=True),
    Param("profundidad", "int", 2, 6),
    Param("min_hess", "float", 1.0, 50.0, log=True),
    Param("reg_lambda", "float", 0.1, 20.0, log=True),
    Param("columnas", "float", 0.4, 1.0),
    Param("submuestreo", "float", 0.5, 1.0),
]

ESPACIO_NGBOOST = [
    Param("n_estimators", "int", 100, 300, log=True),
    Param("learning_rate", "float", 0.02, 0.15, log=True),
    Param("max_depth", "int", 2, 5),
    Param("minibatch_frac", "float", 0.4, 1.0),
]

ESPACIO_DOS_ETAPAS = [
    Param("n_estimators", "int", 80, 600, log=True),
    Param("max_depth", "int", 2, 8),
    Param("learning_rate", "float", 0.01, 0.3, log=True),
    Param("subsample", "float", 0.5, 1.0),
    Param("colsample_bytree", "float", 0.4, 1.0),
    Param("min_child_weight", "float", 1.0, 30.0, log=True),
]

ESPACIO_DLINEAR = [
    Param("longitud", "int", 15, 365, log=True),
    Param("kernel", "int", 5, 61),
    Param("alpha", "float", 1e-4, 10.0, log=True),
    Param("canales", "cat", opciones=("calamar", "ambos")),
]

# Linear y NLinear no descomponen la ventana, por lo que no tienen tamaño de media móvil
ESPACIO_LINEAR = [p for p in ESPACIO_DLINEAR if p.nombre != "kernel"]

ESPACIOS = {"blgb": ESPACIO_BLGB, "ngboost": ESPACIO_NGBOOST, "dos_etapas": ESPACIO_DOS_ETAPAS,
            "dlinear": ESPACIO_DLINEAR, "nlinear": ESPACIO_LINEAR, "linear": ESPACIO_LINEAR}
LTSF = ("dlinear", "nlinear", "linear")


def parametros_banda(estacion: str) -> dict:
    par = json.loads((DATOS_PROC / "parametros.json").read_text(encoding="utf-8"))[estacion]
    return {"L": par["LIM"][0], "U": par["LIM"][1], "ref": f"{estacion}_cuad"}


class ModeloDistribucional:
    """Envoltura común: ajusta un modelo de z y devuelve la probabilidad de banda."""

    def __init__(self, variante: str, params: dict, banda: dict, seed: int = SEED):
        self.variante, self.params, self.banda, self.seed = variante, params, banda, seed

    def fit(self, X, y, q_futuro):
        b = self.banda
        if self.variante == "blgb":
            self.m_ = BandLikelihoodBoosting(banda_inf=b["L"], banda_sup=b["U"],
                                             col_referencia=b["ref"], exponente_referencia=0.5,
                                             random_state=self.seed, **self.params)
            self.m_.fit(X, y, q_futuro=q_futuro)
            return self
        qref = np.sqrt(X[b["ref"]].values)
        z = np.log(np.asarray(q_futuro) / qref)
        if self.variante in LTSF:
            from .dlinear import MODELOS_LTSF
            Modelo = MODELOS_LTSF[self.variante]
            corte = int(len(z) * 0.85)
            # Como en el enfoque de dos etapas, la dispersión del residuo se estima fuera de
            # muestra, en el último 15 % cronológico del entrenamiento.
            aux = Modelo(**self.params).fit(X.iloc[:corte], z[:corte])
            self.sigma_ = float(np.std(z[corte:] - aux.predict(X.iloc[corte:])))
            self.m_ = Modelo(**self.params).fit(X, z)
            return self
        self.mediana_ = np.nanmedian(X.values, axis=0)
        Xi = np.where(np.isnan(X.values), self.mediana_, X.values)
        if self.variante == "ngboost":
            from ngboost import NGBRegressor
            from ngboost.distns import Normal
            from sklearn.tree import DecisionTreeRegressor
            p = dict(self.params)
            base = DecisionTreeRegressor(criterion="friedman_mse", max_depth=p.pop("max_depth"),
                                         random_state=self.seed)
            self.m_ = NGBRegressor(Dist=Normal, Base=base, random_state=self.seed, verbose=False,
                                   **p).fit(Xi, z)
        else:
            from xgboost import XGBRegressor
            n = len(z)
            corte = int(n * 0.85)
            # La dispersión del residuo se estima fuera de muestra (último 15 % cronológico)
            m_aux = XGBRegressor(tree_method="hist", n_jobs=1, random_state=self.seed,
                                 **self.params).fit(Xi[:corte], z[:corte])
            self.sigma_ = float(np.std(z[corte:] - m_aux.predict(Xi[corte:])))
            self.m_ = XGBRegressor(tree_method="hist", n_jobs=1, random_state=self.seed,
                                   **self.params).fit(Xi, z)
        return self

    def predict_proba(self, X):
        if self.variante == "blgb":
            return self.m_.predict_proba(X)
        b = self.banda
        qref = np.sqrt(X[b["ref"]].values)
        if self.variante in LTSF:
            mu, sig = self.m_.predict(X), np.full(len(X), self.sigma_)
            l, u = np.log(b["L"] / qref), np.log(b["U"] / qref)
            p = np.clip(norm.cdf((u - mu) / sig) - norm.cdf((l - mu) / sig), 1e-7, 1 - 1e-7)
            return np.column_stack([1 - p, p])
        Xi = np.where(np.isnan(X.values), self.mediana_, X.values)
        if self.variante == "ngboost":
            d = self.m_.pred_dist(Xi)
            mu, sig = d.params["loc"], d.params["scale"]
        else:
            mu, sig = self.m_.predict(Xi), np.full(len(X), self.sigma_)
        l, u = np.log(b["L"] / qref), np.log(b["U"] / qref)
        p = np.clip(norm.cdf((u - mu) / sig) - norm.cdf((l - mu) / sig), 1e-7, 1 - 1e-7)
        return np.column_stack([1 - p, p])


def _objetivo(variante, banda, X, y, q, seed):
    def f(params):
        punt = []
        for tr, va in cv_interna().split(X):
            m = ModeloDistribucional(variante, params, banda, seed).fit(X.iloc[tr], y.iloc[tr],
                                                                        q.iloc[tr].values)
            punt.append(average_precision_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
        return float(np.mean(punt))
    return f


def ejecutar_pliegue(datos: Datos, estacion: str, variante: str, metodo: str, pliegue: dict,
                     presupuesto: int, destino: Path, seed: int = SEED) -> Path:
    """Optimiza, reajusta y evalúa una variante en un pliegue externo."""
    archivo = destino / f"{variante}__{metodo}__f{pliegue['pliegue']:02d}.pkl"
    if archivo.exists():
        return archivo
    banda = parametros_banda(estacion)
    tr, te = pliegue["train"], pliegue["test"]
    X, y, q = datos.X.iloc[tr], datos.y_clf.iloc[tr], datos.y_reg.iloc[tr]
    t0 = time.perf_counter()
    reg = optimizar(metodo, _objetivo(variante, banda, X, y, q, seed), ESPACIOS[variante],
                    presupuesto, seed)
    t_opt = time.perf_counter() - t0
    params, interno = reg.mejor()
    t1 = time.perf_counter()
    final = ModeloDistribucional(variante, params, banda, seed).fit(X, y, q.values)
    t_fit = time.perf_counter() - t1
    t2 = time.perf_counter()
    pred = final.predict_proba(datos.X.iloc[te])[:, 1]
    t_inf = time.perf_counter() - t2
    yte = datos.y_clf.iloc[te]
    res = {"tarea": "clf", "modelo": variante, "balanceo": "ninguno", "optimizador": metodo,
           "pliegue": pliegue["pliegue"], "bloque": pliegue["bloque"], "params": params,
           "puntaje_interno": interno, "metricas": metricas_clf(yte, pred), "t_opt": t_opt,
           "t_fit": t_fit, "t_inferencia": t_inf, "n_eval": len(reg.filas), "historial": reg.filas,
           "idx_test": te, "y_test": yte.values, "pred": pred, "seed": seed,
           "n_iter": getattr(getattr(final, "m_", None), "n_iter_", None)}
    destino.mkdir(parents=True, exist_ok=True)
    with open(archivo, "wb") as f:
        pickle.dump(res, f)
    return archivo


def ejecutar(variantes: list, metodos: list, estacion: str = "calamar", presupuesto: int = 16,
             n_jobs: int = 10, pliegues_sel: list | None = None, subcarpeta: str | None = None):
    """Lanza en paralelo las variantes × métodos × pliegues externos."""
    datos = cargar_datos(estacion)
    if set(variantes) & set(LTSF):
        if not set(variantes) <= set(LTSF):
            raise ValueError("Los modelos LTSF-Linear usan ventanas de historia como entrada; ejecútelos por separado.")
        from .dlinear import datos_con_ventanas
        datos = datos_con_ventanas(datos)
    destino = RESULTADOS / (subcarpeta or f"original_{estacion}")
    pl = [p for p in datos.pliegues if pliegues_sel is None or p["pliegue"] in pliegues_sel]
    tareas = [(v, m, p) for v in variantes for m in metodos for p in pl]
    tareas.sort(key=lambda t: -len(t[2]["train"]))
    Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(ejecutar_pliegue)(datos, estacion, v, m, p, presupuesto, destino)
        for v, m, p in tareas)
