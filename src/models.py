"""Catálogo de modelos de referencia, técnicas de balanceo y espacios de búsqueda.

``construir_pipeline`` arma el pipeline completo (imputación, escalado, balanceo y
modelo) a partir de un diccionario de hiperparámetros. Todos los pasos se ajustan
únicamente con los datos que recibe ``fit``, de modo que cualquier optimizador que lo
use dentro de su validación interna queda libre de fuga de información.
"""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from .config import SEED
from .spaces import Param

MODELOS_CLF = ["knn", "naive_bayes", "logistica", "arbol", "random_forest", "xgboost", "svm"]
MODELOS_REG = ["knn", "ridge", "lasso", "arbol", "random_forest", "xgboost", "svr"]
BALANCEOS = ["ninguno", "smote", "adasyn", "balanced"]
OPTIMIZADORES = ["grid", "random", "bayes", "genetico"]

NOMBRES_MODELO = {
    "knn": "KNN", "naive_bayes": "Naive Bayes", "logistica": "Regresión logística",
    "arbol": "Árbol de decisión", "random_forest": "Random Forest", "xgboost": "XGBoost",
    "svm": "SVM", "ridge": "Ridge", "lasso": "Lasso", "svr": "SVR",
    "original": "BLGB (modelo propuesto)",
}
NOMBRES_BALANCEO = {"ninguno": "Sin balanceo", "smote": "SMOTE", "adasyn": "ADASYN",
                    "balanced": "class_weight"}
NOMBRES_OPT = {"grid": "Grid", "random": "Random", "bayes": "Bayesiana (TPE)",
               "genetico": "Genética (DEAP)"}


class KNNBalanceado(KNeighborsClassifier):
    """KNN con corrección de priors, análogo de ``class_weight='balanced'``.

    KNN no admite pesos por clase. La corrección divide la fracción de vecinos de cada
    clase por su frecuencia en el entrenamiento y renormaliza, lo que equivale a ponderar
    cada voto por la inversa de la prevalencia de su clase.
    """

    def fit(self, X, y):
        super().fit(X, y)
        self.priors_ = np.bincount(np.searchsorted(self.classes_, y)) / len(y)
        return self

    def predict_proba(self, X):
        p = super().predict_proba(X) / self.priors_
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# Espacios de búsqueda (rangos justificados en el capítulo de modelos de referencia)
# ---------------------------------------------------------------------------
ESPACIOS_CLF = {
    "knn": [Param("n_neighbors", "int", 5, 200, log=True),
            Param("weights", "cat", opciones=("uniform", "distance")),
            Param("p", "cat", opciones=(1, 2))],
    "naive_bayes": [Param("var_smoothing", "float", 1e-11, 1e-1, log=True)],
    "logistica": [Param("C", "float", 1e-3, 1e2, log=True),
                  Param("penalty", "cat", opciones=("l1", "l2"))],
    "arbol": [Param("max_depth", "int", 2, 16),
              Param("min_samples_leaf", "int", 5, 300, log=True),
              Param("criterion", "cat", opciones=("gini", "entropy"))],
    "random_forest": [Param("n_estimators", "int", 50, 250, log=True),
                      Param("max_depth", "int", 3, 16),
                      Param("min_samples_leaf", "int", 1, 60, log=True),
                      Param("max_features", "float", 0.1, 0.6)],
    "xgboost": [Param("n_estimators", "int", 80, 600, log=True),
                Param("max_depth", "int", 2, 8),
                Param("learning_rate", "float", 0.01, 0.3, log=True),
                Param("subsample", "float", 0.5, 1.0),
                Param("colsample_bytree", "float", 0.4, 1.0),
                Param("min_child_weight", "float", 1.0, 30.0, log=True)],
    "svm": [Param("C", "float", 1e-2, 1e2, log=True),
            Param("gamma", "float", 1e-4, 1e-1, log=True)],
}

ESPACIOS_REG = {
    "knn": ESPACIOS_CLF["knn"],
    "ridge": [Param("alpha", "float", 1e-3, 1e3, log=True)],
    "lasso": [Param("alpha", "float", 1e-5, 1e0, log=True)],
    "arbol": [Param("max_depth", "int", 2, 16),
              Param("min_samples_leaf", "int", 5, 300, log=True),
              Param("criterion", "cat", opciones=("squared_error", "friedman_mse"))],
    "random_forest": ESPACIOS_CLF["random_forest"],
    "xgboost": ESPACIOS_CLF["xgboost"],
    "svr": [Param("C", "float", 1e-2, 1e2, log=True),
            Param("gamma", "float", 1e-4, 1e-1, log=True),
            Param("epsilon", "float", 1e-3, 0.5, log=True)],
}


def espacio(tarea: str, modelo: str) -> list[Param]:
    """Espacio de búsqueda de un modelo para la tarea ``"clf"`` o ``"reg"``."""
    return (ESPACIOS_CLF if tarea == "clf" else ESPACIOS_REG)[modelo]


def _clasificador(modelo: str, params: dict, balanceo: str, prevalencia: float, seed: int):
    cw = "balanced" if balanceo == "balanced" else None
    if modelo == "knn":
        cls = KNNBalanceado if balanceo == "balanced" else KNeighborsClassifier
        return cls(n_jobs=1, **params)
    if modelo == "naive_bayes":
        priors = [0.5, 0.5] if balanceo == "balanced" else None
        return GaussianNB(priors=priors, **params)
    if modelo == "logistica":
        return LogisticRegression(solver="liblinear", max_iter=5000, class_weight=cw,
                                  random_state=seed, **params)
    if modelo == "arbol":
        return DecisionTreeClassifier(class_weight=cw, random_state=seed, **params)
    if modelo == "random_forest":
        return RandomForestClassifier(class_weight=cw, random_state=seed, n_jobs=1,
                                      max_samples=0.3, **params)
    if modelo == "xgboost":
        spw = (1 - prevalencia) / prevalencia if balanceo == "balanced" else 1.0
        return XGBClassifier(tree_method="hist", n_jobs=1, random_state=seed,
                             scale_pos_weight=spw, eval_metric="logloss", **params)
    if modelo == "svm":
        # Núcleo RBF aproximado por Nyström (600 componentes) y SVM lineal en ese espacio:
        # costo lineal en n en lugar del O(n²)-O(n³) de la SVC exacta.
        return Pipeline([("nystroem", Nystroem(kernel="rbf", gamma=params["gamma"], n_components=600,
                                                random_state=seed)),
                         ("svc", LinearSVC(C=params["C"], class_weight=cw, dual=False, max_iter=5000))])
    raise ValueError(f"Modelo de clasificación desconocido: {modelo}")


def _regresor(modelo: str, params: dict, seed: int):
    if modelo == "knn":
        return KNeighborsRegressor(n_jobs=1, **params)
    if modelo == "ridge":
        return Ridge(random_state=seed, **params)
    if modelo == "lasso":
        return Lasso(max_iter=20000, random_state=seed, **params)
    if modelo == "arbol":
        return DecisionTreeRegressor(random_state=seed, **params)
    if modelo == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=1, max_samples=0.3, **params)
    if modelo == "xgboost":
        return XGBRegressor(tree_method="hist", n_jobs=1, random_state=seed, **params)
    if modelo == "svr":
        return Pipeline([("nystroem", Nystroem(kernel="rbf", gamma=params["gamma"], n_components=600,
                                                random_state=seed)),
                         ("svr", LinearSVR(C=params["C"], epsilon=params["epsilon"], dual="auto",
                                           max_iter=5000, random_state=seed))])
    raise ValueError(f"Modelo de regresión desconocido: {modelo}")


def construir_pipeline(tarea: str, modelo: str, params: dict, balanceo: str = "ninguno",
                       prevalencia: float = 0.25, seed: int = SEED):
    """Pipeline completo y sin ajustar.

    Parameters
    ----------
    tarea : {"clf", "reg"}
    modelo : str
        Clave del catálogo (``MODELOS_CLF`` o ``MODELOS_REG``).
    params : dict
        Hiperparámetros del estimador final.
    balanceo : str
        ``ninguno``, ``smote``, ``adasyn`` o ``balanced`` (solo clasificación).
    prevalencia : float
        Proporción de la clase positiva, usada por XGBoost en ``scale_pos_weight``.

    Notes
    -----
    El sobremuestreo vive dentro de un ``imblearn.Pipeline``: se aplica solo en ``fit`` y
    nunca sobre los datos de validación o de prueba.
    """
    pasos = [("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]
    if tarea == "reg":
        pipe = Pipeline(pasos + [("modelo", _regresor(modelo, params, seed))])
        # El objetivo se estandariza para que SVR, KNN y Lasso operen en una escala común
        return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
    if balanceo == "smote":
        pasos.append(("balanceo", SMOTE(random_state=seed, k_neighbors=5)))
    elif balanceo == "adasyn":
        pasos.append(("balanceo", ADASYN(random_state=seed, n_neighbors=5)))
    pasos.append(("modelo", _clasificador(modelo, params, balanceo, prevalencia, seed)))
    return ImbPipeline(pasos) if balanceo in ("smote", "adasyn") else Pipeline(pasos)


def puntaje_continuo(estimador, X) -> np.ndarray:
    """Probabilidad de la clase positiva o, en la SVM, la función de decisión transformada.

    La SVC se entrena sin ``probability=True`` (que quintuplica el costo con una validación
    interna); su función de decisión pasa por una logística fija, de modo que el puntaje es
    monótono (válido para AUC) pero sin calibrar, lo que se discute en la Sección 5.1.3.
    """
    if hasattr(estimador, "predict_proba") and not _es_svc(estimador):
        return estimador.predict_proba(X)[:, 1]
    return 1.0 / (1.0 + np.exp(-estimador.decision_function(X)))


def _es_svc(estimador) -> bool:
    final = estimador
    while hasattr(final, "steps"):
        final = final[-1]
    return isinstance(final, (SVC, LinearSVC))
