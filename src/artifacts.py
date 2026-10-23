"""Serialización de modelos finales con su metadata y verificación de ida y vuelta."""
from __future__ import annotations

import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import ARTEFACTOS, SEED

LIBRERIAS = ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost", "scipy", "joblib"]


def esquema_columnas(X: pd.DataFrame) -> list[dict]:
    """Nombre, tipo y rango válido (percentiles 0.5 y 99.5 del entrenamiento) de cada columna."""
    out = []
    for c in X.columns:
        s = X[c].dropna()
        out.append({"nombre": c, "tipo": str(X[c].dtype), "min": float(s.min()), "max": float(s.max()),
                    "p005": float(s.quantile(0.005)), "p995": float(s.quantile(0.995)),
                    "mediana": float(s.median())})
    return out


def guardar_artefacto(nombre: str, estimador, X_ref: pd.DataFrame, X_prueba: pd.DataFrame,
                      predecir, meta: dict, destino: Path = ARTEFACTOS) -> dict:
    """Guarda el estimador comprimido y su metadata; verifica que la recarga sea idéntica.

    Parameters
    ----------
    predecir : callable
        ``predecir(estimador, X)`` → arreglo de predicciones usado en la prueba de ida y vuelta.
    """
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / f"{nombre}.joblib"
    joblib.dump(estimador, ruta, compress=3)
    recargado = joblib.load(ruta)
    p_mem, p_disco = predecir(estimador, X_prueba), predecir(recargado, X_prueba)
    identico = bool(np.array_equal(p_mem, p_disco))
    if not identico:
        raise AssertionError(f"La prueba de ida y vuelta falló para {nombre}.")
    metadata = {"modelo": nombre, "semilla": SEED, "python": platform.python_version(),
                "librerias": {lib: version(lib) for lib in LIBRERIAS},
                "esquema_entrada": esquema_columnas(X_ref), "columnas": list(X_ref.columns),
                "ida_y_vuelta": {"identico": identico, "n_filas": int(len(X_prueba)),
                                 "max_dif_abs": float(np.max(np.abs(p_mem - p_disco)))},
                "tamano_mb": round(ruta.stat().st_size / 2 ** 20, 3), **meta}
    (destino / f"{nombre}.metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=_serializable), encoding="utf-8")
    return metadata


def _serializable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
