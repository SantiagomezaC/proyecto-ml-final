"""Carga y caché de los datos y artefactos que consume el dashboard.

El dashboard nunca entrena ni optimiza: lee tablas livianas de ``data/dashboard`` y los
modelos serializados de ``artifacts``. Cada recurso se carga una sola vez por proceso
(``functools.lru_cache``).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .config import ARTEFACTOS, DATOS, NOMBRES_LEGIBLES

DASH = DATOS / "dashboard"

ORDEN_MODELOS = ["original", "xgboost", "random_forest", "svm", "logistica", "knn", "arbol",
                 "naive_bayes"]


def es(texto):
    """Convierte los números de un texto al formato español (1.234,5) sin tocar la prosa."""
    if not isinstance(texto, str):
        return texto
    texto = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", " ", texto)   # separador de miles provisional
    texto = re.sub(r"(?<=\d)\.(?=\d)", ",", texto)
    return texto.replace(" ", ".")


def etiqueta(col: str) -> str:
    """Nombre legible y unidad de cada predictor, para formularios, ejes y tablas."""
    est = {k: v for k, v in NOMBRES_LEGIBLES.items()}
    otros = {"sen_anual": "Seno del ciclo anual (–)", "cos_anual": "Coseno del ciclo anual (–)",
             "sen_semi": "Seno del ciclo semianual (–)", "cos_semi": "Coseno del ciclo semianual (–)",
             "dist_qef": "Distancia del caudal actual a Q_ef (m³/s)",
             "dist_qef_rel": "Distancia relativa a Q_ef (–)",
             "calamar_cuad": "Caudal actual en Calamar al cuadrado (m⁶/s²)",
             "dist_qef_lag7": "Distancia a Q_ef hace 7 días (m³/s)",
             "dist_qef_lag15": "Distancia a Q_ef hace 15 días (m³/s)"}
    if col in otros:
        return otros[col]
    if col == "y":
        return "Clase (1 = en la banda)"
    if col == "q_futuro":
        return "Caudal en Calamar en t+15 (m³/s)"
    if col in est:
        return f"Caudal en {est[col]} (m³/s)"
    m = re.match(r"(\w+?)_lag(\d+)$", col)
    if m and m.group(1) in est:
        return f"Caudal en {est[m.group(1)]} hace {m.group(2)} días (m³/s)"
    m = re.match(r"(\w+?)_(media|desv)(\d+)$", col)
    if m:
        tipo = "Media" if m.group(2) == "media" else "Desviación"
        return f"{tipo} móvil {m.group(3)} d en {est.get(m.group(1), m.group(1))} (m³/s)"
    m = re.match(r"(\w+?)_d(\d+)$", col)
    if m:
        return f"Cambio en {m.group(2)} d de {est.get(m.group(1), m.group(1))} (m³/s)"
    m = re.match(r"(\w+?)_t(\d+)$", col)
    if m:
        return f"Caudal en {est.get(m.group(1), m.group(1))} desfasado {m.group(2)} d (m³/s)"
    m = re.match(r"(\w+?)_rel$", col)
    if m:
        return f"Caudal relativo a la media anual en {est.get(m.group(1), m.group(1))} (–)"
    return col


def _leer_csv(nombre, **kw):
    return pd.read_csv(DASH / nombre, **kw)


@lru_cache(maxsize=None)
def parametros() -> dict:
    return json.loads((DASH / "parametros.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def caudales() -> pd.DataFrame:
    return _leer_csv("caudales.csv", parse_dates=["fecha"])


@lru_cache(maxsize=None)
def tarea() -> pd.DataFrame:
    return pd.read_parquet(DASH / "tarea_calamar.parquet")


@lru_cache(maxsize=None)
def ubicaciones() -> pd.DataFrame:
    return _leer_csv("ubicaciones.csv")


@lru_cache(maxsize=None)
def tabla(nombre: str) -> pd.DataFrame:
    """Tabla precalculada de ``data/dashboard`` (csv o parquet)."""
    ruta = DASH / nombre
    if ruta.suffix == ".parquet":
        return pd.read_parquet(ruta)
    return pd.read_csv(ruta)


@lru_cache(maxsize=None)
def json_dash(nombre: str) -> dict:
    return json.loads((DASH / nombre).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def modelo(nombre: str):
    """Pipeline final serializado (se carga una vez y queda en memoria)."""
    return joblib.load(ARTEFACTOS / f"{nombre}.joblib")


@lru_cache(maxsize=None)
def recalibrador(nombre: str):
    ruta = ARTEFACTOS / f"{nombre}.recalibrador.joblib"
    return joblib.load(ruta) if ruta.exists() else None


@lru_cache(maxsize=None)
def metadata(nombre: str) -> dict:
    return json.loads((ARTEFACTOS / f"{nombre}.metadata.json").read_text(encoding="utf-8"))


def modelos_disponibles() -> list[str]:
    return [m for m in ORDEN_MODELOS if (ARTEFACTOS / f"{m}.joblib").exists()]


def probabilidad(nombre: str, X: pd.DataFrame) -> np.ndarray:
    """Probabilidad de la clase positiva con la recalibración de Platt cuando existe."""
    from .models import puntaje_continuo
    est = modelo(nombre)
    s = puntaje_continuo(est, X)
    cal = recalibrador(nombre)
    usar = metadata(nombre).get("usar_recalibrado", False)
    return cal.transform(s) if (cal is not None and usar) else s
