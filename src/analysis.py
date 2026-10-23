"""Funciones de análisis compartidas por los capítulos del libro y la exportación del dashboard.

Leen los resultados guardados por el motor de experimentos (un archivo por combinación y
pliegue externo) y construyen tablas de pliegues, predicciones fuera de muestra, curvas de
convergencia y las tablas de pruebas estadísticas.
"""
from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RESULTADOS
from .models import NOMBRES_BALANCEO, NOMBRES_MODELO, NOMBRES_OPT


@lru_cache(maxsize=None)
def cargar_carpeta(carpeta: str) -> tuple:
    """Lee todos los ``.pkl`` de una carpeta de resultados (resultado cacheado)."""
    return tuple(pickle.load(open(f, "rb")) for f in sorted((RESULTADOS / carpeta).glob("*.pkl")))


def tabla_pliegues(carpeta: str) -> pd.DataFrame:
    """Una fila por combinación y pliegue externo, con métricas y tiempos."""
    filas = []
    for r in cargar_carpeta(carpeta):
        filas.append({"modelo": r["modelo"], "balanceo": r["balanceo"], "optimizador": r["optimizador"],
                      "pliegue": r["pliegue"], "bloque": r["bloque"], **r["metricas"],
                      "puntaje_interno": r["puntaje_interno"], "t_opt": r["t_opt"], "t_fit": r["t_fit"],
                      "t_inferencia": r["t_inferencia"], "n_eval": r["n_eval"],
                      "n_test": len(r["y_test"])})
    d = pd.DataFrame(filas)
    d["combo"] = d.modelo + " | " + d.balanceo + " | " + d.optimizador
    return d


def nombres(d: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas legibles de modelo, balanceo y optimizador."""
    d = d.copy()
    d["nombre"] = d.modelo.map(lambda m: NOMBRES_MODELO.get(m, m))
    if "balanceo" in d:
        d["balanceo_nombre"] = d.balanceo.map(lambda b: NOMBRES_BALANCEO.get(b, b))
    if "optimizador" in d:
        d["optimizador_nombre"] = d.optimizador.map(lambda o: NOMBRES_OPT.get(o, o))
    return d


def resumen(d: pd.DataFrame, metricas: list[str], claves=("modelo", "balanceo", "optimizador")) -> pd.DataFrame:
    """Media y desviación estándar por combinación a través de los pliegues externos."""
    g = d.groupby(list(claves))[metricas].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g["t_total"] = d.groupby(list(claves)).t_opt.sum() + d.groupby(list(claves)).t_fit.sum()
    g["n_eval_total"] = d.groupby(list(claves)).n_eval.sum()
    return g.reset_index()


def oof(carpeta: str, modelo: str, balanceo: str, optimizador: str) -> pd.DataFrame:
    """Predicciones fuera de muestra (bucle externo) de una combinación, en orden temporal."""
    filas = [pd.DataFrame({"idx": r["idx_test"], "y": r["y_test"], "pred": r["pred"], "pliegue": r["pliegue"]})
             for r in cargar_carpeta(carpeta)
             if (r["modelo"], r["balanceo"], r["optimizador"]) == (modelo, balanceo, optimizador)]
    return pd.concat(filas).sort_values("idx").reset_index(drop=True)


def historiales(carpeta: str) -> pd.DataFrame:
    """Historial de cada optimización (una fila por evaluación), con la curva *anytime*."""
    filas = []
    for r in cargar_carpeta(carpeta):
        for h in r["historial"]:
            filas.append({"modelo": r["modelo"], "balanceo": r["balanceo"], "optimizador": r["optimizador"],
                          "pliegue": r["pliegue"], "eval": h["eval"], "puntaje": h["puntaje"],
                          "mejor": h["mejor"], "t_eval": h["t_eval"], "t_acum": h["t_acum"],
                          "generacion": h.get("generacion"), "diversidad": h.get("diversidad")})
    return pd.DataFrame(filas)


def matriz_bloques(d: pd.DataFrame, metrica: str, columna: str = "combo") -> pd.DataFrame:
    """Tabla pliegues × modelos de una métrica, para las pruebas de rangos."""
    return d.pivot_table(index="pliegue", columns=columna, values=metrica)


def guardar_json(obj, ruta: Path):
    ruta.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
