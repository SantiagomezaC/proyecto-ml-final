"""Particiones cronológicas para la validación anidada.

El bucle externo usa bloques de calendario consecutivos con ventana de entrenamiento
creciente y una separación de ``GAP_DIAS`` días; el bucle interno usa ``TimeSeriesSplit``
con la misma separación. Las particiones externas se calculan una vez, se guardan en
disco y se reutilizan para todos los modelos, incluido el modelo original.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .config import (ANIOS_POR_BLOQUE, FECHA_CORTE_TEST, GAP_DIAS, INICIO_BLOQUES_EXTERNOS,
                     N_PLIEGUES_INTERNOS)


def separar_prueba(fechas: pd.Series, corte: str = FECHA_CORTE_TEST, gap: int = GAP_DIAS):
    """Índices de entrenamiento (antes del corte) y de prueba (desde corte + gap)."""
    corte = pd.Timestamp(corte)
    tr = np.flatnonzero(fechas < corte)
    te = np.flatnonzero(fechas >= corte + pd.Timedelta(days=gap))
    return tr, te


def pliegues_externos(fechas: pd.Series, inicio: int = INICIO_BLOQUES_EXTERNOS,
                      anios: int = ANIOS_POR_BLOQUE, gap: int = GAP_DIAS,
                      fin: str = FECHA_CORTE_TEST) -> list[dict]:
    """Bloques externos de ``anios`` años con entrenamiento creciente.

    Parameters
    ----------
    fechas : Series
        Fechas del conjunto de entrenamiento, en orden cronológico.

    Returns
    -------
    list of dict
        Cada pliegue con ``train`` y ``test`` (posiciones enteras) y su etiqueta.
    """
    fechas = pd.to_datetime(pd.Series(fechas)).reset_index(drop=True)
    fin_anio = pd.Timestamp(fin).year
    pliegues = []
    for a0 in range(inicio, fin_anio, anios):
        ini, fin_b = pd.Timestamp(f"{a0}-01-01"), pd.Timestamp(f"{a0 + anios}-01-01")
        test = np.flatnonzero((fechas >= ini) & (fechas < fin_b))
        train = np.flatnonzero(fechas < ini - pd.Timedelta(days=gap))
        if len(test) and len(train):
            pliegues.append({"pliegue": len(pliegues), "bloque": f"{a0}-{a0 + anios - 1}",
                             "train": train, "test": test})
    return pliegues


def cv_interna(n_splits: int = N_PLIEGUES_INTERNOS, gap: int = GAP_DIAS) -> TimeSeriesSplit:
    """Validación interna cronológica, usada por todos los optimizadores."""
    return TimeSeriesSplit(n_splits=n_splits, gap=gap)


def guardar_pliegues(pliegues: list[dict], ruta: Path, fechas: pd.Series) -> None:
    """Guarda las particiones externas para garantizar el pareado estricto."""
    fechas = pd.to_datetime(pd.Series(fechas)).reset_index(drop=True)
    salida = [{"pliegue": p["pliegue"], "bloque": p["bloque"],
               "train_fin": str(fechas.iloc[p["train"][-1]].date()),
               "test_inicio": str(fechas.iloc[p["test"][0]].date()),
               "test_fin": str(fechas.iloc[p["test"][-1]].date()),
               "n_train": int(len(p["train"])), "n_test": int(len(p["test"])),
               "train": p["train"].tolist(), "test": p["test"].tolist()} for p in pliegues]
    Path(ruta).write_text(json.dumps(salida), encoding="utf-8")


def cargar_pliegues(ruta: Path) -> list[dict]:
    """Lee las particiones externas guardadas por :func:`guardar_pliegues`."""
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    for p in datos:
        p["train"], p["test"] = np.asarray(p["train"]), np.asarray(p["test"])
    return datos
