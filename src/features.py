"""Ingeniería de características y construcción de las tareas de predicción.

Todas las variables usan información disponible hasta el día t; el objetivo se ubica en
t + h. Las estaciones aguas arriba entran desplazadas por su tiempo de tránsito.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HORIZONTE, INICIO_MODELADO


def construir_features(caudales: pd.DataFrame, objetivo: str, predictoras: list[str],
                       transitos: dict[str, int], q_ef: float) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Construye los predictores del Entregable 1 para una estación objetivo.

    Parameters
    ----------
    caudales : DataFrame
        Tabla diaria con columna ``fecha`` y una columna de caudal por estación.
    objetivo : str
        Estación cuya condición futura se predice.
    predictoras : list of str
        Estaciones aguas arriba que aportan información anticipada.
    transitos : dict
        Rezago (días) de cada estación predictora respecto a la objetivo.
    q_ef : float
        Descarga efectiva de la estación objetivo (m³/s).

    Returns
    -------
    x : DataFrame
        Tabla con ``fecha``, el caudal actual de la objetivo y los predictores.
    base_cols, dom_cols : list of str
        Predictores hidrológicos y predictores de dominio (distancia a la descarga efectiva).
    """
    x = caudales.sort_values("fecha").reset_index(drop=True).copy()
    o = objetivo
    nuevas = {}
    for k in (1, 2, 3, 5, 7, 10, 15, 30, 60):
        nuevas[f"{o}_lag{k}"] = x[o].shift(k)
    base = x[o].shift(1)                       # excluye el día actual de las ventanas
    for w in (7, 30, 90, 365):
        nuevas[f"{o}_media{w}"] = base.rolling(w, min_periods=max(2, w // 3)).mean()
        nuevas[f"{o}_desv{w}"] = base.rolling(w, min_periods=max(2, w // 3)).std()
    nuevas[f"{o}_d1"] = x[o].diff(1)
    nuevas[f"{o}_d7"] = x[o].diff(7)
    for e in predictoras:
        k = transitos.get(e, 0)
        desplazada = x[e].shift(k)
        nuevas[f"{e}_t{k}"] = desplazada
        nuevas[f"{e}_d3"] = desplazada.diff(3)
        nuevas[f"{e}_rel"] = desplazada / x[e].shift(1).rolling(365, min_periods=120).mean()
    doy = x.fecha.dt.dayofyear
    nuevas["sen_anual"] = np.sin(2 * np.pi * doy / 365.25)
    nuevas["cos_anual"] = np.cos(2 * np.pi * doy / 365.25)
    nuevas["sen_semi"] = np.sin(4 * np.pi * doy / 365.25)
    nuevas["cos_semi"] = np.cos(4 * np.pi * doy / 365.25)
    # Variables de dominio: la distancia a Q_ef vuelve monótona la condición de banda
    nuevas["dist_qef"] = (x[o] - q_ef).abs()
    nuevas["dist_qef_rel"] = nuevas["dist_qef"] / q_ef
    nuevas[f"{o}_cuad"] = x[o] ** 2
    for k in (7, 15):
        nuevas[f"dist_qef_lag{k}"] = (nuevas[f"{o}_lag{k}"] - q_ef).abs()
    x = pd.concat([x[["fecha", o]], pd.DataFrame(nuevas)], axis=1)

    base_cols = ([f"{o}_lag{k}" for k in (1, 2, 3, 5, 7, 10, 15, 30, 60)]
                 + [f"{o}_{s}{w}" for w in (7, 30, 90, 365) for s in ("media", "desv")]
                 + [f"{o}_d1", f"{o}_d7"]
                 + [f"{e}_{s}" for e in predictoras for s in (f"t{transitos.get(e, 0)}", "d3", "rel")]
                 + ["sen_anual", "cos_anual", "sen_semi", "cos_semi"])
    dom_cols = ["dist_qef", "dist_qef_rel", f"{o}_cuad", "dist_qef_lag7", "dist_qef_lag15"]
    return x, base_cols, dom_cols


def construir_tarea(caudales: pd.DataFrame, objetivo: str, predictoras: list[str],
                    transitos: dict[str, int], q_ef: float, lim: tuple[float, float],
                    horizonte: int = HORIZONTE, inicio: str = INICIO_MODELADO) -> dict:
    """Arma la tabla de modelado con los dos objetivos del proyecto.

    * ``y``: 1 si el caudal del día t + h cae en la banda de descarga efectiva.
    * ``q_futuro``: caudal del día t + h (m³/s), objetivo de la tarea de regresión.

    Las filas anteriores a ``inicio`` se descartan después de calcular las ventanas, de
    modo que los rezagos de 1976 usan el registro observado de 1975.
    """
    x, base_cols, dom_cols = construir_features(caudales, objetivo, predictoras, transitos, q_ef)
    futuro = x[objetivo].shift(-horizonte)
    x["q_futuro"] = futuro
    x["y"] = futuro.between(*lim).astype(float).where(futuro.notna())
    x = x[(x.fecha >= inicio) & x.y.notna() & x[objetivo].notna() & (x[objetivo] > 0)]
    x = x.reset_index(drop=True)
    return {"datos": x, "base_cols": base_cols, "dom_cols": dom_cols,
            "columnas": base_cols + dom_cols}
