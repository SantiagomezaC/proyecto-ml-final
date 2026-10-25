"""Configuración central del proyecto: semilla, rutas, partición y constantes del dominio.

Todos los módulos importan desde aquí, de modo que una decisión metodológica
(corte temporal, horizonte, separación) se define una sola vez.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Semilla global única, propagada a numpy, scikit-learn, XGBoost, Optuna y DEAP
# ---------------------------------------------------------------------------
SEED = 42


def fijar_semillas(seed: int = SEED) -> None:
    """Fija las fuentes de aleatoriedad de Python y numpy.

    Parameters
    ----------
    seed : int
        Semilla global. Los estimadores reciben además ``random_state=seed``.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Rutas (relativas a la raíz del repositorio; sin rutas absolutas en el código)
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data"
DATOS_CRUDOS = Path(os.environ.get("MAGDALENA_RAW", DATOS / "raw"))
DATOS_PROC = DATOS / "processed"
RESULTADOS = RAIZ / "results"
ARTEFACTOS = RAIZ / "artifacts"
FIGURAS = RAIZ / "book" / "figuras"

# ---------------------------------------------------------------------------
# Delimitación temporal
# ---------------------------------------------------------------------------
# Ningún predictor se imputa en años en que su
# estación no existía. La tarea principal usa Calamar y Puerto Salgar (registro
# desde 1946) y empieza en 1947, lo que da más de 20.000 observaciones. Las
# estaciones del bajo Magdalena (instaladas en 1972-1975) solo intervienen en el
# análisis de sensibilidad y en las tareas adicionales, que empiezan en 1976.
INICIO_MODELADO = "1947-01-01"
INICIO_RED_COMPLETA = "1976-01-01"
PREDICTORAS_CALAMAR = ["pto_salgar"]
FECHA_CORTE_TEST = "2010-01-01"   # conjunto de prueba reservado: 2010-2026
GAP_DIAS = 30                     # separación entre entrenamiento y evaluación
HORIZONTE = 15                    # días de anticipación

# Validación anidada: bloques externos de dos años entre 1986 y 2009
INICIO_BLOQUES_EXTERNOS = 1986
ANIOS_POR_BLOQUE = 2
N_PLIEGUES_INTERNOS = 3

# ---------------------------------------------------------------------------
# Dominio hidrológico
# ---------------------------------------------------------------------------
ESTACION_OBJETIVO = "calamar"
TOLERANCIA = 0.15                 # banda de ±15 % alrededor de la descarga efectiva
N_CLASES_MF = 25                  # clases del análisis magnitud-frecuencia

# Arrancaplumas se excluye de los predictores: su registro termina en 2014.
EXCLUIR_MODELO = ["arrancaplumas"]

ALIAS_ESTACIONES = {
    "arrancaplumas  - aut": "arrancaplumas",
    "arrancaplumas - aut": "arrancaplumas",
    "puerto salgar - aut": "pto_salgar",
    "el banco - aut": "el_banco",
}

NOMBRES_LEGIBLES = {
    "calamar": "Calamar",
    "coyongal": "Coyongal",
    "san_roque": "San Roque",
    "el_banco": "El Banco",
    "tres_cruces": "Tres Cruces",
    "pto_salgar": "Puerto Salgar",
    "arrancaplumas": "Arrancaplumas",
}

# ---------------------------------------------------------------------------
# Métrica principal
# ---------------------------------------------------------------------------
METRICA_CLF = "average_precision"   # PR-AUC: clase positiva minoritaria (~25 %)
METRICA_REG = "neg_root_mean_squared_error"

# ---------------------------------------------------------------------------
# Paleta institucional (máximo cinco colores, coherente entre libro y dashboard)
# ---------------------------------------------------------------------------
PALETA = {
    "rio": "#0B4F6C",       # azul profundo: color principal
    "sedimento": "#C8813A", # ocre: sedimento, énfasis
    "caudal": "#20A39E",    # turquesa: series y elementos secundarios
    "alerta": "#B23A48",    # carmín: banda de descarga efectiva, errores
    "neutro": "#5C6B73",    # gris pizarra: texto secundario y referencias
}
