"""Modelos LTSF-Linear (Zeng et al., 2023) adaptados a la predicción del caudal a quince días.

Zeng et al. propusieron tres modelos lineales para el pronóstico de largo plazo de series de
tiempo (LTSF): Linear, una capa lineal sobre la ventana de historia; NLinear, que antes resta a
la ventana su último valor; y DLinear, que la descompone en tendencia y estacionalidad. Los tres
se implementan aquí; la descripción siguiente corresponde a DLinear y las diferencias de los
otros dos se documentan en sus clases.

DLinear descompone la ventana de historia de cada serie en una tendencia, obtenida con una
media móvil, y un componente estacional, el residuo, y proyecta cada componente sobre el
horizonte con una capa lineal. Zeng et al. mostraron que este modelo lineal supera a
arquitecturas basadas en transformers en los conjuntos de referencia de pronóstico de series
de tiempo, por lo que se usa como referencia obligada para cualquier modelo más complejo.

Adaptación al problema
----------------------
* La entrada es la ventana de los ``longitud`` días más recientes del logaritmo del caudal de
  Calamar y, opcionalmente, de Puerto Salgar como covariable.
* La salida es el cambio logarítmico a quince días, ``z = log(Q(t+15)/Q(t))``, la misma
  variable que modelan el modelo original y NGBoost.
* Las dos capas lineales se ajustan juntas por mínimos cuadrados con una penalización L2
  pequeña; es el óptimo exacto del error cuadrático que DLinear minimiza por descenso de
  gradiente.
* Como DLinear es un pronosticador puntual, la probabilidad de banda se obtiene suponiendo un
  residuo gaussiano cuya dispersión se estima fuera de muestra (ver ``experiment_original``).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge

from .config import DATOS_PROC

LONGITUD_MAXIMA = 365
CANALES = {"calamar": "cal", "pto_salgar": "ps"}


def media_movil(x: np.ndarray, kernel: int) -> np.ndarray:
    """Media móvil centrada con relleno replicado en los extremos, como en DLinear.

    Parameters
    ----------
    x : array (n, L)
        Ventanas ordenadas de la más antigua a la más reciente.
    kernel : int
        Tamaño impar de la ventana de la media móvil.
    """
    m = (kernel - 1) // 2
    relleno = np.concatenate([np.repeat(x[:, :1], m, axis=1), x, np.repeat(x[:, -1:], m, axis=1)], axis=1)
    acumulada = np.cumsum(np.pad(relleno, ((0, 0), (1, 0))), axis=1)
    return (acumulada[:, kernel:] - acumulada[:, :-kernel]) / kernel


def descomponer(x: np.ndarray, kernel: int) -> tuple[np.ndarray, np.ndarray]:
    """Tendencia (media móvil) y componente estacional (residuo) de cada ventana."""
    tendencia = media_movil(x, kernel)
    return tendencia, x - tendencia


class DLinear(RegressorMixin, BaseEstimator):
    """DLinear de un paso: tendencia y estacionalidad proyectadas por dos capas lineales.

    Parameters
    ----------
    longitud : int
        Días de historia de la ventana (≤ 365).
    kernel : int
        Tamaño de la media móvil de la descomposición; se fuerza a impar.
    alpha : float
        Penalización L2 de las capas lineales.
    canales : str
        ``"calamar"`` (univariado, como el DLinear original) o ``"ambos"`` (añade Puerto Salgar
        como covariable).
    """

    def __init__(self, longitud: int = 90, kernel: int = 25, alpha: float = 1e-3, canales: str = "calamar"):
        self.longitud = longitud
        self.kernel = kernel
        self.alpha = alpha
        self.canales = canales

    def _entrada(self, W: pd.DataFrame) -> np.ndarray:
        nombres = ["calamar"] if self.canales == "calamar" else ["calamar", "pto_salgar"]
        L = int(self.longitud)
        k = min(int(self.kernel) | 1, L if L % 2 else L - 1)   # impar y no mayor que la ventana
        partes = []
        for c in nombres:
            pre = CANALES[c]
            cols = [f"{pre}_{i}" for i in range(L - 1, -1, -1)]  # de la más antigua a la más reciente
            x = np.log(W[cols].to_numpy(dtype=np.float64))
            tendencia, estacional = descomponer(x, k)
            partes += [tendencia, estacional]
        return np.hstack(partes)

    def fit(self, W: pd.DataFrame, z: np.ndarray):
        self.lineal_ = Ridge(alpha=self.alpha).fit(self._entrada(W), np.asarray(z, dtype=np.float64))
        return self

    def predict(self, W: pd.DataFrame) -> np.ndarray:
        return self.lineal_.predict(self._entrada(W))


class Linear(DLinear):
    """Linear de LTSF-Linear: una sola capa lineal sobre la ventana sin transformar.

    La capa predice el nivel futuro, ``log Q(t+15)``, a partir de los logaritmos de la ventana;
    el cambio logarítmico se recupera restando ``log Q(t)``. ``kernel`` no interviene.
    """

    def _ventanas(self, W: pd.DataFrame) -> list[np.ndarray]:
        nombres = ["calamar"] if self.canales == "calamar" else ["calamar", "pto_salgar"]
        L = int(self.longitud)
        return [np.log(W[[f"{CANALES[c]}_{i}" for i in range(L - 1, -1, -1)]].to_numpy(dtype=np.float64))
                for c in nombres]

    def _entrada(self, W: pd.DataFrame) -> np.ndarray:
        return np.hstack(self._ventanas(W))

    @staticmethod
    def _nivel_actual(W: pd.DataFrame) -> np.ndarray:
        return np.log(W["cal_0"].to_numpy(dtype=np.float64))

    def fit(self, W: pd.DataFrame, z: np.ndarray):
        nivel = np.asarray(z, dtype=np.float64) + self._nivel_actual(W)
        self.lineal_ = Ridge(alpha=self.alpha).fit(self._entrada(W), nivel)
        return self

    def predict(self, W: pd.DataFrame) -> np.ndarray:
        return self.lineal_.predict(self._entrada(W)) - self._nivel_actual(W)


class NLinear(Linear):
    """NLinear de LTSF-Linear: la capa lineal actúa sobre la ventana menos su último valor.

    La normalización resta a cada canal su valor más reciente y lo suma de nuevo a la salida,
    lo que protege al modelo de los cambios de nivel entre el entrenamiento y la prueba. Como
    el último valor de Calamar es ``log Q(t)``, la salida normalizada es directamente el cambio
    logarítmico ``z``.
    """

    def _entrada(self, W: pd.DataFrame) -> np.ndarray:
        return np.hstack([x - x[:, -1:] for x in self._ventanas(W)])

    def fit(self, W: pd.DataFrame, z: np.ndarray):
        self.lineal_ = Ridge(alpha=self.alpha).fit(self._entrada(W), np.asarray(z, dtype=np.float64))
        return self

    def predict(self, W: pd.DataFrame) -> np.ndarray:
        return self.lineal_.predict(self._entrada(W))


MODELOS_LTSF = {"dlinear": DLinear, "nlinear": NLinear, "linear": Linear}


def construir_ventanas(fechas: pd.Series, referencia: pd.Series,
                       longitud: int = LONGITUD_MAXIMA) -> pd.DataFrame:
    """Ventanas de historia de Calamar y Puerto Salgar alineadas con las filas de la tarea.

    La columna ``cal_k`` es el caudal de Calamar k días antes del día de la predicción
    (``cal_0`` es el día mismo) y ``ps_k`` el de Puerto Salgar. Los huecos se rellenan hacia
    adelante, de modo que cada valor usa solo información disponible en ese momento; los días
    anteriores al primer registro de una estación toman su primer valor observado, lo que solo
    afecta a las primeras ventanas de 1947.
    """
    caudales = pd.read_csv(DATOS_PROC / "caudales.csv", parse_dates=["fecha"]).set_index("fecha")
    serie = caudales[list(CANALES)].asfreq("D").ffill().bfill()
    pos = serie.index.get_indexer(pd.to_datetime(fechas))
    if (pos < longitud).any():
        raise ValueError("La historia disponible no cubre la ventana pedida.")
    columnas = {}
    for c, pre in CANALES.items():
        valores = serie[c].to_numpy(dtype=np.float32)
        for k in range(longitud):
            columnas[f"{pre}_{k}"] = valores[pos - k]
    W = pd.DataFrame(columnas)
    W["calamar_cuad"] = np.asarray(referencia, dtype=np.float64)   # caudal de referencia, como en BLGB
    return W


def datos_con_ventanas(datos):
    """Copia de ``Datos`` cuyas matrices de predictores son las ventanas de DLinear."""
    return replace(datos, X=construir_ventanas(datos.fechas, datos.X["calamar_cuad"].values),
                   X_test=construir_ventanas(datos.fechas_test, datos.X_test["calamar_cuad"].values))
