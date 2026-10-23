"""Calibración de probabilidades y explicación local por valores de Shapley.

La explicación local usa el estimador por permutaciones de Štrumbelj y Kononenko (2014):
para cada permutación de las variables se recorre la cadena desde un individuo de
referencia hasta la observación, y la contribución marginal de cada variable al cambio
en la predicción se promedia. El estimador es insesgado y satisface la propiedad de
eficiencia en cada permutación (la suma de contribuciones es exactamente f(x) − f(ref)),
y solo requiere ``predict_proba``, de modo que sirve para cualquier modelo del proyecto,
incluido el modelo original, sin cargar la biblioteca ``shap`` en el servidor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .config import SEED


def ece(y, p, n_bins: int = 10) -> float:
    """Error de calibración esperado con bins de igual ancho."""
    y, p = np.asarray(y), np.clip(np.asarray(p), 0, 1)
    bordes = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bordes[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def diagrama_confiabilidad(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Frecuencia observada frente a probabilidad media predicha por bin."""
    obs, pred = calibration_curve(y, np.clip(p, 0, 1), n_bins=n_bins, strategy="uniform")
    return pd.DataFrame({"prob_predicha": pred, "frec_observada": obs})


class Recalibrador:
    """Recalibración posterior por escalado de Platt o regresión isotónica.

    Se ajusta con predicciones fuera de muestra del entrenamiento (las del bucle externo),
    nunca con el conjunto de prueba.
    """

    def __init__(self, metodo: str = "platt"):
        self.metodo = metodo

    def fit(self, p, y):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        if self.metodo == "platt":
            self.m_ = LogisticRegression(C=1e6).fit(np.log(p / (1 - p)).reshape(-1, 1), y)
        else:
            self.m_ = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        return self

    def transform(self, p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        if self.metodo == "platt":
            return self.m_.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]
        return self.m_.predict(p)


def shapley_permutaciones(f, x: np.ndarray, fondo: np.ndarray, n_perm: int = 40,
                          seed: int = SEED) -> tuple[np.ndarray, float]:
    """Valores de Shapley de una observación por muestreo de permutaciones antitéticas.

    Parameters
    ----------
    f : callable
        ``f(X) -> array`` con la salida a explicar (probabilidad de la clase positiva o
        predicción de regresión).
    x : array (p,)
        Observación a explicar.
    fondo : array (m, p)
        Muestra de referencia del entrenamiento; define la distribución de "ausencia".
    n_perm : int
        Número de permutaciones (cada una se usa también en sentido inverso).

    Returns
    -------
    phi : array (p,)
        Contribución de cada variable.
    base : float
        Valor esperado de f sobre el fondo, de modo que base + Σφ ≈ f(x).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    fondo = np.asarray(fondo, float)
    p = len(x)
    phi = np.zeros(p)
    filas = []
    ordenes = []
    for _ in range(n_perm):
        perm = rng.permutation(p)
        for orden in (perm, perm[::-1]):
            ref = fondo[rng.integers(len(fondo))].copy()
            cadena = np.empty((p + 1, p))
            cadena[0] = ref
            for j, var in enumerate(orden):
                ref[var] = x[var]
                cadena[j + 1] = ref
            filas.append(cadena)
            ordenes.append(orden)
    salida = np.asarray(f(np.vstack(filas)), float).reshape(len(filas), p + 1)
    for orden, s in zip(ordenes, salida):
        phi[orden] += np.diff(s)
    phi /= len(ordenes)
    base = float(np.mean(f(fondo)))
    # Ajuste de eficiencia: reparte el residuo de muestreo para que base + Σφ = f(x)
    fx = float(np.asarray(f(x[None, :]), float)[0])
    residuo = fx - base - phi.sum()
    phi += residuo * np.abs(phi) / max(np.abs(phi).sum(), 1e-12)
    return phi, base
