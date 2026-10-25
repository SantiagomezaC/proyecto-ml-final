"""Pruebas unitarias de los modelos LTSF-Linear (Linear, NLinear y DLinear)."""
import numpy as np
import pandas as pd
import pytest

from src.dlinear import MODELOS_LTSF, NLinear, descomponer, media_movil

L = 30


def ventanas(n=1500, seed=0):
    """Ventanas sintéticas de caudal positivo y un cambio logarítmico lineal en la historia."""
    rng = np.random.default_rng(seed)
    log_q = np.cumsum(rng.normal(0, 0.05, size=(n, L)), axis=1) + 8.5   # de la más antigua a la más reciente
    cols = {}
    for k in range(L):
        cols[f"cal_{k}"] = np.exp(log_q[:, L - 1 - k])
        cols[f"ps_{k}"] = np.exp(log_q[:, L - 1 - k] - 1.0)
    W = pd.DataFrame(cols)
    z = 0.8 * (log_q[:, -1] - log_q[:, -5]) + rng.normal(0, 0.005, n)
    return W, z


def test_media_movil_de_constante_es_constante():
    x = np.full((3, 20), 2.5)
    assert np.allclose(media_movil(x, 5), 2.5)


def test_descomposicion_suma_la_serie():
    x = np.random.default_rng(1).normal(size=(4, 25))
    tendencia, estacional = descomponer(x, 7)
    assert tendencia.shape == x.shape
    assert np.allclose(tendencia + estacional, x)


@pytest.mark.parametrize("variante", list(MODELOS_LTSF))
def test_recupera_relacion_lineal(variante):
    W, z = ventanas()
    m = MODELOS_LTSF[variante](longitud=L, kernel=5, alpha=1e-6).fit(W.iloc[:1200], z[:1200])
    error = np.sqrt(np.mean((m.predict(W.iloc[1200:]) - z[1200:]) ** 2))
    assert error < 0.02


def test_nlinear_invariante_al_nivel():
    W, z = ventanas()
    m = NLinear(longitud=L, alpha=1e-3).fit(W, z)
    desplazada = W * 3.0   # multiplicar el caudal desplaza el logaritmo de toda la ventana
    assert np.allclose(m.predict(W), m.predict(desplazada))


def test_kernel_par_se_ajusta_a_impar():
    W, z = ventanas(n=300)
    pred_par = MODELOS_LTSF["dlinear"](longitud=L, kernel=6).fit(W, z).predict(W)
    pred_impar = MODELOS_LTSF["dlinear"](longitud=L, kernel=7).fit(W, z).predict(W)
    assert np.allclose(pred_par, pred_impar)
