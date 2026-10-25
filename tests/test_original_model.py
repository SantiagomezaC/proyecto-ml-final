"""Pruebas unitarias del modelo propuesto (BLGB).

Cubren cuatro propiedades básicas: aprendizaje en un problema
sintético con solución conocida, reproducibilidad con la misma semilla, manejo de
entradas inválidas y compatibilidad con ``sklearn.base.clone``. Se añaden pruebas de las
propiedades del algoritmo: gradientes analíticos, descenso monótono y cotas de σ.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from src.original_model import (BandLikelihoodBoosting, construir_arbol, gradientes,
                                perdida_hibrida, probabilidad_banda)


def datos_sinteticos(n=3000, seed=0):
    """Proceso conocido: log(Q_fut/Q_ref) = 0.6·x1 − 0.4·x2 + ruido heterocedástico."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    qref = np.exp(rng.normal(np.log(1000), 0.3, n))
    sigma = 0.1 + 0.15 * (X[:, 2] > 0)
    z = 0.6 * X[:, 0] - 0.4 * X[:, 1] + sigma * rng.normal(size=n)
    qfut = qref * np.exp(z)
    L, U = 900.0, 1300.0
    y = ((qfut >= L) & (qfut <= U)).astype(int)
    Xdf = pd.DataFrame(np.column_stack([X, qref]), columns=["x1", "x2", "x3", "x4", "qref"])
    return Xdf, y, qfut, (L, U)


def modelo(L, U, **kw):
    base = dict(banda_inf=L, banda_sup=U, col_referencia="qref", n_estimadores=150,
                tasa=0.2, profundidad=3, paciencia=20, random_state=7)
    base.update(kw)
    return BandLikelihoodBoosting(**base)


def test_aprende_en_problema_sintetico():
    X, y, q, (L, U) = datos_sinteticos()
    m = modelo(L, U).fit(X[:2000], y[:2000], q_futuro=q[:2000])
    p = m.predict_proba(X[2000:])[:, 1]
    assert roc_auc_score(y[2000:], p) > 0.9
    # La media latente recupera la señal verdadera
    dist = m.predict_distribucion(X[2000:])
    verdadero = 0.6 * X.x1[2000:] - 0.4 * X.x2[2000:]
    assert np.corrcoef(dist["mu"], verdadero)[0, 1] > 0.9


def test_recupera_heterocedasticidad():
    X, y, q, (L, U) = datos_sinteticos(n=4000)
    m = modelo(L, U, n_estimadores=200).fit(X, y, q_futuro=q)
    sig = m.predict_distribucion(X)["sigma"]
    alto, bajo = sig[X.x3 > 0].mean(), sig[X.x3 <= 0].mean()
    assert alto > 1.5 * bajo


def test_reproducibilidad_misma_semilla():
    X, y, q, (L, U) = datos_sinteticos(n=1500)
    p1 = modelo(L, U).fit(X, y, q_futuro=q).predict_proba(X)
    p2 = modelo(L, U).fit(X, y, q_futuro=q).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)


def test_semilla_distinta_cambia_submuestreo():
    X, y, q, (L, U) = datos_sinteticos(n=1500)
    p1 = modelo(L, U, random_state=1).fit(X, y, q_futuro=q).predict_proba(X)
    p2 = modelo(L, U, random_state=2).fit(X, y, q_futuro=q).predict_proba(X)
    assert not np.allclose(p1, p2)


@pytest.mark.parametrize("kw", [dict(banda_inf=10, banda_sup=5), dict(lambda_hibrido=-1),
                                dict(tasa=0), dict(n_bins=300), dict(profundidad=0)])
def test_hiperparametros_invalidos(kw):
    X, y, q, (L, U) = datos_sinteticos(n=300)
    with pytest.raises(ValueError):
        modelo(L, U, **kw).fit(X, y, q_futuro=q)


def test_entradas_invalidas():
    X, y, q, (L, U) = datos_sinteticos(n=300)
    m = modelo(L, U)
    with pytest.raises(ValueError):          # una sola clase
        m.fit(X, np.zeros(len(y)), q_futuro=q)
    with pytest.raises(ValueError):          # caudal futuro no positivo
        m.fit(X, y, q_futuro=-q)
    Xmal = X.copy()
    Xmal.loc[0, "qref"] = np.nan
    with pytest.raises(ValueError):          # referencia con faltantes
        m.fit(Xmal, y, q_futuro=q)
    m.fit(X, y, q_futuro=q)
    with pytest.raises(ValueError):          # número de columnas distinto en predicción
        m.predict_proba(X.iloc[:, :3])


def test_predecir_sin_ajustar():
    from sklearn.exceptions import NotFittedError
    X, y, q, (L, U) = datos_sinteticos(n=100)
    with pytest.raises(NotFittedError):
        modelo(L, U).predict_proba(X)


def test_compatibilidad_clone_y_cross_val():
    X, y, q, (L, U) = datos_sinteticos(n=1200)
    m = modelo(L, U, lambda_hibrido=0.0)
    c = clone(m)
    assert c.get_params() == m.get_params()
    # Sin q_futuro el estimador funciona como un clasificador estándar de scikit-learn
    s = cross_val_score(c, X, y, cv=3, scoring="roc_auc")
    assert np.all(s > 0.6)


def test_faltantes_en_predictores_permitidos():
    X, y, q, (L, U) = datos_sinteticos(n=1000)
    X.iloc[::7, 1] = np.nan
    p = modelo(L, U).fit(X, y, q_futuro=q).predict_proba(X)[:, 1]
    assert np.all(np.isfinite(p)) and np.all((p > 0) & (p < 1))


def test_gradientes_analiticos_coinciden_con_diferencias_finitas():
    rng = np.random.default_rng(3)
    n = 50
    mu, s = rng.normal(0, 0.3, n), rng.normal(-1.5, 0.3, n)
    l, u = rng.normal(-0.3, 0.1, n), rng.normal(0.3, 0.1, n)
    y = rng.integers(0, 2, n).astype(float)
    z = rng.normal(0, 0.3, n)
    lam = 0.7
    g_mu, _, g_s, _ = gradientes(mu, s, l, u, y, z, lam)
    h = 1e-6
    for i in range(5):
        e = np.zeros(n)
        e[i] = h
        num_mu = (perdida_hibrida(mu + e, s, l, u, y, z, lam)
                  - perdida_hibrida(mu - e, s, l, u, y, z, lam)) / (2 * h) * n
        num_s = (perdida_hibrida(mu, s + e, l, u, y, z, lam)
                 - perdida_hibrida(mu, s - e, l, u, y, z, lam)) / (2 * h) * n
        assert num_mu == pytest.approx(g_mu[i], rel=1e-4, abs=1e-6)
        assert num_s == pytest.approx(g_s[i], rel=1e-4, abs=1e-6)


def test_descenso_monotono_de_la_perdida():
    X, y, q, (L, U) = datos_sinteticos(n=2000)
    m = modelo(L, U, paciencia=0, n_estimadores=60).fit(X, y, q_futuro=q)
    perd = [h["perdida_train"] for h in m.historial_]
    assert all(b <= a + 1e-12 for a, b in zip(perd, perd[1:]))


def test_sigma_respeta_cotas():
    X, y, q, (L, U) = datos_sinteticos(n=1000)
    m = modelo(L, U, sigma_min=0.05, sigma_max=0.5).fit(X, y, q_futuro=q)
    sig = m.predict_distribucion(X)["sigma"]
    assert sig.min() >= 0.05 - 1e-12 and sig.max() <= 0.5 + 1e-12


def test_probabilidad_banda_suma_con_complemento():
    p, *_ = probabilidad_banda(np.zeros(3), np.zeros(3), np.array([-1, -2, -np.inf]),
                               np.array([1, 2, np.inf]))
    np.testing.assert_allclose(p, [0.6826895, 0.9544997, 1 - 1e-7], rtol=1e-6)


def test_arbol_recupera_particion_simple():
    rng = np.random.default_rng(0)
    Xb = rng.integers(0, 16, size=(500, 3)).astype(np.uint8)
    objetivo = np.where(Xb[:, 1] <= 7, -1.0, 2.0)
    g, h = -objetivo, np.ones(500)            # paso de Newton = objetivo
    a = construir_arbol(Xb, g, h, 16, 1, 1.0, 0.0, 0.0, np.arange(3))
    assert a.feat[0] == 1 and a.umbral[0] == 7
    np.testing.assert_allclose(a.predecir(Xb), objetivo, atol=0.02)
