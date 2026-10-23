"""Pruebas de las funciones estadísticas frente a valores de referencia conocidos."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from src.stats_tests import (bayes_signed_rank, benjamini_hochberg, bootstrap_bloques, delong, delta_cliff,
                             diebold_mariano, friedman, holm, indices_estacionario, nadeau_bengio, nemenyi)
from src.evaluation import ece, shapley_permutaciones
from src.spaces import Param, rejilla


def test_holm_y_bh_coinciden_con_statsmodels():
    p = np.array([0.001, 0.02, 0.04, 0.3, 0.012])
    np.testing.assert_allclose(holm(p), multipletests(p, method="holm")[1])
    np.testing.assert_allclose(benjamini_hochberg(p), multipletests(p, method="fdr_bh")[1])


def test_diferencia_critica_de_nemenyi_tabla_demsar():
    # Demšar (2006): q_0.05 = 3.031 para k = 8; con N = 12, CD = 3.031·√(8·9/72) = 3.031
    r = pd.Series(np.arange(1, 9, dtype=float), index=list("abcdefgh"))
    assert nemenyi(r, 12)["cd"] == pytest.approx(3.031, abs=2e-3)


def test_friedman_detecta_un_modelo_dominante():
    rng = np.random.default_rng(0)
    t = pd.DataFrame(rng.normal(0, 0.01, (12, 4)), columns=list("abcd"))
    t["a"] += 0.1
    f = friedman(t)
    assert f["p"] < 0.001 and f["rangos"].index[0] == "a"


def test_delong_reproduce_auc():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    s1, s2 = y + rng.normal(0, 1, 400), y + rng.normal(0, 2, 400)
    r = delong(y, s1, s2)
    assert r["auc_a"] == pytest.approx(roc_auc_score(y, s1))
    assert r["auc_b"] == pytest.approx(roc_auc_score(y, s2))
    assert r["p_unilateral"] < 0.05


def test_nadeau_bengio_es_mas_conservadora_que_t_pareada():
    from scipy.stats import ttest_rel
    rng = np.random.default_rng(2)
    a = rng.normal(0.9, 0.02, 12)
    b = a - 0.005 + rng.normal(0, 0.01, 12)
    nb = nadeau_bengio(a, b, 8000, 730)
    assert nb["p"] > ttest_rel(a, b, alternative="greater").pvalue


def test_cliff_extremos():
    assert delta_cliff([2, 3, 4], [0, 1]) == 1.0
    assert delta_cliff([0, 1], [2, 3]) == -1.0


def test_bayes_signed_rank_reconoce_superioridad_y_equivalencia():
    rng = np.random.default_rng(3)
    a = rng.normal(0.9, 0.01, 12)
    assert bayes_signed_rank(a + 0.05, a, rope=0.01)["p_A_mejor"] > 0.95
    assert bayes_signed_rank(a + 0.001, a, rope=0.01)["p_equivalentes"] > 0.9


def test_bootstrap_estacionario_conserva_longitud_y_cubre_media():
    rng = np.random.default_rng(4)
    idx = indices_estacionario(500, 10, rng)
    assert len(idx) == 500 and idx.min() >= 0 and idx.max() < 500
    x = np.cumsum(rng.normal(size=600)) * 0.05 + rng.normal(size=600)
    ic = bootstrap_bloques(np.mean, x, media_bloque=20, n_boot=500)
    assert ic["ic_inf"] < x.mean() < ic["ic_sup"]


def test_diebold_mariano_signo():
    rng = np.random.default_rng(5)
    e1, e2 = rng.normal(0, 1, 1000), rng.normal(0, 1.5, 1000)
    dm = diebold_mariano(e1, e2, h=1)
    assert dm["dm"] < 0 and dm["p"] < 0.01


def test_ece_cero_si_calibrado_perfecto():
    p = np.repeat([0.1, 0.5, 0.9], 1000)
    y = np.concatenate([np.r_[np.ones(100), np.zeros(900)], np.r_[np.ones(500), np.zeros(500)],
                        np.r_[np.ones(900), np.zeros(100)]])
    assert ece(y, p) == pytest.approx(0.0, abs=1e-12)


def test_shapley_eficiencia_y_modelo_aditivo():
    w = np.array([1.0, -2.0, 0.5])
    f = lambda X: X @ w
    fondo = np.zeros((10, 3))
    x = np.array([1.0, 1.0, 2.0])
    phi, base = shapley_permutaciones(f, x, fondo, n_perm=10)
    np.testing.assert_allclose(phi, w * x, atol=1e-10)      # Shapley exacto en modelos aditivos
    assert base + phi.sum() == pytest.approx(f(x[None])[0])


def test_rejilla_respeta_presupuesto():
    esp = [Param("a", "float", 1e-3, 1e2, log=True), Param("b", "cat", opciones=("x", "y")),
           Param("c", "int", 2, 16)]
    g = rejilla(esp, 16)
    assert 1 < len(g) <= 16 and all(set(r) == {"a", "b", "c"} for r in g)
