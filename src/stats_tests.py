"""Pruebas estadísticas para comparar modelos.

Clasificación: Friedman, Nemenyi y diagrama de diferencia crítica, Wilcoxon con Holm,
t corregida de Nadeau-Bengio, prueba bayesiana de rangos con signo (ROPE), DeLong y
McNemar. Regresión con dependencia temporal: Diebold-Mariano con corrección HLN,
Giacomini-White, Clark-West, Model Confidence Set y SPA. Comunes: bootstrap por bloques
(estacionario y BCa por bloques), tamaños del efecto y correcciones por comparaciones
múltiples.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats as st

from .config import SEED


# ---------------------------------------------------------------------------
# Correcciones por comparaciones múltiples y tamaños del efecto
# ---------------------------------------------------------------------------
def holm(p: np.ndarray) -> np.ndarray:
    """Valores p ajustados por Holm-Bonferroni (control del error familiar)."""
    p = np.asarray(p, float)
    m = len(p)
    orden = np.argsort(p)
    ajust = np.empty(m)
    acum = 0.0
    for r, i in enumerate(orden):
        acum = max(acum, (m - r) * p[i])
        ajust[i] = min(1.0, acum)
    return ajust


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Valores p ajustados por Benjamini-Hochberg (control de la tasa de falsos descubrimientos)."""
    p = np.asarray(p, float)
    m = len(p)
    orden = np.argsort(p)[::-1]
    ajust = np.empty(m)
    previo = 1.0
    for r, i in enumerate(orden):
        previo = min(previo, p[i] * m / (m - r))
        ajust[i] = previo
    return ajust


def delta_cliff(a, b) -> float:
    """δ de Cliff: P(A > B) − P(A < B); |δ| < 0.147 despreciable, < 0.33 pequeño, < 0.474 mediano."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    dif = a[:, None] - b[None, :]
    return float(((dif > 0).sum() - (dif < 0).sum()) / dif.size)


def magnitud_cliff(d: float) -> str:
    d = abs(d)
    return ("despreciable" if d < 0.147 else "pequeño" if d < 0.33
            else "mediano" if d < 0.474 else "grande")


def d_cohen_pareado(dif) -> float:
    """d de Cohen para diferencias pareadas: media / desviación estándar."""
    dif = np.asarray(dif, float)
    return float(dif.mean() / dif.std(ddof=1)) if dif.std(ddof=1) > 0 else np.inf


# ---------------------------------------------------------------------------
# Procedimientos globales sobre pliegues: Friedman, Nemenyi, Wilcoxon-Holm
# ---------------------------------------------------------------------------
def friedman(tabla: pd.DataFrame) -> dict:
    """Prueba de Friedman sobre una tabla pliegues × modelos (mayor es mejor).

    Devuelve el estadístico χ²_F, la corrección F de Iman-Davenport y los rangos promedio
    (rango 1 = mejor).
    """
    rangos = tabla.rank(axis=1, ascending=False)
    n, k = tabla.shape
    chi2, p = st.friedmanchisquare(*[tabla[c].values for c in tabla.columns])
    ff = (n - 1) * chi2 / (n * (k - 1) - chi2) if n * (k - 1) != chi2 else np.inf
    p_ff = 1 - st.f.cdf(ff, k - 1, (k - 1) * (n - 1))
    return {"chi2": float(chi2), "gl": k - 1, "p": float(p), "F_iman_davenport": float(ff),
            "p_iman_davenport": float(p_ff), "rangos": rangos.mean().sort_values(),
            "n_bloques": n, "k_modelos": k}


def nemenyi(rangos_promedio: pd.Series, n_bloques: int, alpha: float = 0.05) -> dict:
    """Diferencia crítica de Nemenyi: CD = q_α √(k(k+1)/(6N)) (Demšar, 2006)."""
    k = len(rangos_promedio)
    q = st.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n_bloques))
    pares = []
    for a, b in combinations(rangos_promedio.index, 2):
        dif = abs(rangos_promedio[a] - rangos_promedio[b])
        z = dif / np.sqrt(k * (k + 1) / (6 * n_bloques))
        p = 1 - st.studentized_range.cdf(z * np.sqrt(2), k, np.inf)
        pares.append({"modelo_a": a, "modelo_b": b, "dif_rangos": dif, "p_nemenyi": p,
                      "significativo": dif > cd})
    return {"cd": float(cd), "q_alpha": float(q), "pares": pd.DataFrame(pares)}


def wilcoxon_holm(tabla: pd.DataFrame, referencia: str, alternativa: str = "greater") -> pd.DataFrame:
    """Wilcoxon de rangos con signo de ``referencia`` contra cada modelo, con ajuste de Holm."""
    filas = []
    for c in tabla.columns:
        if c == referencia:
            continue
        dif = tabla[referencia] - tabla[c]
        res = st.wilcoxon(tabla[referencia], tabla[c], alternative=alternativa, zero_method="zsplit")
        filas.append({"modelo": c, "dif_media": dif.mean(), "W": res.statistic, "p": res.pvalue,
                      "delta_cliff": delta_cliff(tabla[referencia], tabla[c]),
                      "victorias": int((dif > 0).sum()), "n": len(dif)})
    out = pd.DataFrame(filas)
    out["p_holm"] = holm(out.p.values)
    return out.sort_values("p")


def nadeau_bengio(a, b, n_train: float, n_test: float, alternativa: str = "greater") -> dict:
    """Prueba t corregida de Nadeau y Bengio (2003) para pliegues dependientes.

    La varianza de la diferencia media se infla por (1/k + n_test/n_train), que corrige la
    correlación inducida por el solapamiento de los conjuntos de entrenamiento.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    k = len(d)
    var = d.var(ddof=1)
    se = np.sqrt((1 / k + n_test / n_train) * var)
    t = d.mean() / se if se > 0 else np.inf
    if alternativa == "greater":
        p = 1 - st.t.cdf(t, k - 1)
    elif alternativa == "less":
        p = st.t.cdf(t, k - 1)
    else:
        p = 2 * (1 - st.t.cdf(abs(t), k - 1))
    return {"t": float(t), "gl": k - 1, "p": float(p), "dif_media": float(d.mean()),
            "se_corregido": float(se), "d_cohen": d_cohen_pareado(d)}


def bayes_signed_rank(a, b, rope: float = 0.01, n_muestras: int = 50000,
                      seed: int = SEED) -> dict:
    """Prueba bayesiana de rangos con signo con región de equivalencia práctica.

    Implementa el modelo de Benavoli et al. (2017): proceso de Dirichlet con pseudo-
    observación en 0. Devuelve P(A ≫ B), P(equivalentes) y P(B ≫ A), donde la
    equivalencia es |diferencia| < ``rope``.
    """
    rng = np.random.default_rng(seed)
    d = np.concatenate([[0.0], np.asarray(a, float) - np.asarray(b, float)])
    n = len(d)
    w = rng.dirichlet(np.ones(n), size=n_muestras)          # (S, n)
    suma = d[:, None] + d[None, :]
    izq = (suma < -2 * rope).astype(float)
    der = (suma > 2 * rope).astype(float)
    centro = 1 - izq - der
    # Probabilidad posterior de cada región: Σ_i Σ_j w_i w_j 1{(d_i + d_j)/2 ∈ región}
    p_izq = np.einsum("si,ij,sj->s", w, izq, w)
    p_der = np.einsum("si,ij,sj->s", w, der, w)
    p_cen = np.einsum("si,ij,sj->s", w, centro, w)
    gana = np.argmax(np.column_stack([p_izq, p_cen, p_der]), axis=1)
    return {"p_B_mejor": float((gana == 0).mean()), "p_equivalentes": float((gana == 1).mean()),
            "p_A_mejor": float((gana == 2).mean()), "rope": rope}


# ---------------------------------------------------------------------------
# Pruebas sobre el conjunto de prueba reservado: DeLong y McNemar
# ---------------------------------------------------------------------------
def _componentes_delong(pos, neg):
    m, n = len(pos), len(neg)
    todos = np.concatenate([pos, neg])
    r_todos = st.rankdata(todos)
    r_pos, r_neg = st.rankdata(pos), st.rankdata(neg)
    v10 = (r_todos[:m] - r_pos) / n
    v01 = 1 - (r_todos[m:] - r_neg) / m
    return v10, v01


def delong(y, s1, s2) -> dict:
    """Prueba de DeLong para dos AUC correlacionadas (DeLong et al., 1988)."""
    y = np.asarray(y).astype(bool)
    v10_1, v01_1 = _componentes_delong(s1[y], s1[~y])
    v10_2, v01_2 = _componentes_delong(s2[y], s2[~y])
    auc1, auc2 = v10_1.mean(), v10_2.mean()
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    S = s10 / y.sum() + s01 / (~y).sum()
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (auc1 - auc2) / np.sqrt(var)
    return {"auc_a": float(auc1), "auc_b": float(auc2), "dif": float(auc1 - auc2), "z": float(z),
            "p": float(2 * (1 - st.norm.cdf(abs(z)))), "p_unilateral": float(1 - st.norm.cdf(z))}


def mcnemar(y, pred_a, pred_b) -> dict:
    """McNemar con corrección de continuidad sobre aciertos discordantes."""
    ca, cb = pred_a == y, pred_b == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 0 else 0.0
    return {"a_acierta_b_falla": b, "b_acierta_a_falla": c, "chi2": float(chi2),
            "p": float(1 - st.chi2.cdf(chi2, 1))}


# ---------------------------------------------------------------------------
# Bootstrap por bloques
# ---------------------------------------------------------------------------
def longitud_bloque(x) -> int:
    """Longitud media de bloque de Politis y White (2004), vía ``arch``."""
    from arch.bootstrap import optimal_block_length
    return max(1, int(np.ceil(optimal_block_length(np.asarray(x, float))["stationary"].iloc[0])))


def indices_estacionario(n: int, media_bloque: float, rng) -> np.ndarray:
    """Índices del bootstrap estacionario de Politis y Romano (1994)."""
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    nuevos = rng.random(n) < 1 / media_bloque
    inicios = rng.integers(n, size=n)
    for t in range(1, n):
        idx[t] = inicios[t] if nuevos[t] else (idx[t - 1] + 1) % n
    return idx


def bootstrap_bloques(estadistico, *arreglos, media_bloque: float | None = None,
                      n_boot: int = 2000, alpha: float = 0.05, seed: int = SEED,
                      metodo: str = "bca") -> dict:
    """Intervalo de confianza por bootstrap estacionario, percentil o BCa por bloques.

    Parameters
    ----------
    estadistico : callable
        ``estadistico(*arreglos)`` → float.
    arreglos : arrays de igual longitud, en orden temporal.
    media_bloque : float, optional
        Longitud media de bloque; por defecto la de Politis-White sobre el primer arreglo.
    metodo : {"bca", "percentil"}
        En BCa, el sesgo se corrige con la proporción de réplicas bajo el estimador y la
        aceleración se estima con un jackknife que elimina bloques completos.
    """
    arreglos = [np.asarray(a) for a in arreglos]
    n = len(arreglos[0])
    rng = np.random.default_rng(seed)
    if media_bloque is None:
        media_bloque = longitud_bloque(arreglos[0].astype(float))
    theta = estadistico(*arreglos)
    reps = []
    for _ in range(n_boot):
        i = indices_estacionario(n, media_bloque, rng)
        try:
            reps.append(estadistico(*[a[i] for a in arreglos]))
        except ValueError:
            continue
    reps = np.asarray(reps, float)
    reps = reps[np.isfinite(reps)]
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if metodo == "bca":
        z0 = st.norm.ppf(np.clip((reps < theta).mean(), 1e-6, 1 - 1e-6))
        L = int(np.ceil(media_bloque))
        bloques = np.array_split(np.arange(n), max(10, n // L))
        jack = []
        for bl in bloques:
            mask = np.ones(n, bool)
            mask[bl] = False
            try:
                jack.append(estadistico(*[a[mask] for a in arreglos]))
            except ValueError:
                continue
        jack = np.asarray(jack)
        dj = jack.mean() - jack
        acel = (dj ** 3).sum() / (6 * ((dj ** 2).sum()) ** 1.5) if (dj ** 2).sum() > 0 else 0.0
        za = st.norm.ppf([alpha / 2, 1 - alpha / 2])
        adj = st.norm.cdf(z0 + (z0 + za) / (1 - acel * (z0 + za)))
        lo, hi = np.percentile(reps, 100 * adj)
    return {"estimacion": float(theta), "ic_inf": float(lo), "ic_sup": float(hi),
            "media_bloque": float(media_bloque), "n_boot": int(len(reps)), "metodo": metodo}


# ---------------------------------------------------------------------------
# Regresión con dependencia temporal
# ---------------------------------------------------------------------------
def _var_hac(d, h: int):
    """Varianza de largo plazo de la media con núcleo rectangular hasta h − 1 rezagos."""
    d = d - d.mean()
    n = len(d)
    gamma0 = np.dot(d, d) / n
    v = gamma0
    for k in range(1, h):
        v += 2 * np.dot(d[k:], d[:-k]) / n
    return v


def diebold_mariano(e1, e2, h: int = 15, potencia: int = 2) -> dict:
    """Diebold-Mariano con la corrección de Harvey, Leybourne y Newbold (1997).

    H0: igual pérdida esperada. Un estadístico negativo favorece al modelo 1.
    """
    d = np.abs(e1) ** potencia - np.abs(e2) ** potencia
    n = len(d)
    v = _var_hac(d, h)
    dm = d.mean() / np.sqrt(v / n)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * corr
    return {"dm": float(stat), "p": float(2 * st.t.sf(abs(stat), n - 1)),
            "p_unilateral_1_mejor": float(st.t.cdf(stat, n - 1)), "dif_perdida": float(d.mean())}


def giacomini_white(e1, e2, h: int = 15, potencia: int = 2) -> dict:
    """Prueba de habilidad predictiva condicional de Giacomini y White (2006).

    Instrumentos: constante y diferencia de pérdida rezagada h periodos. Estadístico de
    Wald n·Z̄'Ω⁻¹Z̄ ~ χ²_q bajo H0.
    """
    d = np.abs(e1) ** potencia - np.abs(e2) ** potencia
    dd = d[h:]
    inst = np.column_stack([np.ones(len(dd)), d[:-h]])
    Z = inst * dd[:, None]
    n, q = Z.shape
    Zc = Z - Z.mean(0)
    omega = Zc.T @ Zc / n
    for k in range(1, h):
        G = Zc[k:].T @ Zc[:-k] / n
        omega += (1 - k / h) * (G + G.T)
    zbar = Z.mean(0)
    w = float(n * zbar @ np.linalg.solve(omega, zbar))
    return {"estadistico": w, "gl": q, "p": float(st.chi2.sf(w, q)), "dif_perdida": float(d.mean())}


def clark_west(y, pred_restringido, pred_general, h: int = 15) -> dict:
    """Clark y West (2007) para modelos anidados: ajusta el sesgo del MSPE del modelo grande.

    H1: el modelo general predice mejor. f_t = e_r² − (e_g² − (ŷ_r − ŷ_g)²).
    """
    er, eg = y - pred_restringido, y - pred_general
    f = er ** 2 - (eg ** 2 - (pred_restringido - pred_general) ** 2)
    n = len(f)
    t = f.mean() / np.sqrt(_var_hac(f, h) / n)
    return {"cw": float(t), "p": float(st.norm.sf(t))}


def model_confidence_set(perdidas: pd.DataFrame, alpha: float = 0.10, bloque: int | None = None,
                         reps: int = 2000, seed: int = SEED) -> pd.DataFrame:
    """Model Confidence Set de Hansen, Lunde y Nason (2011), vía ``arch``.

    Parameters
    ----------
    perdidas : DataFrame
        Pérdida por observación (filas, en orden temporal) y modelo (columnas).
    """
    from arch.bootstrap import MCS
    bloque = bloque or longitud_bloque(perdidas.mean(axis=1).values)
    mcs = MCS(perdidas, size=alpha, reps=reps, block_size=bloque, method="R",
              bootstrap="stationary", seed=seed)
    mcs.compute()
    pv = mcs.pvalues.rename(columns={"Pvalue": "p_mcs"})
    pv["en_conjunto"] = pv.p_mcs >= alpha
    pv["perdida_media"] = perdidas.mean().reindex(pv.index)
    pv["rango_perdida"] = perdidas.rank(axis=1).mean().reindex(pv.index)
    pv["bloque_bootstrap"] = bloque
    return pv.sort_values("perdida_media")


def spa(perdida_benchmark, perdidas_modelos: pd.DataFrame, bloque: int | None = None,
        reps: int = 2000, seed: int = SEED) -> dict:
    """Prueba de habilidad predictiva superior de Hansen (2005) frente a un benchmark."""
    from arch.bootstrap import SPA
    bloque = bloque or longitud_bloque(np.asarray(perdida_benchmark, float))
    s = SPA(np.asarray(perdida_benchmark), perdidas_modelos, block_size=bloque, reps=reps,
            bootstrap="stationary", seed=seed)
    s.compute()
    return {k: float(v) for k, v in s.pvalues.items()} | {"bloque": bloque}
