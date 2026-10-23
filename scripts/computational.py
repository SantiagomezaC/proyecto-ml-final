"""Optimización computacional (Entregable 2, Sección 4).

1. Escalamiento empírico del tiempo de entrenamiento e inferencia al variar n y p, y
   estimación del exponente por regresión log-log, para contrastarlo con la complejidad
   teórica de cada modelo.
2. Variantes eficientes: KNN (fuerza bruta, KD-Tree, Ball Tree, FAISS exacto y HNSW),
   Ridge/Lasso (solver por defecto frente a SAGA), Naive Bayes (``partial_fit`` por lotes),
   XGBoost (``exact`` frente a ``hist`` y parada temprana) y SVM (núcleo RBF frente a
   LinearSVC, SGD y aproximación de Nyström).
3. Paralelización (``n_jobs`` y joblib) y perfilamiento de CPU y memoria.

Uso:  python -m scripts.computational
"""
from __future__ import annotations

import cProfile
import io
import json
import pstats
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Lasso, LogisticRegression, Ridge, SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import RESULTADOS, SEED, fijar_semillas
from src.experiment import cargar_datos
from src.original_model import BandLikelihoodBoosting

SAL = RESULTADOS / "computacional"


def cronometrar(f, rep: int = 3) -> float:
    """Mediana de ``rep`` repeticiones del tiempo de pared."""
    t = []
    for _ in range(rep):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


def memoria_pico(f) -> float:
    """Memoria pico asignada por Python durante ``f`` (MB), con tracemalloc."""
    tracemalloc.start()
    f()
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return pico / 2 ** 20


def preparar():
    d = cargar_datos()
    imp = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    X = imp.fit_transform(d.X)
    Xt = imp.transform(d.X_test)
    return d, X, d.y_clf.values, Xt, d.y_clf_test.values


def modelos_escalamiento(prev):
    return {
        "knn": lambda: KNeighborsClassifier(n_neighbors=30, algorithm="brute", n_jobs=1),
        "naive_bayes": lambda: GaussianNB(),
        "logistica": lambda: LogisticRegression(C=1.0, solver="liblinear", max_iter=5000),
        "arbol": lambda: DecisionTreeClassifier(max_depth=8, random_state=SEED),
        "random_forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=10,
                                                        max_samples=0.6, n_jobs=1, random_state=SEED),
        "xgboost": lambda: XGBClassifier(n_estimators=200, max_depth=4, tree_method="hist",
                                         n_jobs=1, random_state=SEED),
        "svm": lambda: SVC(C=1.0, gamma=0.01, kernel="rbf"),
    }


TEORIA = {
    "knn": ("O(1) (almacena)", "O(n·p) por consulta"),
    "naive_bayes": ("O(n·p)", "O(p) por consulta"),
    "logistica": ("O(n·p) por iteración", "O(p)"),
    "arbol": ("O(p·n log n)", "O(profundidad)"),
    "random_forest": ("O(T·p'·n' log n')", "O(T·profundidad)"),
    "xgboost": ("O(T·(n·p + 2^d·p·B)) con hist", "O(T·d)"),
    "svm": ("O(n²·p) a O(n³)", "O(n_sv·p)"),
    "original": ("O(T·(n·k + 2^d·k·B)) ×2 ensambles", "O(T·d) + O(p log B)"),
}


def escalamiento(X, y, Xt, prev):
    filas = []
    rng = np.random.default_rng(SEED)
    for n in (1000, 2000, 4000, 8000, 12000):
        idx = np.sort(rng.choice(len(X), min(n, len(X)), replace=False))
        for nombre, fab in modelos_escalamiento(prev).items():
            m = fab()
            t_fit = cronometrar(lambda: m.fit(X[idx], y[idx]), rep=2)
            t_inf = cronometrar(lambda: m.predict(Xt[:2000]), rep=2)
            filas.append({"modelo": nombre, "n": len(idx), "p": X.shape[1], "t_fit": t_fit,
                          "t_inf_2000": t_inf, "variable": "n"})
            print("n", nombre, len(idx), round(t_fit, 3), flush=True)
        # Modelo original (usa su propia discretización; se ajusta sin parada temprana)
        Xo = pd.DataFrame(X[idx]).assign(ref=1.0)
        m = BandLikelihoodBoosting(banda_inf=0.5, banda_sup=2.0, col_referencia="ref",
                                   n_estimadores=40, paciencia=0, random_state=SEED)
        Xo.columns = Xo.columns.astype(str)
        qf = np.exp(np.random.default_rng(1).normal(0, 0.5, len(idx)))
        yy = ((qf >= 0.5) & (qf <= 2.0)).astype(int)
        t_fit = cronometrar(lambda: m.fit(Xo, yy, q_futuro=qf), rep=1)
        Xot = pd.DataFrame(Xt[:2000]).assign(ref=1.0)
        Xot.columns = Xot.columns.astype(str)
        t_inf = cronometrar(lambda: m.predict_proba(Xot), rep=2)
        filas.append({"modelo": "original", "n": len(idx), "p": X.shape[1], "t_fit": t_fit,
                      "t_inf_2000": t_inf, "variable": "n"})
    for p in sorted({5, 10, 20, X.shape[1]}):
        cols = np.arange(p)
        for nombre, fab in modelos_escalamiento(prev).items():
            m = fab()
            t_fit = cronometrar(lambda: m.fit(X[:, cols], y), rep=2)
            filas.append({"modelo": nombre, "n": len(X), "p": p, "t_fit": t_fit, "variable": "p"})
            print("p", nombre, p, round(t_fit, 3), flush=True)
    df = pd.DataFrame(filas)
    exps = []
    for (mod, var), g in df.groupby(["modelo", "variable"]):
        x = np.log(g[var].values)
        b = np.polyfit(x, np.log(g.t_fit.values), 1)[0]
        exps.append({"modelo": mod, "variable": var, "exponente_empirico": b,
                     "teoria_entrenamiento": TEORIA[mod][0], "teoria_inferencia": TEORIA[mod][1]})
    return df, pd.DataFrame(exps)


def knn_variantes(X, y, Xt, yt):
    import faiss
    filas = []
    for alg in ("brute", "kd_tree", "ball_tree"):
        m = KNeighborsClassifier(n_neighbors=30, algorithm=alg, n_jobs=1).fit(X, y)
        t = cronometrar(lambda: m.predict_proba(Xt), rep=2)
        mem = memoria_pico(lambda: KNeighborsClassifier(n_neighbors=30, algorithm=alg, n_jobs=1).fit(X, y).predict_proba(Xt[:500]))
        s = m.predict_proba(Xt)[:, 1]
        filas.append({"variante": f"sklearn-{alg}", "t_consulta_ms": t / len(Xt) * 1000, "memoria_pico_mb": mem,
                      "pr_auc": average_precision_score(yt, s), "recall_vecinos": 1.0})
    exactos = KNeighborsClassifier(n_neighbors=30, algorithm="brute").fit(X, y).kneighbors(
        Xt, return_distance=False)
    Xf, Xtf = X.astype("float32"), Xt.astype("float32")
    for nombre, indice in (("faiss-flat (exacto)", faiss.IndexFlatL2(X.shape[1])),
                           ("faiss-hnsw (aprox.)", faiss.IndexHNSWFlat(X.shape[1], 32))):
        indice.add(Xf)
        mem = indice.ntotal * (X.shape[1] * 4 + (32 * 2 * 4 if "hnsw" in nombre else 0)) / 2 ** 20
        t0 = time.perf_counter()
        _, I = indice.search(Xtf, 30)
        t = time.perf_counter() - t0
        s = y[I].mean(axis=1)
        rec = np.mean([len(set(a) & set(b)) / 30 for a, b in zip(I, exactos)])
        filas.append({"variante": nombre, "t_consulta_ms": t / len(Xt) * 1000, "memoria_pico_mb": mem,
                      "pr_auc": average_precision_score(yt, s), "recall_vecinos": rec})
    return pd.DataFrame(filas)


def lineales_variantes(X, q):
    filas = []
    for nombre, m in (("Ridge auto (Cholesky)", Ridge(alpha=1.0)),
                      ("Ridge SAGA", Ridge(alpha=1.0, solver="saga", max_iter=5000, random_state=SEED)),
                      ("Lasso coordenadas", Lasso(alpha=0.01, max_iter=20000)),
                      ("Lasso por SGD (penalización L1)", None)):
        if m is None:
            from sklearn.linear_model import SGDRegressor
            m = SGDRegressor(penalty="l1", alpha=0.01, max_iter=2000, random_state=SEED)
        qs = (q - q.mean()) / q.std()
        t = cronometrar(lambda: m.fit(X, qs), rep=3)
        rmse = float(np.sqrt(np.mean((m.predict(X) - qs) ** 2)))
        filas.append({"variante": nombre, "t_fit": t, "rmse_estandarizado_train": rmse})
    return pd.DataFrame(filas)


def nb_partial_fit(X, y, Xt, yt):
    filas = []
    m = GaussianNB()
    mem = memoria_pico(lambda: m.fit(X, y))
    t = cronometrar(lambda: GaussianNB().fit(X, y))
    s_full = m.predict_proba(Xt)[:, 1]
    filas.append({"variante": "fit completo", "t": t, "memoria_pico_mb": mem,
                  "pr_auc": average_precision_score(yt, s_full)})
    for lote in (500, 2000):
        def incremental():
            mm = GaussianNB()
            for i in range(0, len(X), lote):
                mm.partial_fit(X[i:i + lote], y[i:i + lote], classes=[0, 1])
            return mm
        mem = memoria_pico(incremental)
        t = cronometrar(incremental)
        mm = incremental()
        s = mm.predict_proba(Xt)[:, 1]
        filas.append({"variante": f"partial_fit lotes de {lote}", "t": t, "memoria_pico_mb": mem,
                      "pr_auc": average_precision_score(yt, s),
                      "max_dif_vs_completo": float(np.max(np.abs(s - s_full)))})
    return pd.DataFrame(filas)


def xgb_variantes(X, y, Xt, yt):
    filas = []
    corte = int(len(X) * 0.85)
    for metodo in ("exact", "approx", "hist"):
        m = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, tree_method=metodo,
                          n_jobs=1, random_state=SEED)
        t = cronometrar(lambda: m.fit(X, y), rep=2)
        t_inf = cronometrar(lambda: m.predict_proba(Xt), rep=2) / len(Xt) * 1000
        mem = memoria_pico(lambda: XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                                                 tree_method=metodo, n_jobs=1, random_state=SEED).fit(X, y))
        filas.append({"variante": f"tree_method={metodo}", "t_fit": t, "t_inf_ms_fila": t_inf,
                      "memoria_pico_mb": mem, "n_arboles": 400,
                      "pr_auc": average_precision_score(yt, m.predict_proba(Xt)[:, 1])})
    m = XGBClassifier(n_estimators=2000, max_depth=5, learning_rate=0.05, tree_method="hist",
                      n_jobs=1, random_state=SEED, early_stopping_rounds=50, eval_metric="aucpr")
    t = cronometrar(lambda: m.fit(X[:corte], y[:corte], eval_set=[(X[corte:], y[corte:])],
                                  verbose=False), rep=1)
    filas.append({"variante": "hist + parada temprana (validación cronológica 15 %)", "t_fit": t,
                  "n_arboles": int(m.best_iteration + 1),
                  "pr_auc": average_precision_score(yt, m.predict_proba(Xt)[:, 1])})
    for hilos in (1, 4, 10):
        m = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, tree_method="hist",
                          n_jobs=hilos, random_state=SEED)
        t = cronometrar(lambda: m.fit(X, y), rep=2)
        filas.append({"variante": f"hist n_jobs={hilos}", "t_fit": t, "n_arboles": 400,
                      "pr_auc": average_precision_score(yt, m.predict_proba(Xt)[:, 1])})
    return pd.DataFrame(filas)


def svm_variantes(X, y, Xt, yt):
    filas = []
    for nombre, m in (("SVC RBF (libsvm)", SVC(C=1.0, gamma=0.01)),
                      ("LinearSVC", LinearSVC(C=0.1, dual=False, max_iter=5000)),
                      ("SGDClassifier (hinge)", SGDClassifier(loss="hinge", alpha=1e-4,
                                                              max_iter=2000, random_state=SEED)),
                      ("Nyström (500) + LinearSVC",
                       make_pipeline(Nystroem(gamma=0.01, n_components=500, random_state=SEED),
                                     LinearSVC(C=0.1, dual=False, max_iter=5000)))):
        t = cronometrar(lambda: m.fit(X, y), rep=2)
        t_inf = cronometrar(lambda: m.decision_function(Xt), rep=2)
        from sklearn.base import clone
        mem = memoria_pico(lambda: clone(m).fit(X, y))
        filas.append({"variante": nombre, "t_fit": t, "t_inf_ms_fila": t_inf / len(Xt) * 1000, "memoria_pico_mb": mem,
                      "pr_auc": average_precision_score(yt, m.decision_function(Xt))})
    return pd.DataFrame(filas)


def paralelizacion(X, y):
    from joblib import Parallel, delayed
    filas = []
    for hilos in (1, 2, 4, 8, 10):
        m = RandomForestClassifier(n_estimators=200, max_depth=10, max_samples=0.6, n_jobs=hilos,
                                   random_state=SEED)
        filas.append({"escenario": "Random Forest n_jobs", "trabajadores": hilos,
                      "t": cronometrar(lambda: m.fit(X, y), rep=2)})
    tarea = lambda s: XGBClassifier(n_estimators=150, max_depth=4, tree_method="hist", n_jobs=1,
                                    random_state=s).fit(X, y)
    for hilos in (1, 2, 4, 8, 10):
        filas.append({"escenario": "joblib (loky): 10 ajustes de XGBoost", "trabajadores": hilos,
                      "t": cronometrar(lambda: Parallel(n_jobs=hilos, backend="loky")(
                          delayed(tarea)(s) for s in range(10)), rep=1)})
    df = pd.DataFrame(filas)
    df["aceleracion"] = df.groupby("escenario").t.transform(lambda s: s.iloc[0] / s)
    return df


def perfilamiento(d):
    """Perfil de CPU de una tarea elemental de la validación anidada (XGBoost, TPE, 1 pliegue)."""
    from src.experiment import ejecutar_tarea
    import tempfile
    pr = cProfile.Profile()
    with tempfile.TemporaryDirectory() as tmp:
        pr.enable()
        ejecutar_tarea(d, "clf", "xgboost", "smote", "bayes", d.pliegues[6], 6, Path(tmp))
        pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
    st = pstats.Stats(pr)
    filas = []
    for (archivo, linea, func), (cc, nc, tt, ct, _) in st.stats.items():
        filas.append({"funcion": f"{Path(archivo).name}:{func}", "llamadas": nc,
                      "t_propio": tt, "t_acumulado": ct})
    return pd.DataFrame(filas).sort_values("t_acumulado", ascending=False).head(30), s.getvalue()


def multifidelidad(d):
    """Successive Halving (recurso: número de árboles) frente a búsqueda aleatoria completa."""
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingRandomSearchCV, RandomizedSearchCV, ParameterSampler
    from scipy.stats import loguniform, randint, uniform
    from src.splits import cv_interna
    p = d.pliegues[-1]
    Xtr, ytr = d.X.iloc[p["train"]], d.y_clf.iloc[p["train"]]
    Xte, yte = d.X.iloc[p["test"]], d.y_clf.iloc[p["test"]]
    imp = make_pipeline(SimpleImputer(strategy="median"))
    Xtr_i, Xte_i = imp.fit_transform(Xtr), imp.transform(Xte)
    filas = []
    espacios = {
        "xgboost": (XGBClassifier(tree_method="hist", n_jobs=1, random_state=SEED),
                    {"max_depth": randint(2, 9), "learning_rate": loguniform(0.01, 0.3),
                     "subsample": uniform(0.5, 0.5), "colsample_bytree": uniform(0.4, 0.6),
                     "min_child_weight": loguniform(1, 30)}, 600),
        "random_forest": (RandomForestClassifier(max_samples=0.6, n_jobs=1, random_state=SEED),
                          {"max_depth": randint(3, 17), "min_samples_leaf": randint(1, 61),
                           "max_features": uniform(0.1, 0.5)}, 243),
    }
    for nombre, (est, dist, r_max) in espacios.items():
        for metodo in ("halving (factor 3)", "aleatoria (fidelidad completa)"):
            t0 = time.perf_counter()
            if metodo.startswith("halving"):
                bus = HalvingRandomSearchCV(est, dist, n_candidates=27, factor=3, resource="n_estimators",
                                            min_resources=r_max // 27, max_resources=r_max, cv=cv_interna(),
                                            scoring="average_precision", random_state=SEED, n_jobs=8)
            else:
                bus = RandomizedSearchCV(est.set_params(n_estimators=r_max), dist, n_iter=27, cv=cv_interna(),
                                         scoring="average_precision", random_state=SEED, n_jobs=8)
            bus.fit(Xtr_i, ytr)
            t = time.perf_counter() - t0
            arboles = (sum(int(r) * c for r, c in zip(bus.n_resources_, bus.n_candidates_)) * 3
                       if metodo.startswith("halving") else r_max * 27 * 3)
            filas.append({"modelo": nombre, "metodo": metodo, "tiempo_s": t, "arboles_ajustados": arboles,
                          "mejor_interno": bus.best_score_,
                          "pr_auc_bloque_externo": average_precision_score(
                              yte, bus.best_estimator_.predict_proba(Xte_i)[:, 1]),
                          "rondas": getattr(bus, "n_iterations_", 1)})
            print(filas[-1], flush=True)
    return pd.DataFrame(filas)


def memoria_modelos(X, y, prev):
    filas = []
    for nombre, fab in modelos_escalamiento(prev).items():
        filas.append({"modelo": nombre, "memoria_pico_fit_mb": memoria_pico(lambda: fab().fit(X, y))})
    return pd.DataFrame(filas)


def main():
    fijar_semillas()
    SAL.mkdir(parents=True, exist_ok=True)
    d, X, y, Xt, yt = preparar()
    prev = float(y.mean())
    res = {}
    res["escalamiento"], res["exponentes"] = escalamiento(X, y, Xt, prev)
    res["knn"] = knn_variantes(X, y, Xt, yt)
    res["lineales"] = lineales_variantes(X, d.y_reg.values)
    res["naive_bayes"] = nb_partial_fit(X, y, Xt, yt)
    res["xgboost"] = xgb_variantes(X, y, Xt, yt)
    res["svm"] = svm_variantes(X, y, Xt, yt)
    res["paralelizacion"] = paralelizacion(X, y)
    res["memoria"] = memoria_modelos(X, y, prev)
    res["multifidelidad"] = multifidelidad(d)
    res["perfil"], texto = perfilamiento(d)
    (SAL / "perfil_cprofile.txt").write_text(texto, encoding="utf-8")
    for k, v in res.items():
        v.to_csv(SAL / f"{k}.csv", index=False)
        print(f"\n== {k}\n{v.head(12).to_string()}")
    (SAL / "gpu.json").write_text(json.dumps({"gpu_disponible": False, "detalle":
        "El equipo (Intel Core i7-1255U, 15.7 GB) no tiene GPU NVIDIA; device='cuda' no aplica."}),
        encoding="utf-8")


if __name__ == "__main__":
    main()
