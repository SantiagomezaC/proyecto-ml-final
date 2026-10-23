"""Ablación, sensibilidad y robustez del modelo original frente al mejor benchmark.

Todas las variantes reutilizan, en cada pliegue externo, los hiperparámetros que la
optimización anidada eligió para ese pliegue, de modo que la comparación aísla el efecto
del componente o de la perturbación y no el de una nueva búsqueda.

Uso:  python -m scripts.robustez [n_jobs]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTADOS, SEED, fijar_semillas
from src.experiment import cargar_datos
from src.experiment_original import ModeloDistribucional, parametros_banda
from src.models import construir_pipeline, puntaje_continuo

SAL = RESULTADOS / "robustez"


def params_pliegue(carpeta: Path, patron: str) -> dict:
    return {pickle.load(open(f, "rb"))["pliegue"]: pickle.load(open(f, "rb"))["params"]
            for f in sorted(carpeta.glob(patron))}


def ajustar_blgb(datos, banda, params, tr, te, X=None, Xte=None, y=None, q=None, seed=SEED):
    X = datos.X.iloc[tr] if X is None else X
    y = datos.y_clf.iloc[tr] if y is None else y
    q = datos.y_reg.iloc[tr].values if q is None else q
    Xte = datos.X.iloc[te] if Xte is None else Xte
    m = ModeloDistribucional("blgb", params, banda, seed).fit(X, y, q)
    return average_precision_score(datos.y_clf.iloc[te], m.predict_proba(Xte)[:, 1])


def ajustar_bench(datos, fila, params, tr, te, X=None, Xte=None, y=None, seed=SEED):
    X = datos.X.iloc[tr] if X is None else X
    y = datos.y_clf.iloc[tr] if y is None else y
    Xte = datos.X.iloc[te] if Xte is None else Xte
    est = construir_pipeline("clf", fila.modelo, params, fila.balanceo, float(y.mean()), seed).fit(X, y)
    return average_precision_score(datos.y_clf.iloc[te], puntaje_continuo(est, Xte))


def trabajo(tipo, variante, pliegue, valor, seed):
    datos = cargar_datos()
    banda = parametros_banda("calamar")
    p = datos.pliegues[pliegue]
    tr, te = p["train"], p["test"]
    pb = PARAMS_BLGB[pliegue]
    fila = MEJOR
    pben = PARAMS_BENCH[pliegue]
    rng = np.random.default_rng(seed)
    out = {"tipo": tipo, "variante": variante, "pliegue": pliegue, "valor": valor, "seed": seed}
    if tipo == "ablacion":
        mod = dict(pb)
        if variante == "sin supervisión continua (λ = 0)":
            mod["lambda_hibrido"] = 0.0
        elif variante == "σ constante (homocedástico)":
            mod["heterocedastico"] = False
        elif variante == "sin búsqueda lineal de Armijo":
            mod["busqueda_lineal"] = False
        elif variante == "solo verosimilitud (λ = 100)":
            mod["lambda_hibrido"] = 100.0
        elif variante == "sin reajuste tras parada temprana":
            mod["reajustar"] = False
        out["pr_auc"] = ajustar_blgb(datos, banda, mod, tr, te)
    elif tipo == "lambda":
        out["pr_auc"] = ajustar_blgb(datos, banda, {**pb, "lambda_hibrido": valor}, tr, te)
    elif tipo == "semilla":
        out["modelo"] = variante
        out["pr_auc"] = (ajustar_blgb(datos, banda, pb, tr, te, seed=seed) if variante == "original"
                         else ajustar_bench(datos, fila, pben, tr, te, seed=seed))
    else:
        X, y, q = datos.X.iloc[tr].copy(), datos.y_clf.iloc[tr].copy(), datos.y_reg.iloc[tr].values
        Xte = datos.X.iloc[te].copy()
        if tipo == "ruido":
            # Ruido gaussiano multiplicativo en todos los predictores hidrológicos (error de medición);
            # la columna de referencia del caudal actual también se perturba.
            for Z in (X, Xte):
                for c in Z.columns:
                    if not c.startswith(("sen_", "cos_")):
                        Z[c] = Z[c] * (1 + valor * rng.normal(size=len(Z)))
            Xte["calamar_cuad"] = Xte["calamar_cuad"].abs()
            X["calamar_cuad"] = X["calamar_cuad"].abs()
        elif tipo == "desbalance":
            pos = np.flatnonzero(y.values == 1)
            neg = np.flatnonzero(y.values == 0)
            n_pos = int(valor / (1 - valor) * len(neg))
            keep = np.sort(np.concatenate([neg, rng.choice(pos, min(n_pos, len(pos)), replace=False)]))
            X, y, q = X.iloc[keep], y.iloc[keep], q[keep]
        elif tipo == "muestra":
            k = int(len(X) * valor)
            X, y, q = X.iloc[-k:], y.iloc[-k:], q[-k:]
        out["modelo"] = variante
        out["pr_auc"] = (ajustar_blgb(datos, banda, pb, tr, te, X, Xte, y, q) if variante == "original"
                         else ajustar_bench(datos, fila, pben, tr, te, X, Xte, y))
    return out


if __name__ == "__main__":
    fijar_semillas()
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    SAL.mkdir(parents=True, exist_ok=True)
    mejores = pd.read_csv(RESULTADOS / "mejores_clf.csv")
    MEJOR = mejores.sort_values("pr_auc_mean", ascending=False).iloc[0]
    from src.analysis import tabla_pliegues
    _o = tabla_pliegues("original_calamar")
    OPT_BLGB = _o[_o.modelo == "blgb"].groupby("optimizador").pr_auc.mean().idxmax()
    PARAMS_BLGB = params_pliegue(RESULTADOS / "original_calamar", f"blgb__{OPT_BLGB}__f*.pkl")
    PARAMS_BENCH = params_pliegue(RESULTADOS / "clf_calamar",
                                  f"{MEJOR.modelo}__{MEJOR.balanceo}__{MEJOR.optimizador}__f*.pkl")
    pl = sorted(PARAMS_BLGB)
    tareas = []
    for v in ["completo", "sin supervisión continua (λ = 0)", "σ constante (homocedástico)",
              "sin búsqueda lineal de Armijo", "solo verosimilitud (λ = 100)",
              "sin reajuste tras parada temprana"]:
        tareas += [("ablacion", v, k, None, SEED) for k in pl]
    tareas += [("lambda", "λ", k, lam, SEED) for k in pl for lam in (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)]
    for s in (1, 7, 42, 123, 2024):
        tareas += [("semilla", m, k, None, s) for k in pl for m in ("original", "benchmark")]
    recientes = pl[-6:]
    for tipo, valores in (("ruido", (0.02, 0.05, 0.10)), ("desbalance", (0.15, 0.08)),
                          ("muestra", (0.25, 0.5))):
        tareas += [(tipo, m, k, v, SEED) for k in recientes for v in valores
                   for m in ("original", "benchmark")]
    res = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(delayed(trabajo)(*t) for t in tareas)
    df = pd.DataFrame(res)
    df.to_csv(SAL / "robustez.csv", index=False)
    print(df.groupby(["tipo", "variante", "valor"], dropna=False).pr_auc.agg(["mean", "std", "size"]))
