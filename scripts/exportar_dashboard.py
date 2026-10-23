"""Precalcula las tablas livianas que consume el dashboard (data/dashboard/).

Uso:  python -m scripts.exportar_dashboard
Requiere haber ejecutado el ETL, los benchmarks, el modelo original, los modelos finales,
la robustez y el análisis estadístico (results/estadistica/).
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import cargar_carpeta, nombres, tabla_pliegues
from src.config import ARTEFACTOS, DATOS, DATOS_PROC, RESULTADOS, SEED
from src.evaluation import ece, shapley_permutaciones
from src.experiment import cargar_datos
from src.models import NOMBRES_BALANCEO, NOMBRES_MODELO, NOMBRES_OPT

DASH = DATOS / "dashboard"


def exportar_eda():
    """Tablas de las páginas de Contexto y EDA (no dependen de los modelos)."""
    DASH.mkdir(parents=True, exist_ok=True)
    datos = cargar_datos()

    # --- datos básicos y EDA -------------------------------------------------
    shutil.copy(DATOS_PROC / "caudales.csv", DASH / "caudales.csv")
    shutil.copy(DATOS_PROC / "magnitud_frecuencia.csv", DASH / "magnitud_frecuencia.csv")
    shutil.copy(DATOS_PROC / "fuga.csv", DASH / "fuga.csv")
    tarea = pd.read_parquet(DATOS_PROC / "tarea_calamar.parquet")
    tarea.to_parquet(DASH / "tarea_calamar.parquet", index=False)
    test = tarea[tarea.fecha >= pd.Timestamp("2010-01-31")]
    test.to_parquet(DASH / "tarea_test.parquet", index=False)
    datos.X.sample(40, random_state=SEED).to_parquet(DASH / "fondo.parquet", index=False)

    ubic = pd.read_csv(DATOS_PROC / "ubicaciones.csv")
    ubic.columns = ["estacion", "codigo", "departamento", "municipio", "lat", "lon", "altitud"]
    mapa = {"ARRANCAPLUMAS - AUT": "arrancaplumas", "PUERTO SALGAR - AUT": "pto_salgar",
            "TRES CRUCES": "tres_cruces", "COYONGAL": "coyongal", "EL BANCO - AUT": "el_banco",
            "SAN ROQUE": "san_roque", "CALAMAR": "calamar"}
    ubic["clave"] = ubic.estacion.map(mapa)
    ubic["nombre"] = ubic.estacion.str.replace(" - AUT", "", regex=False).str.title()
    ubic.to_csv(DASH / "ubicaciones.csv", index=False)

    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
    c = pd.read_csv(DATOS_PROC / "caudales.csv", parse_dates=["fecha"])
    s = c[(c.fecha >= "1976-01-01") & (c.fecha < "2010-01-01")].set_index("fecha").calamar
    s = s.interpolate(limit=5).dropna()
    est = {"acf": acf(s, nlags=60, fft=True).tolist(), "pacf": pacf(s, nlags=30).tolist(), "n": int(len(s)),
           "p_adf": float(adfuller(s, autolag="AIC")[1]), "p_kpss": float(kpss(s, regression="c", nlags="auto")[1])}
    (DASH / "estacionariedad.json").write_text(json.dumps(est), encoding="utf-8")
    m = s.resample("MS").mean()
    stl = STL(m, period=12, robust=True).fit()
    pd.DataFrame({"fecha": m.index, "observado": m.values, "tendencia": stl.trend.values,
                  "estacional": stl.seasonal.values, "residuo": stl.resid.values}).to_csv(DASH / "stl.csv", index=False)


def exportar_modelos():
    """Tablas de la página de Modelos: líderes, predicciones, SHAP, estadística."""
    datos = cargar_datos()
    par = json.loads((DATOS_PROC / "parametros.json").read_text(encoding="utf-8"))["calamar"]
    cols = par["BASE_COLS"] + par["DOM_COLS"]

    # --- líderes, predicciones de prueba y calibración -------------------------
    fin = pickle.load(open(RESULTADOS / "final_clf.pkl", "rb"))
    fo = pickle.load(open(RESULTADOS / "final_original.pkl", "rb"))[0]
    maestra = pd.read_parquet(RESULTADOS / "maestra_clf.parquet")
    pl_orig = tabla_pliegues("original_calamar")
    filas, pred, cal = [], {"fecha": datos.fechas_test.values, "y": datos.y_clf_test.values,
                            "q_futuro": datos.y_reg_test.values}, []
    for r in fin:
        fila = maestra[(maestra.modelo == r["modelo"]) & (maestra.balanceo == r["balanceo"])
                       & (maestra.optimizador == r["optimizador"])].iloc[0]
        # La decisión de recalibrar se tomó con las predicciones fuera de muestra del bucle externo
        meta_p = ARTEFACTOS / f"{r['modelo']}.metadata.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        meta["usar_recalibrado"] = bool(r["recalibrar_por_oof"])
        meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        p = r["pred_test_recalibrado"] if meta["usar_recalibrado"] else r["pred_test"]
        pred[f"p_{r['modelo']}"] = p
        mt = r["metricas_test_recalibrado"] if meta["usar_recalibrado"] else r["metricas_test"]
        filas.append({"modelo": r["modelo"], "balanceo": r["balanceo"], "optimizador": r["optimizador"],
                      **{k: fila[k] for k in fila.index if k.endswith(("_mean", "_std"))},
                      "pr_auc_test": mt["pr_auc"], "roc_auc_test": mt["roc_auc"], "f1_test": mt["f1"],
                      "brier_test": mt["brier"], "t_inf_ms": r["t_inf_ms_fila"]})
        cal.append({"modelo": r["modelo"], "brier": mt["brier"], "ece": ece(datos.y_clf_test, p),
                    "ece_crudo": r["ece_test"], "recalibrado": meta["usar_recalibrado"]})
    o = pl_orig[(pl_orig.modelo == "blgb") & (pl_orig.optimizador == fo["optimizador"])]
    agg = {f"{k}_{s}": getattr(o[k], s)() for k in ("pr_auc", "roc_auc", "f1", "brier", "accuracy",
                                                    "precision", "recall") for s in ("mean", "std")}
    filas.append({"modelo": "original", "balanceo": "ninguno", "optimizador": fo["optimizador"], **agg,
                  "pr_auc_test": fo["metricas_test"]["pr_auc"], "roc_auc_test": fo["metricas_test"]["roc_auc"],
                  "f1_test": fo["metricas_test"]["f1"], "brier_test": fo["metricas_test"]["brier"],
                  "t_inf_ms": fo["t_inf_ms_fila"]})
    pred["p_original"] = fo["pred_test"]
    pred["q_mediana"], pred["q_inf"], pred["q_sup"] = fo["q_mediana_test"], fo["q_inf_test"], fo["q_sup_test"]
    cal.append({"modelo": "original", "brier": fo["metricas_test"]["brier"], "ece": fo["ece_test"],
                "ece_crudo": fo["ece_test"], "recalibrado": False})
    lid = nombres(pd.DataFrame(filas))
    lid.to_csv(DASH / "lideres.csv", index=False)
    pd.DataFrame(pred).to_parquet(DASH / "pred_test.parquet", index=False)
    pd.DataFrame(cal).to_csv(DASH / "calibracion.csv", index=False)
    nombres(maestra).to_csv(DASH / "maestra_resumen.csv", index=False)

    # --- SHAP global (permutaciones) sobre 300 días de prueba -------------------
    from src.data import modelo as cargar_modelo  # noqa: F401  (misma ruta de carga que el servidor)
    import joblib
    from src.models import puntaje_continuo
    muestra = datos.X_test.sample(300, random_state=SEED)
    fondo = datos.X.sample(40, random_state=SEED)
    med = fondo.median()
    filas = []
    for nombre in [r["modelo"] for r in fin] + ["original"]:
        est = joblib.load(ARTEFACTOS / f"{nombre}.joblib")
        f = (lambda Z, e=est: puntaje_continuo(e, pd.DataFrame(Z, columns=cols)))
        phis = np.array([shapley_permutaciones(f, x, fondo.fillna(med).values, n_perm=12, seed=i)[0]
                         for i, x in enumerate(muestra.fillna(med).values)])
        imp = np.abs(phis).mean(0)
        filas += [{"modelo": nombre, "variable": v, "importancia": float(i)} for v, i in zip(cols, imp)]
        print("SHAP", nombre, flush=True)
    sh = pd.DataFrame(filas)
    sh.to_csv(DASH / "shap_global.csv", index=False)
    top = sh[sh.modelo == "original"].sort_values("importancia", ascending=False).variable.head(8).tolist()

    # --- estadística, modelo original y regresión ------------------------------
    E = RESULTADOS / "estadistica"
    for nombre in ("cd.json", "cd_112.json", "veredicto.json"):
        shutil.copy(E / nombre, DASH / nombre)
    for nombre in ("pruebas_original.csv", "ablacion.csv", "mcs_regresion.csv", "literatura_cercana.csv",
                   "conv_boosting.csv", "anytime_original.csv"):
        shutil.copy(E / nombre, DASH / nombre)

    lid_b = lid[lid.modelo != "original"].sort_values("pr_auc_mean", ascending=False)
    o = lid[lid.modelo == "original"].iloc[0]
    textos = json.loads((E / "textos_dashboard.json").read_text(encoding="utf-8"))
    dash_par = {**{k: par[k] for k in ("Q_EF", "LIM", "Q_EF_ROBUSTEZ", "n_train", "n_test",
                                       "prevalencia_train", "prevalencia_test")},
                "columnas": cols, "variables_formulario": top,
                "fecha_actualizacion": dt.date.today().isoformat(), **textos}
    (DASH / "parametros.json").write_text(json.dumps(dash_par, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Exportado en", DASH, "| original PR-AUC prueba", round(o.pr_auc_test, 4),
          "| mejor benchmark", lid_b.iloc[0].modelo)


if __name__ == "__main__":
    fase = sys.argv[1] if len(sys.argv) > 1 else "todo"
    if fase in ("eda", "todo"):
        exportar_eda()
    if fase in ("modelos", "todo"):
        exportar_modelos()
