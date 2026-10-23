"""ETL completo: de los archivos crudos del IDEAM a las tablas de modelado.

Uso:  python -m scripts.etl
Genera en data/processed/: caudales.csv, tarea_<estacion>.parquet, parametros.json,
pliegues_externos.json y auditoria_transporte.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (DATOS, DATOS_PROC, ESTACION_OBJETIVO, EXCLUIR_MODELO, FECHA_CORTE_TEST,
                        TOLERANCIA)
from src.etl import auditoria_transporte, cargar_red, descarga_efectiva, tiempos_transito
from src.features import construir_tarea
from src.splits import guardar_pliegues, pliegues_externos, separar_prueba

# Tiempos de tránsito hacia Calamar estimados y verificados geográficamente en el Entregable 1
TRANSITOS_CALAMAR = {"arrancaplumas": 20, "coyongal": 7, "el_banco": 12, "pto_salgar": 21,
                     "san_roque": 10, "tres_cruces": 16}

# Tareas adicionales para la evaluación amplia del modelo original (Entregable 3):
# la misma tarea en otras estaciones del cauce, con regímenes hidrológicos distintos.
TAREAS_ADICIONALES = {
    "el_banco": ["tres_cruces", "pto_salgar"],
    "coyongal": ["el_banco", "tres_cruces", "pto_salgar"],
    "tres_cruces": ["pto_salgar"],
}


def parametros_estacion(red: dict, estacion: str) -> dict:
    """Curva de gasto vigente y descarga efectiva calculadas solo con el entrenamiento."""
    c, t = red["caudales"], red["transporte"]
    corte = pd.Timestamp(FECHA_CORTE_TEST)
    m = c.fecha < corte
    aud = auditoria_transporte(c.fecha[m], c[estacion][m], t[estacion][m])
    a, b = aud.iloc[-1][["a", "b"]]
    q_ef, clase, tabla = descarga_efectiva(c.loc[m, estacion], a, b)
    robustez = [descarga_efectiva(c.loc[m, estacion], r.a, r.b)[0] for r in aud.itertuples()]
    lim = (q_ef * (1 - TOLERANCIA), q_ef * (1 + TOLERANCIA))
    return {"Q_EF": q_ef, "LIM": list(lim), "CLASE_MODAL": list(clase), "A_OF": float(a),
            "B_OF": float(b), "Q_EF_ROBUSTEZ": robustez, "auditoria": aud, "tabla_mf": tabla}


def main() -> None:
    DATOS_PROC.mkdir(parents=True, exist_ok=True)
    red = cargar_red()
    caudales = red["caudales"]
    caudales.to_csv(DATOS_PROC / "caudales.csv", index=False)
    red["transporte"].to_csv(DATOS_PROC / "transporte.csv", index=False)

    parametros = {}
    estaciones = [ESTACION_OBJETIVO] + list(TAREAS_ADICIONALES)
    for est in estaciones:
        p = parametros_estacion(red, est)
        if est == ESTACION_OBJETIVO:
            predictoras = [e for e in TRANSITOS_CALAMAR if e not in EXCLUIR_MODELO]
            transitos = TRANSITOS_CALAMAR
            p["auditoria"].to_csv(DATOS_PROC / "auditoria_transporte.csv", index=False)
            p["tabla_mf"].to_csv(DATOS_PROC / "magnitud_frecuencia.csv", index=False)
        else:
            predictoras = TAREAS_ADICIONALES[est]
            # Tránsito relativo: diferencia de tiempos hacia Calamar, contrastada con la
            # correlación cruzada directa entre la estación predictora y la objetivo.
            transitos = {e: max(1, TRANSITOS_CALAMAR[e] - TRANSITOS_CALAMAR[est]) for e in predictoras}
            p["transitos_ccf"] = tiempos_transito(caudales[caudales.fecha < FECHA_CORTE_TEST],
                                                  est, predictoras)
        tarea = construir_tarea(caudales, est, predictoras, transitos, p["Q_EF"], tuple(p["LIM"]))
        tarea["datos"].to_parquet(DATOS_PROC / f"tarea_{est}.parquet", index=False)
        tr, te = separar_prueba(tarea["datos"].fecha)
        fechas_tr = tarea["datos"].fecha.iloc[tr].reset_index(drop=True)
        pl = pliegues_externos(fechas_tr)
        guardar_pliegues(pl, DATOS_PROC / f"pliegues_{est}.json", fechas_tr)
        parametros[est] = {k: v for k, v in p.items() if k not in ("auditoria", "tabla_mf")}
        parametros[est].update({"PREDICTORAS": predictoras, "TRANSITOS": transitos,
                                "BASE_COLS": tarea["base_cols"], "DOM_COLS": tarea["dom_cols"],
                                "n_train": int(len(tr)), "n_test": int(len(te)),
                                "prevalencia_train": float(tarea["datos"].y.iloc[tr].mean()),
                                "prevalencia_test": float(tarea["datos"].y.iloc[te].mean()),
                                "n_pliegues_externos": len(pl)})
        print(f"{est:12s} Q_ef={p['Q_EF']:8.0f}  train={len(tr):6d} test={len(te):5d} "
              f"prev={parametros[est]['prevalencia_train']:.3f}  pliegues={len(pl)}  "
              f"robustez={[round(v) for v in p['Q_EF_ROBUSTEZ']]}  {p.get('transitos_ccf', '')}")

    (DATOS_PROC / "parametros.json").write_text(json.dumps(parametros, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    ubic = pd.read_csv(DATOS / "UBICACION_ESTACIONES_DESCARGADAS_IDEAM.csv", sep=";", encoding="utf-8-sig")
    ubic.to_csv(DATOS_PROC / "ubicaciones.csv", index=False)


if __name__ == "__main__":
    main()
