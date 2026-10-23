"""Descarga título, año, fuente y resumen de los trabajos más cercanos (OpenAlex).

Uso:  python -m scripts.resumenes_cercanos
"""
import sys, time
from pathlib import Path
import pandas as pd, requests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RESULTADOS

TITULOS = [
    "Uncertainty Estimation in Predicting Suspended Sediment Load Using Probabilistic Machine Learning and Conformal Prediction",
    "Uncertainty Estimation in Predicting River Discharge Using Probabilistic Machine Learning and Conformal Prediction",
    "Using machine learning to improve predictions and provide insight into fluvial sediment transport",
    "Development of a machine learning model for river bed load",
    "Climatic and landscape controls on effective discharge",
    "Distributional Gradient Boosting Machines",
    "Streamflow forecasting using extreme gradient boosting model coupled with Gaussian mixture model",
    "Data-Driven Flood Alert System (FAS) Using Extreme Gradient Boosting (XGBoost) to Forecast Flood Stages",
    "Boosted Generalized Normal Distributions: Integrating Machine Learning with Operations Knowledge",
    "Unraveling hydrological dynamics: climate and human implications in the Magdalena River streamflow",
]


def resumen(inv):
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


filas = []
for t in TITULOS:
    r = requests.get("https://api.openalex.org/works", params={"search": t, "per-page": 1},
                     headers={"User-Agent": "revision-literatura/1.0"}, timeout=60).json()
    w = (r.get("results") or [{}])[0]
    filas.append({"buscado": t, "titulo": w.get("title"), "anio": w.get("publication_year"),
                  "doi": w.get("doi"), "fuente": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                  "resumen": resumen(w.get("abstract_inverted_index"))})
    time.sleep(1)
pd.DataFrame(filas).to_csv(RESULTADOS / "literatura" / "trabajos_cercanos.csv", index=False)
