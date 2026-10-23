"""Revisión sistemática de literatura para la declaración de novedad (Entregable 3, Sección 5.2).

Consulta cuatro bases de datos bibliográficas con APIs públicas, registra la fecha, el
número de resultados y los primeros registros de cada cadena de búsqueda, y guarda la
matriz de novedad y el detalle para su revisión manual (título y resumen).

Uso:  python -m scripts.revision_literatura
"""
from __future__ import annotations

import datetime as dt
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RESULTADOS

CADENAS = [
    # (i) familia del modelo, (ii) problema de aplicación, (iii) sinónimos en español
    ('"effective discharge" machine learning', "Problema + aprendizaje automático"),
    ('"effective discharge" prediction OR forecasting OR classification', "Problema + tarea"),
    ('"distributional gradient boosting" streamflow OR discharge', "Familia del modelo + hidrología"),
    ('NGBoost streamflow OR discharge OR river', "Método afín (NGBoost) + hidrología"),
    ('gradient boosting interval probability threshold exceedance river discharge',
     "Mecanismo (probabilidad de intervalo) + aplicación"),
    ('hybrid loss proper scoring rule boosting classification continuous auxiliary target',
     "Mecanismo (pérdida híbrida) sin restricción de aplicación"),
    ('"Magdalena River" machine learning discharge', "Cuenca de estudio + aprendizaje automático"),
    ('"descarga efectiva" aprendizaje automático OR "caudal dominante"', "Sinónimos en español"),
]

UA = {"User-Agent": "revision-literatura-proyecto-ml/1.0 (uso academico)"}


def openalex(q: str, n: int = 15):
    r = requests.get("https://api.openalex.org/works",
                     params={"search": q, "per-page": n}, headers=UA, timeout=60)
    r.raise_for_status()
    j = r.json()
    regs = [{"titulo": w.get("title") or "", "anio": w.get("publication_year"),
             "doi": w.get("doi") or "", "fuente": ((w.get("primary_location") or {}).get("source")
                                                    or {}).get("display_name", "")}
            for w in j.get("results", [])]
    return j["meta"]["count"], regs


def crossref(q: str, n: int = 15):
    r = requests.get("https://api.crossref.org/works",
                     params={"query.bibliographic": q, "rows": n,
                             "select": "title,DOI,issued,container-title"}, headers=UA, timeout=60)
    r.raise_for_status()
    m = r.json()["message"]
    regs = [{"titulo": (i.get("title") or [""])[0],
             "anio": ((i.get("issued") or {}).get("date-parts") or [[None]])[0][0],
             "doi": i.get("DOI", ""), "fuente": (i.get("container-title") or [""])[0]}
            for i in m.get("items", [])]
    return m["total-results"], regs


def arxiv(q: str, n: int = 15):
    terminos = q.replace(" OR ", " ").replace('"', "")
    consulta = " AND ".join(f"all:{t}" for t in terminos.split()[:6])
    url = ("http://export.arxiv.org/api/query?search_query="
           + urllib.parse.quote(consulta) + f"&max_results={n}")
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
    raiz = ET.fromstring(r.text)
    total = int(raiz.find("o:totalResults", ns).text)
    regs = [{"titulo": " ".join(e.find("a:title", ns).text.split()),
             "anio": e.find("a:published", ns).text[:4], "doi": e.find("a:id", ns).text,
             "fuente": "arXiv"} for e in raiz.findall("a:entry", ns)]
    return total, regs


def semantic_scholar(q: str, n: int = 15):
    for intento in range(8):
        r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",
                         params={"query": q.replace('"', ""), "limit": n,
                                 "fields": "title,year,venue,externalIds"},
                         headers=UA, timeout=60)
        if r.status_code == 429:
            time.sleep(10 * (intento + 1))
            continue
        r.raise_for_status()
        j = r.json()
        regs = [{"titulo": p.get("title", ""), "anio": p.get("year"),
                 "doi": (p.get("externalIds") or {}).get("DOI", ""), "fuente": p.get("venue", "")}
                for p in j.get("data", [])]
        return j.get("total", 0), regs
    raise RuntimeError("Semantic Scholar: límite de solicitudes")


BASES = {"OpenAlex": openalex, "Crossref": crossref, "arXiv": arxiv,
         "Semantic Scholar": semantic_scholar}


def main():
    destino = RESULTADOS / "literatura"
    destino.mkdir(parents=True, exist_ok=True)
    fecha = dt.date.today().isoformat()
    matriz, detalle = [], []
    for cadena, criterio in CADENAS:
        for base, fn in BASES.items():
            try:
                total, regs = fn(cadena)
                estado = "ok"
            except Exception as e:      # se registra el fallo en lugar de ocultarlo
                total, regs, estado = None, [], f"error: {e.__class__.__name__}"
            matriz.append({"fecha_consulta": fecha, "base": base, "cadena": cadena,
                           "criterio": criterio, "resultados": total,
                           "revisados": len(regs), "estado": estado})
            for i, r in enumerate(regs):
                detalle.append({"base": base, "cadena": cadena, "orden": i + 1, **r})
            print(f"{base:17s} {total!s:>10}  {cadena}")
            time.sleep(4)
    pd.DataFrame(matriz).to_csv(destino / "matriz_busqueda.csv", index=False)
    pd.DataFrame(detalle).to_csv(destino / "registros_revisados.csv", index=False)


if __name__ == "__main__":
    main()
