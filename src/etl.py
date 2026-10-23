"""Extracción y transformación de las series diarias del IDEAM (portal DHIME).

El módulo reproduce el ETL del Entregable 1 en forma de funciones reutilizables:
lectura de los archivos crudos, consolidación en una tabla diaria por estación,
auditoría de la serie de transporte, descarga efectiva y tiempos de tránsito.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

from .config import ALIAS_ESTACIONES, DATOS_CRUDOS, N_CLASES_MF

META = ["CodigoEstacion", "NombreEstacion", "Variable", "Parametro", "Unidad", "NivelAprobacion"]


def _nombre_estacion(meta: dict) -> str:
    n = meta["NombreEstacion"].split("[")[0].strip().lower()
    return ALIAS_ESTACIONES.get(n, n.replace(" ", "_"))


def cargar_archivo(ruta: Path) -> tuple[pd.DataFrame, dict]:
    """Lee un archivo DHIME y devuelve la serie (fecha, valor, aprobación) y sus metadatos.

    Parameters
    ----------
    ruta : Path
        Archivo CSV descargado del portal DHIME (codificación UTF-8 con BOM).

    Returns
    -------
    serie : DataFrame
        Columnas ``fecha``, ``valor`` y ``aprobacion``.
    meta : dict
        Metadatos de la primera fila, con la clave adicional ``estacion``.
    """
    df = pd.read_csv(ruta, encoding="utf-8-sig", parse_dates=["Fecha"])
    valor = [c for c in df.columns if c not in META + ["Fecha"]][0]
    meta = {c: df[c].iloc[0] for c in META if c in df}
    meta["estacion"] = _nombre_estacion(meta)
    meta["es_transporte"] = "transporte" in str(meta["Parametro"]).lower()
    serie = df.rename(columns={"Fecha": "fecha", valor: "valor", "NivelAprobacion": "aprobacion"})
    return serie[["fecha", "valor", "aprobacion"]], meta


def cargar_red(directorio: Path = DATOS_CRUDOS, objetivo: str = "calamar") -> dict:
    """Consolida todas las series de caudal y de transporte en tablas diarias anchas.

    El calendario es el de la estación objetivo; los días sin registro quedan como NaN,
    de modo que los huecos son visibles y no se rellenan en esta etapa.

    Returns
    -------
    dict
        ``caudales`` y ``transporte`` (DataFrame con columna ``fecha`` y una columna por
        estación), ``aprobacion`` (nivel de aprobación por día y estación) y ``meta``.
    """
    archivos = sorted(Path(directorio).glob("*.csv"))
    if not archivos:
        raise FileNotFoundError(
            f"No hay archivos en {directorio}. Ver DESCARGA_DATOS.md para obtenerlos del IDEAM."
        )
    caudal, transporte, aprob, metas = {}, {}, {}, {}
    for ruta in archivos:
        serie, meta = cargar_archivo(ruta)
        serie = serie.drop_duplicates("fecha").set_index("fecha")
        e = meta["estacion"]
        if meta["es_transporte"]:
            transporte[e] = serie["valor"]
        else:
            caudal[e] = serie["valor"]
            aprob[e] = serie["aprobacion"]
            metas[e] = meta
    calendario = pd.date_range(caudal[objetivo].index.min(), caudal[objetivo].index.max(), freq="D")

    def ancha(d: dict) -> pd.DataFrame:
        return pd.DataFrame(d).reindex(calendario).rename_axis("fecha").reset_index()

    return {"caudales": ancha(caudal), "transporte": ancha(transporte),
            "aprobacion": ancha(aprob), "meta": metas}


# ---------------------------------------------------------------------------
# Auditoría de la serie de transporte y descarga efectiva
# ---------------------------------------------------------------------------
def curva_gasto(q: pd.Series, qs: pd.Series) -> tuple[float, float, float, int]:
    """Ajusta la curva de gasto sólido ``Qs = a·Q^b`` por mínimos cuadrados en log-log.

    Returns
    -------
    a, b, r2, n : float, float, float, int
    """
    m = (q > 0) & (qs > 0) & q.notna() & qs.notna()
    b, loga, r, _, _ = st.linregress(np.log10(q[m]), np.log10(qs[m]))
    return float(10 ** loga), float(b), float(r ** 2), int(m.sum())


def auditoria_transporte(fechas: pd.Series, q: pd.Series, qs: pd.Series,
                         anios_bloque: int = 5) -> pd.DataFrame:
    """Ajusta la curva de gasto por bloques de ``anios_bloque`` años.

    Un R² igual a uno dentro de cada bloque indica que la serie publicada es una
    transformación determinista del caudal y no una medición independiente.
    """
    d = pd.DataFrame({"fecha": fechas, "Q": q, "Qs": qs}).dropna()
    d = d[(d.Q > 0) & (d.Qs > 0)]
    d["bloque"] = (d.fecha.dt.year // anios_bloque) * anios_bloque
    filas = []
    for b0, g in d.groupby("bloque"):
        if len(g) < 200:
            continue
        a, b, r2, n = curva_gasto(g.Q, g.Qs)
        filas.append({"bloque": f"{b0}-{b0 + anios_bloque - 1}", "a": a, "b": b, "r2": r2, "n": n})
    return pd.DataFrame(filas)


def descarga_efectiva(caudal: pd.Series, a: float, b: float,
                      n_clases: int = N_CLASES_MF) -> tuple[float, tuple[float, float], pd.DataFrame]:
    """Análisis magnitud-frecuencia (Wolman y Miller, 1960; Andrews, 1980).

    El registro se agrupa en ``n_clases`` clases logarítmicas; la carga de cada clase es
    su frecuencia multiplicada por el transporte de un día típico, ``a·Q_c^b``. La
    descarga efectiva es el centro de la clase de carga máxima.

    Returns
    -------
    q_ef : float
        Descarga efectiva (m³/s).
    clase : tuple
        Límites de la clase modal.
    tabla : DataFrame
        Frecuencia y carga relativa por clase.
    """
    x = caudal.dropna()
    x = x[x > 0]
    bordes = np.logspace(np.log10(x.min()), np.log10(x.max()), n_clases + 1)
    frec, _ = np.histogram(x, bins=bordes)
    centros = np.sqrt(bordes[:-1] * bordes[1:])
    carga = frec * (a * centros ** b)
    t = pd.DataFrame({"q_inf": bordes[:-1], "q_sup": bordes[1:], "q_centro": centros,
                      "dias": frec, "frec_rel": frec / frec.sum(), "carga_rel": carga / carga.sum()})
    i = int(t.carga_rel.idxmax())
    return float(t.loc[i, "q_centro"]), (float(t.loc[i, "q_inf"]), float(t.loc[i, "q_sup"])), t


def tiempos_transito(caudales: pd.DataFrame, objetivo: str, estaciones: list[str],
                     max_rezago: int = 40) -> dict[str, int]:
    """Estima el tiempo de tránsito como el rezago que maximiza la correlación cruzada.

    Se usan las diferencias de 7 días para eliminar la estacionalidad común, que de otro
    modo domina la correlación en niveles y desplaza el máximo.
    """
    d = caudales.set_index("fecha")
    y = d[objetivo].diff(7)
    salida = {}
    for e in estaciones:
        x = d[e].diff(7)
        cors = [y.corr(x.shift(k)) for k in range(0, max_rezago + 1)]
        salida[e] = int(np.nanargmax(cors))
    return salida
