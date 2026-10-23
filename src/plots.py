"""Funciones de graficación del dashboard con una plantilla Plotly unificada.

Todas las figuras comparten tipografía, paleta (cinco colores), márgenes y formato de
ejes; cada función recibe datos ya preparados y devuelve un ``go.Figure``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from .config import PALETA
from .data import es

RIO, SED, CAU, ALE, NEU = (PALETA[k] for k in ("rio", "sedimento", "caudal", "alerta", "neutro"))
SECUENCIA = [RIO, SED, CAU, ALE, NEU]
ESCALA_SEQ = [[0, "#f2f7f9"], [0.5, CAU], [1, RIO]]
ESCALA_DIV = [[0, ALE], [0.5, "#f7f7f7"], [1, RIO]]

pio.templates["magdalena"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, system-ui, sans-serif", size=13, color="#13232c"),
        colorway=SECUENCIA,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=56, r=18, t=82, b=48),
        title=dict(font=dict(size=15, color="#13232c"), x=0.0, xanchor="left", y=0.985, yanchor="top"),
        xaxis=dict(showgrid=True, gridcolor="#e9eef0", zeroline=False, linecolor="#c9d3d8",
                   ticks="outside", tickcolor="#c9d3d8", title=dict(standoff=8)),
        yaxis=dict(showgrid=True, gridcolor="#e9eef0", zeroline=False, linecolor="#c9d3d8",
                   title=dict(standoff=8)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        hoverlabel=dict(font=dict(family="Inter, sans-serif", size=12), bgcolor="#ffffff",
                        bordercolor="#c9d3d8"),
        hovermode="closest",
        separators=",.",
    )
)
pio.templates.default = "magdalena"


def _fig(titulo: str, alto: int = 380, **kw) -> go.Figure:
    f = go.Figure(**kw)
    f.update_layout(title=titulo, height=alto)
    return f


def vacia(mensaje: str = "Sin datos para la selección actual") -> go.Figure:
    f = _fig("", 300)
    f.add_annotation(text=mensaje, showarrow=False, font=dict(color=NEU, size=14),
                     xref="paper", yref="paper", x=0.5, y=0.5)
    f.update_xaxes(visible=False)
    f.update_yaxes(visible=False)
    return f


# ---------------------------------------------------------------------------
# Contexto
# ---------------------------------------------------------------------------
def mapa_estaciones(ubic: pd.DataFrame) -> go.Figure:
    objetivo = ubic.clave == "calamar"
    f = go.Figure()
    for m, color, nombre, tam in ((~objetivo, RIO, "Estaciones predictoras", 12),
                                  (objetivo, SED, "Estación objetivo (Calamar)", 18)):
        d = ubic[m]
        f.add_trace(go.Scattermap(
            lat=d.lat, lon=d.lon, mode="markers+text", name=nombre,
            marker=dict(size=tam, color=color), text=d.nombre, textposition="middle right",
            textfont=dict(size=12, color="#13232c"),
            customdata=np.column_stack([d.municipio, d.altitud, d.codigo]),
            hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>Altitud: %{customdata[1]} m s. n. m."
                          "<br>Código IDEAM: %{customdata[2]}<extra></extra>"))
    f.update_layout(map=dict(style="carto-positron", zoom=5.1, center=dict(lat=8.0, lon=-74.6)),
                    height=470, margin=dict(l=0, r=0, t=0, b=0),
                    legend=dict(y=0.02, x=0.01, yanchor="bottom", bgcolor="rgba(255,255,255,.85)"))
    return f


def magnitud_frecuencia(tabla: pd.DataFrame, q_ef: float) -> go.Figure:
    f = _fig("Análisis magnitud-frecuencia en Calamar (1940-2009)", 380)
    f.add_bar(x=tabla.q_centro, y=100 * tabla.frec_rel, name="Frecuencia (% de días)",
              marker_color=CAU, opacity=.55,
              hovertemplate="Clase %{x:,.0f} m³/s<br>%{y:.2f} % de los días<extra></extra>")
    f.add_scatter(x=tabla.q_centro, y=100 * tabla.carga_rel, name="Carga transportada (%)",
                  mode="lines+markers", line=dict(color=SED, width=3),
                  hovertemplate="Clase %{x:,.0f} m³/s<br>%{y:.2f} % de la carga<extra></extra>")
    f.add_vline(x=q_ef, line_dash="dash", line_color=ALE,
                annotation_text=f"Q<sub>ef</sub> = {q_ef:,.0f} m³/s", annotation_font_color=ALE)
    f.update_xaxes(title="Caudal (m³/s, escala logarítmica)", type="log")
    f.update_yaxes(title="Porcentaje")
    return f


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def serie_caudal(d: pd.DataFrame, estaciones: list[str], lim, nombres: dict) -> go.Figure:
    f = _fig("Caudal medio mensual por estación", 400)
    f.add_hrect(y0=lim[0], y1=lim[1], fillcolor=ALE, opacity=.10, line_width=0,
                annotation_text="Banda de descarga efectiva", annotation_position="top left",
                annotation_font_color=ALE)
    for i, e in enumerate(estaciones):
        f.add_scatter(x=d.fecha, y=d[e], name=nombres.get(e, e), mode="lines",
                      line=dict(width=1.6, color=SECUENCIA[i % 5]),
                      hovertemplate="%{x|%b %Y}: %{y:,.0f} m³/s<extra>" + nombres.get(e, e) + "</extra>")
    f.update_xaxes(title="Fecha")
    f.update_yaxes(title="Caudal (m³/s)")
    return f


def balance_anual(d: pd.DataFrame) -> go.Figure:
    a = d.groupby(d.fecha.dt.year).y.agg(["mean", "size"]).reset_index()
    f = _fig("Proporción anual de días en la banda (clase positiva)", 340)
    f.add_bar(x=a.fecha, y=100 * a["mean"], marker_color=SED,
              customdata=a["size"], hovertemplate="%{x}: %{y:.1f} % de %{customdata} días<extra></extra>")
    f.add_hline(y=100 * d.y.mean(), line_dash="dot", line_color=RIO,
                annotation_text=es(f"media {100 * d.y.mean():.1f} %"), annotation_font_color=RIO)
    f.update_xaxes(title="Año")
    f.update_yaxes(title="Días en la banda (%)", rangemode="tozero")
    return f


def histograma(d: pd.DataFrame, var: str, etiqueta: str, por_clase: bool = True) -> go.Figure:
    f = _fig(f"Distribución de {etiqueta}", 340)
    if por_clase:
        for clase, color, nombre in ((0, RIO, "Fuera de la banda"), (1, SED, "En la banda")):
            f.add_histogram(x=d.loc[d.y == clase, var], name=nombre, marker_color=color, opacity=.65,
                            nbinsx=50, histnorm="percent")
        f.update_layout(barmode="overlay")
    else:
        f.add_histogram(x=d[var], marker_color=RIO, nbinsx=50, histnorm="percent")
    f.update_xaxes(title=etiqueta)
    f.update_yaxes(title="Porcentaje de días")
    return f


def caja_mensual(d: pd.DataFrame, var: str, lim, etiqueta: str) -> go.Figure:
    f = _fig(f"Régimen estacional y valores atípicos: {etiqueta}", 360)
    meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    m = d.fecha.dt.month
    for i in range(1, 13):
        f.add_box(y=d.loc[m == i, var], name=meses[i - 1], marker_color=RIO, line_width=1.2,
                  boxpoints="outliers", marker=dict(size=3, color=ALE), showlegend=False)
    if lim is not None:
        f.add_hrect(y0=lim[0], y1=lim[1], fillcolor=SED, opacity=.12, line_width=0)
    f.update_yaxes(title=etiqueta)
    return f


def heatmap_correlacion(c: pd.DataFrame, titulo: str) -> go.Figure:
    f = _fig(titulo, 470)
    f.add_heatmap(z=c.values, x=c.columns, y=c.index, colorscale=ESCALA_DIV, zmin=-1, zmax=1,
                  text=np.round(c.values, 2), texttemplate="%{text}", textfont=dict(size=11),
                  hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>", colorbar=dict(title="ρ", thickness=12))
    f.update_layout(margin=dict(l=110, r=10, t=46, b=90))
    f.update_xaxes(showgrid=False, tickangle=-35)
    f.update_yaxes(showgrid=False, autorange="reversed")
    return f


def faltantes(d: pd.DataFrame, estaciones: list[str], nombres: dict) -> go.Figure:
    pct = (100 * d[estaciones].isna().mean()).sort_values()
    f = _fig("Días sin registro por estación en el periodo filtrado", 320)
    f.add_bar(y=[nombres.get(e, e) for e in pct.index], x=pct.values, orientation="h",
              marker_color=[ALE if v > 10 else CAU for v in pct.values],
              text=[es(f"{v:.1f} %") for v in pct.values], textposition="outside", cliponaxis=False,
              hovertemplate="%{y}: %{x:.2f} % sin dato<extra></extra>")
    f.update_xaxes(title="Días sin dato (%)", range=[0, max(1.0, float(pct.max())) * 1.18])
    f.update_layout(margin=dict(l=110, r=40, t=46, b=48))
    return f


def explorador(d: pd.DataFrame, x: str, y: str, color: str | None, etiquetas: dict) -> go.Figure:
    f = _fig(f"{etiquetas.get(y, y)} frente a {etiquetas.get(x, x)}", 440)
    muestra = d.sample(min(len(d), 4000), random_state=0) if len(d) > 4000 else d
    if color == "y":
        for clase, col, nombre in ((0, RIO, "Fuera de la banda"), (1, SED, "En la banda")):
            s = muestra[muestra.y == clase]
            f.add_scattergl(x=s[x], y=s[y], mode="markers", name=nombre,
                            marker=dict(size=5, color=col, opacity=.45))
    elif color:
        f.add_scattergl(x=muestra[x], y=muestra[y], mode="markers", showlegend=False,
                        marker=dict(size=5, color=muestra[color], colorscale=ESCALA_SEQ, opacity=.6,
                                    colorbar=dict(title=etiquetas.get(color, color), thickness=12)))
    else:
        f.add_scattergl(x=muestra[x], y=muestra[y], mode="markers", showlegend=False,
                        marker=dict(size=5, color=RIO, opacity=.4))
    f.update_xaxes(title=etiquetas.get(x, x))
    f.update_yaxes(title=etiquetas.get(y, y))
    return f


def acf_barras(valores, titulo: str, ic: float) -> go.Figure:
    k = np.arange(len(valores))
    f = _fig(titulo, 320)
    f.add_bar(x=k, y=valores, marker_color=RIO, hovertemplate="rezago %{x}: %{y:.3f}<extra></extra>")
    f.add_hrect(y0=-ic, y1=ic, fillcolor=NEU, opacity=.12, line_width=0)
    f.update_xaxes(title="Rezago (días)")
    f.update_yaxes(title="Correlación", range=[min(-0.2, float(np.min(valores)) - .05), 1.05])
    return f


def descomposicion(stl: pd.DataFrame) -> go.Figure:
    from plotly.subplots import make_subplots
    f = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.03,
                      subplot_titles=("Serie mensual", "Tendencia", "Estacionalidad", "Residuo"))
    for i, (c, col) in enumerate((("observado", RIO), ("tendencia", SED), ("estacional", CAU),
                                  ("residuo", NEU)), start=1):
        f.add_scatter(x=stl.fecha, y=stl[c], mode="lines", line=dict(color=col, width=1.5),
                      showlegend=False, row=i, col=1)
    f.update_layout(height=560, title="Descomposición STL del caudal mensual en Calamar",
                    margin=dict(l=56, r=18, t=70, b=40))
    f.update_yaxes(title_text="m³/s", row=1, col=1)
    return f


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def curva_roc(curvas: dict, auc: dict) -> go.Figure:
    f = _fig("Curva ROC en la prueba (2010-2026)", 380)
    f.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=NEU), name="Azar")
    for i, (nombre, (fpr, tpr)) in enumerate(curvas.items()):
        f.add_scatter(x=fpr, y=tpr, mode="lines", name=es(f"{nombre} (AUC {auc[nombre]:.3f})"),
                      line=dict(width=2.6 if i == 0 else 1.6, color=SECUENCIA[i % 5]))
    f.update_xaxes(title="Tasa de falsos positivos", range=[0, 1])
    f.update_yaxes(title="Tasa de verdaderos positivos (exhaustividad)", range=[0, 1.02])
    return f


def curva_pr(curvas: dict, ap: dict, prevalencia: float) -> go.Figure:
    f = _fig("Curva precisión-exhaustividad en la prueba", 380)
    f.add_hline(y=prevalencia, line_dash="dot", line_color=NEU,
                annotation_text=es(f"azar = prevalencia ({prevalencia:.2f})"))
    for i, (nombre, (rec, prec)) in enumerate(curvas.items()):
        f.add_scatter(x=rec, y=prec, mode="lines", name=es(f"{nombre} (AP {ap[nombre]:.3f})"),
                      line=dict(width=2.6 if i == 0 else 1.6, color=SECUENCIA[i % 5]))
    f.update_xaxes(title="Exhaustividad (recall)", range=[0, 1])
    f.update_yaxes(title="Precisión", range=[0, 1.02])
    return f


def matriz_confusion(cm: np.ndarray, titulo: str) -> go.Figure:
    f = _fig(titulo, 360)
    etiq = [["Verdaderos negativos", "Falsos positivos"], ["Falsos negativos", "Verdaderos positivos"]]
    texto = [[f"{etiq[i][j]}<br><b>{cm[i, j]:,}</b>" for j in range(2)] for i in range(2)]
    f.add_heatmap(z=cm, x=["Predicho: fuera", "Predicho: en banda"], y=["Real: fuera", "Real: en banda"],
                  colorscale=ESCALA_SEQ, showscale=False, text=texto, texttemplate="%{text}",
                  textfont=dict(size=13), hovertemplate="%{y} · %{x}: %{z:,}<extra></extra>")
    f.update_xaxes(showgrid=False)
    f.update_yaxes(showgrid=False, autorange="reversed")
    return f


def confiabilidad(tablas: dict) -> go.Figure:
    f = _fig("Diagrama de confiabilidad (calibración)", 380)
    f.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=NEU),
                  name="Calibración perfecta")
    for i, (nombre, t) in enumerate(tablas.items()):
        f.add_scatter(x=t.prob_predicha, y=t.frec_observada, mode="lines+markers", name=nombre,
                      line=dict(color=SECUENCIA[i % 5], width=2), marker=dict(size=7))
    f.update_xaxes(title="Probabilidad predicha", range=[0, 1])
    f.update_yaxes(title="Frecuencia observada", range=[0, 1])
    return f


def serie_prediccion(d: pd.DataFrame, nombre: str) -> go.Figure:
    f = _fig(f"Probabilidad predicha y días observados en la banda: {nombre}", 360)
    f.add_scatter(x=d.fecha, y=d.y, mode="lines", line=dict(color=SED, width=0), fill="tozeroy",
                  fillcolor="rgba(200,129,58,.25)", name="Día real en la banda", line_shape="hv")
    f.add_scatter(x=d.fecha, y=d.prob, mode="lines", line=dict(color=RIO, width=1.3),
                  name="Probabilidad predicha")
    f.update_yaxes(title="Probabilidad", range=[0, 1.02])
    f.update_xaxes(title="Fecha")
    return f


def shap_barras(df: pd.DataFrame, titulo: str, n: int = 15) -> go.Figure:
    d = df.sort_values("importancia").tail(n)
    f = _fig(titulo, 30 * len(d) + 90)
    f.add_bar(x=d.importancia, y=d.etiqueta, orientation="h", marker_color=RIO,
              hovertemplate="%{y}: %{x:.4f}<extra></extra>")
    f.update_xaxes(title="Media de |valor SHAP| (probabilidad)")
    f.update_layout(margin=dict(l=190, r=20, t=46, b=48))
    return f


def shap_cascada(phi: np.ndarray, nombres: list[str], base: float, valores: list, n: int = 10) -> go.Figure:
    orden = np.argsort(-np.abs(phi))
    top, resto = orden[:n], orden[n:]
    etiquetas = [f"{nombres[i][:42]} = {valores[i]}" for i in top]
    medidas = list(phi[top])
    if len(resto):
        etiquetas.append(f"Otras {len(resto)} variables")
        medidas.append(float(phi[resto].sum()))
    f = _fig("Explicación local (valores de Shapley)", 34 * (len(medidas) + 2) + 80)
    f.add_trace(go.Waterfall(
        orientation="h", base=base, y=["Valor esperado"] + etiquetas[::-1] + ["Predicción"],
        x=[0] + medidas[::-1] + [0], measure=["absolute"] + ["relative"] * len(medidas) + ["total"],
        increasing=dict(marker=dict(color=SED)), decreasing=dict(marker=dict(color=RIO)),
        totals=dict(marker=dict(color=CAU)), connector=dict(line=dict(color="#c9d3d8")),
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    f.update_xaxes(title="Probabilidad de estar en la banda")
    f.update_layout(margin=dict(l=330, r=20, t=60, b=48), showlegend=False)
    f.update_yaxes(automargin=True)
    return f


def diagrama_cd(rangos: pd.Series, cd: float, destacado: str | None = None, titulo: str = "") -> go.Figure:
    """Diagrama de diferencia crítica: rango promedio de cada modelo y barra CD de Nemenyi."""
    r = rangos.sort_values()
    k = len(r)
    f = _fig(titulo or "Diagrama de diferencia crítica (Nemenyi, α = 0.05)", 120 + 34 * k)
    y = np.arange(k)[::-1]
    colores = [SED if n == destacado else RIO for n in r.index]
    f.add_scatter(x=r.values, y=y, mode="markers+text", text=[f"{n}  ({v:.2f})" for n, v in r.items()],
                  textposition="middle right", marker=dict(size=12, color=colores),
                  hovertemplate="%{text}<extra></extra>", showlegend=False)
    # Grupos sin diferencia significativa: barras horizontales que unen modelos a menos de CD
    grupos = []
    vals = r.values
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in grupos):
            grupos.append((i, j))
    for g, (i, j) in enumerate(grupos):
        yy = -0.7 - 0.35 * g
        f.add_shape(type="line", x0=vals[i], x1=vals[j], y0=yy, y1=yy, line=dict(color=NEU, width=4))
    x0 = 1
    f.add_shape(type="line", x0=x0, x1=x0 + cd, y0=k - 0.2, y1=k - 0.2, line=dict(color=ALE, width=3))
    f.add_annotation(x=x0 + cd / 2, y=k + 0.25, text=f"CD = {cd:.2f}", showarrow=False,
                     font=dict(color=ALE))
    f.update_xaxes(title="Rango promedio (1 = mejor)", range=[0.7, k + 1.8])
    f.update_yaxes(visible=False, range=[-0.9 - 0.35 * len(grupos), k + 0.6])
    return f


def convergencia(hist: pd.DataFrame, titulo: str, x: str, ys: dict) -> go.Figure:
    f = _fig(titulo, 340)
    for i, (col, nombre) in enumerate(ys.items()):
        if col not in hist:
            continue
        f.add_scatter(x=hist[x], y=hist[col], mode="lines", name=nombre,
                      line=dict(color=SECUENCIA[i % 5], width=2))
    return f


def intervalo_caudal(q_actual: float, q_med: float, q_inf: float, q_sup: float, lim) -> go.Figure:
    f = _fig("Caudal previsto en t + 15 días (intervalo central del 80 %)", 220)
    f.add_vrect(x0=lim[0], x1=lim[1], fillcolor=SED, opacity=.15, line_width=0,
                annotation_text="Banda de descarga efectiva", annotation_position="top left",
                annotation_font_color="#8a5320")
    f.add_shape(type="line", x0=q_inf, x1=q_sup, y0=0, y1=0, line=dict(color=RIO, width=8))
    f.add_scatter(x=[q_med], y=[0], mode="markers", marker=dict(size=16, color=RIO, symbol="diamond"),
                  name="Mediana prevista", hovertemplate="Mediana: %{x:,.0f} m³/s<extra></extra>")
    f.add_scatter(x=[q_actual], y=[0], mode="markers", marker=dict(size=12, color=CAU, symbol="circle-open",
                                                                    line=dict(width=3)),
                  name="Caudal actual", hovertemplate="Actual: %{x:,.0f} m³/s<extra></extra>")
    lo = min(q_inf, lim[0], q_actual) * 0.85
    hi = max(q_sup, lim[1], q_actual) * 1.1
    f.update_xaxes(title="Caudal (m³/s)", range=[lo, hi])
    f.update_yaxes(visible=False, range=[-1, 1])
    f.update_layout(margin=dict(l=20, r=20, t=46, b=48))
    return f
