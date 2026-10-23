"""Sección 2: análisis exploratorio interactivo con filtros globales."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, callback, dcc, html

from src.config import NOMBRES_LEGIBLES
from src.data import caudales, es, etiqueta, json_dash, parametros, tabla, tarea
from src.plots import (acf_barras, balance_anual, caja_mensual, descomposicion, explorador, faltantes,
                       heatmap_correlacion, histograma, serie_caudal, vacia)

dash.register_page(__name__, path="/eda", name="EDA", title="EDA · Descarga efectiva", order=1)

P = parametros()
COLS = P["columnas"]
ESTACIONES = [e for e in caudales().columns if e != "fecha"]
ANIO_MIN, ANIO_MAX = 1947, int(caudales().fecha.dt.year.max())
VARIABLES = ["calamar"] + COLS + ["q_futuro"]
ETQ = {c: etiqueta(c) for c in VARIABLES + ["y"]}


def _insight(texto, id_):
    texto = [es(t) for t in texto] if isinstance(texto, list) else es(texto)
    if id_ is not None:
        return html.Div(texto, className="insight", id=id_)
    return html.Div(texto, className="insight") if texto else None


def tarjeta(titulo, cuerpo, insight=None, subtitulo=None, id_insight=None):
    return html.Div([html.Div(titulo, className="titulo"),
                     html.Div(subtitulo, className="subtitulo") if subtitulo else None,
                     cuerpo,
                     _insight(insight, id_insight)], className="tarjeta")


def grafico(id_):
    return dcc.Loading(dcc.Graph(id=id_, config={"displayModeBar": False}), type="dot",
                       color="#0B4F6C")


filtros = html.Div(dbc.Row([
    dbc.Col([html.Label("Periodo de análisis"),
             dcc.RangeSlider(id="f-anios", min=ANIO_MIN, max=ANIO_MAX, step=1,
                             value=[ANIO_MIN, ANIO_MAX],
                             marks={a: str(a) for a in range(ANIO_MIN, ANIO_MAX + 1, 10)},
                             tooltip={"placement": "bottom", "always_visible": False})], lg=5),
    dbc.Col([html.Label("Estaciones"),
             dcc.Dropdown(id="f-estaciones", multi=True, value=["calamar", "el_banco", "pto_salgar"],
                          options=[{"label": NOMBRES_LEGIBLES.get(e, e), "value": e} for e in ESTACIONES])],
            lg=4),
    dbc.Col([html.Label("Variable para distribución y régimen"),
             dcc.Dropdown(id="f-variable", value="calamar", clearable=False,
                          options=[{"label": ETQ[c], "value": c} for c in VARIABLES])], lg=3),
], className="g-3 align-items-end"), className="panel-filtros")


def layout():
    est = json_dash("estacionariedad.json")
    stl = tabla("stl.csv")
    stl["fecha"] = stl.fecha.astype("datetime64[ns]")
    fuga = tabla("fuga.csv").head(10)
    return html.Div([
        html.Div([html.H2("Análisis exploratorio"),
                  html.P("Los filtros superiores gobiernan todas las figuras de la página. Cada gráfico "
                         "incluye una lectura que resume qué implica para el modelado.")],
                 className="pagina-titulo mb-2"),
        filtros,
        dbc.Row([
            dbc.Col(tarjeta("Serie de caudal", grafico("g-serie"), id_insight="i-serie"), lg=12),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(tarjeta("Variable objetivo y balance de clases", grafico("g-balance"),
                            id_insight="i-balance"), lg=6),
            dbc.Col(tarjeta("Distribución por clase", grafico("g-hist"), id_insight="i-hist"), lg=6),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(tarjeta("Régimen estacional y atípicos", grafico("g-caja"), id_insight="i-caja"), lg=6),
            dbc.Col(tarjeta("Valores faltantes", grafico("g-faltantes"), id_insight="i-faltantes"), lg=6),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(tarjeta("Matriz de correlación entre estaciones", grafico("g-corr"),
                            id_insight="i-corr", subtitulo="Spearman sobre el periodo y las estaciones "
                                                           "seleccionadas"), lg=6),
            dbc.Col(tarjeta("Explorador bivariado", html.Div([
                dbc.Row([
                    dbc.Col(dcc.Dropdown(id="b-x", value="pto_salgar_t21", clearable=False,
                                         options=[{"label": ETQ[c], "value": c} for c in VARIABLES]), md=4),
                    dbc.Col(dcc.Dropdown(id="b-y", value="q_futuro", clearable=False,
                                         options=[{"label": ETQ[c], "value": c} for c in VARIABLES]), md=4),
                    dbc.Col(dcc.Dropdown(id="b-color", value="y", clearable=True,
                                         placeholder="Color: ninguno",
                                         options=[{"label": "Clase (en la banda)", "value": "y"}]
                                         + [{"label": ETQ[c], "value": c} for c in VARIABLES]), md=4),
                ], className="g-2 mb-1"),
                grafico("g-bivariado")]), id_insight="i-bivariado",
                subtitulo="Elija las variables de los ejes y del color"), lg=6),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(tarjeta("Dependencia temporal", html.Div([
                dcc.Graph(figure=acf_barras(np.array(est["acf"]), "Autocorrelación (ACF) del caudal en "
                                            "Calamar", 1.96 / np.sqrt(est["n"])),
                          config={"displayModeBar": False}),
                dcc.Graph(figure=acf_barras(np.array(est["pacf"]), "Autocorrelación parcial (PACF)",
                                            1.96 / np.sqrt(est["n"])), config={"displayModeBar": False}),
            ]), insight=[f"La autocorrelación vale {est['acf'][1]:.3f} a un día y {est['acf'][15]:.2f} a "
                         f"quince: el tamaño efectivo de muestra es muy inferior al número de filas. ADF "
                         f"(p = {est['p_adf']:.3g}) y KPSS (p = {est['p_kpss']:.2f}) se interpretan juntas "
                         "porque sus hipótesis nulas son opuestas. La PACF es significativa en los primeros "
                         "rezagos y luego decae, lo que justifica rezagos cortos y medias móviles como "
                         "predictores y obliga a usar particiones cronológicas con separación de 30 días."]),
                    lg=6),
            dbc.Col(tarjeta("Tendencia, estacionalidad y residuo",
                            dcc.Graph(figure=descomposicion(stl), config={"displayModeBar": False}),
                            insight=["El componente estacional es bimodal (máximo principal a final de año "
                                     "y secundario a mitad de año), reflejo del doble paso de la Zona de "
                                     "Convergencia Intertropical; por eso el modelo incluye armónicos anual y "
                                     "semianual. La tendencia oscila en escala interanual, coherente con la "
                                     "alternancia de El Niño y La Niña: en 1992 ningún día alcanzó la banda y "
                                     "en 1999 lo hizo más de la mitad, fuente de deriva entre periodos."]),
                    lg=6),
        ], className="g-3 mb-3"),
        tarjeta("Auditoría de fuga de datos", html.Div([
            dbc.Table.from_dataframe(fuga.rename(columns={"variable": "Variable", "AUC": "AUC univariado"})
                                     .assign(**{"AUC univariado": lambda d: d["AUC univariado"].round(3)}),
                                     striped=True, hover=True, size="sm", className="tabla-simple"),
        ]), insight=["Ninguna variable supera el umbral de alerta de 0,95 de AUC univariado. Las más "
                     "altas miden la distancia del caudal actual a la descarga efectiva, información "
                     "legítima disponible en t; el objetivo está quince días adelante."],
            subtitulo="Área bajo la curva ROC de cada predictor por separado, en el entrenamiento"),
    ])


def _filtrar(anios):
    d = tarea()
    return d[(d.fecha.dt.year >= anios[0]) & (d.fecha.dt.year <= anios[1])]


@callback(Output("g-serie", "figure"), Output("i-serie", "children"),
          Input("f-anios", "value"), Input("f-estaciones", "value"))
def act_serie(anios, estaciones):
    if not estaciones:
        return vacia("Seleccione al menos una estación"), "Seleccione al menos una estación."
    c = caudales()
    c = c[(c.fecha.dt.year >= anios[0]) & (c.fecha.dt.year <= anios[1])]
    m = c.set_index("fecha")[estaciones].resample("MS").mean().reset_index()
    dentro = c.calamar.between(*P["LIM"]).mean() * 100
    txt = (f"Entre {anios[0]} y {anios[1]} Calamar estuvo en la banda el {dentro:.1f} % de los días. Las "
           "estaciones del alto Magdalena registran caudales menores y picos más agudos, que la depresión "
           "momposina amortigua antes de llegar a Calamar.")
    return serie_caudal(m, estaciones, P["LIM"], NOMBRES_LEGIBLES), es(txt)


@callback(Output("g-balance", "figure"), Output("i-balance", "children"),
          Input("f-anios", "value"))
def act_balance(anios):
    d = _filtrar(anios)
    if d.empty:
        return vacia(), ""
    a = d.groupby(d.fecha.dt.year).y.mean()
    txt = (f"La clase positiva representa el {100 * d.y.mean():.1f} % de los {len(d):,} días "
           f"(desbalance {(1 - d.y.mean()) / d.y.mean():.1f}:1), pero varía de "
           f"{100 * a.min():.0f} % ({a.idxmin()}) a {100 * a.max():.0f} % ({a.idxmax()}) entre años. Por eso "
           "la métrica principal es la PR-AUC y la validación usa bloques de calendario.")
    return balance_anual(d), es(txt)


@callback(Output("g-hist", "figure"), Output("i-hist", "children"),
          Input("f-anios", "value"), Input("f-variable", "value"))
def act_hist(anios, var):
    d = _filtrar(anios).dropna(subset=[var])
    if d.empty:
        return vacia(), ""
    m1, m0 = d.loc[d.y == 1, var].median(), d.loc[d.y == 0, var].median()
    sk = d[var].skew()
    txt = (f"Mediana de {m1:,.2f} en los días en la banda frente a {m0:,.2f} fuera de ella; asimetría "
           f"de {sk:.2f}. Una separación parcial entre clases y su solapamiento indican que la variable "
           "aporta información pero no basta por sí sola.")
    return histograma(d, var, ETQ[var]), es(txt)


@callback(Output("g-caja", "figure"), Output("i-caja", "children"),
          Input("f-anios", "value"), Input("f-variable", "value"))
def act_caja(anios, var):
    d = _filtrar(anios).dropna(subset=[var])
    if d.empty:
        return vacia(), ""
    q1, q3 = d[var].quantile([.25, .75])
    atip = ((d[var] < q1 - 1.5 * (q3 - q1)) | (d[var] > q3 + 1.5 * (q3 - q1))).mean() * 100
    lim = P["LIM"] if var in ("calamar", "q_futuro") else None
    txt = (f"El {atip:.1f} % de los días queda fuera de 1,5 veces el rango intercuartílico. En hidrología "
           "esos valores corresponden a crecidas y estiajes reales, no a errores, por lo que se conservan "
           "y se usan modelos robustos a la escala (árboles) o escalado estándar.")
    return caja_mensual(d, var, lim, ETQ[var]), es(txt)


@callback(Output("g-faltantes", "figure"), Output("i-faltantes", "children"),
          Input("f-anios", "value"), Input("f-estaciones", "value"))
def act_faltantes(anios, estaciones):
    c = caudales()
    c = c[(c.fecha.dt.year >= anios[0]) & (c.fecha.dt.year <= anios[1])]
    est = estaciones or ESTACIONES
    pct = c[est].isna().mean() * 100
    txt = (f"Faltantes entre {pct.min():.1f} % y {pct.max():.1f} % en la selección. Antes de 1972-1975 la "
           "ausencia de las estaciones del bajo Magdalena es estructural (no existían); por eso el modelo "
           "principal usa solo Calamar y Puerto Salgar, con registro desde 1946, y no imputa años sin "
           "estación. Los huecos operativos restantes dependen del caudal de Calamar (mecanismo MAR) y se "
           "tratan con imputación dentro del pipeline.")
    return faltantes(c, est, NOMBRES_LEGIBLES), es(txt)


@callback(Output("g-corr", "figure"), Output("i-corr", "children"),
          Input("f-anios", "value"), Input("f-estaciones", "value"))
def act_corr(anios, estaciones):
    est = estaciones if estaciones and len(estaciones) > 1 else ESTACIONES
    c = caudales()
    c = c[(c.fecha.dt.year >= anios[0]) & (c.fecha.dt.year <= anios[1])][est]
    m = c.corr(method="spearman")
    m.index = m.columns = [NOMBRES_LEGIBLES.get(e, e) for e in est]
    v = m.where(~np.eye(len(m), dtype=bool)).stack()
    txt = (f"La correlación más alta es {v.max():.2f} ({v.idxmax()[0]}–{v.idxmax()[1]}) y la más baja "
           f"{v.min():.2f}. El alto y el bajo Magdalena forman dos bloques débilmente acoplados: entre ellos "
           "entran el Cauca y el Cesar, y la colinealidad entre estaciones vecinas es física, no un defecto.")
    return heatmap_correlacion(m, "Correlación de Spearman"), es(txt)


@callback(Output("g-bivariado", "figure"), Output("i-bivariado", "children"),
          Input("f-anios", "value"), Input("b-x", "value"), Input("b-y", "value"), Input("b-color", "value"))
def act_bivariado(anios, x, y, color):
    cols = list({x, y, "y"} | ({color} if color else set()))
    d = _filtrar(anios)[cols].dropna()
    if d.empty:
        return vacia(), ""
    rho = d[[x, y]].corr(method="spearman").iloc[0, 1]
    txt = (f"Correlación de Spearman = {rho:.2f} sobre {len(d):,} días (se grafica una muestra de hasta "
           "4.000 puntos). Coloree por la clase para ver dónde se concentran los días en la banda.")
    return explorador(d, x, y, color, ETQ), es(txt)
