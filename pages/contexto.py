"""Sección 1: contexto del problema, pregunta de investigación, datos y metodología."""
import dash
import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from src.data import es, etiqueta, parametros, tabla, ubicaciones
from src.plots import magnitud_frecuencia, mapa_estaciones

dash.register_page(__name__, path="/", name="Contexto", title="Contexto · Descarga efectiva",
                   order=0)

P = parametros()


def kpi(valor, nombre, nota=None, clase=""):
    return html.Div([html.Div(nombre, className="nombre"), html.Div(es(valor), className="valor"),
                     html.Div(es(nota), className="nota") if nota else None], className=f"kpi {clase}")


def tarjeta(titulo, *hijos, subtitulo=None):
    hijos = [es(h) if isinstance(h, str) else h for h in hijos]
    return html.Div([html.Div(titulo, className="titulo"),
                     html.Div(subtitulo, className="subtitulo") if subtitulo else None, *hijos],
                    className="tarjeta")


def _diccionario():
    cols = P["columnas"]
    filas = [{"Variable": c, "Descripción": etiqueta(c),
              "Grupo": ("Estado actual y reciente de Calamar" if c.startswith("calamar")
                        else "Calendario" if c.startswith(("sen", "cos"))
                        else "Dominio (descarga efectiva)" if c.startswith("dist")
                        else "Estaciones aguas arriba")} for c in cols]
    filas = [{"Variable": "y", "Descripción": "Objetivo: 1 si Q(t+15) en Calamar cae en la banda "
              + es("[{:,.0f}; {:,.0f}] m³/s".format(*P["LIM"])), "Grupo": "Variable objetivo"}] + filas
    return dash_table.DataTable(
        data=filas, columns=[{"name": c, "id": c} for c in ("Variable", "Descripción", "Grupo")],
        page_size=10, sort_action="native", filter_action="native",
        style_cell={"textAlign": "left", "padding": "6px 10px", "whiteSpace": "normal"},
        style_table={"overflowX": "auto"})


AUTORES = [
    ("Kevin Clemente Rosario", "Geólogo · Estudiante de M.Sc.", "autor_kevin.jpg"),
    ("Manuel Santiago Meza Castro", "Estudiante de Geología y de M.Sc.", "autor_manuel.jpg"),
]


def autor(nombre, rol, foto):
    return html.Div([
        html.Img(src=dash.get_asset_url(foto), alt=f"Fotografía de {nombre}", className="autor-foto"),
        html.Div([html.Div(nombre, className="autor-nombre"), html.Div(rol, className="autor-rol")]),
    ], className="autor")


PASOS = [
    ("01", "ETL", "7 series diarias del IDEAM, consolidación y auditoría de la serie de transporte"),
    ("02", "EDA", "Magnitud-frecuencia, tránsito de crecidas, fuga y estructura temporal"),
    ("03", "140 modelos", "7 algoritmos × 4 balanceos × 4 optimizadores, más 28 de regresión"),
    ("04", "Validación anidada", "12 bloques externos de 2 años, validación interna cronológica"),
    ("05", "Comparación estadística", "Friedman-Nemenyi, Wilcoxon-Holm, Nadeau-Bengio, MCS"),
    ("06", "Modelo original", "Boosting de verosimilitud de banda (BLGB), programado en numpy"),
    ("07", "Producción", "Artefactos serializados, dashboard y despliegue en Render"),
]


def layout():
    lid = tabla("lideres.csv")
    orig = lid[lid.modelo == "original"].iloc[0] if (lid.modelo == "original").any() else None
    mejor_bench = lid[lid.modelo != "original"].sort_values("pr_auc_mean", ascending=False).iloc[0]
    mf = tabla("magnitud_frecuencia.csv")
    return html.Div([
        html.Section(dbc.Row([
            dbc.Col([
                html.Div([
                    html.Img(src=dash.get_asset_url("logo_uninorte_blanco.png"),
                             alt="Logotipo de la Universidad del Norte", className="hero-logo"),
                    html.Span("Proyecto de investigación · Machine Learning", className="etiqueta"),
                ], className="hero-marca mb-3"),
                html.H1("¿Cuándo volverá el Magdalena a mover la mayor parte de su sedimento?"),
                html.P("Este tablero predice, con quince días de anticipación, si el caudal del río en "
                       "Calamar caerá dentro de la banda de descarga efectiva: el rango de caudales que, "
                       "a lo largo de las décadas, transporta la mayor fracción de la carga sedimentaria "
                       "y modela la geometría del cauce.", className="lead mt-3"),
                dbc.Row([
                    dbc.Col(dcc.Link(dbc.Button("Explorar los datos", color="light", className="me-2 mt-2"),
                                     href="/eda"), width="auto"),
                    dbc.Col(dcc.Link(dbc.Button("Probar los modelos", outline=True, color="light",
                                                className="mt-2"), href="/modelos"), width="auto"),
                ], className="g-0"),
            ], lg=6, className="hero-texto"),
            dbc.Col(html.Figure([
                html.Img(src=dash.get_asset_url("transporte_sedimentos.png"), className="hero-img",
                         alt="Esquema del transporte de sedimentos en un cauce: aporte desde la cuenca, "
                             "transporte en suspensión de limos y arcillas, saltación de arenas y arrastre "
                             "de gravas y bloques en la dirección del flujo"),
                html.Figcaption("Mecanismos de transporte de sedimentos en un cauce aluvial",
                                className="hero-pie"),
            ], className="hero-figura mb-0"), lg=6),
        ], className="align-items-center g-4"), className="hero mb-4"),

        dbc.Row([
            dbc.Col(kpi(f"{P['Q_EF']:,.0f} m³/s", "Descarga efectiva",
                        f"Banda ±15 %: {P['LIM'][0]:,.0f}–{P['LIM'][1]:,.0f} m³/s"), md=3, sm=6),
            dbc.Col(kpi(f"{100 * P['prevalencia_train']:.1f} %", "Días en la banda",
                        "Clase positiva en el entrenamiento 1947-2009", "acento"), md=3, sm=6),
            dbc.Col(kpi("15 días", "Horizonte de predicción",
                        "Compromiso entre utilidad operativa y dificultad"), md=3, sm=6),
            dbc.Col(kpi(f"{orig.pr_auc_test:.3f}" if orig is not None else "—",
                        "PR-AUC del modelo original",
                        f"Mejor benchmark: {mejor_bench.nombre} {mejor_bench.pr_auc_test:.3f}",
                        "alerta"), md=3, sm=6),
        ], className="g-3 mb-4"),

        html.Section([
            html.Div("Autores", className="titulo-seccion"),
            dbc.Row([dbc.Col(autor(*a), md=6) for a in AUTORES], className="g-3"),
            html.P("Universidad del Norte, Barranquilla, Colombia", className="autores-pie"),
        ], className="autores mb-4"),

        dbc.Row([
            dbc.Col(tarjeta(
                "Por qué importa",
                html.P(["El Magdalena drena cerca de la cuarta parte de Colombia y entrega al Caribe del "
                        "orden de 144 millones de toneladas de sedimento al año, con un caudal medio "
                        "cercano a 7.200 m³/s, lo que lo sitúa entre los ríos de mayor rendimiento "
                        "sedimentario de Suramérica (Restrepo y Kjerfve, 2000). Ese sedimento condiciona "
                        "la vida útil de los embalses, la profundidad del canal navegable entre "
                        "Barrancabermeja y Barranquilla y el crecimiento del delta."]),
                html.P("La descarga efectiva describe qué caudal hace el trabajo geomorfológico a largo "
                       "plazo, pero no dice cuándo volverá a presentarse. Anticiparlo permite programar "
                       "purgas de embalses, dragados y campañas de muestreo de sedimentos en los días "
                       "de mayor transporte."),
                html.Div([html.B("Pregunta de investigación. "),
                          "¿Es posible clasificar, con la información hidrológica disponible hasta el "
                          "día t, si el día t+15 pertenecerá al rango de descarga efectiva del río "
                          "Magdalena en Calamar, y puede un modelo que aprovecha la naturaleza continua "
                          "del caudal superar con significancia estadística a los clasificadores "
                          "convencionales?"], className="insight"),
            ), lg=7),
            dbc.Col(tarjeta(
                "Objetivos",
                html.Ol([
                    html.Li("Estimar la descarga efectiva de Calamar y verificar su robustez frente a la "
                            "curva de gasto sólido."),
                    html.Li("Entrenar y comparar 140 combinaciones de modelos, balanceo y optimización "
                            "con validación anidada cronológica."),
                    html.Li("Diseñar e implementar un modelo original que use el caudal futuro como "
                            "supervisión continua y validarlo con al menos tres pruebas de hipótesis."),
                    html.Li("Poner los mejores modelos a disposición de usuarios no expertos mediante "
                            "este tablero."),
                ], className="mb-0"),
            ), lg=5),
        ], className="g-3 mb-4"),

        dbc.Row([
            dbc.Col(tarjeta("Red de estaciones del IDEAM", dcc.Graph(figure=mapa_estaciones(ubicaciones()),
                                                                     config={"displayModeBar": False}),
                            html.Div(["La onda de crecida tarda de 7 días (Coyongal) a 21 días (Puerto Salgar) en "
                                      "llegar a Calamar. Puerto Salgar, con registro desde 1946, entra al "
                                      "modelo principal desfasada 21 días; las estaciones del bajo "
                                      "Magdalena, instaladas en 1972-1975, se usan en las tres tareas "
                                      "adicionales para no imputar años sin registro."], className="insight"),
                            subtitulo="Cauce principal del Magdalena, de Guaduas (Cundinamarca) a Calamar "
                                      "(Bolívar)"), lg=6),
            dbc.Col(tarjeta("Cómo se definió la variable objetivo",
                            dcc.Graph(figure=magnitud_frecuencia(mf, P["Q_EF"]),
                                      config={"displayModeBar": False}),
                            html.Div([f"La frecuencia de los caudales decrece hacia los valores altos y el "
                                      f"transporte por día crece; su producto alcanza el máximo en "
                                      f"{P['Q_EF']:,.0f} m³/s. Las {len(P['Q_EF_ROBUSTEZ'])} curvas de gasto "
                                      "históricas del IDEAM conducen a la misma clase modal."],
                                     className="insight"),
                            subtitulo="Producto de la frecuencia de cada clase de caudal por el transporte "
                                      "de un día típico (Wolman y Miller, 1960)"), lg=6),
        ], className="g-3 mb-4"),

        tarjeta("Datos",
                dbc.Row([
                    dbc.Col([
                        html.Table([html.Tbody([
                            html.Tr([html.Th("Fuente"), html.Td("IDEAM, portal DHIME (caudal medio diario)")]),
                            html.Tr([html.Th("Periodo"), html.Td("1940-2026; modelado desde 1947 con "
                                                                 "Calamar y Puerto Salgar, sin imputar "
                                                                 "estaciones inexistentes")]),
                            html.Tr([html.Th("Observaciones"),
                                     html.Td(f"{P['n_train']:,} días de entrenamiento (1947-2009) y "
                                             f"{P['n_test']:,} de prueba (2010-2026)")]),
                            html.Tr([html.Th("Predictores"), html.Td(f"{len(P['columnas'])} variables")]),
                            html.Tr([html.Th("Tarea"), html.Td("Clasificación binaria; tarea auxiliar de "
                                                               "regresión del caudal en t+15")]),
                            html.Tr([html.Th("Licencia"), html.Td("Uso personal y no comercial; los archivos "
                                                                  "originales no se redistribuyen")]),
                        ])], className="table table-sm tabla-simple"),
                    ], lg=5),
                    dbc.Col(_diccionario(), lg=7),
                ]),
                subtitulo="Diccionario de variables: todas usan información disponible hasta el día t"),

        html.Div(className="mb-4"),
        tarjeta("Metodología",
                html.Div([html.Div([html.Div(n, className="n"), html.Div(t, className="t"),
                                    html.Div(d, className="d")],
                                   className="paso destacado" if n == "06" else "paso")
                          for n, t, d in PASOS], className="flujo"),
                subtitulo="Del registro crudo a la predicción en producción"),

        html.Div(className="mb-4"),
        dbc.Row([
            dbc.Col(tarjeta("Limitaciones y consideraciones éticas", html.Ul([
                html.Li("El 89,6 % del registro de Calamar tiene nivel de aprobación preliminar."),
                html.Li("La autocorrelación diaria (0,999) reduce el tamaño efectivo de muestra; por eso "
                        "toda partición es cronológica y los intervalos usan bootstrap por bloques."),
                html.Li("Sin precipitación ni índices ENSO, el desempeño cae en horizontes largos."),
                html.Li("La regulación por embalses y el cambio de uso del suelo pueden hacer que el pasado "
                        "no represente el futuro (deriva)."),
                html.Li("No hay datos personales; el riesgo ético principal es el uso operativo de una "
                        "predicción sin comunicar su incertidumbre, que el tablero siempre muestra."),
            ], className="mb-0")), lg=7),
            dbc.Col(tarjeta("Referencias",
                html.Ul([
                    html.Li("Andrews, E. D. (1980). Effective and bankfull discharges of streams in the "
                            "Yampa River basin. Journal of Hydrology, 46, 311-330."),
                    html.Li("Restrepo, J. D., y Kjerfve, B. (2000). Magdalena river: interannual variability "
                            "(1975-1995) and revised water discharge and sediment load estimates. Journal "
                            "of Hydrology, 235, 137-149."),
                    html.Li("Wolman, M. G., y Miller, J. P. (1960). Magnitude and frequency of forces in "
                            "geomorphic processes. The Journal of Geology, 68, 54-74."),
                ], className="small mb-0")), lg=5),
        ], className="g-3 mb-4"),
    ])
