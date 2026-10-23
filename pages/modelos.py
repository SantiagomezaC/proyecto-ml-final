"""Sección 3: modelos, predicción en vivo, diagnóstico, comparación estadística y modelo original."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dash_table, dcc, html, no_update
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

from src.data import (es, etiqueta, json_dash, metadata, modelo, modelos_disponibles, parametros, probabilidad,
                      tabla)
from src.evaluation import diagrama_confiabilidad, shapley_permutaciones
from src.models import NOMBRES_MODELO
from src.plots import (confiabilidad, convergencia, curva_pr, curva_roc, diagrama_cd, intervalo_caudal,
                       matriz_confusion, serie_prediccion, shap_barras, shap_cascada, vacia)

dash.register_page(__name__, path="/modelos", name="Modelos", title="Modelos · Descarga efectiva", order=2)

P = parametros()
COLS = P["columnas"]
MODELOS = modelos_disponibles()
OPC_MODELOS = [{"label": NOMBRES_MODELO.get(m, m), "value": m} for m in MODELOS]
PRINCIPALES = P.get("variables_formulario", COLS[:8])


def _insight(texto, id_):
    texto = [es(t) for t in texto] if isinstance(texto, list) else es(texto)
    if id_ is not None:
        return html.Div(texto, className="insight", id=id_)
    return html.Div(texto, className="insight") if texto else None


def tarjeta(titulo, cuerpo, insight=None, subtitulo=None, id_insight=None):
    return html.Div([html.Div(titulo, className="titulo"),
                     html.Div(subtitulo, className="subtitulo") if subtitulo else None, cuerpo,
                     _insight(insight, id_insight)], className="tarjeta")


def grafico(id_):
    return dcc.Loading(dcc.Graph(id=id_, config={"displayModeBar": False}), type="dot", color="#0B4F6C")


# ---------------------------------------------------------------------------
# Pestaña 1: tabla de líderes
# ---------------------------------------------------------------------------
def tab_lideres():
    lid = tabla("lideres.csv").copy()
    fmt = lambda m, s: f"{m:.3f} ± {s:.3f}"
    filas = pd.DataFrame({
        "Modelo": lid.nombre, "Balanceo": lid.balanceo_nombre, "Optimizador": lid.optimizador_nombre,
        "PR-AUC externo": [fmt(a, b) for a, b in zip(lid.pr_auc_mean, lid.pr_auc_std)],
        "ROC-AUC externo": [fmt(a, b) for a, b in zip(lid.roc_auc_mean, lid.roc_auc_std)],
        "F1 externo": [fmt(a, b) for a, b in zip(lid.f1_mean, lid.f1_std)],
        "Brier externo": [fmt(a, b) for a, b in zip(lid.brier_mean, lid.brier_std)],
        "PR-AUC prueba": lid.pr_auc_test.round(3), "ROC-AUC prueba": lid.roc_auc_test.round(3),
        "Inferencia (ms/fila)": lid.t_inf_ms.round(4),
        "_orden": lid.pr_auc_mean})
    filas = filas.sort_values("_orden", ascending=False).drop(columns="_orden")
    est_cond = [{"if": {"filter_query": '{Modelo} contains "BLGB"'},
                 "backgroundColor": "#f8efe5", "fontWeight": 700}]
    maestra = tabla("maestra_resumen.csv")
    return html.Div([
        tarjeta("Mejores combinaciones por modelo", dash_table.DataTable(
            data=filas.to_dict("records"), columns=[{"name": c, "id": c} for c in filas.columns],
            sort_action="native", filter_action="native", style_data_conditional=est_cond,
            style_cell={"padding": "7px 10px", "textAlign": "center"},
            style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "left"}
                                    for c in ("Modelo", "Balanceo", "Optimizador")],
            style_table={"overflowX": "auto"}),
            subtitulo="Métricas del bucle externo de la validación anidada (media ± desviación en 12 bloques "
                      "de dos años, 1986-2009) y de la prueba reservada 2010-2026. Ordenable y filtrable.",
            insight=P.get("texto_lideres", "")),
        html.Div(className="mb-3"),
        tarjeta("Las 112 combinaciones del Entregable 2", dcc.Graph(
            figure=_fig_combinaciones(maestra), config={"displayModeBar": False}),
            subtitulo="PR-AUC externa media de cada combinación modelo × balanceo × optimizador",
            insight="La elección del algoritmo pesa mucho más que la del balanceo o el optimizador: las "
                    "combinaciones se agrupan por modelo y, dentro de cada uno, las diferencias entre "
                    "técnicas son del orden de la variabilidad entre bloques."),
    ])


def _fig_combinaciones(m):
    import plotly.graph_objects as go
    from src.plots import SECUENCIA
    f = go.Figure()
    simbolos = {"grid": "square", "random": "circle", "bayes": "diamond", "genetico": "triangle-up"}
    for i, (b, g) in enumerate(m.groupby("balanceo_nombre")):
        f.add_scatter(x=g.pr_auc_mean, y=g.nombre, mode="markers", name=b,
                      marker=dict(size=10, color=SECUENCIA[i % 5], opacity=.85,
                                  symbol=[simbolos.get(o, "circle") for o in g.optimizador]),
                      customdata=np.column_stack([g.optimizador_nombre, g.pr_auc_std]),
                      hovertemplate="%{y} · " + b + " · %{customdata[0]}<br>PR-AUC %{x:.3f} ± "
                                                    "%{customdata[1]:.3f}<extra></extra>")
    f.update_layout(height=420, title="Cada punto es una combinación (símbolo = optimizador)",
                    margin=dict(l=150, r=20, t=60, b=48))
    f.update_xaxes(title="PR-AUC externa media")
    return f


# ---------------------------------------------------------------------------
# Pestaña 2: predicción en vivo
# ---------------------------------------------------------------------------
def _entrada(col, valor=None):
    esq = {e["nombre"]: e for e in metadata(MODELOS[0])["esquema_entrada"]}[col]
    paso = "any"
    return dbc.Col([
        dbc.Label(etiqueta(col), html_for=f"in-{col}", className="small fw-semibold mb-1"),
        dbc.Input(id={"tipo": "entrada", "col": col}, type="number", value=valor, step=paso,
                  placeholder=f"{esq['p005']:,.1f} a {esq['p995']:,.1f}", debounce=True, size="sm"),
    ], md=6, lg=3, className="mb-2")


def tab_prediccion():
    pred = tabla("pred_test.parquet")
    fechas = pred.fecha
    return html.Div([
        dbc.Row([
            dbc.Col(tarjeta("1. Elija el modelo y un día de partida", html.Div([
                dbc.Label("Modelo", className="small fw-semibold"),
                dcc.Dropdown(id="p-modelo", options=OPC_MODELOS, value=MODELOS[0], clearable=False),
                dbc.Label("Cargar las condiciones observadas de un día (2010-2026)",
                          className="small fw-semibold mt-3"),
                dcc.DatePickerSingle(id="p-fecha", min_date_allowed=fechas.min(),
                                     max_date_allowed=fechas.max(), date="2011-05-01",
                                     display_format="YYYY-MM-DD", className="d-block"),
                html.Div("El formulario se llena con los valores reales de ese día; puede editar cualquier "
                         "variable para explorar escenarios.", className="small text-muted mt-2"),
            ])), lg=4),
            dbc.Col(tarjeta("2. Ajuste las variables de entrada", html.Div([
                html.Div("Variables con mayor contribución media (SHAP) del modelo original:",
                         className="small text-muted mb-2"),
                dbc.Row([_entrada(c) for c in PRINCIPALES], className="g-2"),
                dbc.Accordion([dbc.AccordionItem(
                    dbc.Row([_entrada(c) for c in COLS if c not in PRINCIPALES], className="g-2"),
                    title=f"Las demás {len(COLS) - len(PRINCIPALES)} variables")],
                    start_collapsed=True, className="mt-2"),
                dbc.Button("Predecir", id="p-boton", color="primary", className="mt-3 px-4"),
            ])), lg=8),
        ], className="g-3 mb-3"),
        html.Div(id="p-alertas"),
        dbc.Row([
            dbc.Col(html.Div(id="p-resultado"), lg=4),
            dbc.Col(tarjeta("Caudal previsto por el modelo original", grafico("p-intervalo"),
                            id_insight="p-intervalo-txt"), lg=8),
        ], className="g-3 mb-3"),
        tarjeta("Por qué el modelo predice esto", grafico("p-shap"), id_insight="p-shap-txt",
                subtitulo="Valores de Shapley por permutaciones (40 permutaciones antitéticas, fondo de 40 "
                          "días del entrenamiento)"),
    ])


@callback(Output({"tipo": "entrada", "col": dash.ALL}, "value"), Input("p-fecha", "date"))
def precargar(fecha):
    pred = tabla("pred_test.parquet")
    d = tabla("tarea_test.parquet")
    fila = d[d.fecha == pd.Timestamp(fecha)]
    if fila.empty:
        return [no_update] * len(COLS)
    orden = [c for c in PRINCIPALES] + [c for c in COLS if c not in PRINCIPALES]
    vals = fila.iloc[0]
    return [None if pd.isna(vals[c]) else round(float(vals[c]), 4) for c in orden]


@callback(Output("p-resultado", "children"), Output("p-intervalo", "figure"),
          Output("p-intervalo-txt", "children"), Output("p-shap", "figure"), Output("p-shap-txt", "children"),
          Output("p-alertas", "children"),
          Input("p-boton", "n_clicks"), Input("p-modelo", "value"),
          State({"tipo": "entrada", "col": dash.ALL}, "value"), State("p-fecha", "date"))
def predecir(_, nombre, valores, fecha):
    orden = [c for c in PRINCIPALES] + [c for c in COLS if c not in PRINCIPALES]
    if not valores or all(v is None for v in valores):
        return (html.Div(), vacia("Cargue un día o complete el formulario"), "", vacia(""), "",
                dbc.Alert("Elija un día de partida para llenar el formulario.", color="info"))
    entrada = dict(zip(orden, valores))
    alertas = []
    esquema = {e["nombre"]: e for e in metadata(nombre)["esquema_entrada"]}
    fuera = [etiqueta(c) for c, v in entrada.items()
             if v is not None and not (esquema[c]["min"] <= v <= esquema[c]["max"])]
    if fuera:
        alertas.append(dbc.Alert(["Valores fuera del rango observado en el entrenamiento (la predicción es "
                                  "una extrapolación): ", ", ".join(fuera[:5]),
                                  "…" if len(fuera) > 5 else ""], color="warning"))
    ref = entrada.get("calamar_cuad")
    if ref is None or ref <= 0:
        alertas.append(dbc.Alert("El caudal actual al cuadrado (calamar_cuad) debe ser positivo: es la "
                                 "referencia del modelo original.", color="danger"))
        return html.Div(), vacia(""), "", vacia(""), "", alertas
    X = pd.DataFrame([{c: (np.nan if entrada[c] is None else float(entrada[c])) for c in COLS}])
    try:
        p = float(probabilidad(nombre, X)[0])
        orig = modelo("original")
        dist = orig.predict_distribucion(X)
    except Exception as e:      # la aplicación no se detiene ante una entrada inválida
        alertas.append(dbc.Alert(f"No fue posible predecir con esa entrada: {e}", color="danger"))
        return html.Div(), vacia(""), "", vacia(""), "", alertas
    q_act = float(np.sqrt(ref))
    en = p >= 0.5
    res = html.Div([
        html.Div("Probabilidad de estar en la banda dentro de 15 días", className="lbl"),
        html.Div(f"{100 * p:.1f} %", className="prob my-2"),
        html.Div(("Se anticipa transporte dominante" if en else "No se anticipa transporte dominante")
                 + f" · {NOMBRES_MODELO.get(nombre, nombre)}", className="fw-semibold"),
        html.Div(f"Caudal actual: {q_act:,.0f} m³/s · banda {P['LIM'][0]:,.0f}–{P['LIM'][1]:,.0f} m³/s",
                 className="small mt-2", style={"opacity": .9}),
    ], className=f"resultado {'si' if en else 'no'}")
    fig_int = intervalo_caudal(q_act, float(dist["mediana"][0]), float(dist["inferior"][0]),
                               float(dist["superior"][0]), P["LIM"])
    txt_int = (f"BLGB modela la distribución del caudal futuro: mediana {dist['mediana'][0]:,.0f} m³/s e "
               f"intervalo del 80 % entre {dist['inferior'][0]:,.0f} y {dist['superior'][0]:,.0f} m³/s. La "
               "probabilidad de banda es el área de esa distribución dentro de la franja ocre.")
    fondo = tabla("fondo.parquet")[COLS]
    med = fondo.median()
    Xi = X.fillna(med)
    phi, base = shapley_permutaciones(lambda Z: probabilidad(nombre, pd.DataFrame(Z, columns=COLS)),
                                      Xi.values[0], fondo.fillna(med).values, n_perm=40)
    vals = [f"{v:,.2f}" for v in Xi.values[0]]
    fig_shap = shap_cascada(phi, [etiqueta(c) for c in COLS], base, vals)
    i = int(np.argmax(np.abs(phi)))
    txt_shap = (f"Partiendo de la probabilidad media {100 * base:.1f} %, la variable que más mueve esta "
                f"predicción es «{etiqueta(COLS[i])}» ({100 * phi[i]:+.1f} puntos). Las barras ocres empujan "
                "hacia la banda; las azules, en contra.")
    return res, fig_int, es(txt_int), fig_shap, es(txt_shap), alertas


# ---------------------------------------------------------------------------
# Pestaña 3: diagnóstico
# ---------------------------------------------------------------------------
def tab_diagnostico():
    return html.Div([
        dbc.Row([dbc.Col([dbc.Label("Modelo a diagnosticar", className="small fw-semibold"),
                          dcc.Dropdown(id="d-modelo", options=OPC_MODELOS, value=MODELOS[0],
                                       clearable=False)], md=4)], className="mb-3"),
        dbc.Row([
            dbc.Col(tarjeta("Matriz de confusión (umbral 0,5)", grafico("d-cm"), id_insight="d-cm-txt"), lg=4),
            dbc.Col(tarjeta("Curva ROC", grafico("d-roc")), lg=4),
            dbc.Col(tarjeta("Curva precisión-exhaustividad", grafico("d-pr")), lg=4),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(tarjeta("Calibración", grafico("d-cal"), id_insight="d-cal-txt"), lg=5),
            dbc.Col(tarjeta("Interpretabilidad global", grafico("d-shap"), id_insight="d-shap-txt",
                            subtitulo="Media del valor absoluto de Shapley sobre 300 días de la prueba"), lg=7),
        ], className="g-3 mb-3"),
        tarjeta("Probabilidad predicha en el tiempo", grafico("d-serie"),
                insight="La banda ocre marca los días que realmente estuvieron en el rango de descarga "
                        "efectiva; un buen modelo eleva la probabilidad antes de que cada episodio comience."),
    ])


@callback(Output("d-cm", "figure"), Output("d-cm-txt", "children"), Output("d-roc", "figure"),
          Output("d-pr", "figure"), Output("d-cal", "figure"), Output("d-cal-txt", "children"),
          Output("d-shap", "figure"), Output("d-shap-txt", "children"), Output("d-serie", "figure"),
          Input("d-modelo", "value"))
def diagnosticar(nombre):
    pred = tabla("pred_test.parquet")
    y = pred.y.values
    p = pred[f"p_{nombre}"].values
    cm = confusion_matrix(y, (p >= 0.5).astype(int))
    tn, fp, fn, tp = cm.ravel()
    txt_cm = (f"Detecta {tp:,} de {tp + fn:,} días en la banda (exhaustividad {tp / (tp + fn):.3f}) con "
              f"{fp:,} falsas alarmas (precisión {tp / max(tp + fp, 1):.3f}).")
    comparar = [nombre] + [m for m in ("original", "xgboost", "logistica") if m != nombre][:2]
    roc_c, pr_c, auc, ap = {}, {}, {}, {}
    for m in comparar:
        s = pred[f"p_{m}"].values
        n = NOMBRES_MODELO.get(m, m)
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        k1, k2 = max(1, len(fpr) // 400), max(1, len(prec) // 400)
        roc_c[n], pr_c[n] = (fpr[::k1], tpr[::k1]), (rec[::k2], prec[::k2])
        auc[n], ap[n] = roc_auc_score(y, s), average_precision_score(y, s)
    tablas = {NOMBRES_MODELO.get(m, m): diagrama_confiabilidad(y, pred[f"p_{m}"].values) for m in comparar}
    cal = tabla("calibracion.csv").set_index("modelo")
    c = cal.loc[nombre]
    txt_cal = (f"Brier = {c.brier:.4f} y ECE = {c.ece:.4f} en la prueba"
               + (f"; la recalibración de Platt, ajustada con las predicciones fuera de muestra del bucle "
                  f"externo, lleva el ECE de {c.ece_crudo:.4f} a {c.ece:.4f}." if c.recalibrado else "."))
    sh = tabla("shap_global.csv")
    sh = sh[sh.modelo == nombre].assign(etiqueta=lambda d: d.variable.map(etiqueta))
    top = sh.sort_values("importancia", ascending=False).head(3)
    txt_sh = ("Variables más influyentes: " + "; ".join(f"{e} ({v:.3f})" for e, v in
                                                      zip(top.etiqueta, top.importancia)) + ".")
    serie = pd.DataFrame({"fecha": pred.fecha, "y": y, "prob": p})
    return (matriz_confusion(cm, NOMBRES_MODELO.get(nombre, nombre)), es(txt_cm), curva_roc(roc_c, auc),
            curva_pr(pr_c, ap, y.mean()), confiabilidad(tablas), es(txt_cal),
            shap_barras(sh, f"Importancia SHAP: {NOMBRES_MODELO.get(nombre, nombre)}"), es(txt_sh),
            serie_prediccion(serie, NOMBRES_MODELO.get(nombre, nombre)))


# ---------------------------------------------------------------------------
# Pestaña 4: comparación estadística
# ---------------------------------------------------------------------------
def tab_comparacion():
    cd = json_dash("cd.json")
    rangos = pd.Series(cd["rangos"])
    rangos.index = [NOMBRES_MODELO.get(m, m) for m in rangos.index]
    cd112 = json_dash("cd_112.json")
    mcs = tabla("mcs_regresion.csv")
    return html.Div([
        dbc.Row([
            dbc.Col(tarjeta("Diagrama de diferencia crítica", dcc.Graph(
                figure=diagrama_cd(rangos, cd["cd"], NOMBRES_MODELO["original"]),
                config={"displayModeBar": False}),
                subtitulo=f"Mejor combinación de cada modelo y modelo original; {cd['n_bloques']} bloques "
                          "externos, PR-AUC",
                insight=(f"Friedman: χ² = {cd['friedman']['chi2']:.2f}, p = {cd['friedman']['p']:.2g} "
                         f"(Iman-Davenport F = {cd['friedman']['F_iman_davenport']:.2f}). Los modelos unidos "
                         f"por una barra gris difieren en menos de CD = {cd['cd']:.2f} rangos y no son "
                         "distinguibles según Nemenyi.")), lg=7),
            dbc.Col(tarjeta("Prueba global sobre las 112 combinaciones", html.Div([
                html.Table(html.Tbody([
                    html.Tr([html.Th("Estadístico de Friedman χ²"), html.Td(f"{cd112['chi2']:.1f}")]),
                    html.Tr([html.Th("Grados de libertad"), html.Td(cd112["gl"])]),
                    html.Tr([html.Th("p-valor"), html.Td(f"{cd112['p']:.2g}")]),
                    html.Tr([html.Th("Diferencia crítica (Nemenyi)"), html.Td(f"{cd112['cd']:.1f} rangos")]),
                    html.Tr([html.Th("Pares significativos"),
                             html.Td(f"{cd112['pares_signif']:,} de {cd112['pares_total']:,}")]),
                ]), className="table table-sm tabla-simple"),
            ]), insight="Con 112 modelos y 12 bloques la diferencia crítica es tan amplia que solo los "
                        "extremos del ranking se separan; por eso las comparaciones dirigidas se hacen entre "
                        "finalistas y con corrección de Holm."), lg=5),
        ], className="g-3 mb-3"),
        tarjeta("Tarea de regresión: Model Confidence Set (α = 0,10)", dash_table.DataTable(
            data=mcs.to_dict("records"), columns=[{"name": c, "id": c} for c in mcs.columns],
            sort_action="native", style_cell={"padding": "6px 10px"}, style_table={"overflowX": "auto"}),
            subtitulo="Pérdida cuadrática diaria de las predicciones del bucle externo; bootstrap estacionario",
            insight=P.get("texto_mcs", "")),
    ])


# ---------------------------------------------------------------------------
# Pestaña 5: modelo original
# ---------------------------------------------------------------------------
PSEUDO = """Entrada: X (n×p), y ∈ {0,1}ⁿ, caudal futuro q, referencia Q_ref, banda [L, U], λ, η, T, d
Salida: ensambles F_μ y F_s;  p(x) = Φ((u−μ)/σ) − Φ((l−μ)/σ)

1  z ← log(q / Q_ref);  l ← log(L / Q_ref);  u ← log(U / Q_ref)
2  μ ← media(z);  s ← log desv(z)                  ▷ s = log σ, acotado a [log σ_min, log σ_max]
3  Discretizar cada columna de X en B bins por cuantiles
4  para t = 1, …, T:
5      calcular g_μ, h_μ, g_s, h_s de  ℒ = BCE(y, p) + λ·NLL(z | μ, σ)
6      f_μ ← ÁrbolNewton(X, g_μ, h_μ, d);  f_s ← ÁrbolNewton(X, g_s, h_s, d)
7      η_t ← η;  mientras ℒ(μ + η_t f_μ, s + η_t f_s) > ℒ(μ, s) + c·η_t·∇ℒᵀf:  η_t ← η_t / 2
8      si no hubo descenso en 8 intentos: terminar
9      μ ← μ + η_t f_μ;  s ← clip(s + η_t f_s)
10     si la BCE de validación no mejora en P rondas: terminar
11 devolver F_μ, F_s (truncados en la mejor ronda de validación)"""


def tab_original():
    pr = tabla("pruebas_original.csv")
    ab = tabla("ablacion.csv")
    lit = tabla("literatura_cercana.csv")
    conv = tabla("conv_boosting.csv")
    anytime = tabla("anytime_original.csv")
    ver = json_dash("veredicto.json")
    return html.Div([
        dbc.Row([
            dbc.Col(tarjeta("Band-Likelihood Gradient Boosting (BLGB)", html.Div([
                html.P("Los clasificadores convencionales solo ven si cada día estuvo o no en la banda. BLGB "
                       "aprovecha además cuánto se alejó el caudal: modela la distribución condicional del "
                       "cambio logarítmico del caudal a 15 días, con media y dispersión dependientes de las "
                       "condiciones actuales, y calcula la probabilidad de banda como el área de esa "
                       "distribución entre los límites, que se trasladan a cada observación."),
                html.P("La media y el logaritmo de la dispersión son dos ensambles de árboles ajustados por "
                       "boosting de segundo orden sobre una pérdida híbrida: la entropía cruzada del evento "
                       "de interés más λ veces la verosimilitud gaussiana del caudal futuro. Para λ > 0 la "
                       "pérdida es una regla de puntuación estrictamente propia, y la búsqueda lineal de "
                       "Armijo garantiza que la pérdida de entrenamiento decrece en cada ronda."),
                html.P("Todo el núcleo (discretización, histogramas de gradiente, ganancia de segundo orden, "
                       "búsqueda lineal y parada temprana) está programado en numpy, sin librerías de "
                       "boosting.", className="mb-0"),
            ])), lg=6),
            dbc.Col(tarjeta("Pseudocódigo", html.Pre(PSEUDO, className="pseudo mb-0")), lg=6),
        ], className="g-3 mb-3"),
        html.Div(ver["texto"], className=f"veredicto {'ok' if ver['superior'] else 'no'} mb-3"),
        tarjeta("Pruebas de hipótesis frente al mejor benchmark", dash_table.DataTable(
            data=pr.to_dict("records"), columns=[{"name": c, "id": c} for c in pr.columns],
            style_cell={"padding": "6px 10px", "whiteSpace": "normal", "textAlign": "left",
                        "minWidth": "90px"}, style_table={"overflowX": "auto"}),
            subtitulo="p-valores crudos y ajustados por Holm; tamaño del efecto con δ de Cliff o d de Cohen"),
        html.Div(className="mb-3"),
        dbc.Row([
            dbc.Col(tarjeta("Convergencia del boosting", dcc.Graph(
                figure=convergencia(conv, "Pérdida por ronda (modelo final)", "iter",
                                    {"perdida_train": "Pérdida híbrida de entrenamiento",
                                     "bce_validacion": "Entropía cruzada de validación"}).update_xaxes(
                    title="Ronda de boosting").update_yaxes(title="Pérdida", type="log"),
                config={"displayModeBar": False}),
                insight="La pérdida de entrenamiento decrece de forma monótona, como garantiza la búsqueda de "
                        "Armijo; la parada temprana corta donde la entropía cruzada de validación deja de "
                        "mejorar."), lg=6),
            dbc.Col(tarjeta("Optimización bayesiana del modelo original", dcc.Graph(
                figure=convergencia(anytime, "Mejor PR-AUC interna hasta la evaluación t", "eval",
                                    {"mejor": "BLGB (TPE)"}).update_xaxes(title="Evaluación").update_yaxes(
                    title="PR-AUC interna"), config={"displayModeBar": False}),
                insight="Curva de desempeño en cualquier momento (anytime) del pliegue más reciente."), lg=6),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(tarjeta("Ablación", dash_table.DataTable(
                data=ab.to_dict("records"), columns=[{"name": c, "id": c} for c in ab.columns],
                style_cell={"padding": "6px 10px", "textAlign": "left", "whiteSpace": "normal"}),
                subtitulo="Cada componente se retira por separado; PR-AUC externa media en los 12 bloques"),
                lg=6),
            dbc.Col(tarjeta("Diferencias con los métodos afines", dash_table.DataTable(
                data=lit.to_dict("records"), columns=[{"name": c, "id": c} for c in lit.columns],
                style_cell={"padding": "6px 10px", "textAlign": "left", "whiteSpace": "normal"},
                page_size=6, style_table={"overflowX": "auto"})), lg=6),
        ], className="g-3"),
    ])


def layout():
    return html.Div([
        html.Div([html.H2("Modelos y resultados"),
                  html.P("Los modelos se entrenaron fuera de línea; el tablero solo carga los artefactos "
                         "serializados. La prueba 2010-2026 nunca se usó para seleccionar ni ajustar.")],
                 className="pagina-titulo mb-3"),
        dbc.Tabs([
            dbc.Tab(tab_lideres(), label="Tabla de líderes", tab_id="t1", className="pt-3"),
            dbc.Tab(tab_prediccion(), label="Predicción en vivo", tab_id="t2", className="pt-3"),
            dbc.Tab(tab_diagnostico(), label="Diagnóstico", tab_id="t3", className="pt-3"),
            dbc.Tab(tab_comparacion(), label="Comparación estadística", tab_id="t4", className="pt-3"),
            dbc.Tab(tab_original(), label="Modelo original", tab_id="t5", className="pt-3"),
        ], active_tab="t1"),
    ])
