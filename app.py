"""Punto de entrada del dashboard: crea la aplicación multipágina y expone ``server``.

Local:      python app.py
Producción: gunicorn app:server
"""
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

from src.data import parametros

VERSION = "v3.0"

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Descarga efectiva · Río Magdalena",
    update_title=None,
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"},
               {"name": "description", "content": "Predicción a 15 días de la descarga efectiva "
                "del río Magdalena en Calamar con modelos de machine learning."}],
)
server = app.server

enlaces = [("Contexto", "/"), ("EDA", "/eda"), ("Modelos", "/modelos")]

navbar = dbc.Navbar(
    dbc.Container([
        dcc.Link(html.Span([html.Img(src=dash.get_asset_url("logo.svg"), alt="Logotipo: onda de río"),
                            html.Span("Descarga efectiva · Río Magdalena")], className="navbar-brand"),
                 href="/", style={"textDecoration": "none", "color": "#fff"}),
        dbc.NavbarToggler(id="toggler", n_clicks=0),
        dbc.Collapse(
            dbc.Nav([dbc.NavItem(dbc.NavLink(t, href=h, active="exact")) for t, h in enlaces]
                    + [html.Span(VERSION, className="version ms-lg-3 align-self-center")],
                    className="ms-auto align-items-lg-center", navbar=True),
            id="collapse", is_open=False, navbar=True),
    ], fluid="xl"),
    dark=True, fixed="top", className="barra",
)

p = parametros()
pie = html.Footer(dbc.Container([
    html.Div([
        html.Span("Kevin Clemente Rosario · Manuel Santiago Meza Castro · Universidad del Norte · "
                  "Machine Learning 2026-30"),
        html.Span(f" · Datos: IDEAM (DHIME) · Actualizado {p.get('fecha_actualizacion', '')} · {VERSION}"),
    ]),
], fluid="xl"), className="pie")

app.layout = html.Div([
    navbar,
    dbc.Container([dash.page_container, pie], fluid="xl", className="contenido"),
])


@app.callback(Output("collapse", "is_open"), Input("toggler", "n_clicks"),
              State("collapse", "is_open"), prevent_initial_call=True)
def alternar_menu(n, abierto):
    return not abierto


if __name__ == "__main__":
    app.run(debug=True)
