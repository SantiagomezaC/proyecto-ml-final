---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python (ml-final)
  language: python
  name: ml-final
---

# Datos, análisis exploratorio y preparación

Este capítulo describe los datos, construye la variable objetivo y examina las propiedades de
las series que condicionan el diseño de los modelos: la cobertura de la red de estaciones, el
mecanismo de los datos faltantes, la naturaleza de la serie de transporte de sedimentos, la
dependencia temporal, la deriva entre periodos y la posibilidad de fuga de información. Cada
sección termina con la decisión de preparación que se deriva de su hallazgo, y la última reúne
esas decisiones. El procesamiento está implementado como módulos reproducibles (`src/etl.py`,
`src/features.py`), y las celdas de código de este capítulo, que pueden desplegarse con el botón
«Show code cell source», solo leen sus resultados y los analizan.

```{code-cell} ipython3
:tags: [hide-input]
# Configuración común del capítulo: rutas, estilo de las figuras, semilla y tablas procesadas
import sys, json, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from myst_nb import glue
from src import estilo
from src.config import DATOS_PROC, FECHA_CORTE_TEST, INICIO_MODELADO, SEED, NOMBRES_LEGIBLES, fijar_semillas
from src.estilo import RIO, SED, CAU, ALE, NEU
fijar_semillas()
caudales = pd.read_csv(DATOS_PROC / "caudales.csv", parse_dates=["fecha"])      # caudal diario de las 7 estaciones
par = json.loads((DATOS_PROC / "parametros.json").read_text(encoding="utf-8"))["calamar"]  # Q_ef, banda, tránsitos
tarea = pd.read_parquet(DATOS_PROC / "tarea_calamar.parquet")                    # tabla de modelado
EST = [e for e in caudales.columns if e != "fecha"]
NL = NOMBRES_LEGIBLES
```

## Fuente, estructura y ETL

Los datos provienen del portal DHIME del IDEAM {cite}`ideam2026` y corresponden a series de
caudal medio diario y de transporte medio diario de sedimentos en suspensión de siete
estaciones del cauce principal del Magdalena. La unidad de observación es el día en una
estación; el conjunto es un panel corto en entidades (siete) y largo en tiempo, de modo que su
tamaño efectivo depende sobre todo de la autocorrelación y no del número de filas. La licencia
del portal autoriza la descarga para uso personal y no comercial, por lo que el repositorio
documenta el procedimiento de descarga y no redistribuye los archivos originales. La tabla
siguiente resume el periodo de registro, la proporción de días sin dato y la forma de la
distribución de cada estación.

```{code-cell} ipython3
:tags: [hide-input]
# Inventario de las series: periodo de registro, faltantes por periodo, media y asimetría
filas = []
for e in EST:
    s = caudales.set_index("fecha")[e]
    filas.append({"Estación": NL[e], "Inicio": s.first_valid_index().date(), "Fin": s.last_valid_index().date(),
                  "Faltantes 1940-2026 (%)": round(100 * s.isna().mean(), 1),
                  "Faltantes desde 1947 (%)": round(100 * s["1947":].isna().mean(), 1),
                  "Faltantes desde 1976 (%)": round(100 * s["1976":].isna().mean(), 1),
                  "Media (m³/s)": round(s.mean()), "Asimetría": round(s.skew(), 2)})
inv = pd.DataFrame(filas)
glue("n_dias", len(caudales), display=False)
inv
```

El ETL lee los catorce archivos (caudal y transporte de cada estación), corrige la marca de
orden de bytes y los metadatos repetidos en cada fila, unifica los nombres de las estaciones y
construye una tabla ancha sobre el calendario diario de Calamar, con {glue:text}`n_dias:,` días
entre 1940 y 2026. Los días sin registro quedan como valores faltantes explícitos; ninguna serie
se rellena en esta etapa. El inventario muestra dos situaciones distintas: Calamar, Puerto Salgar
y Arrancaplumas tienen registros casi completos desde la década de 1940, mientras que las
estaciones del bajo Magdalena acumulan más del 40 % de faltantes en el periodo completo, porque
empezaron a operar décadas después.

## Periodo de modelado y mecanismo de los faltantes

Esa diferencia exige distinguir dos tipos de faltante. Los faltantes operativos son días aislados
o rachas cortas en que una estación en funcionamiento no reportó. Los faltantes estructurales
son los años anteriores a la instalación de una estación: entre el 40 % y el 48 % de los valores
de El Banco, San Roque, Tres Cruces y Coyongal en el periodo completo. Estos últimos no son
aleatorios en ningún sentido útil, porque dependen de forma determinista de la fecha, e
imputarlos (por ejemplo, con la mediana) convertiría cuatro predictores en constantes durante
tres décadas del entrenamiento, mientras que la prueba se evaluaría con valores reales. La
{numref}`fig-cobertura` muestra la cobertura anual de cada estación.

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Cobertura anual del registro por estación. El bloque vacío anterior a 1972-1975 corresponde a
      estaciones aún no instaladas. La línea continua marca el inicio del diseño adoptado (1947) y la
      punteada, el del diseño alternativo con la red completa (1976).
    name: fig-cobertura
---
# Porcentaje de días con dato por estación y año
cob = caudales.set_index("fecha")[EST].notna().resample("YS").mean() * 100
fig, ax = plt.subplots(figsize=(10, 3.4))
im = ax.imshow(cob.T, aspect="auto", cmap=estilo.mapa_seq(), vmin=0, vmax=100,
               extent=[cob.index.year.min(), cob.index.year.max() + 1, len(EST), 0])
ax.set_yticks(np.arange(len(EST)) + .5, [NL[e] for e in EST]); ax.grid(False)
ax.axvline(1947, color=SED, lw=2); ax.axvline(1976, color=ALE, ls=":", lw=2)
ax.set_xlabel("Año"); ax.set_title("Porcentaje de días con dato por año")
fig.colorbar(im, ax=ax, label="% de días con dato", fraction=.025)
plt.show()
```

La figura confirma que el problema es estructural y no de calidad de la medición: las cuatro
estaciones del bajo Magdalena aparecen entre 1972 y 1975 y, a partir de entonces, su cobertura es
comparable a la de las demás. Hay dos formas de evitar la imputación de estaciones inexistentes.
La primera conserva las seis estaciones predictoras y recorta el periodo de modelado a 1976,
cuando la última de ellas (Coyongal, marzo de 1975) acumula un año de registro. La segunda
conserva el periodo largo y restringe los predictores a las estaciones con registro continuo:
Calamar, la estación objetivo, y Puerto Salgar, con registro desde 1946, de modo que el modelado
empieza el 1 de enero de 1947. La tabla siguiente compara ambas con un diseño de referencia que
imputa las estaciones inexistentes, usando los mismos dos modelos (XGBoost y regresión
logística) y la misma prueba reservada.

```{code-cell} ipython3
---
tags: [hide-input]
---
# Comparación de tres diseños de la base de modelado con los mismos modelos y la misma prueba
from src.features import construir_tarea
from src.splits import separar_prueba
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from src.models import construir_pipeline
disenos = {
    "Seis estaciones desde 1940, imputadas": (["coyongal", "el_banco", "pto_salgar", "san_roque", "tres_cruces"], "1940-01-01"),
    "Seis estaciones desde 1976": (["coyongal", "el_banco", "pto_salgar", "san_roque", "tres_cruces"], "1976-01-01"),
    "Adoptado: Calamar y Puerto Salgar desde 1947": (["pto_salgar"], "1947-01-01"),
}
filas = []
for nombre, (pred, ini) in disenos.items():
    # Misma construcción de variables y misma partición entrenamiento/prueba en los tres diseños
    t = construir_tarea(caudales, "calamar", pred, par["TRANSITOS"], par["Q_EF"], tuple(par["LIM"]), inicio=ini)
    d, cs = t["datos"], t["columnas"]
    a_, b_ = separar_prueba(d.fecha)
    ytr_, yte_ = d.y.iloc[a_].astype(int), d.y.iloc[b_].astype(int)
    xg = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=.05, n_jobs=4, random_state=SEED).fit(d.iloc[a_][cs], ytr_)
    lr = construir_pipeline("clf", "logistica", {"C": 0.1, "penalty": "l1"}, "balanced").fit(d.iloc[a_][cs], ytr_)
    est_cols = [c for c in cs if not c.startswith(("calamar", "sen", "cos", "dist"))]
    filas.append({"diseño": nombre, "observaciones": len(d), "entrenamiento": len(a_), "prueba": len(b_), "predictores": len(cs),
                  "faltantes en predictores de estaciones (%)": 100 * d.iloc[a_][est_cols].isna().mean().mean(),
                  "PR-AUC XGBoost": average_precision_score(yte_, xg.predict_proba(d.iloc[b_][cs])[:, 1]),
                  "PR-AUC logística": average_precision_score(yte_, lr.predict_proba(d.iloc[b_][cs])[:, 1])})
dis = pd.DataFrame(filas).set_index("diseño")
for k, n in zip(("e1", "r76", "a47"), dis.index):
    glue(f"n_{k}", int(dis.loc[n, "observaciones"]), display=False)
    glue(f"xg_{k}", float(dis.loc[n, "PR-AUC XGBoost"]), display=False)
    glue(f"lr_{k}", float(dis.loc[n, "PR-AUC logística"]), display=False)
    glue(f"fe_{k}", float(dis.loc[n, "faltantes en predictores de estaciones (%)"]), display=False)
dis.round(3)
```

El recorte a 1976 elimina la imputación, pero deja {glue:text}`n_r76:,` observaciones y pierde
tres décadas de registro. El diseño adoptado no imputa ninguna estación en años en que no existía
y dispone de {glue:text}`n_a47:,` observaciones. La comparación empírica lo respalda: con
XGBoost, la PR-AUC en la prueba es {glue:text}`xg_a47:.3f` con Calamar y Puerto Salgar desde 1947,
frente a {glue:text}`xg_r76:.3f` con las seis estaciones desde 1976 y {glue:text}`xg_e1:.3f` con el
diseño imputado; con la regresión logística, menos sensible al tamaño de la muestra, los tres
valores ({glue:text}`lr_a47:.3f`, {glue:text}`lr_r76:.3f` y {glue:text}`lr_e1:.3f`) difieren en
menos de una centésima. En conjunto, la ventaja del modelo no lineal proviene de los treinta años
adicionales de registro, que abarcan más ciclos de El Niño y La Niña, y no de las estaciones del
bajo Magdalena. El diseño imputado alcanza casi el mismo desempeño porque también usa el periodo
largo, pero lo hace alimentando cuatro predictores con medianas durante tres décadas; el diseño
adoptado obtiene el mismo resultado sin esa imputación. La información anticipada de las
estaciones del bajo Magdalena está en buena parte contenida en el estado reciente de Calamar: la
más informativa, Coyongal, tiene un tiempo de tránsito de siete días, inferior al horizonte de
predicción. Las seis estaciones no se descartan del estudio: intervienen en las tres tareas
adicionales con las que se evalúa el modelo propuesto en {doc}`06_validacion`, que empiezan en
1976. El análisis magnitud-frecuencia de la descarga efectiva usa el periodo 1940-2009, porque
solo requiere la serie de Calamar.

Dentro de su periodo operativo, cada estación conserva faltantes operativos. Para establecer su
mecanismo, la tabla siguiente compara el caudal de Calamar en los días con y sin registro de cada
estación, con la prueba de Mann-Whitney y el tamaño de efecto A de Vargha y Delaney.

```{code-cell} ipython3
---
tags: [hide-input]
---
# Mecanismo de los faltantes operativos: ¿dependen del caudal observado en Calamar?
inicio_op = {"pto_salgar": "1947-01-01", "arrancaplumas": "1947-01-01"}
filas = []
for e in EST:
    if e == "calamar":
        continue
    d_e = caudales[caudales.fecha >= inicio_op.get(e, "1976-01-01")]       # solo el periodo operativo
    m = d_e[e].isna()
    a, b = d_e.loc[m, "calamar"].dropna(), d_e.loc[~m, "calamar"].dropna()  # Calamar sin y con dato de la estación
    if len(a) < 30:
        continue
    u = st.mannwhitneyu(a, b)
    rachas = (m.ne(m.shift()).cumsum()[m].value_counts())                 # longitud de cada racha de faltantes
    filas.append({"Estación": NL[e], "Periodo": "desde " + inicio_op.get(e, "1976-01-01")[:4], "Días sin dato": int(m.sum()),
                  "% sin dato": round(100 * m.mean(), 1), "Rachas": len(rachas),
                  "Racha mediana (d)": int(rachas.median()), "Racha máxima (d)": int(rachas.max()),
                  "Q Calamar sin dato (mediana)": round(a.median()), "Q Calamar con dato (mediana)": round(b.median()),
                  "A de Vargha-Delaney": round(u.statistic / (len(a) * len(b)), 3), "p": u.pvalue})
mec = pd.DataFrame(filas)
from src.stats_tests import holm
mec["p (Holm)"] = holm(mec.p.values)                                       # varias estaciones a la vez
d47 = caudales[caudales.fecha >= INICIO_MODELADO]
glue("a_ps", float(mec.set_index("Estación").loc["Puerto Salgar", "A de Vargha-Delaney"]), display=False)
glue("falt_ps", float(mec.set_index("Estación").loc["Puerto Salgar", "% sin dato"]), display=False)
glue("falt_cal", float(100 * d47.calamar.isna().mean()), display=False)
glue("a_coyongal", float(mec.set_index("Estación").loc["Coyongal", "A de Vargha-Delaney"]), display=False)
mec.style.format({"p": "{:.2g}", "p (Holm)": "{:.2g}"})
```

Para la tarea principal importan los faltantes de Calamar ({glue:text}`falt_cal:.1f` % de los días
desde 1947) y de Puerto Salgar ({glue:text}`falt_ps:.1f` %). Con muestras de este tamaño casi toda
diferencia resulta significativa, de modo que la lectura se apoya en el tamaño de efecto: A es la
probabilidad de que el caudal de Calamar en un día sin dato supere al de un día con dato, y un
valor de 0,5 indicaría que la ausencia no se relaciona con el caudal. Puerto Salgar
(A = {glue:text}`a_ps:.3f`) y Coyongal (A = {glue:text}`a_coyongal:.3f`) tienen sus huecos
concentrados en aguas bajas. El mecanismo es, por tanto, MAR respecto del caudal observado en
Calamar y no MCAR: la probabilidad de faltar depende de una variable observada. De ese diagnóstico
se siguen dos decisiones: la imputación se ajusta solo con el entrenamiento y dentro del pipeline,
y su efecto sobre el desempeño se mide al final de este capítulo comparando la mediana con una
imputación condicional.

## Auditoría de la serie de transporte y descarga efectiva

La serie de transporte de sedimentos solo es útil si es una medición independiente del caudal.
Para comprobarlo, se ajusta por quinquenios la relación potencial entre transporte y caudal,
$Q_s = a\,Q^{b}$, por mínimos cuadrados en escala logarítmica.

```{code-cell} ipython3
:tags: [hide-input]
# Ajuste de la curva de gasto sólido por quinquenio (calculado en src/etl.py)
aud = pd.read_csv(DATOS_PROC / "auditoria_transporte.csv")
glue("r2_min", float(aud.r2.min()), display=False)
glue("n_r2_alto", int((aud.r2 > 0.997).sum()), display=False)
glue("bloque_r2_min", aud.loc[aud.r2.idxmin(), "bloque"], display=False)
glue("n_bloques_aud", len(aud), display=False)
glue("q_ef", par["Q_EF"], display=False)
glue("lim_inf", par["LIM"][0], display=False); glue("lim_sup", par["LIM"][1], display=False)
aud.rename(columns={"bloque": "Bloque", "a": "a", "b": "Exponente b", "r2": "R²", "n": "Días"}).round(
    {"Exponente b": 3, "R²": 5})
```

En {glue:text}`n_r2_alto` de los {glue:text}`n_bloques_aud` quinquenios la relación entre el
transporte publicado y el caudal es una potencia con R² superior a 0,997, un ajuste imposible
para dos variables medidas por separado. El quinquenio restante ({glue:text}`bloque_r2_min`,
R² = {glue:text}`r2_min:.3f`) es aquel en que el exponente pasa de 1,44 a 1,94, de modo que mezcla
dos ecuaciones. Los cambios de exponente entre quinquenios delatan las sucesivas curvas de gasto
con que el IDEAM calcula el transporte a partir del caudal. La serie de transporte es, por tanto,
una transformación determinista del caudal y no aporta información independiente: queda
excluida de los predictores y del objetivo y se usa solo para localizar la descarga efectiva
mediante el análisis magnitud-frecuencia de la {numref}`fig-mf`.

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Análisis magnitud-frecuencia en Calamar (1940-2009). La carga relativa de cada clase es su
      frecuencia por el transporte de un día típico; su máximo define la descarga efectiva.
    name: fig-mf
---
# Frecuencia de cada clase de caudal y carga de sedimento que transporta (frecuencia × a·Q^b)
mf = pd.read_csv(DATOS_PROC / "magnitud_frecuencia.csv")
fig, ax = plt.subplots(figsize=(9.5, 3.8))
ax.bar(mf.q_centro, 100 * mf.frec_rel, width=mf.q_sup - mf.q_inf, color=CAU, alpha=.45, label="Frecuencia")
ax.plot(mf.q_centro, 100 * mf.carga_rel, "-o", color=SED, ms=4, label="Carga transportada")
ax.axvline(par["Q_EF"], color=ALE, ls="--", label=f"$Q_{{ef}}$ = {par['Q_EF']:,.0f} m³/s")
ax.axvspan(*par["LIM"], color=ALE, alpha=.08, label="Banda ±15 %")
ax.set_xscale("log"); ax.set_xlabel("Caudal (m³/s)"); ax.set_ylabel("% del total"); ax.legend(ncol=4, loc="upper left")
ax.set_title("Frecuencia y carga por clase de caudal")
plt.show()
```

La frecuencia de los caudales tiene su máximo por debajo de la descarga efectiva, porque los
caudales bajos e intermedios son los más comunes, mientras que la carga se desplaza hacia
caudales mayores, que transportan más sedimento por día. El producto de ambos tiene un máximo en
una clase intermedia: la descarga efectiva de Calamar es {glue:text}`q_ef:,.0f` m³/s. Las ocho
ecuaciones históricas conducen a la misma clase modal, de modo que la posición del máximo depende
de la distribución de frecuencias del caudal y no de la escala de la curva {cite}`nash1994`. La
variable objetivo vale 1 cuando el caudal del día *t* + 15 cae en la banda de ±15 % alrededor de
ese valor, entre {glue:text}`lim_inf:,.0f` y {glue:text}`lim_sup:,.0f` m³/s. La auditoría y la
descarga efectiva se calculan con las funciones siguientes de `src/etl.py`: `curva_gasto` ajusta
la potencia en escala logarítmica, `auditoria_transporte` la repite en cada quinquenio y
`descarga_efectiva` construye el análisis magnitud-frecuencia y devuelve la clase de mayor carga.

```{code-cell} ipython3
---
tags: [remove-input]
---
# Código fuente de la auditoría y del análisis magnitud-frecuencia
import inspect
from IPython.display import Code
from src import etl
Code("\n\n".join(inspect.getsource(f) for f in (etl.curva_gasto, etl.auditoria_transporte, etl.descarga_efectiva)),
     language="python")
```

## Variable objetivo

La {numref}`fig-prev` muestra la proporción de días de cada año en que el caudal, quince días
después, estuvo en la banda.

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Proporción anual de días en la banda de descarga efectiva. La línea vertical separa el
      entrenamiento (1947-2009) de la prueba reservada (2010-2026).
    name: fig-prev
---
# Prevalencia anual de la clase positiva en el entrenamiento y en la prueba
tr = tarea[tarea.fecha < FECHA_CORTE_TEST]; te = tarea[tarea.fecha >= pd.Timestamp(FECHA_CORTE_TEST) + pd.Timedelta(days=30)]
anual = tarea.groupby(tarea.fecha.dt.year).y.mean() * 100
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.bar(anual.index, anual.values, color=[SED if a < 2010 else CAU for a in anual.index])
ax.axhline(100 * tr.y.mean(), color=RIO, ls=":", label=f"media de entrenamiento {100*tr.y.mean():.1f} %")
ax.axvline(2009.5, color=NEU, lw=1)
ax.set_ylabel("% de días en la banda"); ax.set_xlabel("Año"); ax.legend(loc="upper left")
ax.set_title("Prevalencia anual de la clase positiva")
plt.show()
a = tr.groupby(tr.fecha.dt.year).y.mean()
glue("prev_tr", 100 * tr.y.mean(), display=False); glue("prev_te", 100 * te.y.mean(), display=False)
glue("n_pos", int(tr.y.sum()), display=False); glue("n_tr", len(tr), display=False); glue("n_te", len(te), display=False)
glue("anio_min", int(a.idxmin()), display=False); glue("anio_max", int(a.idxmax()), display=False)
glue("prev_max", 100 * a.max(), display=False)
```

La clase positiva representa el {glue:text}`prev_tr:.1f` % de los {glue:text}`n_tr:,` días de
entrenamiento ({glue:text}`n_pos:,` casos) y el {glue:text}`prev_te:.1f` % de los
{glue:text}`n_te:,` días de prueba. El desbalance global (unos tres días negativos por cada
positivo) es moderado, pero la prevalencia anual varía entre cero en {glue:text}`anio_min`,
año de El Niño, y {glue:text}`prev_max:.0f` % en {glue:text}`anio_max`, año de La Niña. Esa
heterogeneidad tiene dos implicaciones metodológicas. La primera es que la exactitud no distingue
a un buen modelo de la regla trivial, por lo que la métrica de selección es el área bajo la curva
precisión-exhaustividad {cite}`saito2015`. La segunda es que los bloques de validación deben
abarcar años completos, para que cada uno contenga fases hidrológicas distintas.

## Análisis univariado y bivariado

La tabla siguiente resume la distribución del caudal de cada estación desde 1947, con la
proporción de valores que caen fuera de 1,5 veces el rango intercuartílico.

```{code-cell} ipython3
:tags: [hide-input]
# Estadística descriptiva por estación desde 1947, con asimetría, curtosis y atípicos por IQR
cols_q = [e for e in EST if e != "arrancaplumas"]
d47m = caudales[caudales.fecha >= INICIO_MODELADO]
desc = d47m[cols_q].describe(percentiles=[.05, .25, .5, .75, .95]).T
desc["asimetría"] = d47m[cols_q].skew(); desc["curtosis"] = d47m[cols_q].kurt()
iqr = desc["75%"] - desc["25%"]
desc["% atípicos (IQR)"] = [100 * ((d47m[c] < desc.loc[c, "25%"] - 1.5 * iqr[c]) | (d47m[c] > desc.loc[c, "75%"] + 1.5 * iqr[c])).mean() for c in cols_q]
desc.index = [NL[c] for c in desc.index]
glue("asim_cal", float(d47m.calamar.skew()), display=False)
desc.drop(columns="count").round(2)
```

Las series de caudal son positivas y sin valores imposibles (no hay ceros ni negativos). Las
estaciones del alto Magdalena presentan asimetría positiva marcada, propia de cuencas de montaña
con crecidas rápidas, mientras que Calamar es casi simétrica (asimetría de
{glue:text}`asim_cal:.2f`) porque la depresión momposina amortigua los picos. Los valores fuera de
1,5 veces el rango intercuartílico corresponden a crecidas y estiajes reales y se conservan; los
modelos de árboles son invariantes a transformaciones monótonas y en los demás se aplica
escalado estándar dentro del pipeline. La {numref}`fig-corr` muestra cómo se relacionan las
estaciones entre sí el mismo día.

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Correlación de Spearman entre estaciones el mismo día (1976-2009). Las estaciones del alto y del
      bajo Magdalena forman dos bloques débilmente acoplados.
    name: fig-corr
---
# Correlación de rangos entre estaciones, sin desfase, en el periodo con la red completa
d76 = caudales[(caudales.fecha >= "1976-01-01") & (caudales.fecha < FECHA_CORTE_TEST)]
cm = d76[EST].corr(method="spearman")
glue("corr_cal_ps", float(cm.loc["calamar", "pto_salgar"]), display=False)
glue("corr_cal_coy", float(cm.loc["calamar", "coyongal"]), display=False)
fig, ax = plt.subplots(figsize=(6.4, 5.2))
im = ax.imshow(cm, cmap=estilo.mapa_div(), vmin=-1, vmax=1)
ax.set_xticks(range(len(EST)), [NL[e] for e in EST], rotation=40, ha="right"); ax.set_yticks(range(len(EST)), [NL[e] for e in EST])
for i in range(len(EST)):
    for j in range(len(EST)):
        ax.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if abs(cm.iloc[i, j]) > .6 else "#13232c")
ax.grid(False); ax.set_title("Correlación de Spearman")
plt.show()
```

La matriz separa dos grupos. Las estaciones del bajo Magdalena están fuertemente correlacionadas
con Calamar, que se ubica aguas abajo de ellas (Coyongal alcanza {glue:text}`corr_cal_coy:.2f`),
mientras que la correlación del mismo día entre Calamar y Puerto Salgar es apenas
{glue:text}`corr_cal_ps:.2f`. Esa baja correlación no indica falta de relación: la onda de crecida
tarda unas tres semanas en recorrer el cauce desde Puerto Salgar, de modo que la información útil
de esa estación está en su caudal desfasado, y por eso entra en los modelos con un rezago igual a
su tiempo de tránsito. La tabla siguiente mide la asociación de cada predictor con la variable
objetivo y su redundancia con los demás.

```{code-cell} ipython3
:tags: [hide-input]
# Asociación de cada predictor con la clase (información mutua, punto-biserial, A) y colinealidad (VIF)
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
cols = par["BASE_COLS"] + par["DOM_COLS"]
Xtr = tr[cols].fillna(tr[cols].median())
mi = mutual_info_classif(Xtr, tr.y.astype(int), random_state=SEED)
pb = [st.pointbiserialr(tr.y, Xtr[c])[0] for c in cols]
mw = [st.mannwhitneyu(Xtr.loc[tr.y == 1, c], Xtr.loc[tr.y == 0, c]) for c in cols]
A = [m.statistic / ((tr.y == 1).sum() * (tr.y == 0).sum()) for m in mw]
from src.stats_tests import benjamini_hochberg
asoc = pd.DataFrame({"variable": cols, "información mutua": mi, "r punto-biserial": pb,
                     "A de Vargha-Delaney": A, "p (BH)": benjamini_hochberg(np.array([m.pvalue for m in mw]))})
Xs = (Xtr - Xtr.mean()) / Xtr.std()
asoc["VIF"] = [variance_inflation_factor(Xs.values, i) for i in range(len(cols))]
asoc = asoc.sort_values("información mutua", ascending=False)
glue("mi_top", asoc.iloc[0].variable, display=False)
glue("n_cols", len(cols), display=False)
glue("vif_altos", int((asoc.VIF > 10).sum()), display=False)
asoc.head(15).round(3)
```

La asociación se mide con la información mutua, que capta relaciones no lineales, junto con la
correlación punto-biserial y el tamaño de efecto A, con control de la tasa de falsos
descubrimientos de Benjamini y Hochberg {cite}`benjamini1995` por tratarse de
{glue:text}`n_cols` pruebas simultáneas. La variable más informativa es `{glue:text}`mi_top``, y
las variables de dominio que miden la distancia del caudal actual a la descarga efectiva superan a
los rezagos del caudal en bruto. Esto confirma que la condición de banda no es monótona en el
caudal: un día con caudal muy alto está tan lejos de la banda como uno con caudal muy bajo. El
factor de inflación de varianza supera 10 en {glue:text}`vif_altos` variables, y es infinito para
`dist_qef` y `dist_qef_rel`, que difieren solo en la escala (la segunda es la primera dividida por
$Q_{ef}$); esa redundancia exacta es inocua para los árboles y la absorbe la regularización en los
modelos lineales. El resto de la colinealidad es física, porque proviene de rezagos sucesivos de
una serie con autocorrelación de 0,999 y de estaciones del mismo cauce, y no se elimina por
selección de variables: se trata con regularización en los modelos lineales.

## Componente temporal

La dependencia temporal del caudal determina tanto el diseño de la validación como el cálculo de
la incertidumbre. La figura siguiente muestra la autocorrelación simple y parcial del caudal
diario de Calamar en el periodo de entrenamiento.

```{code-cell} ipython3
:tags: [hide-input]
# Autocorrelación, pruebas de estacionariedad (ADF y KPSS) y descomposición STL del caudal de Calamar
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL
s = caudales[(caudales.fecha >= INICIO_MODELADO) & (caudales.fecha < FECHA_CORTE_TEST)].set_index("fecha").calamar
s = s.interpolate(limit=5).dropna()                       # huecos de hasta 5 días, solo para estas pruebas
p_adf = adfuller(s, autolag="AIC")[1]; p_kpss = kpss(s, regression="c", nlags="auto")[1]
ac, pac = acf(s, nlags=90, fft=True), pacf(s, nlags=40)
mens = s.resample("MS").mean().interpolate(limit_direction="both")  # meses sin dato en las décadas tempranas
stl = STL(mens, period=12, robust=True).fit()
glue("p_adf", p_adf, display=False); glue("p_kpss", p_kpss, display=False)
glue("acf1", ac[1], display=False); glue("acf15", ac[15], display=False); glue("acf30", ac[30], display=False)
glue("pacf2", pac[2], display=False)
var_est = 1 - np.var(stl.resid) / np.var(stl.resid + stl.seasonal)   # fuerza de la estacionalidad
glue("fuerza_est", var_est, display=False)
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
perfil = stl.seasonal.groupby(stl.seasonal.index.month).mean()
picos = [m for m in perfil.index if perfil[m] >= perfil.get(m - 1, perfil[12]) and perfil[m] >= perfil.get(m + 1, perfil[1])]
picos = sorted(picos, key=lambda m: -perfil[m])[:2]
glue("mes_pico1", MESES[picos[0] - 1], display=False); glue("mes_pico2", MESES[picos[1] - 1], display=False)
glue("mes_min", MESES[int(perfil.idxmin()) - 1], display=False)
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.2))
axs[0].bar(range(91), ac, color=RIO, width=.8); axs[0].set_title("ACF del caudal diario en Calamar")
axs[0].axhspan(-1.96 / np.sqrt(len(s)), 1.96 / np.sqrt(len(s)), color=NEU, alpha=.2); axs[0].set_xlabel("Rezago (días)")
axs[1].bar(range(41), pac, color=SED, width=.8); axs[1].set_title("PACF"); axs[1].set_xlabel("Rezago (días)")
plt.show()
```

La autocorrelación del caudal diario es {glue:text}`acf1:.3f` a un día, {glue:text}`acf15:.2f` a
quince y {glue:text}`acf30:.2f` a treinta; la autocorrelación parcial a dos días es
{glue:text}`pacf2:.2f`, señal de que la dinámica de corto plazo está dominada por los dos primeros
rezagos y que los siguientes aportan correcciones decrecientes. La prueba ADF rechaza la raíz
unitaria (p = {glue:text}`p_adf:.2g`) y la KPSS no rechaza la estacionariedad
(p = {glue:text}`p_kpss:.2f`); como sus hipótesis nulas son opuestas, la coincidencia respalda que
la serie es estacionaria en nivel alrededor de un ciclo estacional estable. Dos consecuencias
guían el resto del estudio: con una autocorrelación de 0,92 al horizonte de predicción, ninguna
partición aleatoria es válida, y los días consecutivos no son observaciones independientes para
el cálculo de intervalos de confianza. La {numref}`fig-stl` separa el ciclo estacional de la
variación de largo plazo.

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Descomposición STL robusta del caudal medio mensual en Calamar (1947-2009).
    name: fig-stl
---
# Componentes de la descomposición STL: serie observada, tendencia, estacionalidad y residuo
fig, axs = plt.subplots(4, 1, figsize=(10, 6.4), sharex=True)
for ax, (serie, nombre, c) in zip(axs, [(mens, "Observado", RIO), (stl.trend, "Tendencia", SED),
                                        (stl.seasonal, "Estacional", CAU), (stl.resid, "Residuo", NEU)]):
    ax.plot(serie.index, serie.values, color=c, lw=1.3); ax.set_ylabel(nombre, fontsize=9)
axs[0].set_title("Descomposición STL (m³/s)")
plt.show()
```

La descomposición STL {cite}`cleveland1990` separa una componente estacional bimodal, con máximo
principal en {glue:text}`mes_pico1`, máximo secundario en {glue:text}`mes_pico2` y mínimo en
{glue:text}`mes_min`, de una tendencia que oscila en escala interanual. La fuerza de la
estacionalidad, medida como la fracción de la varianza sin tendencia que explica la componente
estacional, es {glue:text}`fuerza_est:.2f`; por eso los modelos reciben armónicos anual y
semianual del día del año. Las oscilaciones de la tendencia coinciden con la alternancia de El
Niño y La Niña documentada para la cuenca {cite}`restrepo2000` y constituyen la principal fuente
de deriva entre periodos, que se cuantifica en la sección siguiente.

## Deriva entre periodos

Un modelo entrenado con el pasado solo es útil si la relación que aprende se mantiene en el
futuro. La tabla siguiente compara la distribución de cada serie entre tres periodos con el
índice de estabilidad poblacional (PSI) y el estadístico de Kolmogorov-Smirnov.

```{code-cell} ipython3
:tags: [hide-input]
# Índice de estabilidad poblacional (PSI) y Kolmogorov-Smirnov entre 1976-1992, 1993-2009 y 2010-2026
def psi(a, b, bins=10):
    q = np.quantile(a, np.linspace(0, 1, bins + 1)); q[0], q[-1] = -np.inf, np.inf   # deciles del periodo de referencia
    pa, pb_ = np.histogram(a, q)[0] / len(a), np.histogram(b, q)[0] / len(b)
    pa, pb_ = np.clip(pa, 1e-4, None), np.clip(pb_, 1e-4, None)
    return float(((pa - pb_) * np.log(pa / pb_)).sum())
cq = caudales.set_index("fecha")
p1, p2, p3 = cq["1976":"1992"], cq["1993":"2009"], cq["2010":]
filas = []
for v in ["calamar", "pto_salgar", "el_banco", "tres_cruces", "coyongal", "san_roque"]:
    a, b, c = p1[v].dropna(), p2[v].dropna(), p3[v].dropna()
    filas.append({"variable": v, "media 1976-92": a.mean(), "media 1993-09": b.mean(), "media 2010-26": c.mean(),
                  "PSI 76-92 → 93-09": psi(a, b), "PSI 93-09 → 10-26": psi(b, c),
                  "KS 93-09 vs 10-26": st.ks_2samp(b, c).statistic})
deriva = pd.DataFrame(filas)
ty = tarea.set_index("fecha").y
filas_y = {"variable": "y (prevalencia)", "media 1976-92": ty["1976":"1992"].mean(), "media 1993-09": ty["1993":"2009"].mean(), "media 2010-26": ty["2010":].mean()}
glue("psi_max", float(deriva[["PSI 76-92 → 93-09", "PSI 93-09 → 10-26"]].max().max()), display=False)
glue("psi_tc", float(deriva.set_index("variable").loc["tres_cruces", "PSI 93-09 → 10-26"]), display=False)
glue("psi_coy", float(deriva.set_index("variable").loc["coyongal", "PSI 76-92 → 93-09"]), display=False)
glue("psi_ps", float(deriva.set_index("variable").loc["pto_salgar", "PSI 93-09 → 10-26"]), display=False)
glue("psi_cal", float(deriva.set_index("variable").loc["calamar", "PSI 93-09 → 10-26"]), display=False)
pd.concat([deriva, pd.DataFrame([filas_y])]).round(3)
```

Con miles de días la prueba KS rechaza casi cualquier diferencia, por lo que la interpretación se
apoya en el PSI: por debajo de 0,1 el desplazamiento se considera menor, entre 0,1 y 0,25
moderado y por encima de 0,25 severo. El caudal de Calamar y el objetivo son estables (PSI de
{glue:text}`psi_cal:.3f` entre el entrenamiento reciente y la prueba), y la prevalencia de la clase
positiva es prácticamente la misma en ambos periodos, de modo que no hay desplazamiento de la
distribución a priori. Puerto Salgar, la estación predictora, también es estable (PSI de
{glue:text}`psi_ps:.3f`). Dos estaciones del bajo Magdalena, en cambio, muestran desplazamientos
severos: Coyongal entre las dos mitades del entrenamiento (PSI = {glue:text}`psi_coy:.2f`) y Tres
Cruces entre el entrenamiento y la prueba (PSI = {glue:text}`psi_tc:.2f`, con una media que pasa
de unos 2.850 a 1.880 m³/s). Como el caudal de Calamar, que integra el aporte de esas estaciones,
no cambia en la misma proporción, el desplazamiento apunta a cambios en la curva de calibración o
en la ubicación del sensor y no a un cambio hidrológico: es una deriva de medición. Ese hallazgo
es un argumento adicional a favor del diseño adoptado, que no usa esas estaciones como
predictores de Calamar; en las tareas adicionales de {doc}`06_validacion`, donde sí intervienen,
la deriva afecta por igual a todos los modelos comparados.

## Tiempos de tránsito

Los rezagos con que entra cada estación aguas arriba se estimaron por correlación cruzada, sin
información geográfica. La tabla siguiente contrasta esos tiempos con la ubicación de las
estaciones, como verificación externa.

```{code-cell} ipython3
:tags: [hide-input]
# Verificación geográfica de los tiempos de tránsito estimados por correlación cruzada
from src.etl import tiempos_transito
ubic = pd.read_csv(DATOS_PROC / "ubicaciones.csv")
ubic.columns = ["estacion", "codigo", "departamento", "municipio", "lat", "lon", "altitud"]
mapa = {"ARRANCAPLUMAS - AUT": "arrancaplumas", "PUERTO SALGAR - AUT": "pto_salgar", "TRES CRUCES": "tres_cruces",
        "COYONGAL": "coyongal", "EL BANCO - AUT": "el_banco", "SAN ROQUE": "san_roque", "CALAMAR": "calamar"}
ubic["clave"] = ubic.estacion.map(mapa)
ubic["transito"] = ubic.clave.map({**par["TRANSITOS"], "calamar": 0})
g = ubic.dropna(subset=["transito"])
rs = st.spearmanr(g.altitud, g.transito)[0]; rp = st.pearsonr(g.lat, g.transito)[0]
glue("rs_alt", rs, display=False); glue("rp_lat", rp, display=False)
g[["estacion", "municipio", "altitud", "lat", "transito"]].sort_values("transito", ascending=False)
```

Los tiempos de tránsito crecen con la distancia al mar: su correlación de Spearman con la altitud
es {glue:text}`rs_alt:.2f` y la de Pearson con la latitud es {glue:text}`rp_lat:.2f`. Que una
estimación puramente estadística reproduzca el orden geográfico de las estaciones respalda el uso
de cada una desfasada por su tiempo de tránsito; Puerto Salgar, en particular, entra con un rezago
de 21 días.

## Análisis multivariado

La {numref}`fig-pca` resume la estructura conjunta de los {glue:text}`n_cols` predictores
estandarizados con un análisis de componentes principales.

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Análisis de componentes principales de los predictores estandarizados (entrenamiento). (a)
      Varianza explicada acumulada; (b) proyección en las dos primeras componentes, coloreada por clase.
    name: fig-pca
---
# Componentes principales, atípicos multivariados (Isolation Forest) y distancia de Mahalanobis robusta
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.covariance import MinCovDet
pca = PCA().fit(Xs)
cum = np.cumsum(pca.explained_variance_ratio_)
k90 = int(np.searchsorted(cum, .90) + 1)                  # componentes para el 90 % de la varianza
Z = pca.transform(Xs)[:, :2]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
axs[0].plot(range(1, len(cum) + 1), cum, "-o", ms=3, color=RIO); axs[0].axhline(.9, color=NEU, ls=":")
axs[0].set_xlabel("Número de componentes"); axs[0].set_ylabel("Varianza acumulada"); axs[0].set_title("(a) Varianza explicada")
idx = np.random.default_rng(0).choice(len(Z), 3000, replace=False)
for cl, c, n in ((0, RIO, "fuera de la banda"), (1, SED, "en la banda")):
    m = tr.y.values[idx] == cl
    axs[1].scatter(Z[idx][m, 0], Z[idx][m, 1], s=5, alpha=.4, color=c, label=n)
axs[1].set_xlabel("CP1"); axs[1].set_ylabel("CP2"); axs[1].legend(markerscale=3); axs[1].set_title("(b) Proyección")
plt.show()
iso = IsolationForest(random_state=SEED, contamination="auto").fit(Xs)
pct_iso = 100 * (iso.predict(Xs) == -1).mean()
mcd = MinCovDet(random_state=SEED).fit(Xs.iloc[:, :20])
d2 = mcd.mahalanobis(Xs.iloc[:, :20])
pct_mah = 100 * (d2 > st.chi2.ppf(.999, 20)).mean()
marcados = iso.predict(Xs) == -1
q_cal = tr.calamar.values
glue("iso_extremos", 100 * np.mean((q_cal[marcados] > np.quantile(q_cal, .9)) | (q_cal[marcados] < np.quantile(q_cal, .1))), display=False)
glue("k90", k90, display=False); glue("pc1", pca.explained_variance_ratio_[0], display=False)
glue("pct_iso", pct_iso, display=False); glue("pct_mah", pct_mah, display=False)
```

Las {glue:text}`k90` primeras componentes principales explican el 90 % de la varianza de los
{glue:text}`n_cols` predictores, y la primera concentra el {glue:text}`pc1:.0%`, lo que confirma la
fuerte redundancia entre rezagos y estaciones. La proyección en el plano de las dos primeras
componentes muestra que la clase positiva ocupa una franja de la primera componente, la que
resume el nivel general de caudal, con días negativos a ambos lados: una frontera que ningún
clasificador lineal puede trazar sin las variables de distancia a la descarga efectiva. Isolation
Forest marca el {glue:text}`pct_iso:.1f` % de los días como atípicos multivariados, y el
{glue:text}`iso_extremos:.0f` % de ellos corresponde a días con el caudal de Calamar en el decil
superior o inferior: son crecidas y estiajes, eventos hidrológicamente válidos que se conservan.
La distancia de Mahalanobis robusta (determinante de covarianza mínima sobre las primeras 20
variables, umbral χ² al 99,9 %) marca el {glue:text}`pct_mah:.1f` %, una proporción incompatible
con un umbral del 0,1 %. Ese resultado indica que la distribución conjunta, con colas pesadas y
relaciones no lineales entre rezagos, se aparta de la normalidad multivariada que el umbral
supone, y no que una cuarta parte de los días sea atípica. Por esa razón la detección de atípicos
se apoya en Isolation Forest, que no requiere ese supuesto.

## Auditoría de fuga de información

Una variable con fuga contiene información posterior al momento de la predicción. Como prueba
complementaria a la revisión de la construcción de cada variable, la tabla siguiente muestra el
área bajo la curva ROC que alcanza cada predictor por sí solo.

```{code-cell} ipython3
:tags: [hide-input]
# AUC de cada predictor por separado: un valor cercano a 1 delataría información del futuro
from sklearn.metrics import roc_auc_score
fuga = pd.DataFrame([{"variable": c, "AUC": max(roc_auc_score(tr.y, Xtr[c]), 1 - roc_auc_score(tr.y, Xtr[c]))}
                     for c in cols]).sort_values("AUC", ascending=False)
glue("auc_max", float(fuga.AUC.max()), display=False)
fuga.to_csv(DATOS_PROC / "fuga.csv", index=False)
fuga.head(8).round(3)
```

Todas las variables se construyen con información hasta el día *t*, el objetivo se ubica en
*t* + 15 y las ventanas móviles excluyen el propio día, de modo que estarían disponibles en el
momento de la predicción. El AUC individual alcanza un máximo de {glue:text}`auc_max:.3f`, por
debajo del umbral de alerta de 0,95; las variables más altas miden la cercanía del caudal actual a
la banda, información legítima cuyo poder decae con el horizonte. No hay identificadores ni
entidades repetidas entre particiones, porque la partición es cronológica y la única entidad es la
red de estaciones.

## Modelo de referencia lineal: regularización e intervalos por bloques

Antes de los modelos no lineales se estableció un modelo de referencia, una regresión logística
regularizada, con dos cuidados metodológicos: el parámetro de regularización se elige por
validación cronológica y los intervalos de confianza respetan la dependencia temporal. La tabla
siguiente compara ese modelo con la persistencia y con la misma regresión sin ajustar.

```{code-cell} ipython3
:tags: [hide-input]
# Regresión logística: C y penalización por validación cronológica; IC por bootstrap de bloques frente a i.i.d.
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import average_precision_score, f1_score
from src.models import construir_pipeline
from src.stats_tests import bootstrap_bloques, longitud_bloque
from src.experiment import cargar_datos
D = cargar_datos()
base = construir_pipeline("clf", "logistica", {"C": 1.0, "penalty": "l2"}, "balanced")
grid = GridSearchCV(base, {"modelo__C": np.logspace(-3, 2, 11), "modelo__penalty": ["l1", "l2"]},
                    scoring="average_precision", cv=TimeSeriesSplit(5, gap=30), n_jobs=4).fit(D.X, D.y_clf)
C_opt, pen = grid.best_params_["modelo__C"], grid.best_params_["modelo__penalty"]
p_ajust = grid.best_estimator_.predict_proba(D.X_test)[:, 1]
p_def = base.fit(D.X, D.y_clf).predict_proba(D.X_test)[:, 1]
y = D.y_clf_test.values
pers = D.X_test["dist_qef"].values <= 0.15 * par["Q_EF"]   # persistencia: en la banda hoy → en la banda en 15 días
pers = pers.astype(float)
L = longitud_bloque(y.astype(float))                        # longitud media de bloque de Politis y White
filas = []
for nombre, s in (("Persistencia", pers), ("Logística C = 1, sin ajustar", p_def), (f"Logística C = {C_opt:.3g}, {pen}", p_ajust)):
    ic_b = bootstrap_bloques(average_precision_score, y, s, media_bloque=L, n_boot=1000)
    rng = np.random.default_rng(SEED); iid = []
    for _ in range(1000):                                   # bootstrap ingenuo, que trata los días como independientes
        i = rng.integers(0, len(y), len(y)); iid.append(average_precision_score(y[i], s[i]))
    filas.append({"modelo": nombre, "PR-AUC": ic_b["estimacion"],
                  "IC 95 % bloques (BCa)": f"[{ic_b['ic_inf']:.3f}; {ic_b['ic_sup']:.3f}]",
                  "IC 95 % filas i.i.d.": f"[{np.percentile(iid, 2.5):.3f}; {np.percentile(iid, 97.5):.3f}]",
                  "ancho bloques / ancho i.i.d.": (ic_b["ic_sup"] - ic_b["ic_inf"]) / (np.percentile(iid, 97.5) - np.percentile(iid, 2.5)),
                  "F1": f1_score(y, (s >= .5).astype(int))})
base_tab = pd.DataFrame(filas)
glue("c_opt", C_opt, display=False); glue("pen", pen, display=False); glue("L_bloque", L, display=False)
glue("ratio_ic", float(base_tab["ancho bloques / ancho i.i.d."].iloc[-1]), display=False)
glue("ap_base", float(base_tab["PR-AUC"].iloc[-1]), display=False)
base_tab.round(3)
```

El parámetro de regularización se elige por búsqueda en rejilla con validación cronológica de
cinco pliegues y separación de 30 días sobre el entrenamiento, entre once valores de C de
$10^{-3}$ a $10^{2}$ y las penalizaciones L1 y L2. El óptimo es C = {glue:text}`c_opt:.3g` con
penalización {glue:text}`pen`, que coincide con el valor por defecto; la diferencia es que la
elección se sustenta en la validación. En {doc}`03_benchmarks` este ajuste se generaliza a la
validación anidada de todos los modelos de referencia. Los intervalos de confianza se calculan
con el bootstrap estacionario {cite}`politis1994`, cuya longitud media de bloque
({glue:text}`L_bloque` días) se elige con el criterio de Politis y White {cite}`politis2004`, y con
corrección BCa estimada mediante un jackknife que elimina bloques completos. El intervalo por
bloques es {glue:text}`ratio_ic:.1f` veces más ancho que el que remuestrea días independientes: un
bootstrap ingenuo subestimaría la incertidumbre en esa proporción. El modelo regularizado alcanza
una PR-AUC de {glue:text}`ap_base:.3f` y supera con amplitud a la persistencia, que es la referencia
mínima que cualquier modelo debe superar.

## Sensibilidad a la imputación

Como los faltantes residuales son MAR, la imputación teóricamente apropiada es la condicional. La
tabla siguiente compara, en los doce bloques de validación externa, la mediana con una imputación
iterativa que condiciona cada variable a las demás y con la mediana acompañada de indicadores de
faltante.

```{code-cell} ipython3
:tags: [hide-input]
# Sensibilidad de la PR-AUC a la estrategia de imputación en los 12 bloques externos
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
res = []
for nombre, imp in (("mediana", SimpleImputer(strategy="median")),
                    ("mediana + indicador", SimpleImputer(strategy="median", add_indicator=True)),
                    ("iterativa (condicional)", IterativeImputer(max_iter=10, random_state=SEED))):
    for p in D.pliegues:
        pipe = Pipeline([("imp", imp), ("esc", StandardScaler()),
                         ("m", LogisticRegression(C=C_opt, penalty=pen, solver="liblinear", class_weight="balanced", max_iter=5000))])
        pipe.fit(D.X.iloc[p["train"]], D.y_clf.iloc[p["train"]])
        res.append({"imputación": nombre, "pliegue": p["pliegue"],
                    "pr_auc": average_precision_score(D.y_clf.iloc[p["test"]], pipe.predict_proba(D.X.iloc[p["test"]])[:, 1])})
res = pd.DataFrame(res)
tab_imp = res.groupby("imputación").pr_auc.agg(["mean", "std"])
w = st.wilcoxon(res[res["imputación"] == "mediana"].pr_auc.values, res[res["imputación"] == "iterativa (condicional)"].pr_auc.values)
glue("p_imp", w.pvalue, display=False)
glue("imp_lectura", "no permite distinguirlas" if w.pvalue >= 0.05 else "las distingue", display=False)
glue("dif_imp", float(tab_imp.loc["iterativa (condicional)", "mean"] - tab_imp.loc["mediana", "mean"]), display=False)
tab_imp.round(4)
```

La diferencia media de PR-AUC entre la imputación iterativa y la mediana es
{glue:text}`dif_imp:+.4f`, y la prueba de Wilcoxon pareada por bloques {glue:text}`imp_lectura`
(p = {glue:text}`p_imp:.2f`). Con faltantes residuales de esta magnitud la elección del imputador no
cambia las conclusiones, y se conserva la mediana por su menor costo computacional en la
validación anidada.

## Síntesis: decisiones de preparación

La tabla reúne las decisiones de preparación que se derivan de este capítulo y el hallazgo que
motiva cada una.

| Decisión | Hallazgo que la motiva |
|---|---|
| Predictores restringidos a Calamar y Puerto Salgar; modelado desde 1947 | Faltantes estructurales anteriores a la instalación de la red ({numref}`fig-cobertura`) y mayor desempeño con el registro largo |
| Imputación por mediana dentro del pipeline | Faltantes operativos MAR de baja incidencia; la imputación condicional no cambia el desempeño |
| Escalado estándar, sin transformación logarítmica | Calamar casi simétrica; atípicos válidos |
| Variables de distancia a $Q_{ef}$ | Objetivo no monótono en el caudal ({numref}`fig-pca`) |
| Puerto Salgar desfasada 21 días | Tiempo de tránsito verificado con la geografía; baja correlación del mismo día ({numref}`fig-corr`) |
| Armónicos anual y semianual | Estacionalidad bimodal ({numref}`fig-stl`) |
| Serie de transporte excluida de los predictores | Transformación determinista del caudal (R² > 0,997) |
| Regularización en los modelos lineales | Colinealidad física (VIF > 10) |
| Bloques de validación de dos años y separación de 30 días | Autocorrelación de 0,92 a 15 días y heterogeneidad interanual |
| Intervalos por bootstrap estacionario | Días consecutivos no independientes |
| PR-AUC como métrica principal | Clase positiva del 26 % y prevalencia anual variable |
