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

# Revisión de literatura y novedad

La condición R1 exige que el modelo propuesto no haya sido usado en la literatura para el mismo
fenómeno y la misma tarea de predicción, y que la novedad se demuestre con una búsqueda
documentada. Este capítulo describe el protocolo, presenta la matriz de novedad, analiza los
trabajos más cercanos por familia algorítmica y concluye con una declaración de novedad
acotada a lo que la búsqueda permite afirmar.

```{code-cell} ipython3
---
tags: [hide-input]
---
import sys; sys.path.insert(0, "..")
import pandas as pd
from myst_nb import glue
from src.config import RESULTADOS
mat = pd.read_csv(RESULTADOS / "literatura" / "matriz_busqueda.csv")
reg = pd.read_csv(RESULTADOS / "literatura" / "registros_revisados.csv")
cerc = pd.read_csv(RESULTADOS / "literatura" / "trabajos_cercanos.csv")
glue("fecha_busqueda", mat.fecha_consulta.iloc[0], display=False)
glue("n_cadenas", mat.cadena.nunique(), display=False)
glue("n_bases_ok", int(mat[mat.estado == "ok"].base.nunique()), display=False)
glue("n_revisados", int(len(reg)), display=False)
```

## Protocolo

La búsqueda se ejecutó el {glue:text}`fecha_busqueda` mediante el script
`scripts/revision_literatura.py`, que consulta las interfaces públicas de cuatro bases
bibliográficas: OpenAlex, índice abierto y multidisciplinario que incorpora el antiguo Microsoft
Academic Graph; Crossref, el registro de DOI de las editoriales; arXiv, para
preimpresos de aprendizaje automático y estadística; y Semantic Scholar. Scopus y Web of
Science no se incluyeron porque no ofrecen una interfaz abierta que permita registrar la
consulta de forma reproducible, y Google Scholar prohíbe la consulta automatizada; OpenAlex
cumple el papel de índice multidisciplinario general. El script guarda la fecha, el número de
resultados de cada cadena en cada base y los quince primeros registros por relevancia, cuyo
título y resumen se revisaron manualmente.

Las {glue:text}`n_cadenas` cadenas combinan los tres ejes que pide la guía: el problema de
aplicación (descarga efectiva, caudal dominante, río Magdalena), la familia y el mecanismo del
modelo (boosting distribucional, NGBoost, probabilidad de intervalo, pérdida híbrida con
objetivo auxiliar) y sinónimos en español. Semantic Scholar devolvió errores por límite de
solicitudes en la mayoría de las consultas pese a los reintentos con espera creciente; el
fallo queda registrado en la matriz en lugar de ocultarse, y las otras tres bases cumplen el
mínimo exigido. En total se revisaron {glue:text}`n_revisados` registros.

## Matriz de novedad

```{code-cell} ipython3
---
tags: [hide-input]
---
# Juicio de relevancia de la revisión manual (título y resumen) por cadena y base
relevancia = {
    ('"effective discharge" machine learning', "OpenAlex"): (2, "Dos trabajos predicen transporte de sedimentos con XGBoost o redes; ninguno predice el régimen de descarga efectiva"),
    ('"effective discharge" machine learning', "arXiv"): (0, "Resultados de descarga eléctrica, baterías y caudales sin relación con la descarga efectiva"),
    ('"effective discharge" machine learning', "Crossref"): (0, "Coincidencias léxicas fuera del dominio"),
    ('"effective discharge" prediction OR forecasting OR classification', "OpenAlex"): (1, "Estudio de 2015 en Geophysical Research Letters que explica la descarga efectiva con un marco analítico; no la predice en el tiempo"),
    ('"effective discharge" prediction OR forecasting OR classification', "arXiv"): (0, "Sin relación"),
    ('"effective discharge" prediction OR forecasting OR classification', "Crossref"): (0, "Sin relación"),
    ('"distributional gradient boosting" streamflow OR discharge', "OpenAlex"): (2, "DGBM (März y Kneib) y una variante de normal generalizada: boosting distribucional genérico, sin pérdida de evento"),
    ('"distributional gradient boosting" streamflow OR discharge', "Crossref"): (2, "XGBoost con mezcla gaussiana para pronóstico de caudales: dos etapas, sin clasificación de banda"),
    ('"distributional gradient boosting" streamflow OR discharge', "Semantic Scholar"): (0, "Regresión puntual de caudales"),
    ('"distributional gradient boosting" streamflow OR discharge', "arXiv"): (0, "Sin resultados"),
    ("NGBoost streamflow OR discharge OR river", "OpenAlex"): (2, "NGBoost con predicción conforme para carga de sedimentos y caudal (2026): regresión probabilística, sin evento de intervalo"),
    ("NGBoost streamflow OR discharge OR river", "Crossref"): (0, "Sin relación directa"),
    ("NGBoost streamflow OR discharge OR river", "arXiv"): (0, "Sin resultados"),
    ("gradient boosting interval probability threshold exceedance river discharge", "OpenAlex"): (1, "Alerta de inundaciones con XGBoost: regresión de niveles comparada con umbrales"),
    ("gradient boosting interval probability threshold exceedance river discharge", "Crossref"): (0, "Teoría de excedencias sin aprendizaje automático"),
    ("gradient boosting interval probability threshold exceedance river discharge", "arXiv"): (0, "Sin resultados"),
    ("hybrid loss proper scoring rule boosting classification continuous auxiliary target", "OpenAlex"): (0, "Clasificación con boosting estándar"),
    ("hybrid loss proper scoring rule boosting classification continuous auxiliary target", "Crossref"): (1, "Estimación de incertidumbre con boosting y reglas propias, sin objetivo de intervalo"),
    ("hybrid loss proper scoring rule boosting classification continuous auxiliary target", "arXiv"): (0, "Sin resultados"),
    ('"Magdalena River" machine learning discharge', "OpenAlex"): (1, "Dinámica del caudal del Magdalena y ENSO; sin modelos de clasificación de banda"),
    ('"descarga efectiva" aprendizaje automático OR "caudal dominante"', "OpenAlex"): (0, "Coincidencias con 'descarga' en contextos médicos"),
    ('"descarga efectiva" aprendizaje automático OR "caudal dominante"', "Crossref"): (0, "Sin relación"),
}
m = mat.copy()
m["relevantes"] = [relevancia.get((c, b), (0, ""))[0] for c, b in zip(m.cadena, m.base)]
m["observación"] = [relevancia.get((c, b), (0, "Sin resultados por error de la interfaz" if e != "ok" else ""))[1]
                    for c, b, e in zip(m.cadena, m.base, m.estado)]
m.loc[m.estado != "ok", "observación"] = "No disponible: " + m.loc[m.estado != "ok", "estado"]
m = m.rename(columns={"fecha_consulta": "fecha", "base": "base de datos", "cadena": "cadena de búsqueda",
                      "resultados": "resultados", "revisados": "revisados"})
glue("n_relevantes", int(m.relevantes.sum()), display=False)
m[["base de datos", "cadena de búsqueda", "resultados", "revisados", "relevantes", "observación"]]
```

La columna de resultados de Crossref debe leerse con cautela: su motor de búsqueda
bibliográfica devuelve cualquier registro que contenga alguno de los términos, de modo que los
totales son del orden de millones y solo los primeros registros por relevancia son
informativos. De los registros revisados, {glue:text}`n_relevantes` resultaron cercanos al
problema o al mecanismo, y ninguno aplica un modelo de aprendizaje automático a la predicción
de la pertenencia del caudal a la banda de descarga efectiva.

## Trabajos más cercanos

```{code-cell} ipython3
---
tags: [hide-input]
---
cerc[["anio", "titulo", "fuente", "doi"]].rename(columns={"anio": "año", "titulo": "título"})
```

Los trabajos recuperados se agrupan en tres líneas. La primera aplica aprendizaje automático al
transporte de sedimentos: modelos de XGBoost entrenados con cerca de 400 descriptores de
cuenca para predecir carga suspendida y de fondo en Minnesota, o una red neuronal entrenada con
8.117 mediciones de carga de fondo en 134 ríos. Ambos estiman la cantidad de sedimento, no el
régimen de caudal que domina su transporte, y no tienen componente temporal de anticipación.
La segunda línea estudia la propia descarga efectiva: un trabajo publicado en 2015 en
*Geophysical Research Letters* relaciona su
cociente con el caudal medio con el exponente de la curva de gasto y la variabilidad del
régimen mediante un marco analítico, una caracterización de largo plazo como la de los
antecedentes del capítulo 1, sin predicción. La tercera línea, la más próxima en lo metodológico,
es la regresión probabilística de caudales y sedimentos con boosting: NGBoost y PGBM combinados
con predicción conforme para la carga suspendida y para el caudal de un embalse en Irán
(ambos de 2026), y XGBoost acoplado a mezclas gaussianas para pronóstico de caudales. En todos
ellos el objetivo es la variable continua; las probabilidades de umbral, si se calculan, se
derivan después del ajuste.

## Diferenciación por familia algorítmica

```{code-cell} ipython3
---
tags: [hide-input]
---
familias = pd.DataFrame([
    ("NGBoost (Duan et al., 2020)", "Boosting distribucional", "Árboles para los parámetros de una distribución; gradiente natural", "Ajusta solo la verosimilitud del continuo; la probabilidad de banda se obtiene después, sin incentivo en los límites", "Se incluye como competidor en el capítulo 6"),
    ("DGBM / XGBoostLSS (März y Kneib, 2022)", "Boosting distribucional", "Newton multiparámetro con hessianos observados", "Igual objetivo de verosimilitud; no contempla un evento definido por la variable", "BLGB añade la entropía cruzada del evento y límites por observación"),
    ("GAMLSS por boosting (Mayr et al., 2012)", "Regresión distribucional aditiva", "Media, escala y forma como funciones aditivas", "Componentes aditivos sin interacciones de árbol; pérdida de verosimilitud", "Se conserva la parametrización de la escala en logaritmo"),
    ("XGBoost AFT con censura en intervalo (Barnwal et al., 2022)", "Supervivencia / censura", "Verosimilitud de intervalos censurados", "Requiere conocer el intervalo que contiene la respuesta; no modela el complemento de una banda, que es la unión de dos semirrectas", "La probabilidad de intervalo de BLGB coincide con su término para los positivos"),
    ("Regresión ordinal de enlace acumulado (McCullagh, 1980)", "Modelos de umbral latente", "Variable latente cortada por umbrales fijos", "Umbrales globales; la banda de BLGB se desplaza con el caudal de referencia de cada día", "Se conserva la idea de una latente gaussiana cortada por límites"),
    ("EMOS y BFS en hidrometeorología (Gneiting et al., 2005; Krzysztofowicz, 1999)", "Posprocesamiento probabilístico", "Distribución predictiva calibrada a partir de un pronóstico determinista", "Requieren un pronóstico físico previo; se ajustan con CRPS o verosimilitud, no con la pérdida del evento", "BLGB aprende la distribución directamente de los predictores"),
    ("Aprendizaje multitarea (Caruana, 1997)", "Supervisión auxiliar", "Una tarea auxiliar regulariza la principal", "Usualmente con representaciones compartidas en redes; sin regla de puntuación propia que conecte ambas tareas", "En BLGB ambas tareas comparten la misma distribución predictiva"),
    ("Regresión cuantílica (Koenker y Bassett, 1978)", "Regresión de cuantiles", "Cuantiles condicionales sin supuesto distribucional", "Estimar la masa de un intervalo exige muchos cuantiles y no garantiza monotonía", "BLGB obtiene la masa de la banda en forma cerrada"),
    ("XGBoost + mezcla gaussiana para caudales (2020)", "Dos etapas", "Regresión de caudal y modelado del residuo", "La dispersión no depende del estado del río en el ajuste conjunto; sin pérdida del evento", "Se evalúa como 'dos etapas' en el capítulo 6"),
], columns=["Método afín", "Familia", "Qué hace", "Limitación frente al problema", "Relación con BLGB"])
familias
```

La comparación por familias precisa qué conserva y qué añade la propuesta. De la regresión
distribucional por boosting conserva la parametrización de la media y del logaritmo de la
escala mediante ensambles de árboles de segundo orden. De los modelos de umbral latente conserva
la idea de una variable gaussiana cortada por límites. Lo que ninguna de estas familias combina
es la pérdida del evento de interés con la verosimilitud del continuo dentro de un mismo
procedimiento de boosting: los métodos distribucionales optimizan solo la verosimilitud y dejan
la probabilidad de banda como un subproducto, mientras que los clasificadores optimizan solo el
evento y descartan el continuo. A eso se suma que los límites de la banda se trasladan a cada
observación a través del caudal de referencia, de modo que el modelo aprende la dinámica del
cambio y no el nivel absoluto, y que el algoritmo incluye una búsqueda lineal con garantía de
descenso, ausente en las implementaciones de NGBoost y XGBoostLSS. La limitación que la
propuesta aborda es concreta: con una muestra efectiva pequeña, el clasificador desperdicia la
información del continuo, y el regresor distribucional no concentra su capacidad en los
límites que deciden la clase.

## Declaración de novedad

Al mejor de nuestro conocimiento y según la búsqueda descrita, realizada el
{glue:text}`fecha_busqueda` en OpenAlex, Crossref y arXiv con {glue:text}`n_cadenas` cadenas en
inglés y en español, no se encontraron estudios que apliquen un modelo de aprendizaje
automático a la predicción anticipada de la pertenencia del caudal a la banda de descarga
efectiva, ni en el río Magdalena ni en otra cuenca, ni estudios que propongan un procedimiento
de boosting que combine la entropía cruzada de un evento de intervalo con la verosimilitud del
continuo que lo define. Los trabajos más cercanos son la regresión probabilística de caudales y
sedimentos con NGBoost y PGBM (2026), de la que la propuesta se diferencia por optimizar el
evento y por los límites dependientes de cada observación; el boosting distribucional genérico
(DGBM), del que se diferencia por la pérdida híbrida y la búsqueda lineal; y los estudios de la
descarga efectiva como descriptor de largo plazo, que caracterizan el fenómeno pero no lo
anticipan. La novedad que se afirma es, por tanto, doble y acotada: una aplicación nueva, la
anticipación del régimen de transporte dominante, y un mecanismo de aprendizaje nuevo,
transferible a cualquier problema en el que la clase se defina por la pertenencia de una
variable continua observable a un intervalo.
