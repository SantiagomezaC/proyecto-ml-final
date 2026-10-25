# Descarga efectiva en el río Magdalena

**Predicción probabilística a quince días de la banda de caudales que más sedimento transporta**

<div class="autores-libro">
  <div class="autor">
    <img src="_static/autor_kevin.jpg" alt="Fotografía de Kevin Clemente Rosario">
    <div><div class="nombre">Kevin Clemente Rosario</div><div class="rol">Geólogo · Estudiante de M.Sc.</div></div>
  </div>
  <div class="autor">
    <img src="_static/autor_manuel.jpg" alt="Fotografía de Manuel Santiago Meza Castro">
    <div><div class="nombre">Manuel Santiago Meza Castro</div><div class="rol">Estudiante de Geología y de M.Sc.</div></div>
  </div>
</div>
<p class="afiliacion">Universidad del Norte, Barranquilla, Colombia</p>

```{admonition} Recursos del estudio
:class: tip
- **Aplicación interactiva:** <https://descarga-efectiva-magdalena.onrender.com>
- **Código y datos procesados:** <https://github.com/SantiagomezaC/proyecto-ml-final>
```

## Resumen

La descarga efectiva es el caudal que, a lo largo de los años, transporta la mayor parte del
sedimento de un río. La literatura la trata como un descriptor de largo plazo; este estudio la
convierte en un objetivo de predicción operativa: anticipar, con quince días de antelación, si
el caudal del río Magdalena en Calamar estará dentro de una banda de ±15 % alrededor de su
descarga efectiva, estimada en 9.954 m³/s. Con 28.558 días de registro del IDEAM (1947-2026) en
Calamar y Puerto Salgar, el problema es una clasificación binaria con 26 % de casos positivos
sobre una serie temporal con una autocorrelación de 0,92 al horizonte de predicción.

Se compararon primero 140 combinaciones de modelos de referencia, técnicas de balanceo y
métodos de optimización de hiperparámetros con validación anidada cronológica en doce bloques
de dos años y pruebas estadísticas que respetan la dependencia temporal. El mejor modelo de
referencia es Random Forest (PR-AUC externa de 0,910 ± 0,075), estadísticamente indistinguible
de XGBoost, SVM y KNN. El análisis de sus errores, concentrados en los días que cruzan los
límites de la banda por poco, motivó el modelo propuesto, Band-Likelihood Gradient Boosting
(BLGB): un boosting de segundo orden programado desde cero que modela la distribución del cambio
logarítmico del caudal futuro, con media y dispersión dependientes del estado del río, traslada
la banda a cada observación y optimiza una pérdida que combina la entropía cruzada del evento con
la verosimilitud del caudal. Se demuestra que esa pérdida es una regla de puntuación
estrictamente propia y que el algoritmo desciende de forma monótona.

El modelo propuesto supera a los siete modelos de referencia en los doce bloques (0,943 ± 0,053)
y en la prueba reservada 2010-2026 (0,965 frente a 0,950), con significancia en tres pruebas
confirmatorias y un efecto entre mediano y grande, y produce probabilidades calibradas sin
recalibración. Supera también con amplitud a la familia LTSF-Linear (Linear, NLinear y DLinear), referencia
actual del pronóstico de largo plazo de series de tiempo, pero queda empatado con métodos distribucionales como NGBoost, lo que indica que
la mayor parte de la ganancia proviene de modelar la distribución del caudal futuro. Los modelos
se ponen a disposición de usuarios no especialistas en una aplicación interactiva con predicción
en vivo, intervalo del caudal previsto y explicación de cada predicción.

**Palabras clave:** descarga efectiva, río Magdalena, pronóstico probabilístico, gradient
boosting distribucional, reglas de puntuación propias, validación anidada, series de tiempo.

## Contribuciones

El estudio hace cuatro contribuciones. La primera es la formulación del problema: la descarga
efectiva, un concepto geomorfológico de largo plazo, se convierte en un evento predecible a
quince días, útil para programar purgas de embalses, dragados del canal navegable y campañas de
muestreo de sedimentos. La segunda es una comparación sistemática de 140 modelos de referencia
con un protocolo que evita la fuga de información y la sobreestimación por selección de
hiperparámetros. La tercera es el modelo propuesto, con su formulación, dos resultados teóricos
demostrados y una implementación propia verificada con pruebas unitarias. La cuarta es una
validación estadística que incluye métodos recientes de regresión distribucional y de pronóstico
de series de tiempo, evaluación en otras tres estaciones del cauce, análisis de robustez y
análisis de errores, con los resultados desfavorables reportados como tales.

## Problema

El río Magdalena drena cerca de la cuarta parte de Colombia y entrega al mar Caribe del orden
de 144 millones de toneladas de sedimento al año {cite}`restrepo2000`, uno de los rendimientos
sedimentarios más altos de Suramérica {cite}`milliman1992`. La mayor parte de ese trabajo
geomorfológico no la realizan las crecidas excepcionales ni los caudales bajos y frecuentes,
sino un rango intermedio de caudales, la descarga efectiva {cite}`wolman1960`. Saber con
antelación cuándo el río estará en ese rango permite programar las operaciones que dependen del
transporte de sedimentos en el momento en que este es más intenso.

## Objetivos

1. Construir una base de datos reproducible del caudal del Magdalena y definir la variable
   objetivo a partir de la descarga efectiva.
2. Establecer el desempeño de los modelos de referencia con una validación que respete la
   dependencia temporal y comparar estadísticamente algoritmos, técnicas de balanceo y métodos
   de optimización.
3. Diseñar, implementar y validar un modelo que aproveche la naturaleza continua de la variable
   que define la clase, y verificar si supera a los modelos de referencia con significancia
   estadística.
4. Poner los modelos a disposición de usuarios no especialistas y documentar el estudio de forma
   reproducible.

## Organización del libro

El capítulo {doc}`02_datos` describe los datos, construye la variable objetivo y deriva del
análisis exploratorio las decisiones de preparación. {doc}`03_benchmarks` y sus secciones
presentan los modelos de referencia: diseño experimental, resultados de clasificación,
comparación de optimizadores, optimización computacional, regresión, comparación estadística e
interpretabilidad. {doc}`04_literatura` documenta la búsqueda bibliográfica y la declaración de
novedad, {doc}`05_modelo_original` desarrolla el modelo propuesto y {doc}`06_validacion` lo valida.
{doc}`07_dashboard` describe la aplicación interactiva, {doc}`08_conclusiones` reúne los hallazgos,
las limitaciones y el trabajo futuro, y {doc}`09_reproducibilidad` documenta el entorno y la
bibliografía. El {doc}`A_revisiones` registra los cambios metodológicos respecto de una versión
preliminar del estudio.

Las celdas de código de cada capítulo están ocultas para facilitar la lectura y pueden
desplegarse con el botón «Show code cell source»; las funciones principales de cada etapa se
muestran completas en el capítulo correspondiente.

## Cómo citar

Clemente Rosario, K. y Meza Castro, M. S. (2026). *Descarga efectiva en el río Magdalena:
predicción probabilística a quince días de la banda de caudales que más sedimento transporta*.
Universidad del Norte. <https://santiagomezac.github.io/proyecto-ml-final/>

## Disponibilidad de datos y código

Los datos de caudal y transporte de sedimentos son del IDEAM y se descargan del portal DHIME; su
licencia no permite redistribuirlos, por lo que el repositorio documenta el procedimiento de
descarga e incluye las tablas procesadas necesarias para reproducir los resultados. Todo el
código, los resultados intermedios y las fuentes de este libro están en el repositorio.
