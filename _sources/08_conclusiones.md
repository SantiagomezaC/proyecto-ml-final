# Conclusiones y limitaciones

## Hallazgos

El proyecto convirtió un descriptor geomorfológico de largo plazo, la descarga efectiva, en un
objetivo de predicción operativa: anticipar con quince días de antelación si el Magdalena estará
en Calamar dentro de la banda de caudales que más sedimento transporta. Cinco hallazgos resumen
el trabajo.

El primero es metodológico y proviene de las correcciones al Entregable 1. La imputación de
estaciones que aún no existían inflaba la información aparente del modelo; al restringir los
predictores a las estaciones que existen en todo el periodo (Calamar y Puerto Salgar, desde 1947)
el problema queda libre de imputaciones estructurales, conserva 28.558 observaciones y no pierde
desempeño, porque treinta años adicionales de registro aportan más que las estaciones del bajo
Magdalena. Los intervalos de confianza calculados con bootstrap por bloques resultaron unas cuatro
veces más anchos que los del informe original, que subestimaba la incertidumbre en esa
proporción.

El segundo proviene de los 140 modelos del Entregable 2. Con validación anidada cronológica en
doce bloques de dos años, el mejor benchmark de clasificación es Random Forest con ponderación por
clase (PR-AUC externa de 0,910 ± 0,075), pero Random Forest, XGBoost, SVM y KNN son
estadísticamente indistinguibles. El balanceo de clases no mejora la discriminación, solo desplaza
el punto de operación, y ADASYN la empeora. Con 16 evaluaciones por pliegue los cuatro
optimizadores llegan a la misma calidad final; TPE y el algoritmo genético llegan antes que la
rejilla, con efecto grande, y solo marginalmente antes que la búsqueda aleatoria. En regresión,
SVR y XGBoost forman el conjunto de confianza y reducen el error de la persistencia en un tercio.

El tercero es el diagnóstico que motivó el modelo original: los errores de los clasificadores se
concentran en los días que cruzan los límites de la banda por un margen pequeño, porque la
etiqueta binaria trata igual un caudal que roza el umbral y uno que cae en el centro de la banda.

El cuarto es el resultado del modelo original. Band-Likelihood Gradient Boosting modela la
distribución del cambio logarítmico del caudal con media y dispersión dependientes del estado del
río, traslada la banda a cada observación y optimiza una pérdida estrictamente propia que combina
la entropía cruzada del evento con la verosimilitud del caudal futuro. Supera al mejor benchmark
en los doce bloques externos (PR-AUC de 0,943 ± 0,053 frente a 0,910 ± 0,075) y en la prueba
reservada 2010-2026 (0,965 frente a 0,950), con probabilidades mejor calibradas. La ablación
atribuye la ganancia a la supervisión continua; la ventaja crece cuando la muestra se reduce y se
mantiene bajo ruido y desbalance.

El quinto matiza el anterior y debe leerse con la misma importancia. Frente a métodos
distribucionales recientes, NGBoost y un enfoque de dos etapas, el modelo original queda
empatado: la mayor parte de la ventaja sobre los clasificadores se debe a modelar la
distribución del caudal futuro, idea que comparten esos métodos, y no a la pérdida híbrida en sí.
En otras tres estaciones del cauce la ventaja se reproduce en dos y desaparece en la tercera.

## Veredicto sobre las condiciones del modelo original

| Condición | Evidencia | Veredicto |
|---|---|---|
| R1. Novedad en la aplicación | Revisión en OpenAlex, Crossref y arXiv con ocho cadenas; ningún estudio predice la pertenencia a la banda de descarga efectiva ni combina en boosting la verosimilitud del evento con la del continuo (capítulo 4) | Se cumple, con la novedad acotada a la aplicación y a la formulación |
| R2. Implementación algorítmica propia | Clase `BandLikelihoodBoosting` programada con numpy: discretización, árboles de segundo orden por histogramas, gradientes analíticos verificados, búsqueda de Armijo y parada temprana; dos teoremas demostrados y 18 pruebas unitarias (capítulo 5) | Se cumple |
| R3. Superioridad estadística | Media externa mayor que la de los siete benchmarks; Wilcoxon-Holm (p = 0,0017), t corregida de Nadeau-Bengio (p = 0,026) y bootstrap estacionario BCa en la prueba (p = 0,0015), con δ de Cliff de 0,35 y d de Cohen de 0,99 (capítulo 6) | Se cumple frente a los benchmarks del curso; no frente a NGBoost |

## Limitaciones

El conjunto de predictores carece de variables meteorológicas. Sin precipitación ni índices del
ENSO, ningún modelo dispone de la causa de los ascensos rápidos desde el estiaje, que concentran
los errores de todos los modelos, incluido el original.

La autocorrelación del caudal reduce el tamaño efectivo de muestra a una fracción de los días
disponibles, y con doce bloques externos la potencia de las pruebas no paramétricas es limitada:
Nemenyi no separa al original del segundo modelo y la prueba bayesiana con región de
equivalencia de ±0,01 no alcanza el umbral convencional de superioridad práctica.

El 89,6 % del registro de Calamar tiene nivel de aprobación preliminar, y las estaciones de Tres
Cruces y Coyongal muestran desplazamientos de su distribución compatibles con cambios en la curva
de calibración, que afectan a las tareas adicionales.

La regulación por embalses y el cambio de uso del suelo implican que el pasado puede no
representar el futuro. La evaluación en el periodo 2010-2026 sugiere que la relación aprendida se
mantiene, pero no garantiza que lo haga ante cambios mayores en la cuenca.

La comparación con el estado del arte se limitó a dos métodos distribucionales con presupuesto de
optimización igual al del modelo original; no se evaluaron modelos de pronóstico hidrológico con
base física, que serían la referencia en un contexto operativo.

## Trabajo futuro

La incorporación de precipitación y del índice oceánico de El Niño es la extensión con mayor
potencial, porque ataca la fuente principal de error. En el plano metodológico, la familia
gaussiana del cambio logarítmico puede reemplazarse por una distribución con colas o asimetría
ajustables; es el escenario en que el término de la tarea de la pérdida híbrida debería aportar
más frente a la verosimilitud pura, según el análisis del capítulo 5. La formulación es además
transferible a cualquier problema en que la clase se defina por la pertenencia de una variable
continua observable a un intervalo, como los rangos de calidad de agua o de temperatura, donde
convendría evaluarla. Finalmente, un pronóstico de la banda para varios horizontes a la vez daría a
los usuarios una curva de probabilidad en el tiempo y no un único valor a quince días.

## Declaración de autoría y uso de herramientas

Esta sección debe ser completada y firmada por los autores antes de la entrega, conforme a la
Sección 13 de la guía del Entregable 3. Debe indicar qué partes del trabajo (código, análisis,
redacción, figuras, dashboard) se apoyaron en herramientas de software o de inteligencia
artificial, para qué se usaron, y declarar que los autores comprenden y pueden defender cada
línea del código y cada resultado presentado.

| Autor | Contribución | Firma |
|---|---|---|
| Kevin Clemente Rosario | | |
| Manuel Santiago Meza Castro | | |

**Herramientas utilizadas y propósito:** *(completar por el equipo)*
