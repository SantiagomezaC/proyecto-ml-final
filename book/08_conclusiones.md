# Conclusiones y limitaciones

## Hallazgos

El estudio convirtió un descriptor geomorfológico de largo plazo, la descarga efectiva, en un
objetivo de predicción operativa: anticipar con quince días de antelación si el Magdalena estará
en Calamar dentro de la banda de caudales que más sedimento transporta. Seis hallazgos resumen
el trabajo.

El primero se refiere a los datos. Restringir los predictores a las estaciones con registro
continuo, Calamar y Puerto Salgar desde 1947, evita imputar estaciones en años en que aún no
existían, conserva 28.558 observaciones y no pierde desempeño, porque treinta años adicionales de
registro aportan más que las estaciones del bajo Magdalena. La dependencia temporal del caudal es
tan fuerte (autocorrelación de 0,92 a quince días) que un bootstrap que trata los días como
independientes subestima la incertidumbre en un factor de cuatro; todos los intervalos del
estudio se calculan por bloques.

El segundo proviene de los 140 modelos de referencia. Con validación anidada cronológica en doce
bloques de dos años, el mejor es Random Forest con ponderación por clase (PR-AUC externa de
0,910 ± 0,075), pero Random Forest, XGBoost, SVM y KNN son estadísticamente indistinguibles. El
balanceo de clases no mejora la discriminación, solo desplaza el punto de operación, y ADASYN la
empeora porque concentra los sintéticos en la frontera de la banda. Con 16 evaluaciones por
pliegue, los cuatro métodos de optimización llegan a la misma calidad final; TPE y el algoritmo
genético llegan antes que la rejilla. En regresión, SVR y XGBoost forman el conjunto de confianza
de modelos y reducen en un tercio el error de la persistencia.

El tercero explica la variabilidad entre periodos, que supera con amplitud la diferencia entre
modelos: la dificultad de cada bloque depende de cuántos días exigen anticipar una entrada a la
banda (correlación de Spearman de −0,93 con la PR-AUC del bloque). Los errores de todos los
modelos se concentran en los días que cruzan los límites por poco, porque la etiqueta binaria trata
igual un caudal que roza el umbral y uno que cae en el centro de la banda.

El cuarto es el resultado del modelo propuesto. Band-Likelihood Gradient Boosting modela la
distribución del cambio logarítmico del caudal con media y dispersión dependientes del estado del
río, traslada la banda a cada observación y optimiza una pérdida estrictamente propia que combina
la entropía cruzada del evento con la verosimilitud del caudal futuro. Supera a los siete modelos
de referencia en los doce bloques externos (PR-AUC de 0,943 ± 0,053 frente a 0,910 ± 0,075) y en
la prueba reservada 2010-2026 (0,965 frente a 0,950), con probabilidades mejor calibradas. La
ablación atribuye la ganancia a la supervisión continua; la ventaja crece cuando la muestra se
reduce y se mantiene bajo ruido y desbalance.

El quinto sitúa ese resultado frente a los métodos recientes. Frente a la familia LTSF-Linear
de pronóstico de largo plazo de series de tiempo (Linear, NLinear y DLinear), la ventaja es amplia:
PR-AUC externa de 0,943 frente a valores entre 0,847 y 0,871, con victoria en los doce bloques
contra cada uno. Los tres miembros son indistinguibles entre sí, y su desventaja común muestra que
un modelo lineal en la historia del caudal y con dispersión constante no capta que la dinámica del
río depende de su estado. Frente a los métodos
distribucionales, NGBoost y un enfoque de dos etapas, el modelo propuesto queda empatado: la mayor
parte de la ventaja sobre los clasificadores se debe a modelar la distribución del caudal futuro,
idea que comparten esos métodos, y no a la pérdida híbrida en sí.

El sexto se refiere a la generalidad. En otras tres estaciones del cauce la ventaja se reproduce
en dos y desaparece en la tercera, Tres Cruces, cuya serie muestra una deriva de medición.

## Alcance de las contribuciones

| Criterio | Evidencia | Conclusión |
|---|---|---|
| Novedad | Búsqueda en OpenAlex, Crossref y arXiv con ocho cadenas: ningún estudio predice la pertenencia a la banda de descarga efectiva ni combina en boosting la verosimilitud del evento con la del continuo ({doc}`04_literatura`) | Novedad en la aplicación y en la formulación de la pérdida |
| Implementación propia | Clase `BandLikelihoodBoosting` programada con numpy: discretización, árboles de segundo orden por histogramas, gradientes analíticos verificados, búsqueda de Armijo y parada temprana; dos teoremas demostrados y 18 pruebas unitarias ({doc}`05_modelo_original`) | Implementación completa y verificada |
| Superioridad estadística | Media externa mayor que la de los siete modelos de referencia; Wilcoxon-Holm (p = 0,0017), t corregida de Nadeau-Bengio (p = 0,026) y bootstrap estacionario en la prueba (p = 0,0015), con δ de Cliff de 0,35 y d de Cohen de 0,99 ({doc}`06_validacion`) | Superior a los modelos de referencia y a la familia LTSF-Linear; empatado con NGBoost |

## Limitaciones

El conjunto de predictores carece de variables meteorológicas. Sin precipitación ni índices del
ENSO, ningún modelo dispone de la causa de los ascensos rápidos desde el estiaje, que concentran
los errores de todos los modelos, incluido el propuesto.

La autocorrelación del caudal reduce el tamaño efectivo de muestra a una fracción de los días
disponibles, y con doce bloques externos la potencia de las pruebas no paramétricas es limitada:
Nemenyi no separa al modelo propuesto del segundo modelo y la prueba bayesiana con región de
equivalencia de ±0,01 no alcanza el umbral convencional de superioridad práctica.

El 89,6 % del registro de Calamar tiene nivel de aprobación preliminar, y las estaciones de Tres
Cruces y Coyongal muestran desplazamientos de su distribución compatibles con cambios en la curva
de calibración, que afectan a las tareas adicionales.

La regulación por embalses y el cambio de uso del suelo implican que el pasado puede no
representar el futuro. La evaluación en el periodo 2010-2026 sugiere que la relación aprendida se
mantiene, pero no garantiza que lo haga ante cambios mayores en la cuenca.

La comparación con el estado del arte incluyó cinco métodos recientes, dos distribucionales y
los tres de la familia LTSF-Linear, con el mismo presupuesto de optimización que el modelo
propuesto; quedaron fuera los pronosticadores profundos basados en transformers, que la propia
literatura de LTSF-Linear encuentra inferiores a los lineales, y no se evaluaron modelos de pronóstico hidrológico con base
física, que serían la referencia en un contexto operativo.

## Trabajo futuro

La incorporación de precipitación y del índice oceánico de El Niño es la extensión con mayor
potencial, porque ataca la fuente principal de error. En el plano metodológico, la familia
gaussiana del cambio logarítmico puede reemplazarse por una distribución con colas o asimetría
ajustables; es el escenario en que el término de la tarea de la pérdida híbrida debería aportar
más frente a la verosimilitud pura, según el análisis de {doc}`05_modelo_original`. La formulación
es además transferible a cualquier problema en que la clase se defina por la pertenencia de una
variable continua observable a un intervalo, como los rangos de calidad de agua o de temperatura,
donde convendría evaluarla. Finalmente, un pronóstico de la banda para varios horizontes a la vez
daría a los usuarios una curva de probabilidad en el tiempo y no un único valor a quince días.

## Contribuciones de los autores y uso de herramientas

El estudio partió de una revisión bibliográfica sobre hidrología y geomorfología fluvial, en
particular sobre el transporte de sedimentos, el análisis magnitud-frecuencia y el concepto de
descarga efectiva, así como sobre los antecedentes del río Magdalena, que ambos autores
realizaron conjuntamente y a partir de la cual plantearon el problema. Kevin Clemente Rosario
construyó la base de datos, el análisis exploratorio y el modelo base de la versión preliminar.
Manuel Santiago Meza Castro diseñó y ejecutó la comparación de los modelos de referencia, la
optimización de hiperparámetros y la comparación estadística. Ambos desarrollaron el modelo
propuesto, su validación, la aplicación interactiva y este documento.

Las herramientas de inteligencia artificial generativa se emplearon como apoyo en el diseño del
modelo propuesto. Las decisiones sobre el problema, los datos, la formulación hidrológica y la
interpretación de los resultados corresponden a los autores, que declaran comprender y poder
defender cada parte del código, del análisis y de los resultados presentados en este documento y
en la aplicación interactiva.
