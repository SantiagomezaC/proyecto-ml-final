# Trazabilidad de correcciones

La retroalimentación del Entregable 1 (nota 4,5) valoró la auditoría de la serie de transporte,
la estimación robusta de la descarga efectiva, la verificación geográfica de los tiempos de
tránsito y el uso de la persistencia como referencia, y señaló tres debilidades del modelo base
y dos tareas pendientes. La tabla siguiente registra cada observación, la acción realizada, su
ubicación en este libro y la evidencia que permite verificarla. A la fecha de cierre de este
informe no se dispone de retroalimentación escrita del Entregable 2, que se entrega integrado en
los capítulos 3 y 9; la segunda parte de la tabla recoge las correcciones que el propio equipo
detectó al revisar el material anterior, que se reportan con el mismo criterio.

## Observaciones del docente

| Observación del docente | Acción realizada | Ubicación | Evidencia |
|---|---|---|---|
| Los intervalos de confianza remuestrean filas pese a una autocorrelación de 0,999 | Todos los intervalos se calculan con bootstrap estacionario (Politis y Romano), longitud media de bloque de Politis y White y corrección BCa con jackknife por bloques; las comparaciones entre modelos usan pruebas sobre bloques externos y la t corregida de Nadeau-Bengio | Capítulo 2, sección «Corrección 2 y 3»; capítulos 3.5 y 6 | Tabla del modelo base con la razón de anchos entre el intervalo por bloques y el i.i.d.; función `bootstrap_bloques` en `src/stats_tests.py` |
| El parámetro de regularización no se ajustó | C y la penalización (L1/L2) se eligen por búsqueda con validación cronológica; en el Entregable 2 la regresión logística, Ridge y Lasso se optimizan dentro de la validación anidada con cuatro métodos | Capítulo 2; capítulo 3 | Valor óptimo de C reportado en el capítulo 2; tabla maestra de 140 combinaciones con hiperparámetros por pliegue |
| La imputación de estaciones que aún no existían queda sin justificar | Ningún predictor se imputa en años en que su estación no existía. Se compararon dos soluciones: recortar a 1976 con las seis estaciones (18.359 observaciones, por debajo del mínimo de 20.000 de la guía) o conservar 1947-2026 con las estaciones que existen en todo el periodo, Calamar y Puerto Salgar (28.558 observaciones). Se adoptó la segunda, que además obtiene mayor PR-AUC en la prueba; las seis estaciones se usan en las tareas adicionales del capítulo 6. El mecanismo de los faltantes residuales se diagnostica (MAR respecto del caudal de Calamar) y se mide la sensibilidad a la imputación | Capítulo 2, sección «Corrección 1» y última sección | Figura de cobertura; tabla comparativa de los tres diseños; tabla de Mann-Whitney con tamaño de efecto A y Holm; comparación de imputación por mediana, mediana con indicador e iterativa en los 12 bloques |
| Conviene actualizar el README | Nuevo README con descripción, estructura del repositorio, instalación, reproducción paso a paso, pruebas y despliegue | Raíz del repositorio | `README.md` |
| Explorar los modelos no lineales | 112 combinaciones de clasificación (KNN, Naive Bayes, logística, árbol, Random Forest, XGBoost, SVM RBF) y 28 de regresión, y un modelo original no lineal | Capítulos 3, 5 y 6 | Tabla maestra; diagrama de diferencia crítica |

## Correcciones detectadas por el equipo

| Hallazgo | Acción realizada | Ubicación | Evidencia |
|---|---|---|---|
| El dashboard del Entregable 1 afirmaba que la imputación por mediana era adecuada porque el mecanismo era MAR, argumento que no distinguía los faltantes estructurales de los operativos | El texto se reescribió a partir del nuevo diagnóstico: faltantes estructurales anteriores a 1976 excluidos y faltantes operativos MAR con análisis de sensibilidad | Dashboard, página EDA; capítulo 2 | Leyenda de la figura de faltantes del dashboard |
| La comparación de sobreajuste entre pliegues usaba una prueba t pareada, que supone independencia entre pliegues que comparten entrenamiento | Se reemplazó por la t corregida de Nadeau y Bengio en todas las comparaciones sobre pliegues | Capítulos 3.5 y 6 | Función `nadeau_bengio` en `src/stats_tests.py` |
| El ETL y la construcción de variables vivían en celdas de cuaderno con rutas absolutas | El ETL se convirtió en los módulos `src/etl.py` y `src/features.py`, con rutas relativas centralizadas en `src/config.py` y un único punto de entrada | Capítulo 2; capítulo 9 | `python -m scripts.etl` reproduce las tablas de `data/processed/` |
| Las particiones de validación se recalculaban en cada cuaderno | Las particiones externas se calculan una vez, se guardan en disco y las leen todos los modelos, lo que garantiza el pareado estricto que exigen las pruebas del capítulo 6 | Capítulo 3 | `data/processed/pliegues_calamar.json` |

## Consistencia de los resultados

Las correcciones modifican resultados del Entregable 1, y todos los que dependen de ellas se
recalcularon con el diseño corregido: el conjunto de predictores (de 43 a 31 variables), la
prevalencia (26,3 % en el entrenamiento 1947-2009), el modelo base, sus intervalos y las figuras
del dashboard. La base de modelado conserva más de 20.000 observaciones, como exige la guía. El único resultado que no cambia es la
descarga efectiva, porque su cálculo usa solo la serie de Calamar, completa desde 1940. La
comparación directa entre el modelo base del Entregable 1 y el corregido se presenta en el
capítulo 2, incluido el caso en que la corrección no mejora la métrica, que se reporta tal como
resultó.
