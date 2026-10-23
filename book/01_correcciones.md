# Respuesta a la evaluación del Entregable 1

La evaluación del Entregable 1 reconoció la auditoría de la serie de transporte de sedimentos,
la estimación robusta de la descarga efectiva, la verificación geográfica de los tiempos de
tránsito y el uso de la persistencia como modelo de referencia. Identificó, además, tres
problemas metodológicos del modelo base: los intervalos de confianza se obtuvieron remuestreando
días como si fueran independientes, pese a una autocorrelación de 0,999; el parámetro de
regularización de la regresión logística no se ajustó; y se imputaron valores de estaciones en
años en que aún no existían. Pidió también actualizar el README y explorar modelos no lineales.
En este capítulo se discute cada problema, la decisión metodológica adoptada y su efecto
cuantitativo sobre los resultados. Los análisis completos se encuentran en el capítulo 2, en la
sección 3.5 y en el capítulo 6.

## Dependencia temporal e intervalos de confianza

El caudal diario de Calamar tiene una autocorrelación de 0,999 a un día de rezago, que se
mantiene en 0,92 a quince días y en 0,76 a treinta. Con esa persistencia, días consecutivos
contienen prácticamente la misma información, y un bootstrap que remuestrea días de forma
independiente produce intervalos cuyo ancho refleja el número nominal de observaciones y no el
número efectivo. Los intervalos se recalcularon con el bootstrap estacionario de Politis y Romano
{cite}`politis1994`, que remuestrea bloques de longitud aleatoria con distribución geométrica. La
longitud media del bloque se eligió con el criterio automático de Politis y White
{cite}`politis2004` y resultó de 133 días; con ese bloque, los 5.946 días del periodo de prueba
(2010-2026) equivalen a unos 45 bloques independientes. La consecuencia es directa: el intervalo
de 95 % de la PR-AUC del modelo base es 4,0 veces más ancho que el obtenido con remuestreo
independiente, de modo que el informe anterior subestimaba la incertidumbre en ese factor. Los
intervalos incluyen la corrección BCa, cuya constante de aceleración se estimó con un jackknife
que elimina bloques completos; un jackknife por observaciones individuales heredaría el mismo
sesgo que se buscaba corregir.

La misma dependencia afecta la comparación entre modelos. Los pliegues de una validación
cronológica comparten la mayor parte del conjunto de entrenamiento, por lo que sus puntajes no
son independientes y la prueba t pareada del Entregable 1 subestimaba la varianza de las
diferencias. Las comparaciones sobre los doce bloques externos se basan ahora en la prueba de
Wilcoxon con corrección de Holm y en la t corregida de Nadeau y Bengio {cite}`nadeau2003`, que infla la varianza por
el factor $1/k + n_{\text{prueba}}/n_{\text{entrenamiento}}$. Para que estas pruebas pareadas sean
válidas, las particiones externas se calculan una sola vez y se almacenan en
`data/processed/pliegues_calamar.json`, de modo que todos los modelos se evalúan exactamente en
los mismos bloques.

## Regularización del modelo base

En el Entregable 1 la regresión logística se entrenó con el valor por defecto C = 1. En esta
versión, C se selecciona por búsqueda en rejilla entre once valores espaciados
logarítmicamente de $10^{-3}$ a $10^{2}$, con penalización L1 o L2, mediante validación
cronológica de cinco pliegues sobre el entrenamiento 1947-2009 y una separación de 30 días entre
entrenamiento y validación, mayor que el horizonte de quince días de la variable objetivo, para
evitar que la etiqueta de un día de entrenamiento dependa de caudales del periodo de validación.
El óptimo es C = 1 con penalización L2, igual al valor empleado antes, y la PR-AUC en la prueba
es 0,881. La coincidencia no hace irrelevante la corrección: el valor de C pasó de ser un supuesto
no examinado a una estimación respaldada por validación. En el capítulo 3 este ajuste se
incorpora a la validación anidada, en la que la regresión logística, Ridge y Lasso se optimizan
dentro de cada bloque externo con cuatro métodos de búsqueda, de forma que el desempeño reportado
no incluye el optimismo de seleccionar hiperparámetros con los mismos datos con que se evalúa.

## Estaciones sin registro y periodo de modelado

La revisión mostró que el problema era mayor de lo reconocido en el informe anterior. Entre el
40 % y el 48 % de los valores de El Banco, San Roque, Tres Cruces y Coyongal eran medianas
asignadas a años anteriores a su instalación, ocurrida entre 1972 y 1975. Ese faltante es
estructural: depende de la fecha y no del estado hidrológico, y la mediana impuesta convertía
cuatro predictores en constantes durante tres décadas del entrenamiento, mientras que la prueba
se evaluaba con valores observados. Las distribuciones de esos predictores diferían, por
construcción, entre entrenamiento y prueba.

Se evaluaron dos diseños que eliminan la imputación estructural. El primero conserva las seis
estaciones y recorta el periodo de modelado a 1976, lo que deja 18.359 observaciones, por debajo
del mínimo de 20.000 que exige la guía. El segundo conserva el periodo 1947-2026 y restringe los
predictores a las estaciones con registro en todo ese intervalo, Calamar y Puerto Salgar (en
operación desde 1946), con 28.558 observaciones después de descontar la separación de 30 días
entre entrenamiento y prueba. Con los mismos modelos de referencia y la misma prueba, XGBoost
obtiene una PR-AUC de 0,941 con el segundo diseño, frente a 0,919 con el recorte a 1976 y 0,940
con el diseño imputado del Entregable 1; en la regresión logística los tres valores (0,880, 0,882
y 0,875) difieren en menos de una centésima. La ventaja del modelo no lineal proviene, por tanto,
de los treinta años adicionales de registro, que abarcan más ciclos de El Niño y La Niña y un
intervalo más amplio de estados hidrológicos, y no de las estaciones del bajo Magdalena. Esto es
coherente con la hidráulica del sistema: la información anticipada de esas estaciones está en
buena parte contenida en el estado reciente de Calamar, ya que la más informativa, Coyongal,
tiene un tiempo de tránsito de siete días, inferior al horizonte de predicción. Se adoptó el
segundo diseño; las seis estaciones se conservan solo en el análisis de sensibilidad.

En el diseño adoptado el faltante residual es operativo: el 1,2 % de los días en Calamar y el
3,8 % en Puerto Salgar. Su mecanismo se examinó comparando el caudal de Calamar en los días con y
sin registro de cada estación mediante la prueba de Mann-Whitney, el tamaño de efecto A de Vargha
y Delaney y la corrección de Holm. En Puerto Salgar A = 0,379, lo que indica que los huecos se
concentran en aguas bajas; el mecanismo es MAR respecto del caudal de Calamar y no MCAR. Este
diagnóstico corrige también un argumento del Entregable 1, que invocaba el mecanismo MAR para
justificar la mediana sin distinguir los faltantes estructurales de los operativos. Bajo un
mecanismo MAR la imputación apropiada es la condicional, y la mediana solo es defendible si su
efecto sobre el desempeño es despreciable. En los doce bloques externos, la diferencia media de
PR-AUC entre una imputación iterativa condicional y la mediana es de −0,002, y la prueba de
Wilcoxon no las distingue (p = 0,57). Con faltantes de esta magnitud el imputador no modifica las
conclusiones, y se conservó la mediana por su menor costo, ajustada siempre dentro del pipeline y
solo con los datos de entrenamiento.

## Modelos no lineales

El capítulo 3 evalúa siete clasificadores con cuatro técnicas de balanceo y cuatro métodos de
optimización (112 combinaciones) y siete regresores con los cuatro optimizadores (28
combinaciones). Los modelos no lineales superan al lineal: Random Forest alcanza una PR-AUC externa
de 0,910 ± 0,075, XGBoost 0,905, KNN 0,888 y la SVM con núcleo RBF 0,886, frente a 0,850 ± 0,136
de la regresión logística. Las diferencias entre los cuatro modelos no lineales no son
significativas, mientras que la ventaja de Random Forest sobre la regresión logística sí lo es
tras la corrección de Holm, con un tamaño de efecto pequeño. El resultado tiene una explicación
física. La pertenencia a la banda de descarga efectiva no es monótona en el caudal, porque un
caudal demasiado bajo y uno demasiado alto quedan ambos fuera de ella; un modelo lineal solo
puede representar esa relación a través de variables construidas, como la distancia del caudal
actual a la descarga efectiva, mientras que los árboles y los núcleos la aprenden directamente.
La interpretación con SHAP y LIME del capítulo 3 confirma que el modelo no lineal utiliza esa
estructura. El análisis de los errores de estos modelos, concentrados en los días que cruzan los
límites de la banda por un margen pequeño, motivó a su vez el modelo original no lineal de los
capítulos 5 y 6, que modela la distribución del caudal futuro en lugar de la etiqueta binaria.

## Reproducibilidad

El README se reescribió con la descripción del problema, la estructura del repositorio, la
instalación del entorno, los comandos de reproducción en orden y la ejecución de las pruebas.
El ETL y la construcción de variables, que en el Entregable 1 residían en celdas de cuaderno con
rutas absolutas, se trasladaron a los módulos `src/etl.py` y `src/features.py`, con rutas
relativas centralizadas en `src/config.py`; el comando `python -m scripts.etl` regenera todas las
tablas de `data/processed/`, y el capítulo 9 documenta el entorno, la semilla y la tabla maestra
de experimentos. El texto de la página de análisis exploratorio del dashboard, que repetía el
argumento del mecanismo MAR sin distinguir los dos tipos de faltante, se reescribió a partir del
diagnóstico anterior.

## Efecto sobre los resultados anteriores

Todos los resultados que dependen de las correcciones se recalcularon con el diseño corregido. El
número de predictores pasó de 43 a 31 y la prevalencia de la clase positiva es de 26,3 % en el
entrenamiento (1947-2009) y de 25,2 % en la prueba (2010-2026). La descarga efectiva, 9.954 m³/s,
no cambia, porque su cálculo usa solo la serie de Calamar, completa desde 1940. Las figuras del
dashboard se regeneraron con el diseño corregido. Dos resultados de
la corrección no favorecen al diseño adoptado y se reportan como tales: el valor óptimo de C
coincide con el que ya se usaba, y la regresión logística obtiene una PR-AUC ligeramente mayor con
el recorte a 1976 que con el diseño adoptado (0,882 frente a 0,880), diferencia que está muy por
debajo de la incertidumbre del intervalo por bloques.
