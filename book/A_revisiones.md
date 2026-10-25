# Apéndice A. Revisiones respecto de la versión preliminar

Una versión preliminar de este trabajo, limitada al análisis exploratorio y a un modelo base de
regresión logística, fue sometida a revisión académica. La revisión valoró la auditoría de la
serie de transporte de sedimentos, la estimación robusta de la descarga efectiva, la
verificación geográfica de los tiempos de tránsito y el uso de la persistencia como referencia,
y señaló tres problemas metodológicos del modelo base: los intervalos de confianza se obtenían
remuestreando días como si fueran independientes, pese a una autocorrelación de 0,999; el
parámetro de regularización no se ajustaba; y se imputaban valores de estaciones en años en que
aún no existían. Recomendó además explorar modelos no lineales y documentar mejor la
reproducción del análisis. Este apéndice registra cómo se atendió cada observación y en qué
parte de la versión actual se encuentra la evidencia. Los cambios no se presentan en el cuerpo
del texto como correcciones, sino como decisiones metodológicas justificadas en el lugar que les
corresponde.

## Dependencia temporal e intervalos de confianza

El caudal diario de Calamar tiene una autocorrelación de 0,999 a un día, que se mantiene en 0,92
a quince días y en 0,76 a treinta. Con esa persistencia, días consecutivos contienen
prácticamente la misma información, y un bootstrap que remuestrea días de forma independiente
produce intervalos cuyo ancho refleja el número nominal de observaciones y no el efectivo. En la
versión actual todos los intervalos se calculan con el bootstrap estacionario de Politis y
Romano {cite}`politis1994`, que remuestrea bloques de longitud aleatoria con distribución
geométrica, con una longitud media de 133 días elegida por el criterio automático de Politis y
White {cite}`politis2004` y con corrección BCa estimada por un jackknife que elimina bloques
completos. El intervalo de la PR-AUC del modelo base resulta 4,0 veces más ancho que el obtenido
con remuestreo independiente, de modo que la versión preliminar subestimaba la incertidumbre en
ese factor. La misma dependencia afecta la comparación entre modelos: los pliegues de una
validación cronológica comparten la mayor parte del entrenamiento, y la prueba t pareada que se
usaba subestimaba la varianza de las diferencias. Las comparaciones se basan ahora en la prueba
de Wilcoxon con corrección de Holm y en la t corregida de Nadeau y Bengio {cite}`nadeau2003`, y
las particiones externas se calculan una sola vez y se almacenan, de modo que todos los modelos
se evalúan exactamente en los mismos bloques.

Ubicación: {doc}`02_datos`, sección «Modelo de referencia lineal»; {doc}`03e_comparacion`;
{doc}`06_validacion`.

## Regularización del modelo base

El modelo base usaba el valor por defecto C = 1. En la versión actual, C y el tipo de penalización
se eligen por búsqueda en rejilla entre once valores de $10^{-3}$ a $10^{2}$ y las penalizaciones
L1 y L2, con validación cronológica de cinco pliegues y una separación de 30 días. El óptimo,
C = 1 con penalización L2, coincide con el valor anterior; la diferencia es que ahora es una
estimación respaldada por validación y no un supuesto. En los modelos de referencia, la
regresión logística, Ridge y Lasso se optimizan además dentro de la validación anidada con cuatro
métodos de búsqueda.

Ubicación: {doc}`02_datos`, sección «Modelo de referencia lineal»; {doc}`03_benchmarks`.

## Estaciones sin registro y periodo de modelado

Entre el 40 % y el 48 % de los valores de El Banco, San Roque, Tres Cruces y Coyongal eran
medianas asignadas a años anteriores a su instalación, entre 1972 y 1975. Ese faltante es
estructural: depende de la fecha y no del estado hidrológico, y la mediana impuesta convertía
cuatro predictores en constantes durante tres décadas del entrenamiento. Se evaluaron dos
diseños sin imputación estructural. El recorte del periodo a 1976 con las seis estaciones deja
18.359 observaciones y pierde tres décadas de registro, con ellas varios ciclos de El Niño y La
Niña; restringir los predictores a las estaciones con registro continuo, Calamar y Puerto
Salgar, conserva el periodo 1947-2026 y 28.558 observaciones. Con los mismos modelos y
la misma prueba, XGBoost obtiene una PR-AUC de 0,941 con este último diseño, frente a 0,919 con el
recorte a 1976 y 0,940 con el diseño imputado, y se adoptó. El faltante residual (1,2 % en
Calamar y 3,8 % en Puerto Salgar) se concentra en aguas bajas, un mecanismo MAR respecto del
caudal de Calamar; una imputación iterativa condicional no cambia el desempeño respecto de la
mediana (diferencia de −0,002 en PR-AUC, p = 0,57).

Ubicación: {doc}`02_datos`, secciones «Periodo de modelado» y «Modelo de referencia lineal».

## Modelos no lineales

Los modelos de referencia comprenden siete clasificadores con cuatro técnicas de balanceo y
cuatro métodos de optimización, y siete regresores con los cuatro métodos: 140 combinaciones
evaluadas con validación anidada. Los modelos no lineales superan al lineal (Random Forest
0,910 ± 0,075 frente a 0,850 ± 0,136 de la regresión logística) porque la pertenencia a la banda
no es monótona en el caudal. El análisis de sus errores motivó además el modelo propuesto, que
modela la distribución del caudal futuro.

Ubicación: {doc}`03_benchmarks` y sus secciones; {doc}`05_modelo_original`.

## Reproducibilidad

El ETL y la construcción de variables, que residían en cuadernos con rutas absolutas, se
trasladaron a módulos de Python con rutas relativas y un único punto de entrada
(`python -m scripts.etl`). El repositorio incluye un README con la instalación, los comandos de
reproducción en orden y las pruebas unitarias, y el capítulo {doc}`09_reproducibilidad` documenta
el entorno, la semilla y la tabla maestra de experimentos.

## Efecto sobre los resultados preliminares

Todos los resultados que dependen de estos cambios se recalcularon. El número de predictores
pasó de 43 a 31, y la prevalencia de la clase positiva es de 26,3 % en el entrenamiento
(1947-2009) y de 25,2 % en la prueba (2010-2026). La descarga efectiva, 9.954 m³/s, no cambia,
porque su cálculo solo usa la serie de Calamar, completa desde 1940. Dos resultados de la
revisión no favorecen al diseño adoptado y se reportan como tales: el valor óptimo de C coincide
con el que ya se usaba, y la regresión logística obtiene una PR-AUC ligeramente mayor con el
recorte a 1976 que con el diseño adoptado (0,882 frente a 0,880), una diferencia muy inferior a la
incertidumbre del intervalo por bloques.
