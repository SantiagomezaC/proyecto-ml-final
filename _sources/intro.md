# Descarga efectiva en el río Magdalena

**Kevin Clemente Rosario · Manuel Santiago Meza Castro**
Maestría · Universidad del Norte · Machine Learning, 2026-30

```{admonition} Productos del proyecto
:class: tip
- **Dashboard en línea:** <https://descarga-efectiva-magdalena.onrender.com>
- **Repositorio:** <https://github.com/SantiagomezaC/proyecto-ml-final>
```

## Resumen ejecutivo

Este libro documenta los entregables 2 y 3 del proyecto. A partir de 28.558 días de registro del
IDEAM (1947-2026) en Calamar y Puerto Salgar, se predice si el caudal del día *t* + 15 caerá en
la banda de ±15 % alrededor de la descarga efectiva de Calamar (9.954 m³/s), una tarea de
clasificación con 26 % de casos positivos, y como tarea auxiliar se predice el propio caudal.

Se entrenaron y compararon 140 combinaciones de modelos, técnicas de balanceo y métodos de
optimización de hiperparámetros con validación anidada cronológica en doce bloques de dos años.
El mejor benchmark es Random Forest (PR-AUC externa de 0,910 ± 0,075), estadísticamente
indistinguible de XGBoost, SVM y KNN. El modelo original, Band-Likelihood Gradient Boosting,
programado en numpy, modela la distribución del caudal futuro y optimiza una pérdida
estrictamente propia que combina la probabilidad del evento con la verosimilitud del caudal.
Supera a todos los benchmarks en los doce bloques (0,943 ± 0,053) y en la prueba reservada
2010-2026 (0,965 frente a 0,950), con significancia en tres pruebas (Wilcoxon-Holm, t corregida de
Nadeau-Bengio y bootstrap estacionario) y un efecto mediano a grande, aunque queda empatado con
métodos distribucionales recientes como NGBoost. Los modelos se ponen a disposición de usuarios
no expertos en un dashboard con predicción en vivo, intervalo del caudal previsto y explicación
local de cada predicción.

## Problema

El río Magdalena drena cerca de la cuarta parte de Colombia y entrega al mar Caribe del orden
de 144 millones de toneladas de sedimento al año {cite}`restrepo2000`, uno de los rendimientos
sedimentarios más altos de Suramérica {cite}`milliman1992`. La mayor parte de ese trabajo
geomorfológico no la realizan las crecidas excepcionales ni los caudales bajos y frecuentes,
sino un rango intermedio de caudales, la descarga efectiva {cite}`wolman1960`, que en la
estación Calamar se ubica alrededor de 9.954 m³/s. La literatura usa la descarga efectiva como
un descriptor de largo plazo; este proyecto la convierte en un objetivo de predicción:
anticipar, con quince días de antelación, si el río estará dentro de la banda de ±15 % alrededor
de ese valor, información útil para programar purgas de embalses, dragados del canal navegable
y campañas de muestreo de sedimentos.

## Objetivos

1. Corregir el análisis del Entregable 1 según la retroalimentación recibida y ampliar el
   análisis exploratorio.
2. Entrenar, optimizar y comparar estadísticamente 140 combinaciones de modelos, técnicas de
   balanceo y métodos de optimización con validación anidada cronológica (Entregable 2).
3. Diseñar, implementar y validar un modelo original que aproveche la naturaleza continua de la
   variable que define la clase, y verificar si supera a los benchmarks con significancia
   estadística (Entregable 3).
4. Llevar los mejores modelos a un dashboard desplegado y documentar todo de forma
   reproducible.

## Mapa del libro

El capítulo 1 registra las correcciones al Entregable 1. El capítulo 2 reconstruye el ETL y el
análisis exploratorio. El capítulo 3 y sus secciones presentan el Entregable 2: diseño
experimental, resultados de clasificación, comparación de optimizadores, optimización
computacional, regresión, comparación estadística e interpretabilidad. Los capítulos 4 a 6
desarrollan el modelo original: revisión de literatura y declaración de novedad, formulación,
teoría e implementación, y validación estadística. El capítulo 7 describe el dashboard y su
despliegue, el capítulo 8 reúne las conclusiones y el veredicto sobre las condiciones R1, R2 y
R3, y el capítulo 9 documenta la reproducibilidad y la bibliografía.
