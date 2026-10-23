# Descarga efectiva en el río Magdalena

**Kevin Clemente Rosario · Manuel Santiago Meza Castro**
Maestría · Universidad del Norte · Machine Learning, 2026-30

```{admonition} Productos del proyecto
:class: tip
- **Dashboard en línea:** URL_DASHBOARD
- **Repositorio:** <https://github.com/SantiagomezaC/proyecto-ml-final>
```

## Resumen ejecutivo

RESUMEN_EJECUTIVO

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
