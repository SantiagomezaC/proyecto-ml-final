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

# Modelos benchmark: diseño experimental

Este capítulo y sus secciones desarrollan el Entregable 2. Aquí se fija el diseño común a las
140 combinaciones (validación anidada, espacios de búsqueda, optimizadores y técnicas de
balanceo); las secciones siguientes presentan los resultados de clasificación, la comparación
de los optimizadores, la optimización computacional, la tarea de regresión, la comparación
estadística y la interpretabilidad.

```{code-cell} ipython3
---
tags: [hide-input]
---
import sys, json; sys.path.insert(0, "..")
import numpy as np, pandas as pd
from myst_nb import glue
from src.config import DATOS_PROC, SEED
from src.splits import cargar_pliegues
from src.models import ESPACIOS_CLF, ESPACIOS_REG, NOMBRES_MODELO
from src.spaces import rejilla
pl = cargar_pliegues(DATOS_PROC / "pliegues_calamar.json")
tabla_pl = pd.DataFrame([{"pliegue": p["pliegue"], "bloque de prueba": p["bloque"], "n entrenamiento": len(p["train"]),
                          "n prueba": len(p["test"])} for p in pl])
glue("n_pliegues", len(pl), display=False)
glue("ntr_min", int(tabla_pl["n entrenamiento"].min()), display=False)
glue("ntr_max", int(tabla_pl["n entrenamiento"].max()), display=False)
tabla_pl
```

## El pipeline combinatorio

Cada modelo base se combina con cada técnica de balanceo y cada método de optimización. En
clasificación hay siete modelos (KNN, Naive Bayes, regresión logística con penalización L1 o
L2, árbol de decisión, Random Forest, XGBoost y SVM con núcleo RBF aproximado), cuatro estrategias de
balanceo (ninguna, SMOTE, ADASYN y ponderación por clase) y cuatro optimizadores (rejilla,
aleatorio, bayesiano con TPE y genético con DEAP), es decir, 7 × 4 × 4 = 112 combinaciones.
En regresión, donde el balanceo no aplica, los siete modelos (KNN, Ridge, Lasso, árbol, Random
Forest, XGBoost y SVR) se combinan con los cuatro optimizadores, 28 combinaciones más. Cada
combinación se entrena y evalúa en los {glue:text}`n_pliegues` pliegues externos, de modo que el
experimento completo comprende 140 × 12 = 1.680 optimizaciones independientes. La ejecución se
automatiza con `src/experiment.py`, que despacha cada tupla (modelo, balanceo, optimizador,
pliegue) a un proceso de `joblib`, guarda su resultado en un archivo propio y permite reanudar
una corrida interrumpida sin repetir trabajo.

## Validación anidada cronológica

Reportar la métrica del mejor modelo sobre el mismo esquema que sirvió para elegir sus
hiperparámetros produce una estimación optimista, el sesgo de selección descrito por Varma y
Simon {cite}`varma2006` y por Cawley y Talbot {cite}`cawley2010`. El protocolo separa dos
niveles. El bucle externo divide el periodo 1986-2009 en doce bloques de calendario de dos años;
para cada bloque, el entrenamiento comprende todos los días anteriores menos una separación de
30 días, de modo que la ventana de entrenamiento crece de {glue:text}`ntr_min:,` a
{glue:text}`ntr_max:,` días. El bucle interno,
que ejecuta cada optimizador, aplica `TimeSeriesSplit` con tres pliegues y la misma separación
dentro del entrenamiento externo. La métrica que se reporta proviene siempre del bloque externo,
que el optimizador nunca vio.

La elección de los bloques responde a tres hallazgos del análisis exploratorio. La
autocorrelación (0,92 a quince días) impide cualquier partición aleatoria, que dejaría días
casi idénticos a ambos lados del corte {cite}`bergmeir2012`; la separación de 30 días supera
el horizonte de predicción y el tramo en que la autocorrelación cae por debajo de 0,75. La
prevalencia anual varía de 0 % a 54 % con las fases de ENSO, de modo que un bloque de dos años
contiene al menos un ciclo hidrológico completo y, en todos los casos, días positivos. Por
último, doce bloques ofrecen a las pruebas de rangos y de Wilcoxon una potencia razonable: con
cinco bloques, el p-valor mínimo alcanzable por Wilcoxon sería 0,031, insuficiente tras una
corrección de Holm. La prueba reservada 2010-2026 no interviene en ninguna decisión de este
capítulo; se usa solo para evaluar una vez los modelos finales.

La métrica de selección en el bucle interno es la PR-AUC (precisión promedio), coherente con el
desbalance y con el interés operativo en la clase positiva; en regresión es la raíz del error
cuadrático medio.

## Espacios de búsqueda

```{code-cell} ipython3
---
tags: [hide-input]
---
filas = []
for tarea, esp in (("clasificación", ESPACIOS_CLF), ("regresión", ESPACIOS_REG)):
    for m, params in esp.items():
        for p in params:
            filas.append({"tarea": tarea, "modelo": NOMBRES_MODELO.get(m, m), "hiperparámetro": p.nombre,
                          "tipo": p.tipo, "rango": (", ".join(map(str, p.opciones)) if p.tipo == "cat"
                                                    else f"[{p.bajo:g}, {p.alto:g}]"),
                          "escala": "log" if p.log else ("—" if p.tipo == "cat" else "lineal"),
                          "puntos de rejilla": len(rejilla(params, 16))})
pd.DataFrame(filas)
```

Cada espacio se declara una sola vez en `src/models.py` como una lista de objetos `Param`, y
`src/spaces.py` lo traduce a los cuatro optimizadores, de modo que todos exploran exactamente
el mismo dominio. Los rangos se fijaron con criterios del problema y no de forma arbitraria. En
KNN, el número de vecinos va de 5 a 200 en escala logarítmica: con la autocorrelación del caudal,
los vecinos más cercanos de un día son sus propios días contiguos, y valores pequeños sobreajustan.
En la regresión logística y la SVM, C y γ recorren cuatro o cinco órdenes de magnitud en escala
logarítmica, como recomienda la guía para parámetros de regularización y de escala. En los
árboles, la profundidad máxima (2 a 16) y el tamaño mínimo de hoja (5 a 300 días, logarítmico)
controlan la complejidad frente a un tamaño efectivo reducido. En Random Forest se fijó
`max_samples = 0,3`, con un efecto doble: la submuestra decorrelaciona los árboles y reduce el
costo de cada ajuste en proporción, decisivo con más de 22.000 días de entrenamiento en esta
máquina. La SVM y la SVR usan el núcleo RBF aproximado por el método de Nyström con 600
componentes, seguido de una SVM lineal, para evitar la complejidad cuadrática o cúbica del
núcleo completo con una muestra de este tamaño; la sección de optimización
computacional cuantifica el ahorro y la pérdida de exactitud frente a la SVC exacta. En XGBoost, la tasa de aprendizaje (0,01 a 0,3) y el número de árboles (80 a 600)
se exploran conjuntamente porque su producto determina la capacidad efectiva.

## Métodos de optimización

Los cuatro métodos reciben el mismo presupuesto de 16 evaluaciones por pliegue externo, cada
una con tres ajustes internos, y comparten la función objetivo `evaluar_config` de
`src/optim.py`. Esa función reconstruye el pipeline completo (imputación, escalado, balanceo y
modelo) en cada pliegue interno, de modo que el preprocesamiento y el sobremuestreo se ajustan
solo con la porción de entrenamiento también en Optuna y DEAP, que no integran esa protección
de forma nativa.

**Rejilla.** La rejilla se construye con todas las opciones de cada categórico y niveles
equiespaciados (en la escala del parámetro) para los numéricos, añadidos de forma rotativa
mientras el producto no supere el presupuesto. Su debilidad conocida es que, con varios
parámetros, asigna pocos niveles a cada uno y desperdicia evaluaciones en dimensiones poco
influyentes {cite}`bergstra2012`.

**Aleatoria.** Dieciséis configuraciones muestreadas de forma uniforme o log-uniforme. Con el
mismo presupuesto explora más valores distintos de cada parámetro que la rejilla, lo que
importa cuando pocos parámetros dominan la respuesta.

**Bayesiana (Optuna, TPE).** El modelo sustituto es el *Tree-structured Parzen Estimator*
{cite}`bergstra2011`, que no modela $p(y \mid x)$ sino dos densidades sobre las configuraciones:
$\ell(x)$, ajustada a las que superan un cuantil $\gamma$ del puntaje, y $g(x)$, ajustada al
resto. La función de adquisición es la mejora esperada, que bajo esta parametrización es
proporcional a $\left[\gamma + (1-\gamma)\,g(x)/\ell(x)\right]^{-1}$, de modo que maximizarla
equivale a elegir el candidato con mayor cociente $\ell(x)/g(x)$. El balance entre exploración y
explotación lo determinan las seis evaluaciones iniciales aleatorias, que exploran el espacio
antes de que el cociente concentre la búsqueda, y el carácter multivariado del muestreador, que
modela las dependencias entre parámetros (por ejemplo, entre la tasa de aprendizaje y el número
de árboles) en lugar de tratarlos como independientes.

**Genética (DEAP).** Cada individuo es un vector de genes en $[0,1]^d$ que se decodifica al
espacio de cada parámetro con su escala {cite}`fortin2012`. La selección es por torneo de
tamaño 3, que mantiene una presión selectiva moderada y, a diferencia de la ruleta, no depende
de la escala del puntaje, propiedad importante cuando las configuraciones difieren en
milésimas de PR-AUC. El cruce es
de mezcla (`cxBlend`, α = 0,5) con probabilidad 0,7, adecuado para genes continuos porque genera
hijos dentro y ligeramente fuera del segmento entre los padres. La mutación es gaussiana
(σ = 0,2 sobre el gen, probabilidad por gen 0,3) con probabilidad 0,3, con recorte a $[0,1]$. Con
un presupuesto de 16, la población es de seis individuos y evoluciona durante dos generaciones;
el elitismo de un individuo preserva el mejor hallado. La diversidad genética se mide en cada
generación como la distancia euclídea media entre pares de la población, indicador de
convergencia prematura.

**Multifidelidad.** Como técnica adicional se implementa *Successive Halving*
{cite}`jamieson2016` con el número de árboles como recurso para XGBoost y Random Forest, y se
compara con la búsqueda aleatoria sobre el mismo conjunto de candidatos (sección de optimización).

## Técnicas de balanceo

SMOTE {cite}`chawla2002` genera instancias sintéticas de la clase positiva interpolando entre
cada caso y uno de sus cinco vecinos más cercanos de la misma clase; ADASYN {cite}`he2008`
reparte la generación según la proporción de vecinos de la clase contraria, concentrándola en
los casos difíciles, cerca de la frontera. Ambas se aplican dentro de un `imblearn.Pipeline`,
después del escalado y solo durante `fit`, de modo que nunca se generan sintéticos con
información de validación o de prueba. La ponderación por clase usa `class_weight='balanced'`
donde el estimador la admite; en XGBoost se traduce en `scale_pos_weight` igual al cociente
entre negativos y positivos; en Naive Bayes, en probabilidades a priori iguales; y en KNN, que no
admite pesos, se implementó la clase `KNNBalanceado`, que divide la fracción de vecinos de cada
clase por su prevalencia y renormaliza, lo que equivale a ponderar cada voto por la inversa de
la frecuencia de su clase.

El sobremuestreo en series de tiempo merece una advertencia. SMOTE interpola entre días que la
autocorrelación vuelve casi idénticos, de modo que los sintéticos se parecen a días reales
cercanos y aportan menos diversidad de la que aportarían en datos independientes. El efecto se
evalúa empíricamente en la sección siguiente.

## Regla de ausencia de fuga

Todo paso que aprende de los datos (imputación, escalado, sobremuestreo, discretización del
modelo original, recalibración de probabilidades) se ajusta dentro del pliegue que corresponde:
el bucle interno con la porción interna de entrenamiento, el reentrenamiento externo con el
entrenamiento del bloque, y el modelo final con 1947-2009. Las particiones externas se
calculan una vez y se guardan en `data/processed/pliegues_calamar.json`; todos los modelos,
incluido el modelo original del capítulo 5, leen ese mismo archivo.
