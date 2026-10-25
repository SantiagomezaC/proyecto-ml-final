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

# Optimización computacional

El experimento completo exigió del orden de 82.000 ajustes de modelos en un portátil con un
procesador Intel Core i7-1255U (dos núcleos de rendimiento y ocho de eficiencia, doce hilos) y
15,7 GB de memoria, sin GPU. Esta sección contrasta la complejidad teórica de cada modelo con su
tiempo empírico, evalúa variantes eficientes de cada familia de modelos y documenta la
paralelización y el perfilamiento que hicieron viable el experimento. Los tiempos los mide
`scripts/computational.py` en condiciones controladas (mismos datos, un solo hilo salvo que se
indique lo contrario) y este capítulo los analiza.

```{code-cell} ipython3
---
tags: [hide-input]
---
# Mediciones de tiempo, memoria y exactitud producidas por scripts/computational.py
import sys, json, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from myst_nb import glue
from src import estilo
from src.estilo import RIO, SED, CAU, ALE, NEU, COLORES
from src.config import RESULTADOS
from src.models import NOMBRES_MODELO
C = RESULTADOS / "computacional"
esc = pd.read_csv(C / "escalamiento.csv"); exps = pd.read_csv(C / "exponentes.csv")
```

## Complejidad teórica frente a tiempo empírico

La {numref}`fig-escalamiento` muestra cómo crece el tiempo de entrenamiento de cada modelo con el
número de observaciones (a) y de variables (b), en escala logarítmica, de modo que la pendiente
de cada recta es el exponente empírico de su complejidad.

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Tiempo de entrenamiento en función del número de observaciones (escala log-log). La pendiente de cada
      recta estima el exponente empírico de la complejidad.
    name: fig-escalamiento
---
# Tiempo de entrenamiento frente a n y frente a p, y exponentes empíricos por regresión log-log
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for i, (m, g) in enumerate(esc[esc.variable == "n"].groupby("modelo")):
    c = (COLORES + ["#7a5c99", "#2f6b3a", "#9c6b30"])[i]
    axs[0].loglog(g.n, g.t_fit, "-o", ms=4, color=c, label=NOMBRES_MODELO.get(m, m))
for i, (m, g) in enumerate(esc[esc.variable == "p"].groupby("modelo")):
    c = (COLORES + ["#7a5c99", "#2f6b3a", "#9c6b30"])[i]
    axs[1].loglog(g.p, g.t_fit, "-o", ms=4, color=c, label=NOMBRES_MODELO.get(m, m))
axs[0].set(xlabel="n (observaciones)", ylabel="Tiempo de entrenamiento (s)", title="(a) Escalamiento en n")
axs[1].set(xlabel="p (variables)", ylabel="Tiempo de entrenamiento (s)", title="(b) Escalamiento en p")
axs[0].legend(fontsize=7.5, ncol=2)
plt.tight_layout(); plt.show()
t = exps.pivot_table(index="modelo", columns="variable", values="exponente_empirico")
t = t.join(exps.drop_duplicates("modelo").set_index("modelo")[["teoria_entrenamiento", "teoria_inferencia"]])
t.index = [NOMBRES_MODELO.get(m, m) for m in t.index]
t = t.rename(columns={"n": "exponente empírico en n", "p": "exponente empírico en p",
                      "teoria_entrenamiento": "O(·) entrenamiento", "teoria_inferencia": "O(·) inferencia"})
for m, v in exps[exps.variable == "n"].set_index("modelo").exponente_empirico.items():
    glue(f"exp_{m}", float(v), display=False)
```

La tabla siguiente compara esos exponentes con el orden teórico del entrenamiento y de la
inferencia de cada modelo.

```{code-cell} ipython3
---
tags: [hide-input]
---
# Exponentes empíricos en n y en p frente a la complejidad teórica
t.round(2)
```

Los exponentes empíricos se estimaron con cinco tamaños de muestra entre 1.000 y 12.000 días y
deben leerse como pendientes locales en ese rango, no como órdenes asintóticos. Con esa
salvedad, la concordancia con la teoría es buena en los casos informativos. La SVC con núcleo
RBF escala con un exponente de {glue:text}`exp_svm:.2f`, muy cerca del comportamiento cuadrático
esperado para el problema dual de libsvm, y es la razón por la que en el experimento se usó la
aproximación de Nyström. El árbol de decisión ({glue:text}`exp_arbol:.2f`) y Random Forest
({glue:text}`exp_random_forest:.2f`) siguen el orden $n\log n$ del ordenamiento de los cortes, y
la regresión logística ({glue:text}`exp_logistica:.2f`) el orden lineal por iteración más un
número de iteraciones que crece levemente con n. El modelo propuesto escala con un exponente de
{glue:text}`exp_original:.2f`, lineal como predice la {ref}`proposición 1 <prop-complejidad-ref>`
de {doc}`05_modelo_original`. Los exponentes menores que uno de KNN ({glue:text}`exp_knn:.2f`), que en el
entrenamiento solo almacena los datos, y de XGBoost con histogramas ({glue:text}`exp_xgboost:.2f`)
no contradicen la teoría: reflejan costos fijos (validación de datos, construcción de los
histogramas, gestión de hilos) que dominan a tamaños pequeños, de modo que la pendiente en este
rango subestima el orden asintótico.

Los tiempos de esta sección los mide `scripts/computational.py`. La función `escalamiento`
entrena cada modelo con muestras de 1.000 a 12.000 días y registra el tiempo; el exponente
empírico es la pendiente de la regresión de log(tiempo) sobre log(n).

```{code-cell} ipython3
---
tags: [remove-input]
---
# Código fuente de la medición del escalamiento
import inspect
from IPython.display import Code
from scripts import computational as comp
Code("\n\n".join(inspect.getsource(f) for f in (comp.cronometrar, comp.escalamiento)), language="python")
```

## Variantes eficientes por modelo

### KNN: estructuras de indexación y vecinos aproximados

La tabla compara el tiempo por consulta, la memoria y la exactitud de cinco formas de buscar los
vecinos más cercanos.

```{code-cell} ipython3
---
tags: [hide-input]
---
# Búsqueda de vecinos: fuerza bruta, KD-Tree, Ball Tree, FAISS exacto y FAISS HNSW aproximado
knn = pd.read_csv(C / "knn.csv")
base = knn.loc[knn.variante == "sklearn-brute", "t_consulta_ms"].iloc[0]
knn["aceleración vs fuerza bruta"] = base / knn.t_consulta_ms
for v, k in (("faiss-hnsw (aprox.)", "hnsw"), ("sklearn-kd_tree", "kd"), ("faiss-flat (exacto)", "flat")):
    fila = knn[knn.variante == v].iloc[0]
    glue(f"knn_ac_{k}", float(fila["aceleración vs fuerza bruta"]), display=False)
    glue(f"knn_rec_{k}", float(fila.recall_vecinos), display=False)
glue("knn_lento_kd", float(knn.loc[knn.variante == "sklearn-kd_tree", "t_consulta_ms"].iloc[0] / base), display=False)
knn.round(4)
```

La búsqueda exhaustiva cuesta $O(np)$ por consulta. Los KD-Trees y Ball Trees reducen ese costo
a $O(\log n)$ cuando la dimensión es baja, pero con 31 variables correlacionadas el número de
ramas que deben visitarse crece y la ventaja desaparece: el KD-Tree resulta más lento que la
fuerza bruta (su tiempo por consulta es {glue:text}`knn_lento_kd:.0f` veces mayor), porque la
búsqueda exhaustiva vectorizada sobre una matriz densa aprovecha mejor la memoria caché que el
recorrido de un árbol que poda poco. FAISS {cite}`johnson2019` con índice plano calcula las mismas
distancias exactas con instrucciones vectorizadas en precisión simple
({glue:text}`knn_ac_flat:.1f` veces), y el grafo HNSW {cite}`malkov2020` encuentra vecinos
aproximados con {glue:text}`knn_ac_hnsw:.1f` veces menos tiempo y recupera el
{glue:text}`knn_rec_hnsw:.1%` de los 30 vecinos exactos. El compromiso entre exactitud y velocidad
es favorable en este problema porque la predicción promedia 30 vecinos: perder algunos y
reemplazarlos por otros casi igual de cercanos apenas cambia la proporción de positivos.

### Ridge y Lasso: solvers

La tabla compara el tiempo y el error de los solvers disponibles para los modelos lineales
regularizados.

```{code-cell} ipython3
---
tags: [hide-input]
---
# Solver directo frente a solvers iterativos en Ridge y Lasso
pd.read_csv(C / "lineales.csv").round(4)
```

Con n ≈ 22.000 y p = 31, el solver directo de Ridge (Cholesky sobre la matriz de Gram de
$p\times p$, costo $O(np^2 + p^3)$) es más rápido que SAGA {cite}`defazio2014`, cuyo costo por
época es $O(np)$ pero necesita muchas épocas para converger con variables muy colineales. SAGA
compensa cuando n crece en órdenes de magnitud o los datos son dispersos, condiciones que no se
dan aquí; la decisión de usar el solver por defecto se apoya en esta medición.

### Naive Bayes: aprendizaje incremental

La tabla compara el ajuste en un solo paso con el ajuste por lotes de distinto tamaño.

```{code-cell} ipython3
---
tags: [hide-input]
---
# fit frente a partial_fit por lotes: tiempo, memoria pico y diferencia en las probabilidades
pd.read_csv(C / "naive_bayes.csv").round(5)
```

`partial_fit` actualiza las medias y varianzas por clase lote a lote con las fórmulas de
actualización de momentos, de modo que el modelo final es el mismo que con `fit` (la diferencia
máxima en las probabilidades es del orden del error de redondeo) y la memoria pico depende del
tamaño del lote y no de n: con lotes de 500 días la memoria pico cae de unos 8 MB a 0,3 MB sin
cambio alguno en las predicciones ni en el tiempo. En este problema todo cabe en memoria y la
ventaja práctica es pequeña; la técnica sería la adecuada si el registro se ampliara a series
subdiarias de toda la red.

### XGBoost: histogramas y parada temprana

La tabla compara los tres métodos de construcción de árboles de XGBoost, la parada temprana y el
efecto del número de hilos.

```{code-cell} ipython3
---
tags: [hide-input]
---
# tree_method exact, approx e hist; parada temprana cronológica; número de hilos
xg = pd.read_csv(C / "xgboost.csv")
t_exact = xg.loc[xg.variante == "tree_method=exact", "t_fit"].iloc[0]
t_hist = xg.loc[xg.variante == "tree_method=hist", "t_fit"].iloc[0]
glue("xgb_ac_hist", float(t_exact / t_hist), display=False)
es = xg[xg.variante.str.contains("parada")].iloc[0]
glue("xgb_es_arboles", int(es.n_arboles), display=False)
xg.round(4)
```

El método exacto ordena cada variable en cada nodo y evalúa todos los cortes posibles; el método
por histogramas discretiza una vez cada variable en un número fijo de bins y acumula gradientes
por bin, lo que reduce el costo por nivel de $O(n\,p\log n)$ a $O(np + p\cdot\text{bins})$. La
aceleración medida es de {glue:text}`xgb_ac_hist:.1f` veces con PR-AUC prácticamente igual. La
parada temprana sobre el último 15 % cronológico del entrenamiento detiene el ajuste en
{glue:text}`xgb_es_arboles` árboles. El modelo propuesto usa la misma idea de histogramas,
programada desde cero.

### SVM: núcleo completo frente a aproximaciones lineales

La tabla compara la SVC con núcleo RBF completo con tres alternativas de costo lineal.

```{code-cell} ipython3
---
tags: [hide-input]
---
# SVC RBF frente a LinearSVC, descenso estocástico y aproximación de Nyström
sv = pd.read_csv(C / "svm.csv")
t_rbf = sv.loc[sv.variante.str.startswith("SVC"), "t_fit"].iloc[0]
sv["aceleración vs SVC"] = t_rbf / sv.t_fit
ny = sv[sv.variante.str.startswith("Nyström")].iloc[0]
glue("ny_ac", float(ny["aceleración vs SVC"]), display=False)
glue("ny_dif", float(ny.pr_auc - sv.loc[sv.variante.str.startswith("SVC"), "pr_auc"].iloc[0]), display=False)
glue("ny_mem", float(ny.memoria_pico_mb), display=False)
sv.round(4)
```

La SVC con núcleo RBF resuelve un problema cuadrático dual cuyo costo crece entre $O(n^2p)$ y
$O(n^3)$, y su inferencia es $O(n_{sv}\,p)$ por consulta. LinearSVC y el descenso de gradiente
estocástico eliminan el núcleo y escalan linealmente, pero pierden la frontera no lineal que el
problema requiere. La aproximación de Nyström {cite}`williams2001` con 500 componentes proyecta los
datos en un espacio de características donde el núcleo es aproximadamente un producto interno, y
recupera la capacidad no lineal con un costo lineal en n: es {glue:text}`ny_ac:.1f` veces más
rápida que la SVC con una diferencia de {glue:text}`ny_dif:+.4f` en PR-AUC, y su inferencia es
un orden de magnitud más rápida porque no depende del número de vectores de soporte. El costo se
traslada a la memoria: la matriz de características de Nyström ($n \times 500$ en doble
precisión) eleva la memoria pico del ajuste de unos 7 MB a {glue:text}`ny_mem:.0f` MB, un
compromiso aceptable en esta máquina pero que habría que vigilar con muestras mucho mayores. Esta
comparación de velocidad usa 500 componentes; en la validación anidada se usaron 600, para perder
menos exactitud frente al núcleo completo.

Las variantes eficientes de KNN, XGBoost y la SVM se miden con las funciones siguientes del
mismo script, que entrenan la versión estándar y la optimizada sobre los mismos datos y
registran tiempo, memoria y PR-AUC.

```{code-cell} ipython3
---
tags: [remove-input]
---
# Código fuente de la medición de las variantes eficientes
Code("\n\n".join(inspect.getsource(f) for f in (comp.knn_variantes, comp.xgb_variantes, comp.svm_variantes)),
     language="python")
```

## Paralelización

La {numref}`fig-paralelo` mide la aceleración que se obtiene al aumentar el número de procesos o
hilos en dos escenarios: el paralelismo interno de Random Forest y el paralelismo entre ajustes
independientes.

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Aceleración en función del número de trabajadores, para el paralelismo interno de Random Forest
      (n_jobs) y para el paralelismo entre ajustes independientes con joblib (backend loky).
    name: fig-paralelo
---
# Aceleración por número de trabajadores; horas de CPU y de reloj del experimento real
par = pd.read_csv(C / "paralelizacion.csv")
fig, ax = plt.subplots(figsize=(7.5, 3.6))
for (esc_, g), c in zip(par.groupby("escenario"), (RIO, SED)):
    ax.plot(g.trabajadores, g.aceleracion, "-o", color=c, label=esc_)
ax.plot([1, 10], [1, 10], ":", color=NEU, label="Aceleración ideal")
ax.set(xlabel="Trabajadores", ylabel="Aceleración", title="Escalamiento del paralelismo"); ax.legend(fontsize=8)
plt.show()
for e, g in par.groupby("escenario"):
    glue("acel_" + ("rf" if "Random" in e else "joblib"), float(g.aceleracion.max()), display=False)
gpu = json.loads((C / "gpu.json").read_text(encoding="utf-8"))
import re
from datetime import datetime
log = (RESULTADOS / "log_cadena.txt").read_text(encoding="utf-8")
t_ini = re.findall(r"== (\d\d:\d\d:\d\d) INICIO clf", log)[-1]; t_fin = re.findall(r"== (\d\d:\d\d:\d\d) INICIO reg", log)[-1]
dur = (datetime.strptime(t_fin, "%H:%M:%S") - datetime.strptime(t_ini, "%H:%M:%S")).seconds / 3600
if dur <= 0: dur += 24
pc = pd.read_parquet(RESULTADOS / "pliegues_clf.parquet")
horas_cpu = (pc.t_opt.sum() + pc.t_fit.sum()) / 3600
glue("horas_tarea", horas_cpu, display=False); glue("horas_reloj", dur, display=False)
glue("concurrencia", horas_cpu / dur, display=False)
```

El experimento se paralelizó entre tareas y no dentro de ellas: cada tupla (modelo, balanceo,
optimizador, pliegue) es independiente, de modo que diez procesos de `joblib` con el backend
`loky` ejecutan diez optimizaciones a la vez, cada una con estimadores de un solo hilo. Esta
estrategia evita la sobresuscripción (diez procesos con Random Forest de diez hilos competirían
por doce hilos) y resultó eficaz en el experimento real: las 1.344 optimizaciones de
clasificación sumaron {glue:text}`horas_tarea:.1f` horas de tiempo de tarea y se completaron en
{glue:text}`horas_reloj:.1f` horas de reloj, una concurrencia efectiva de
{glue:text}`concurrencia:.1f` procesos simultáneos. La {numref}`fig-paralelo` muestra el
reverso de la medalla: con tareas muy cortas (diez ajustes de XGBoost de menos de medio segundo)
el paralelismo entre procesos no acelera nada, porque el costo de lanzar los procesos y
serializar los datos supera al del cómputo, mientras que el paralelismo interno de Random Forest
alcanza una aceleración de {glue:text}`acel_rf:.1f` con diez hilos. Ninguna curva se acerca a la
ideal porque ocho de los diez núcleos del procesador son de eficiencia, más lentos que los dos de
rendimiento, y la frecuencia baja con la carga sostenida. La lección es que el paralelismo debe
aplicarse al nivel de grano grueso, el de las tareas de minutos, que es como se organizó el
experimento. Con 1.680 optimizaciones y una duración típica de minutos, herramientas distribuidas como Dask o Ray
no se justificaron: el cuello de botella es el número de núcleos, no la coordinación. El equipo
no tiene GPU NVIDIA, de modo que la aceleración con `device='cuda'` de XGBoost no pudo medirse.

```{code-cell} ipython3
---
tags: [remove-input]
---
# Código fuente de la medición de la paralelización
Code(inspect.getsource(comp.paralelizacion), language="python")
```

## Perfilamiento

Para saber dónde se gasta el tiempo dentro de una tarea, se perfiló una tarea elemental con
`cProfile` y se midió la memoria pico del ajuste de cada modelo. Las dos tablas siguientes
muestran las funciones que más tiempo acumulan y la memoria por modelo.

```{code-cell} ipython3
---
tags: [hide-input]
---
# Perfil de CPU de una tarea elemental y memoria pico del ajuste de cada modelo
perf = pd.read_csv(C / "perfil.csv")
mem = pd.read_csv(C / "memoria.csv"); mem["modelo"] = mem.modelo.map(lambda m: NOMBRES_MODELO.get(m, m))
display(perf.head(12).round(3))
mem.round(2)
```

El perfil de CPU de una tarea elemental (XGBoost con SMOTE y TPE en un pliegue) muestra que cerca
de las tres cuartas partes del tiempo se consumen en la actualización de los árboles de XGBoost
(`core.update`) dentro de la validación interna, mientras que la generación de sintéticos de
SMOTE, la construcción de las matrices de datos y la lógica del optimizador (muestreo de TPE,
registro de Optuna) son marginales. La conclusión práctica fue que la única forma efectiva de acortar el
experimento era reducir el costo de cada ajuste: de ahí la submuestra de 0,3 en Random Forest, la aproximación de Nyström en la SVM, el
método `hist` en XGBoost y el despacho de las tareas más costosas primero para equilibrar la
carga de los procesos.

## Modelo estándar frente a optimizado

La tabla final reúne, para los tres modelos en que la optimización computacional cambió la
decisión de diseño, la versión estándar y la optimizada con su complejidad, sus tiempos, su
memoria y su efecto sobre la PR-AUC.

```{code-cell} ipython3
---
tags: [hide-input]
---
# Resumen: versión estándar frente a optimizada en KNN, XGBoost y SVM
filas = [
    ("KNN", "fuerza bruta", "FAISS HNSW", "O(n·p) por consulta", "O(log n) aprox.", knn.loc[knn.variante == "sklearn-brute", "t_consulta_ms"].iloc[0],
     knn.loc[knn.variante == "faiss-hnsw (aprox.)", "t_consulta_ms"].iloc[0], "ms/consulta",
     knn.loc[knn.variante == "faiss-hnsw (aprox.)", "pr_auc"].iloc[0] - knn.loc[knn.variante == "sklearn-brute", "pr_auc"].iloc[0]),
    ("XGBoost", "exact", "hist", "O(n·p log n) por nivel", "O(n·p + p·B) por nivel", t_exact, t_hist, "s (entrenamiento)",
     xg.loc[xg.variante == "tree_method=hist", "pr_auc"].iloc[0] - xg.loc[xg.variante == "tree_method=exact", "pr_auc"].iloc[0]),
    ("SVM", "SVC RBF", "Nyström + LinearSVC", "O(n²·p) a O(n³)", "O(n·m·p)", t_rbf, ny.t_fit, "s (entrenamiento)", ny.pr_auc - sv.loc[sv.variante.str.startswith("SVC"), "pr_auc"].iloc[0]),
]
tabla_so = pd.DataFrame(filas, columns=["Modelo", "Estándar", "Optimizado", "O(·) estándar", "O(·) optimizado", "Tiempo estándar",
                                        "Tiempo optimizado", "Unidad", "Δ PR-AUC"])
mem_std = [knn.loc[knn.variante == "sklearn-brute", "memoria_pico_mb"].iloc[0],
           xg.loc[xg.variante == "tree_method=exact", "memoria_pico_mb"].iloc[0],
           sv.loc[sv.variante.str.startswith("SVC"), "memoria_pico_mb"].iloc[0]]
mem_opt = [knn.loc[knn.variante == "faiss-hnsw (aprox.)", "memoria_pico_mb"].iloc[0],
           xg.loc[xg.variante == "tree_method=hist", "memoria_pico_mb"].iloc[0], ny.memoria_pico_mb]
inf_std = [knn.loc[knn.variante == "sklearn-brute", "t_consulta_ms"].iloc[0],
           xg.loc[xg.variante == "tree_method=exact", "t_inf_ms_fila"].iloc[0], sv.loc[sv.variante.str.startswith("SVC"), "t_inf_ms_fila"].iloc[0]]
inf_opt = [knn.loc[knn.variante == "faiss-hnsw (aprox.)", "t_consulta_ms"].iloc[0],
           xg.loc[xg.variante == "tree_method=hist", "t_inf_ms_fila"].iloc[0], ny.t_inf_ms_fila]
tabla_so.insert(7, "Inferencia estándar (ms/fila)", inf_std); tabla_so.insert(8, "Inferencia optimizada (ms/fila)", inf_opt)
tabla_so["Memoria estándar (MB)"] = mem_std; tabla_so["Memoria optimizada (MB)"] = mem_opt
tabla_so.round(4)
```

En los tres casos la versión optimizada reduce el tiempo en uno o más órdenes de magnitud con un
cambio de PR-AUC del orden de milésimas, que es menor que la variación entre semillas del propio
modelo. El costo aparece en otra dimensión: HNSW cambia exactitud por velocidad al recuperar solo
una parte de los vecinos exactos, y Nyström cambia memoria por tiempo. Ninguno de esos costos
afectó las conclusiones del estudio, y sin estas optimizaciones la validación anidada de las 140
combinaciones no habría sido viable en un computador portátil.
