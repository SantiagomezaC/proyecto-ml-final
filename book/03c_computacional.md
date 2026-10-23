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
tiempo empírico, evalúa las variantes eficientes que propone la guía y documenta la
paralelización y el perfilamiento que hicieron viable el experimento.

```{code-cell} ipython3
---
tags: [hide-input]
---
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

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Tiempo de entrenamiento en función del número de observaciones (escala log-log). La pendiente de cada
      recta estima el exponente empírico de la complejidad.
    name: fig-escalamiento
---
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
t.round(2)
```

INTERPRETACION_COMPLEJIDAD

## Variantes eficientes por modelo

### KNN: estructuras de indexación y vecinos aproximados

```{code-cell} ipython3
---
tags: [hide-input]
---
knn = pd.read_csv(C / "knn.csv")
base = knn.loc[knn.variante == "sklearn-brute", "t_consulta_ms"].iloc[0]
knn["aceleración vs fuerza bruta"] = base / knn.t_consulta_ms
for v, k in (("faiss-hnsw (aprox.)", "hnsw"), ("sklearn-kd_tree", "kd"), ("faiss-flat (exacto)", "flat")):
    fila = knn[knn.variante == v].iloc[0]
    glue(f"knn_ac_{k}", float(fila["aceleración vs fuerza bruta"]), display=False)
    glue(f"knn_rec_{k}", float(fila.recall_vecinos), display=False)
knn.round(4)
```

La búsqueda exhaustiva cuesta $O(np)$ por consulta. Los KD-Trees y Ball Trees reducen ese costo
a $O(\log n)$ cuando la dimensión es baja, pero con 43 variables correlacionadas el número de
ramas que deben visitarse crece y la ventaja se diluye; el KD-Tree consigue una aceleración de
{glue:text}`knn_ac_kd:.1f` veces. FAISS {cite}`johnson2019` con índice plano calcula las mismas
distancias exactas con instrucciones vectorizadas en precisión simple
({glue:text}`knn_ac_flat:.1f` veces), y el grafo HNSW {cite}`malkov2020` encuentra vecinos
aproximados con {glue:text}`knn_ac_hnsw:.1f` veces menos tiempo y recupera el
{glue:text}`knn_rec_hnsw:.1%` de los 30 vecinos exactos. El compromiso entre exactitud y velocidad
es favorable en este problema porque la predicción promedia 30 vecinos: perder algunos y
reemplazarlos por otros casi igual de cercanos apenas cambia la proporción de positivos.

### Ridge y Lasso: solvers

```{code-cell} ipython3
---
tags: [hide-input]
---
pd.read_csv(C / "lineales.csv").round(4)
```

Con n ≈ 12.000 y p = 43, el solver directo de Ridge (Cholesky sobre la matriz de Gram de
$p\times p$, costo $O(np^2 + p^3)$) es más rápido que SAGA {cite}`defazio2014`, cuyo costo por
época es $O(np)$ pero necesita muchas épocas para converger con variables muy colineales. SAGA
compensa cuando n crece en órdenes de magnitud o los datos son dispersos, condiciones que no se
dan aquí; la decisión de usar el solver por defecto se apoya en esta medición.

### Naive Bayes: aprendizaje incremental

```{code-cell} ipython3
---
tags: [hide-input]
---
pd.read_csv(C / "naive_bayes.csv").round(5)
```

`partial_fit` actualiza las medias y varianzas por clase lote a lote con las fórmulas de
actualización de momentos, de modo que el modelo final es el mismo que con `fit` (la diferencia
máxima en las probabilidades es del orden del error de redondeo) y la memoria pico depende del
tamaño del lote y no de n. En este problema todo cabe en memoria y la ventaja es nula; la
técnica sería la adecuada si el registro se ampliara a series subdiarias de toda la red.

### XGBoost: histogramas y parada temprana

```{code-cell} ipython3
---
tags: [hide-input]
---
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
{glue:text}`xgb_es_arboles` árboles. El modelo original usa la misma idea de histogramas,
programada desde cero.

### SVM: núcleo completo frente a aproximaciones lineales

```{code-cell} ipython3
---
tags: [hide-input]
---
sv = pd.read_csv(C / "svm.csv")
t_rbf = sv.loc[sv.variante.str.startswith("SVC"), "t_fit"].iloc[0]
sv["aceleración vs SVC"] = t_rbf / sv.t_fit
ny = sv[sv.variante.str.startswith("Nyström")].iloc[0]
glue("ny_ac", float(ny["aceleración vs SVC"]), display=False)
glue("ny_dif", float(ny.pr_auc - sv.loc[sv.variante.str.startswith("SVC"), "pr_auc"].iloc[0]), display=False)
sv.round(4)
```

La SVC con núcleo RBF resuelve un problema cuadrático dual cuyo costo crece entre $O(n^2p)$ y
$O(n^3)$, y su inferencia es $O(n_{sv}\,p)$ por consulta. LinearSVC y el descenso de gradiente
estocástico eliminan el núcleo y escalan linealmente, pero pierden la frontera no lineal que el
problema requiere. La aproximación de Nyström {cite}`williams2001` con 500 componentes proyecta los
datos en un espacio de características donde el núcleo es aproximadamente un producto interno, y
recupera la capacidad no lineal con un costo lineal en n: es {glue:text}`ny_ac:.1f` veces más
rápida que la SVC con una diferencia de {glue:text}`ny_dif:+.4f` en PR-AUC.

## Paralelización

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Aceleración en función del número de trabajadores, para el paralelismo interno de Random Forest
      (n_jobs) y para el paralelismo entre ajustes independientes con joblib (backend loky).
    name: fig-paralelo
---
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
```

El experimento se paralelizó entre tareas y no dentro de ellas: cada tupla (modelo, balanceo,
optimizador, pliegue) es independiente, de modo que diez procesos de `joblib` con el backend
`loky` ejecutan diez optimizaciones a la vez, cada una con estimadores de un solo hilo. Esta
estrategia evita la sobresuscripción (diez procesos con Random Forest de diez hilos competirían
por doce hilos) y alcanza una aceleración de {glue:text}`acel_joblib:.1f` con diez trabajadores
frente a {glue:text}`acel_rf:.1f` del paralelismo interno de Random Forest. Ninguna de las dos
llega a diez porque ocho de los diez núcleos del procesador son de eficiencia, más lentos que los
dos de rendimiento, y porque la frecuencia del procesador baja con la carga sostenida. Con
1.680 optimizaciones y una duración típica de minutos, herramientas distribuidas como Dask o Ray
no se justificaron: el cuello de botella es el número de núcleos, no la coordinación. El equipo
no tiene GPU NVIDIA, de modo que la aceleración con `device='cuda'` de XGBoost no pudo medirse.

## Perfilamiento

```{code-cell} ipython3
---
tags: [hide-input]
---
perf = pd.read_csv(C / "perfil.csv")
mem = pd.read_csv(C / "memoria.csv"); mem["modelo"] = mem.modelo.map(lambda m: NOMBRES_MODELO.get(m, m))
display(perf.head(12).round(3))
mem.round(2)
```

El perfil de CPU de una tarea elemental (XGBoost con SMOTE y TPE en un pliegue) muestra que el
tiempo se concentra en el ajuste de los estimadores dentro de la validación interna y en la
generación de sintéticos, mientras que la lógica del optimizador (muestreo de TPE, bookkeeping
de Optuna) es despreciable. La conclusión práctica fue que la única forma efectiva de acortar el
experimento era reducir el costo de cada ajuste: de ahí la submuestra de 0,6 en Random Forest, el
método `hist` en XGBoost y el despacho de las tareas más costosas primero para equilibrar la
carga de los procesos.

## Modelo estándar frente a optimizado

```{code-cell} ipython3
---
tags: [hide-input]
---
filas = [
    ("KNN", "fuerza bruta", "FAISS HNSW", "O(n·p) por consulta", "O(log n) aprox.", knn.loc[knn.variante == "sklearn-brute", "t_consulta_ms"].iloc[0],
     knn.loc[knn.variante == "faiss-hnsw (aprox.)", "t_consulta_ms"].iloc[0], "ms/consulta",
     knn.loc[knn.variante == "faiss-hnsw (aprox.)", "pr_auc"].iloc[0] - knn.loc[knn.variante == "sklearn-brute", "pr_auc"].iloc[0]),
    ("XGBoost", "exact", "hist", "O(n·p log n) por nivel", "O(n·p + p·B) por nivel", t_exact, t_hist, "s (entrenamiento)",
     xg.loc[xg.variante == "tree_method=hist", "pr_auc"].iloc[0] - xg.loc[xg.variante == "tree_method=exact", "pr_auc"].iloc[0]),
    ("SVM", "SVC RBF", "Nyström + LinearSVC", "O(n²·p) a O(n³)", "O(n·m·p)", t_rbf, ny.t_fit, "s (entrenamiento)", ny.pr_auc - sv.loc[sv.variante.str.startswith("SVC"), "pr_auc"].iloc[0]),
]
pd.DataFrame(filas, columns=["Modelo", "Estándar", "Optimizado", "O(·) estándar", "O(·) optimizado", "Tiempo estándar",
                             "Tiempo optimizado", "Unidad", "Δ PR-AUC"]).round(4)
```
