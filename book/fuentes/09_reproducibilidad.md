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

# Reproducibilidad y bibliografía

Este capítulo documenta lo necesario para reproducir los resultados: el entorno de software,
la semilla, la tabla maestra de experimentos y el orden de ejecución.

## Entorno

La tabla lista las versiones de las bibliotecas principales con que se obtuvieron los resultados.

```{code-cell} ipython3
---
tags: [hide-input]
---
# Versiones de Python y de las bibliotecas principales
import sys, platform, json
sys.path.insert(0, "..")
from importlib.metadata import version
import pandas as pd
from src.config import SEED, RESULTADOS
libs = ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "xgboost", "optuna", "deap", "shap", "lime",
        "statsmodels", "arch", "ngboost", "faiss-cpu", "dash", "dash-bootstrap-components", "plotly", "jupyter-book"]
pd.DataFrame({"versión": {l: version(l) for l in libs} | {"python": platform.python_version()}})
```

Todas las versiones están fijadas en `requirements-dev.txt` (experimentación y libro) y en
`requirements.txt` (aplicación interactiva en producción). Los resultados se obtuvieron en Windows 11 con un
procesador Intel Core i7-1255U y 15,7 GB de memoria, sin GPU.

## Semillas

La semilla global es `SEED = 42`, definida una sola vez en `src/config.py`. La función
`fijar_semillas` la propaga a los generadores de Python y numpy; cada estimador la recibe en
`random_state` (scikit-learn, XGBoost, imbalanced-learn y el modelo propuesto), el muestreador
TPE de Optuna en `seed` y el algoritmo genético la usa tanto en el generador de numpy que crea
la población inicial como en el módulo `random` que emplea DEAP en la selección, el cruce y la
mutación. La sensibilidad a la semilla se evaluó explícitamente con cinco semillas ({doc}`03a_clasificacion` y
{doc}`06_validacion`).

## Tabla maestra de experimentos

La celda siguiente reúne los resultados de todas las optimizaciones en una tabla maestra, la
guarda en disco y muestra sus primeras filas.

```{code-cell} ipython3
---
tags: [hide-input]
---
# Tabla maestra: una fila por combinación con media y desviación de cada métrica, tiempo y semilla
from src.analysis import tabla_pliegues
partes = []
for carpeta, tarea in (("clf_calamar", "clasificación"), ("reg_calamar", "regresión"), ("original_calamar", "clasificación")):
    t = tabla_pliegues(carpeta); t["tarea"] = tarea
    partes.append(t)
largo = pd.concat(partes, ignore_index=True)
from src.analysis import cargar_carpeta
params = {(r["modelo"], r["balanceo"], r["optimizador"], r["pliegue"]): json.dumps(r["params"])
          for c in ("clf_calamar", "reg_calamar", "original_calamar") for r in cargar_carpeta(c)}
largo["hiperparametros"] = [params[(a, b, c, d)] for a, b, c, d in zip(largo.modelo, largo.balanceo, largo.optimizador, largo.pliegue)]
largo["semilla"] = SEED
metricas = [c for c in ("pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy", "brier", "rmse", "mae", "r2", "mape") if c in largo]
agg = largo.groupby(["tarea", "modelo", "balanceo", "optimizador"]).agg(
    **{f"{m}_media": (m, "mean") for m in metricas}, **{f"{m}_desv": (m, "std") for m in metricas},
    tiempo_total_s=("t_opt", "sum"), evaluaciones=("n_eval", "sum"), pliegues=("pliegue", "nunique")).reset_index()
agg["semilla"] = SEED
largo.to_csv(RESULTADOS / "tabla_maestra_por_pliegue.csv", index=False)
agg.to_csv(RESULTADOS / "tabla_maestra.csv", index=False)
print(f"{len(agg)} combinaciones ({(agg.tarea == 'clasificación').sum()} de clasificación, "
      f"{(agg.tarea == 'regresión').sum()} de regresión) y {len(largo):,} registros por pliegue.")
agg.head(10)
```

La tabla maestra (`results/tabla_maestra.csv`) contiene una fila por combinación con la media y
la desviación de cada métrica del bucle externo, el tiempo total, el número de evaluaciones y la
semilla; `results/tabla_maestra_por_pliegue.csv` agrega los hiperparámetros elegidos en cada
pliegue externo. Las 112 combinaciones de clasificación, las 28 de regresión y las variantes del
modelo propuesto, de los métodos distribucionales y de los modelos LTSF-Linear comparten el mismo esquema.

## Cómo reproducir los resultados

El README del repositorio enumera los comandos en orden. Los cómputos costosos (la validación
anidada de los 140 modelos, del modelo propuesto y de los métodos recientes) se guardan como resultados versionados en
`results/`, y los cuadernos de este libro los leen y regeneran tablas y figuras; el código que
los produce está en `scripts/` y `src/` y se ejecuta de principio a fin con los comandos del
README. Las pruebas unitarias se ejecutan con `python -m pytest`.

## Bibliografía

```{bibliography}
:style: unsrt
```
