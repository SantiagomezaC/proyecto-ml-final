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

# Comparación estadística de los benchmarks

Con 140 modelos, el número de comparaciones pareadas posibles es 9.730, y aplicarlas sin control
produciría cerca de 487 falsos positivos al 5 %. La comparación sigue por eso una secuencia
jerárquica: una prueba global, un análisis post-hoc con control del error familiar, comparaciones
dirigidas entre finalistas con corrección de Holm y tamaños del efecto.

```{code-cell} ipython3
---
tags: [hide-input]
---
import sys, json, pickle, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from myst_nb import glue
from sklearn.metrics import average_precision_score, roc_auc_score
from src import estilo
from src.estilo import RIO, SED, CAU, ALE, NEU
from src.config import RESULTADOS
from src.analysis import tabla_pliegues
from src.models import NOMBRES_MODELO
from src.stats_tests import (friedman, nemenyi, delong, holm, bootstrap_bloques, longitud_bloque, delta_cliff,
                             magnitud_cliff, wilcoxon_holm)
from src.experiment import cargar_datos
D = cargar_datos(); E = RESULTADOS / "estadistica"
pl = tabla_pliegues("clf_calamar")
mejores = pd.read_csv(RESULTADOS / "mejores_clf.csv")
fin = {r["modelo"]: r for r in pickle.load(open(RESULTADOS / "final_clf.pkl", "rb"))}
```

## Clasificación

### Etapa 1: prueba global sobre las 112 combinaciones

```{code-cell} ipython3
---
tags: [hide-input]
---
c112 = json.loads((E / "cd_112.json").read_text(encoding="utf-8"))
for k in ("chi2", "p", "cd", "pares_signif", "pares_total", "gl"):
    glue(f"c112_{k}", c112[k], display=False)
pd.DataFrame({"rango medio": c112["rangos_top"]}).rename(index=lambda c: " · ".join(c.split(" | "))).round(2)
```

La prueba de Friedman sobre los doce bloques externos rechaza que las 112 combinaciones tengan
el mismo rango medio (χ² = {glue:text}`c112_chi2:.1f` con {glue:text}`c112_gl` grados de
libertad, p = {glue:text}`c112_p:.2g`). La diferencia crítica de Nemenyi con 112 modelos y doce
bloques es de {glue:text}`c112_cd:.1f` rangos, y solo {glue:text}`c112_pares_signif:,` de los
{glue:text}`c112_pares_total:,` pares la superan. Con doce bloques y 112 modelos, el control del
error familiar vuelve indistinguibles casi todas las combinaciones vecinas; la potencia de la
prueba se concentra en las diferencias extremas, y la comparación informativa debe hacerse entre
finalistas.

### Etapa 2: diagrama de diferencia crítica entre los mejores de cada modelo

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Diagrama de diferencia crítica de la mejor combinación de cada modelo (PR-AUC en 12 bloques). Las
      barras grises unen modelos cuya diferencia de rango medio no supera la CD de Nemenyi.
    name: fig-cd-bench
---
tabla = pd.read_csv(E / "universo_pliegues.csv", index_col=0).drop(columns="original")
fr = friedman(tabla); ne = nemenyi(fr["rangos"], fr["n_bloques"])
from src.estilo import dibujar_cd
fig, ax = plt.subplots(figsize=(10, 3.4)); dibujar_cd(fr["rangos"], ne["cd"], ax=ax, nombres=NOMBRES_MODELO); plt.show()
glue("fr7_chi2", fr["chi2"], display=False); glue("fr7_p", fr["p"], display=False); glue("cd7", ne["cd"], display=False)
glue("fr7_ff", fr["F_iman_davenport"], display=False); glue("fr7_pff", fr["p_iman_davenport"], display=False)
```

```{code-cell} ipython3
---
tags: [hide-input]
---
ne["pares"].assign(modelo_a=lambda d: d.modelo_a.map(NOMBRES_MODELO), modelo_b=lambda d: d.modelo_b.map(NOMBRES_MODELO)).sort_values("p_nemenyi").round(4)
```

Entre las mejores combinaciones de los siete modelos, Friedman rechaza la igualdad
(χ² = {glue:text}`fr7_chi2:.1f`, p = {glue:text}`fr7_p:.2g`; la corrección F de Iman y Davenport
da {glue:text}`fr7_ff:.1f`), pero la diferencia crítica de {glue:text}`cd7:.2f` rangos solo separa
a Naive Bayes de los cuatro mejores modelos. Random Forest, XGBoost, SVM y KNN forman un grupo
estadísticamente indistinguible, al que se suman la regresión logística y el árbol en el límite
de la significancia. El ordenamiento de los benchmarks es, por tanto, robusto en sus extremos y
difuso en el centro.

### Comparaciones dirigidas en la prueba reservada: DeLong con Holm

```{code-cell} ipython3
---
tags: [hide-input]
---
y = D.y_clf_test.values
top = mejores.sort_values("pr_auc_mean", ascending=False).modelo.head(3).tolist()
filas = []
L = longitud_bloque(y.astype(float))
for i in range(3):
    for j in range(i + 1, 3):
        a, b = top[i], top[j]
        dl = delong(y, fin[a]["pred_test"], fin[b]["pred_test"])
        ic = bootstrap_bloques(lambda yy, s1, s2: roc_auc_score(yy, s1) - roc_auc_score(yy, s2), y,
                               fin[a]["pred_test"], fin[b]["pred_test"], media_bloque=L, n_boot=1000)
        filas.append({"Modelo A": NOMBRES_MODELO[a], "Modelo B": NOMBRES_MODELO[b], "Δ AUC": dl["dif"],
                      "IC 95 % bootstrap por bloques (BCa)": f"[{ic['ic_inf']:.4f}; {ic['ic_sup']:.4f}]",
                      "z (DeLong)": dl["z"], "p crudo": dl["p"]})
dt = pd.DataFrame(filas); dt["p ajustado (Holm)"] = holm(dt["p crudo"].values)
glue("L_test", L, display=False)
dt.round(4)
```

La prueba de DeLong sobre la prueba reservada no distingue a Random Forest de XGBoost
(p ajustado de 0,14), pero sí declara significativas sus ventajas sobre KNN, y la de XGBoost
sobre KNN, tras la corrección de Holm. El intervalo de confianza por bootstrap estacionario de la
misma diferencia de AUC, en cambio, contiene el cero en los tres casos. La discrepancia se debe a
que DeLong estima la varianza del AUC suponiendo que los días de prueba son independientes; con
una autocorrelación de 0,99 la muestra efectiva es mucho menor que los 5.946 días, y sus p-valores
resultan optimistas. Por esa razón DeLong se reporta solo como prueba complementaria, y las
conclusiones se apoyan en el bootstrap por bloques (con longitud media de {glue:text}`L_test`
días) y en las pruebas sobre bloques externos.

### Intervalos de confianza y tamaños del efecto

```{code-cell} ipython3
---
tags: [hide-input]
---
filas = []
for m in mejores.sort_values("pr_auc_mean", ascending=False).modelo:
    s = fin[m]["pred_test"]
    for nombre, f in (("PR-AUC", average_precision_score), ("ROC-AUC", roc_auc_score)):
        ic = bootstrap_bloques(f, y, s, media_bloque=L, n_boot=1000)
        filas.append({"Modelo": NOMBRES_MODELO[m], "Métrica": nombre, "Estimación": ic["estimacion"],
                      "IC 95 % inferior": ic["ic_inf"], "IC 95 % superior": ic["ic_sup"]})
ics = pd.DataFrame(filas)
ref = mejores.sort_values("pr_auc_mean", ascending=False).iloc[0].modelo
w = wilcoxon_holm(tabla, ref)
w["modelo"] = w.modelo.map(NOMBRES_MODELO); w["magnitud"] = w.delta_cliff.map(magnitud_cliff)
glue("ref_bench", NOMBRES_MODELO[ref], display=False)
display(ics.round(4))
w[["modelo", "dif_media", "victorias", "p", "p_holm", "delta_cliff", "magnitud"]].round(4)
```

Los intervalos de la PR-AUC en la prueba se solapan ampliamente entre los cuatro mejores modelos.
La comparación de {glue:text}`ref_bench` con cada benchmark mediante Wilcoxon con Holm sobre los
doce bloques confirma la estructura del diagrama de diferencia crítica: la ventaja es
significativa y de tamaño no despreciable frente a Naive Bayes (δ de Cliff grande), el árbol de
decisión y la regresión logística (pequeño), y no lo es frente a KNN, XGBoost ni SVM, con tamaños
del efecto despreciables. Entre los mejores benchmarks, por tanto, las diferencias observadas son
menores que la variabilidad entre bloques, y elegir a Random Forest como referencia para el modelo
original es una decisión por media, no una superioridad demostrada.

## Regresión: secuencia jerárquica

### Model Confidence Set

```{code-cell} ipython3
---
tags: [hide-input]
---
mcs = pd.read_csv(E / "mcs_regresion.csv")
glue("mcs_n", int((mcs["¿En el conjunto?"] == "Sí").sum()), display=False)
glue("mcs_mejor", mcs.iloc[0]["Combinación"], display=False)
mcs
```

El Model Confidence Set {cite}`hansen2011` parte de las 28 combinaciones y, con un bootstrap
estacionario sobre la pérdida cuadrática diaria de las predicciones fuera de muestra
(1986-2009), elimina iterativamente la peor mientras la hipótesis de igual capacidad predictiva
se rechace. Con α = 0,10, el conjunto de confianza retiene {glue:text}`mcs_n` combinaciones.

### Comparaciones dirigidas: Giacomini-White, Clark-West y Diebold-Mariano

```{code-cell} ipython3
---
tags: [hide-input]
---
dirig = pd.read_csv(E / "regresion_dirigidas.csv")
cw = pd.read_csv(E / "regresion_clark_west.csv")
sp = json.loads((E / "spa_regresion.json").read_text(encoding="utf-8"))
display(dirig.round(4))
display(cw.round(4))
pd.Series(sp, name="SPA frente a Ridge").to_frame()
```

La secuencia de regresión confirma lo que sugerían las métricas. El conjunto de confianza retiene
las combinaciones de SVR y XGBoost con optimización bayesiana, aleatoria y genética, y excluye a
Random Forest, a los modelos lineales, a KNN y a los árboles, cuyas pérdidas son significativamente
mayores. Entre los finalistas, ni Diebold-Mariano con la corrección de Harvey, Leybourne y Newbold
{cite}`harvey1997` ni Giacomini-White {cite}`giacomini2006` encuentran diferencias significativas
tras Holm, y los intervalos del bootstrap estacionario de la diferencia de pérdida contienen el
cero: SVR y XGBoost son estadísticamente equivalentes para predecir el caudal a quince días, con
tamaños de efecto despreciables (|d| < 0,08). La prueba de Clark-West, aplicada al único
anidamiento natural del problema, el de la persistencia ($\hat Q_{t+15} = Q_t$) dentro de los
modelos que usan el caudal actual, rechaza con claridad que los finalistas no mejoren sobre ella
{cite}`clark2007`. Finalmente, la prueba SPA de Hansen {cite}`hansen2005` con Ridge como
benchmark da un p-valor consistente de 0,0005: hay al menos un modelo que supera de forma genuina
a la regresión lineal regularizada, y esa conclusión no es producto de haber probado muchos
modelos.
