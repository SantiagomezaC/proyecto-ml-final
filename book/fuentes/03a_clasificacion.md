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

# Resultados de clasificación

Esta sección presenta los resultados de las 112 combinaciones de clasificación: su desempeño en
la validación externa, el efecto de las técnicas de balanceo, el desempeño de los modelos finales
en la prueba reservada, la calidad de sus probabilidades y su estabilidad frente a la semilla.

```{code-cell} ipython3
---
tags: [hide-input]
---
# Resultados de la validación anidada de clasificación y modelos finales reentrenados
import sys, json, pickle, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from myst_nb import glue
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from src import estilo
from src.estilo import RIO, SED, CAU, ALE, NEU, COLORES
from src.config import RESULTADOS
from src.analysis import tabla_pliegues, resumen, nombres, oof
from src.models import NOMBRES_MODELO, NOMBRES_BALANCEO, NOMBRES_OPT
from src.evaluation import ece, diagrama_confiabilidad, Recalibrador
from src.experiment import cargar_datos
D = cargar_datos()
pl = tabla_pliegues("clf_calamar")
met = ["pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy", "brier"]
res = nombres(resumen(pl, met))
mejores = pd.read_csv(RESULTADOS / "mejores_clf.csv")
fin = {r["modelo"]: r for r in pickle.load(open(RESULTADOS / "final_clf.pkl", "rb"))}
glue("n_combos", len(res), display=False)
glue("n_tareas", len(pl), display=False)
```

## Tabla maestra

Las {glue:text}`n_combos` combinaciones se evaluaron en los doce bloques externos
({glue:text}`n_tareas` optimizaciones). La tabla muestra las veinte mejores según la PR-AUC
media del bucle externo; la tabla completa, con los hiperparámetros de cada pliegue, el tiempo
y la semilla, está en `results/tabla_maestra.csv` y se describe en {doc}`09_reproducibilidad`.

```{code-cell} ipython3
---
tags: [hide-input]
---
# Veinte mejores combinaciones según la PR-AUC media de los 12 bloques externos (media ± desviación)
fmt = lambda m, s: f"{m:.3f} ± {s:.3f}"
vista = res.sort_values("pr_auc_mean", ascending=False).head(20)
pd.DataFrame({"Modelo": vista.nombre, "Balanceo": vista.balanceo_nombre, "Optimizador": vista.optimizador_nombre,
              "PR-AUC": [fmt(a, b) for a, b in zip(vista.pr_auc_mean, vista.pr_auc_std)],
              "ROC-AUC": [fmt(a, b) for a, b in zip(vista.roc_auc_mean, vista.roc_auc_std)],
              "F1": [fmt(a, b) for a, b in zip(vista.f1_mean, vista.f1_std)],
              "Recall": [fmt(a, b) for a, b in zip(vista.recall_mean, vista.recall_std)],
              "Brier": [fmt(a, b) for a, b in zip(vista.brier_mean, vista.brier_std)],
              "Tiempo total (s)": vista.t_total.round(0)}).reset_index(drop=True)
```

La tabla muestra medias muy próximas entre las mejores combinaciones y desviaciones mucho
mayores que esas diferencias. La {numref}`fig-cajas-modelos` muestra esa dispersión bloque por
bloque para la mejor combinación de cada modelo.

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: PR-AUC externa por bloque de cada modelo (mejor combinación de balanceo y optimizador). Cada punto es
      un bloque de dos años; la caja resume los doce bloques.
    name: fig-cajas-modelos
---
# PR-AUC de cada bloque externo para la mejor combinación de cada modelo
orden = mejores.sort_values("pr_auc_mean", ascending=False).modelo.tolist()
fig, ax = plt.subplots(figsize=(10, 3.9))
datos_caja = []
for i, m in enumerate(orden):
    f = mejores.set_index("modelo").loc[m]
    v = pl[(pl.modelo == m) & (pl.balanceo == f.balanceo) & (pl.optimizador == f.optimizador)].sort_values("pliegue")
    datos_caja.append(v.pr_auc.values)
    ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-.12, .12, len(v)), v.pr_auc, s=14, color=SED, zorder=3)
ax.boxplot(datos_caja, positions=range(len(orden)), widths=.5, showfliers=False,
           medianprops=dict(color=RIO, lw=2), boxprops=dict(color=NEU), whiskerprops=dict(color=NEU), capprops=dict(color=NEU))
ax.set_xticks(range(len(orden)), [NOMBRES_MODELO[m] for m in orden], rotation=15)
ax.set_ylabel("PR-AUC en el bloque externo"); ax.set_title("Variabilidad entre bloques de la mejor combinación de cada modelo")
plt.show()
mej = mejores.sort_values("pr_auc_mean", ascending=False)
glue("mejor_modelo", NOMBRES_MODELO[mej.iloc[0].modelo], display=False)
glue("mejor_pr", float(mej.iloc[0].pr_auc_mean), display=False)
glue("mejor_pr_sd", float(mej.iloc[0].pr_auc_std), display=False)
glue("peor_modelo", NOMBRES_MODELO[mej.iloc[-1].modelo], display=False)
glue("peor_pr", float(mej.iloc[-1].pr_auc_mean), display=False)
peor_bloque = pl.groupby("bloque").pr_auc.mean().idxmin()
glue("peor_bloque", peor_bloque, display=False)
glue("peor_bloque_pr", float(pl.groupby("bloque").pr_auc.mean().min()), display=False)
```

La mejor combinación es {glue:text}`mejor_modelo` con ponderación por clase y búsqueda
aleatoria, con una PR-AUC externa de {glue:text}`mejor_pr:.3f` ± {glue:text}`mejor_pr_sd:.3f`, y
las diez primeras posiciones de la tabla maestra son variantes de Random Forest separadas por
menos de una centésima. XGBoost aparece como el segundo algoritmo, con la combinación ADASYN y
búsqueda aleatoria, y su desviación entre bloques es la menor de las mejores combinaciones. En el
otro extremo, {glue:text}`peor_modelo` ({glue:text}`peor_pr:.3f`) paga el supuesto de
independencia condicional entre predictores que son, en buena parte, rezagos casi idénticos de la
misma serie.

La dispersión de la figura es mayor entre bloques que entre modelos, y todos los modelos tienen
sus peores resultados en los mismos bloques. Para identificar qué hace difícil a un bloque, la
tabla siguiente lo caracteriza por su prevalencia, por la proporción de positivos que exigen
anticipar una entrada a la banda y por su caudal máximo.

```{code-cell} ipython3
---
tags: [hide-input]
---
# Dificultad de cada bloque externo: proporción de días positivos que exigen anticipar una entrada
# a la banda (caudal fuera de ella en t y dentro en t + 15)
import json
from scipy.stats import spearmanr
from src.config import DATOS_PROC
LIM = json.loads((DATOS_PROC / "parametros.json").read_text(encoding="utf-8"))["calamar"]["LIM"]
q_hoy = np.sqrt(D.X.calamar_cuad.values)
en_hoy = (q_hoy >= LIM[0]) & (q_hoy <= LIM[1])
yb = D.y_clf.values.astype(int)
filas = []
for p in D.pliegues:
    te = p["test"]
    filas.append({"pliegue": p["pliegue"], "prevalencia": yb[te].mean(),
                  "positivos que entran a la banda": yb[te][~en_hoy[te]].sum() / max(yb[te].sum(), 1),
                  "caudal máximo en t+15 (m³/s)": D.y_reg.values[te].max()})
dif_blq = (pl.groupby("pliegue").agg(bloque=("bloque", "first"), **{"PR-AUC media (112 combinaciones)": ("pr_auc", "mean")})
           .join(pd.DataFrame(filas).set_index("pliegue")))
rho_e, p_e = spearmanr(dif_blq["PR-AUC media (112 combinaciones)"], dif_blq["positivos que entran a la banda"])
rho_p, p_p = spearmanr(dif_blq["PR-AUC media (112 combinaciones)"], dif_blq.prevalencia)
peor = dif_blq.loc[dif_blq["PR-AUC media (112 combinaciones)"].idxmin()]
glue("rho_entrada", rho_e, display=False); glue("p_entrada", p_e, display=False)
glue("rho_prev", rho_p, display=False); glue("p_prev", p_p, display=False)
glue("entrada_peor", float(peor["positivos que entran a la banda"]), display=False)
glue("prev_peor", float(peor.prevalencia), display=False)
glue("entrada_med", float(dif_blq["positivos que entran a la banda"].median()), display=False)
glue("qmax_peor", float(peor["caudal máximo en t+15 (m³/s)"]), display=False)
dif_blq.set_index("bloque").round(3)
```

La {numref}`fig-cajas-modelos` muestra que la variabilidad entre bloques supera con amplitud la
diferencia entre los mejores modelos, y la tabla anterior permite identificar su causa. La
dificultad de un bloque depende menos de cuántos días están en la banda que de cuántos de ellos
exigen anticipar una entrada: días en que el caudal todavía está fuera de la banda en *t* y la
alcanza en *t* + 15. La correlación de Spearman entre la PR-AUC media del bloque y la proporción
de positivos de ese tipo es de {glue:text}`rho_entrada:.2f` (p = {glue:text}`p_entrada:.1e`, doce
bloques), mientras que la correlación con la prevalencia ({glue:text}`rho_prev:.2f`,
p = {glue:text}`p_prev:.2f`) no es significativa. El bloque {glue:text}`peor_bloque`, el más
difícil para todos los modelos (PR-AUC media de {glue:text}`peor_bloque_pr:.2f` entre las 112
combinaciones), reúne dos condiciones desfavorables: una prevalencia de apenas
{glue:text}`prev_peor:.1%`, que reduce la PR-AUC alcanzable, y un {glue:text}`entrada_peor:.0%` de
positivos que son entradas a la banda, frente a una mediana de {glue:text}`entrada_med:.0%` en los
doce bloques. Hidrológicamente, el bloque contiene la crecida de 2008, cuyo caudal máximo
({glue:text}`qmax_peor:,.0f` m³/s) es el tercero más alto del periodo de entrenamiento, y el río
atraviesa la banda en ascensos y descensos rápidos que el estado actual anticipa mal. La
variabilidad entre bloques tiene, por tanto, un origen físico identificable y no es ruido de
estimación; por esa razón las conclusiones de este capítulo se apoyan en pruebas sobre bloques
pareados y no en comparaciones de medias.

## Efecto del balanceo de clases

Con un 26 % de días positivos, cabe preguntarse si el sobremuestreo o la ponderación de la clase
minoritaria mejoran los modelos. La {numref}`fig-balanceo` muestra el cambio medio de tres
métricas respecto de no balancear, para cada modelo y técnica.

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Cambio medio de cada métrica respecto de no balancear, por modelo y técnica (promedio sobre
      optimizadores y bloques). Valores positivos indican mejora.
    name: fig-balanceo
---
# Diferencia de cada métrica entre cada técnica de balanceo y no balancear, pareada por modelo, optimizador y bloque
base = pl[pl.balanceo == "ninguno"].set_index(["modelo", "optimizador", "pliegue"])
filas = []
for b in ("smote", "adasyn", "balanced"):
    otro = pl[pl.balanceo == b].set_index(["modelo", "optimizador", "pliegue"])
    dif = (otro[["pr_auc", "recall", "f1", "precision", "brier"]] - base[["pr_auc", "recall", "f1", "precision", "brier"]]).reset_index()
    dif["balanceo"] = b
    filas.append(dif)
dif = pd.concat(filas)
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, metr, tit in zip(axs, ("pr_auc", "recall", "f1"), ("Δ PR-AUC", "Δ exhaustividad", "Δ F1")):
    t = dif.groupby(["modelo", "balanceo"])[metr].mean().unstack()
    t.index = [NOMBRES_MODELO[m] for m in t.index]
    t = t[["smote", "adasyn", "balanced"]]
    y = np.arange(len(t))
    for j, (b, col) in enumerate(zip(t.columns, (RIO, CAU, SED))):
        ax.barh(y + (j - 1) * .26, t[b], height=.26, color=col, label=NOMBRES_BALANCEO[b])
    ax.axvline(0, color=NEU, lw=1); ax.set_title(tit); ax.set_yticks(y, t.index)
axs[0].legend(loc="lower left", fontsize=8)
plt.tight_layout(); plt.show()
from scipy.stats import wilcoxon
from src.stats_tests import holm
tab_b = []
for b in ("smote", "adasyn", "balanced"):
    g = dif[dif.balanceo == b]
    for metr in ("pr_auc", "recall", "f1"):
        tab_b.append({"balanceo": NOMBRES_BALANCEO[b], "métrica": metr, "Δ media": g[metr].mean(),
                      "fracción de mejoras": (g[metr] > 0).mean(), "p (Wilcoxon)": wilcoxon(g[metr]).pvalue})
tab_b = pd.DataFrame(tab_b)
tab_b["p (Holm)"] = holm(tab_b["p (Wilcoxon)"].values)
for b in ("SMOTE", "ADASYN", "class_weight"):
    for metr in ("pr_auc", "recall"):
        glue(f"bal_{b}_{metr}", float(tab_b[(tab_b.balanceo == b) & (tab_b["métrica"] == metr)]["Δ media"].iloc[0]), display=False)
```

La tabla siguiente resume esas diferencias para todos los modelos juntos, con la prueba de
Wilcoxon sobre las diferencias pareadas y la corrección de Holm.

```{code-cell} ipython3
---
tags: [hide-input]
---
# Cambio medio, fracción de casos con mejora y significancia de cada técnica de balanceo
tab_b.round(4)
```

El balanceo no mejora la discriminación. La PR-AUC, que evalúa el orden de los puntajes y no
depende del umbral, apenas cambia con SMOTE ({glue:text}`bal_SMOTE_pr_auc:+.4f`, no
significativo) o con la ponderación por clase ({glue:text}`bal_class_weight_pr_auc:+.4f`), y
empeora con ADASYN ({glue:text}`bal_ADASYN_pr_auc:+.4f`, significativo tras Holm). Lo que sí
cambia es el punto de operación con el umbral de 0,5: la exhaustividad sube entre
{glue:text}`bal_SMOTE_recall:.2f` y {glue:text}`bal_ADASYN_recall:.2f` puntos a costa de la
precisión, y el F1 queda igual o baja. Este comportamiento es el esperado para un desbalance
moderado (26 % de positivos): las tres técnicas desplazan el umbral efectivo hacia la clase minoritaria, lo
que equivale a cambiar el costo relativo de los errores, pero no aportan información nueva sobre
qué días están en la banda. Para entender por qué ADASYN, en particular, empeora la
discriminación, la tabla siguiente examina dónde concentra sus sintéticos en el entrenamiento del
último bloque.

```{code-cell} ipython3
---
tags: [hide-input]
---
# Mecanismo de ADASYN en el entrenamiento del último bloque: cercanía temporal de los vecinos que usa
# para interpolar y concentración de su peso en los positivos próximos al límite de la banda
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
p = D.pliegues[-1]; tr = p["train"]
Xa = D.X.iloc[tr]; Za = StandardScaler().fit_transform(Xa.fillna(Xa.median()))
ya = D.y_clf.values[tr].astype(int); pos = np.flatnonzero(ya == 1)
_, iv = NearestNeighbors(n_neighbors=6).fit(Za[pos]).kneighbors(Za[pos])
dias = np.abs(pos[iv[:, 1:]] - pos[:, None])
_, ia = NearestNeighbors(n_neighbors=6).fit(Za).kneighbors(Za[pos])
r_ad = (ya[ia[:, 1:]] == 0).mean(1)                  # dificultad local con que ADASYN pondera cada positivo
w_ad = r_ad / r_ad.sum()
q15 = D.y_reg.values[tr][pos]
QEF = json.loads((DATOS_PROC / "parametros.json").read_text(encoding="utf-8"))["calamar"]["Q_EF"]
borde = np.minimum(np.abs(q15 - LIM[0]), np.abs(q15 - LIM[1])) / QEF < 0.05
glue("ad_med_dias", float(np.median(dias)), display=False)
glue("ad_frac7", float((dias <= 7).mean()), display=False)
glue("ad_frac_r", float((r_ad > 0).mean()), display=False)
glue("ad_peso_borde", float(w_ad[borde].sum()), display=False)
glue("ad_pos_borde", float(borde.mean()), display=False)
pd.DataFrame({"indicador": ["Mediana de la separación temporal con los 5 vecinos positivos (días)",
                            "Vecinos positivos a 7 días o menos", "Positivos con al menos un vecino negativo",
                            "Positivos a menos del 5 % de Q_ef de un límite", "Peso de ADASYN asignado a esos positivos"],
              "valor": [np.median(dias), (dias <= 7).mean(), (r_ad > 0).mean(), borde.mean(), w_ad[borde].sum()]}).round(3)
```

El efecto negativo de ADASYN se explica por la interacción entre su regla de muestreo y la
estructura temporal de los datos. ADASYN genera más sintéticos alrededor de los positivos que
tienen negativos entre sus vecinos, y solo el {glue:text}`ad_frac_r:.0%` de los positivos cumple
esa condición; como esos días se ubican cerca de los límites de la banda, el
{glue:text}`ad_peso_borde:.0%` del peso de muestreo recae en positivos situados a menos del 5 % de
$Q_{ef}$ de un límite, aunque estos representan solo el {glue:text}`ad_pos_borde:.0%` de la clase.
Además, los vecinos positivos con que interpola son, en la práctica, días contiguos del mismo
episodio: la separación temporal mediana es de {glue:text}`ad_med_dias:.0f` días y el
{glue:text}`ad_frac7:.0%` de los vecinos está a una semana o menos. El método produce, por tanto,
casi duplicados de los casos más ambiguos de cada episodio y desplaza la frontera hacia la región
donde la etiqueta es menos fiable. SMOTE, que reparte los sintéticos de manera uniforme entre los
positivos, no presenta esa concentración, y su efecto sobre la PR-AUC no es significativo.

## Desempeño en la prueba reservada

Con la combinación ganadora de cada modelo se repitió la optimización sobre todo el
entrenamiento (1947-2009), se reentrenó el pipeline y se evaluó una única vez en 2010-2026. La
tabla reúne las métricas de los siete modelos finales.

```{code-cell} ipython3
---
tags: [hide-input]
---
# Métricas de los siete modelos finales en la prueba reservada 2010-2026 (umbral 0,5 donde aplica)
y = D.y_clf_test.values
filas = []
for m in orden:
    r = fin[m]
    mt = r["metricas_test"]
    filas.append({"Modelo": NOMBRES_MODELO[m], "Balanceo": NOMBRES_BALANCEO[r["balanceo"]], "Optimizador": NOMBRES_OPT[r["optimizador"]],
                  "Accuracy": mt["accuracy"], "Precision": mt["precision"], "Recall": mt["recall"], "F1": mt["f1"],
                  "ROC-AUC": mt["roc_auc"], "PR-AUC": mt["pr_auc"], "Brier": mt["brier"],
                  "Inferencia (ms/fila)": r["t_inf_ms_fila"]})
tabla_test = pd.DataFrame(filas)
glue("test_mejor_pr", float(tabla_test["PR-AUC"].max()), display=False)
glue("test_mejor_modelo", tabla_test.loc[tabla_test["PR-AUC"].idxmax(), "Modelo"], display=False)
tabla_test.round(4)
```

La {numref}`fig-roc-pr` muestra las curvas completas, que no dependen del umbral, y la
{numref}`fig-confusion`, los aciertos y errores con el umbral de 0,5.

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Curvas ROC y de precisión-exhaustividad en la prueba 2010-2026 de la mejor combinación de cada modelo.
    name: fig-roc-pr
---
# Curvas ROC y precisión-exhaustividad en la prueba reservada
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for i, m in enumerate(orden):
    s = fin[m]["pred_test"]
    fpr, tpr, _ = roc_curve(y, s); pr_, rc_, _ = precision_recall_curve(y, s)
    c = (COLORES + ["#7a5c99", "#2f6b3a"])[i]
    axs[0].plot(fpr, tpr, color=c, lw=1.6, label=f"{NOMBRES_MODELO[m]} ({fin[m]['metricas_test']['roc_auc']:.3f})")
    axs[1].plot(rc_, pr_, color=c, lw=1.6, label=f"{NOMBRES_MODELO[m]} ({fin[m]['metricas_test']['pr_auc']:.3f})")
axs[0].plot([0, 1], [0, 1], ":", color=NEU); axs[1].axhline(y.mean(), ls=":", color=NEU)
axs[0].set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos", title="ROC")
axs[1].set(xlabel="Exhaustividad", ylabel="Precisión", title="Precisión-exhaustividad")
axs[0].legend(fontsize=7.5, loc="lower right"); axs[1].legend(fontsize=7.5, loc="lower left")
plt.tight_layout(); plt.show()
```

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Matrices de confusión en la prueba (umbral 0,5) de la mejor combinación de cada modelo.
    name: fig-confusion
---
# Matrices de confusión en la prueba reservada con umbral 0,5
fig, axs = plt.subplots(1, len(orden), figsize=(14, 2.6))
cm_ref = confusion_matrix(y, (fin[orden[0]]["pred_test"] >= .5).astype(int))
glue("cm_fp", int(cm_ref[0, 1]), display=False); glue("cm_fn", int(cm_ref[1, 0]), display=False)
for ax, m in zip(axs, orden):
    cm = confusion_matrix(y, (fin[m]["pred_test"] >= .5).astype(int))
    ax.imshow(cm, cmap=estilo.mapa_seq()); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=9, color="white" if cm[i, j] > cm.max() / 2 else "#13232c")
    ax.set_xticks([0, 1], ["no", "sí"]); ax.set_yticks([0, 1], ["no", "sí"]); ax.set_title(NOMBRES_MODELO[m], fontsize=9)
axs[0].set_ylabel("Real"); fig.supxlabel("Predicho", fontsize=10)
plt.tight_layout(); plt.show()
```

En la prueba reservada, Random Forest y SVM empatan en PR-AUC ({glue:text}`test_mejor_pr:.3f`),
seguidos de XGBoost y KNN; Random Forest obtiene además el mejor F1 y el menor Brier. Los valores
de la prueba superan a las medias externas porque el periodo 2010-2026 no contiene un tramo tan
desfavorable como el bloque 2008-2009, lo que refuerza la conveniencia de evaluar sobre muchos
bloques y no sobre una única partición. Todos los modelos superan con amplitud a la persistencia
(PR-AUC de 0,65, {doc}`02_datos`), que es la referencia exigente en una serie con autocorrelación
de 0,92 a quince días. Con el umbral de 0,5, Random Forest comete {glue:text}`cm_fp` falsos
positivos (días que anuncia en la banda y no lo están) y {glue:text}`cm_fn` falsos negativos (días
en la banda que no anticipa); como usa ponderación por clase, su punto de operación favorece la
exhaustividad. Los tiempos de inferencia son del orden de milésimas de milisegundo por día, salvo
KNN, que debe calcular distancias a todo el entrenamiento.

### Justificación de la métrica principal

El criterio de selección es la PR-AUC porque el interés operativo está en la clase positiva y
porque la clase es minoritaria: la PR-AUC de un clasificador aleatorio es igual a la
prevalencia (0,25), mientras que la ROC-AUC de ese mismo clasificador es 0,5 y se mantiene alta
aunque el modelo produzca muchas falsas alarmas, porque los verdaderos negativos son abundantes
{cite}`saito2015`. Entre precisión y exhaustividad, el costo relativo depende del uso. Para
programar una campaña de muestreo de sedimentos o una purga de embalse, un falso negativo
(no anticipar un periodo de transporte dominante) significa perder la ventana de operación,
mientras que un falso positivo cuesta una movilización innecesaria; en ese escenario conviene
privilegiar la exhaustividad, y el umbral puede bajarse a partir de la curva de la
{numref}`fig-roc-pr`. Para la planificación de dragados, en cambio, las falsas alarmas tienen un
costo directo y conviene la precisión. Al reportar la PR-AUC, que integra todos los umbrales, la
comparación entre modelos no depende de esa elección, que corresponde a cada usuario según su
aplicación.

Las métricas de todas las combinaciones se calculan con `metricas_clf` (`src/experiment.py`) a
partir del puntaje continuo de la clase positiva; la PR-AUC es la precisión promedio de
scikit-learn. El error de calibración se calcula con `ece` (`src/evaluation.py`).

```{code-cell} ipython3
---
tags: [remove-input]
---
# Código fuente de las métricas de clasificación y del error de calibración
import inspect
from IPython.display import Code
from src import experiment as exp_mod, evaluation as ev_mod
Code(inspect.getsource(exp_mod.metricas_clf) + "\n\n" + inspect.getsource(ev_mod.ece), language="python")
```

## Calibración de probabilidades

Una probabilidad útil debe ser también confiable: si el modelo anuncia un 70 %, el evento debería
ocurrir cerca del 70 % de las veces. La {numref}`fig-calibracion` compara la probabilidad predicha
con la frecuencia observada, antes y después de recalibrar.

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Diagramas de confiabilidad en la prueba, antes (línea continua) y después (línea discontinua) de la
      recalibración de Platt ajustada con las predicciones fuera de muestra del bucle externo.
    name: fig-calibracion
---
# Diagramas de confiabilidad y recalibración de Platt e isotónica ajustadas con predicciones fuera de muestra
filas = []
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for i, m in enumerate(orden):
    r = fin[m]; s = r["pred_test"]
    oo = oof("clf_calamar", m, r["balanceo"], r["optimizador"])
    iso = Recalibrador("isotonica").fit(oo.pred.values, oo.y.values).transform(s)
    pla = r["pred_test_recalibrado"]
    from sklearn.metrics import brier_score_loss
    filas.append({"Modelo": NOMBRES_MODELO[m], "Brier": brier_score_loss(y, np.clip(s, 0, 1)), "ECE": ece(y, s),
                  "Brier Platt": brier_score_loss(y, pla), "ECE Platt": ece(y, pla),
                  "Brier isotónica": brier_score_loss(y, iso), "ECE isotónica": ece(y, iso),
                  "Recalibrar (decisión en OOF)": "sí" if r["recalibrar_por_oof"] else "no"})
    c = (COLORES + ["#7a5c99", "#2f6b3a"])[i]
    t = diagrama_confiabilidad(y, s); tp = diagrama_confiabilidad(y, pla)
    ax = axs[0] if i < 4 else axs[1]
    ax.plot(t.prob_predicha, t.frec_observada, "-o", ms=3, color=c, label=NOMBRES_MODELO[m])
    ax.plot(tp.prob_predicha, tp.frec_observada, "--", color=c, lw=1)
for ax in axs:
    ax.plot([0, 1], [0, 1], ":", color=NEU); ax.set(xlabel="Probabilidad predicha", ylabel="Frecuencia observada"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
cal = pd.DataFrame(filas)
peor = cal.loc[cal.ECE.idxmax()]
glue("cal_peor", peor.Modelo, display=False); glue("cal_peor_ece", float(peor.ECE), display=False)
glue("cal_peor_ece_platt", float(peor["ECE Platt"]), display=False)
```

La tabla cuantifica la calibración de cada modelo con el puntaje de Brier y el error de
calibración esperado (ECE), con y sin recalibración.

```{code-cell} ipython3
---
tags: [hide-input]
---
# Brier y ECE en la prueba, sin recalibrar y con las dos recalibraciones
cal.round(4)
```

La calibración separa a los modelos con más claridad que la discriminación. La SVM produce
puntajes sin calibrar, porque su función de decisión se transforma con una logística fija, y su
error de calibración esperado es alto; lo mismo ocurre con {glue:text}`cal_peor`
(ECE = {glue:text}`cal_peor_ece:.3f`), cuyas probabilidades se concentran en los extremos por el
supuesto de independencia. La recalibración de Platt, ajustada con las predicciones fuera de
muestra del bucle externo y nunca con la prueba, reduce el ECE de {glue:text}`cal_peor` a
{glue:text}`cal_peor_ece_platt:.3f` y el de la SVM por debajo de 0,011, y mejora el Brier de todos
los modelos salvo la regresión logística, que ya estaba bien calibrada por construcción: su
pérdida logarítmica es una regla propia y la decisión tomada en las predicciones fuera de muestra
fue, correctamente, no recalibrarla. La regresión isotónica logra reducciones similares; se
prefirió Platt por tener solo dos parámetros, menos propensos a sobreajustar con la dependencia
temporal de las predicciones. Los modelos finales y la aplicación interactiva usan la versión
recalibrada cuando la decisión en las predicciones fuera de muestra fue favorable.

## Robustez y estabilidad

Una diferencia entre modelos solo es interpretable si supera la variación que produce el azar del
propio algoritmo. La tabla siguiente compara, para el mejor modelo de referencia y para el modelo
propuesto, la variación entre cinco semillas de inicialización con la variación entre bloques.

```{code-cell} ipython3
---
tags: [hide-input]
---
# Desviación de la PR-AUC entre semillas (mismo bloque) frente a la desviación entre bloques (misma semilla)
rob = pd.read_csv(RESULTADOS / "robustez" / "robustez.csv")
sem = rob[rob.tipo == "semilla"]
var_sem = sem.groupby(["modelo", "pliegue"]).pr_auc.std().groupby("modelo").mean()
var_blq = sem.groupby(["modelo", "seed"]).pr_auc.std().groupby("modelo").mean()
ref = mejores.sort_values("pr_auc_mean", ascending=False).iloc[0].modelo
tab_sem = pd.DataFrame({"desviación entre semillas (media por bloque)": var_sem,
                        "desviación entre bloques (media por semilla)": var_blq})
tab_sem.index = [NOMBRES_MODELO["original"] if i == "original" else f"{NOMBRES_MODELO[ref]} (mejor modelo de referencia)" for i in tab_sem.index]
glue("sd_sem_bench", float(var_sem["benchmark"]), display=False); glue("sd_blq_bench", float(var_blq["benchmark"]), display=False)
dif_top = res.sort_values("pr_auc_mean", ascending=False).pr_auc_mean.head(10)
glue("rango_top10", float(dif_top.max() - dif_top.min()), display=False)
tab_sem.round(4)
```

Todas las métricas de este capítulo se reportan como media ± desviación estándar sobre los doce
bloques externos y no como valores puntuales. Para el mejor modelo de referencia, la sensibilidad a la
semilla de inicialización (cinco semillas, con los hiperparámetros elegidos en cada bloque) es
pequeña: la desviación media entre semillas dentro de un bloque es de {glue:text}`sd_sem_bench:.4f`,
frente a una desviación entre bloques de {glue:text}`sd_blq_bench:.4f`. La aleatoriedad del
algoritmo pesa, por tanto, mucho menos que la variabilidad hidrológica entre periodos. Al mismo
tiempo, las diez mejores combinaciones de la tabla maestra se separan en apenas
{glue:text}`rango_top10:.4f` de PR-AUC media, una diferencia del mismo orden que la variabilidad
entre semillas y muy inferior a la variabilidad entre bloques. Las diferencias entre los mejores
modelos de referencia no son, por tanto, mayores que la variabilidad intrínseca de cada uno, y
decidir entre ellos exige las pruebas formales de {doc}`03e_comparacion`. La estabilidad del
modelo propuesto se discute en {doc}`06_validacion`.
