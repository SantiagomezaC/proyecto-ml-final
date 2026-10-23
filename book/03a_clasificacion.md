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

```{code-cell} ipython3
---
tags: [hide-input]
---
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
y la semilla, está en `results/tabla_maestra.csv` (capítulo 9).

```{code-cell} ipython3
---
tags: [hide-input]
---
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

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: PR-AUC externa por bloque de cada modelo (mejor combinación de balanceo y optimizador). Cada punto es
      un bloque de dos años; la caja resume los doce bloques.
    name: fig-cajas-modelos
---
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

INTERPRETACION_MODELOS

## Efecto del balanceo de clases

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Cambio medio de cada métrica respecto de no balancear, por modelo y técnica (promedio sobre
      optimizadores y bloques). Valores positivos indican mejora.
    name: fig-balanceo
---
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
tab_b.round(4)
```

INTERPRETACION_BALANCEO

## Desempeño en la prueba reservada

Con la combinación ganadora de cada modelo se repitió la optimización sobre todo el
entrenamiento (1947-2009), se reentrenó el pipeline y se evaluó una única vez en 2010-2026.

```{code-cell} ipython3
---
tags: [hide-input]
---
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

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Curvas ROC y de precisión-exhaustividad en la prueba 2010-2026 de la mejor combinación de cada modelo.
    name: fig-roc-pr
---
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
fig, axs = plt.subplots(1, len(orden), figsize=(14, 2.6))
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

INTERPRETACION_PRUEBA

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
comparación entre modelos no depende de esa elección, que se deja al usuario del dashboard.

## Calibración de probabilidades

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Diagramas de confiabilidad en la prueba, antes (línea continua) y después (línea discontinua) de la
      recalibración de Platt ajustada con las predicciones fuera de muestra del bucle externo.
    name: fig-calibracion
---
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
cal.round(4)
```

INTERPRETACION_CALIBRACION

## Robustez y estabilidad

```{code-cell} ipython3
---
tags: [hide-input]
---
rob = pd.read_csv(RESULTADOS / "robustez" / "robustez.csv")
sem = rob[rob.tipo == "semilla"]
var_sem = sem.groupby(["modelo", "pliegue"]).pr_auc.std().groupby("modelo").mean()
var_blq = sem.groupby(["modelo", "seed"]).pr_auc.std().groupby("modelo").mean()
ref = mejores.sort_values("pr_auc_mean", ascending=False).iloc[0].modelo
tab_sem = pd.DataFrame({"desviación entre semillas (media por bloque)": var_sem,
                        "desviación entre bloques (media por semilla)": var_blq})
tab_sem.index = [NOMBRES_MODELO["original"] if i == "original" else f"{NOMBRES_MODELO[ref]} (mejor benchmark)" for i in tab_sem.index]
glue("sd_sem_bench", float(var_sem["benchmark"]), display=False); glue("sd_blq_bench", float(var_blq["benchmark"]), display=False)
dif_top = res.sort_values("pr_auc_mean", ascending=False).pr_auc_mean.head(10)
glue("rango_top10", float(dif_top.max() - dif_top.min()), display=False)
tab_sem.round(4)
```

INTERPRETACION_ROBUSTEZ
