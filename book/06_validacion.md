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

# Validación estadística del modelo original

```{code-cell} ipython3
---
tags: [hide-input]
---
import sys, json, pickle, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import scipy.stats as st
from myst_nb import glue
from sklearn.metrics import average_precision_score, roc_auc_score
from src import estilo
from src.estilo import RIO, SED, CAU, ALE, NEU, COLORES
from src.config import RESULTADOS, SEED
from src.analysis import tabla_pliegues, cargar_carpeta
from src.models import NOMBRES_MODELO
from src.stats_tests import (friedman, nemenyi, wilcoxon_holm, nadeau_bengio, bootstrap_bloques, longitud_bloque,
                             delta_cliff, magnitud_cliff, holm)
from src.experiment import cargar_datos
D = cargar_datos(); E = RESULTADOS / "estadistica"
tabla = pd.read_csv(E / "universo_pliegues.csv", index_col=0)
ver = json.loads((E / "veredicto.json").read_text(encoding="utf-8"))
fo = pickle.load(open(RESULTADOS / "final_original.pkl", "rb"))[0]
fin = {r["modelo"]: r for r in pickle.load(open(RESULTADOS / "final_clf.pkl", "rb"))}
ref = ver["ref"]
NM = {**NOMBRES_MODELO, "original": "BLGB (original)"}
```

## Protocolo

El modelo original pasó por exactamente el mismo protocolo que los benchmarks: las mismas doce
particiones externas leídas del mismo archivo, la misma validación interna cronológica, la misma
semilla global, el mismo preprocesamiento y el mismo presupuesto de 16 evaluaciones por pliegue.
Sus hiperparámetros se optimizaron con TPE y con el algoritmo genético, y para el veredicto se
toma el método con mejor PR-AUC interna en el entrenamiento completo, igual que en los
benchmarks. El universo de comparación está formado por la mejor combinación de cada uno de los
siete modelos benchmark y el modelo original.

## Resultados por bloque

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: PR-AUC por bloque externo del modelo original y de la mejor combinación de cada benchmark.
    name: fig-bloques-original
---
fig, ax = plt.subplots(figsize=(11, 4))
bloques = tabla_pliegues("clf_calamar").drop_duplicates("pliegue").set_index("pliegue").bloque
x = np.arange(len(tabla))
for i, m in enumerate(tabla.mean().sort_values(ascending=False).index):
    if m == "original":
        ax.plot(x, tabla[m], "-o", color=SED, lw=2.6, ms=6, label=NM[m], zorder=5)
    else:
        ax.plot(x, tabla[m], "-", color=RIO if m == ref else NEU, lw=1.6 if m == ref else .8,
                alpha=1 if m == ref else .55, label=NM[m] if m == ref else None)
ax.set_xticks(x, bloques.loc[tabla.index].values, rotation=30)
ax.set_ylabel("PR-AUC en el bloque externo"); ax.legend(loc="lower left")
ax.set_title("Desempeño por bloque (gris: demás benchmarks)")
plt.show()
med = tabla.mean(); sd = tabla.std()
victorias = int((tabla["original"] > tabla.drop(columns="original").max(axis=1)).sum())
glue("orig_media", float(med["original"]), display=False); glue("orig_sd", float(sd["original"]), display=False)
glue("ref_nombre", NM[ref], display=False); glue("ref_media", float(med[ref]), display=False)
glue("ref_sd", float(sd[ref]), display=False); glue("victorias_todos", victorias, display=False)
glue("victorias_ref", int((tabla["original"] > tabla[ref]).sum()), display=False)
glue("dif_media", float(med["original"] - med[ref]), display=False)
```

## Tabla resumen

```{code-cell} ipython3
---
tags: [hide-input]
---
y = D.y_clf_test.values
L = longitud_bloque(y.astype(float))
filas = []
for m in tabla.mean().sort_values(ascending=False).index:
    s = fo["pred_test"] if m == "original" else fin[m]["pred_test"]
    ic = bootstrap_bloques(average_precision_score, y, s, media_bloque=L, n_boot=1000)
    t_fit = fo["t_fit"] if m == "original" else fin[m]["t_fit"]
    t_inf = fo["t_inf_ms_fila"] if m == "original" else fin[m]["t_inf_ms_fila"]
    filas.append({"Modelo": NM[m], "PR-AUC externa": f"{med[m]:.4f} ± {sd[m]:.4f}",
                  "PR-AUC prueba": ic["estimacion"], "IC 95 % (bloques, BCa)": f"[{ic['ic_inf']:.3f}; {ic['ic_sup']:.3f}]",
                  "ROC-AUC prueba": roc_auc_score(y, s), "Entrenamiento final (s)": t_fit, "Inferencia (ms/fila)": t_inf})
resumen = pd.DataFrame(filas)
glue("orig_test", float(resumen.set_index("Modelo").loc[NM["original"], "PR-AUC prueba"]), display=False)
glue("ref_test", float(resumen.set_index("Modelo").loc[NM[ref], "PR-AUC prueba"]), display=False)
resumen.round(4)
```

## Supuestos de las pruebas

```{code-cell} ipython3
---
tags: [hide-input]
---
dif = tabla["original"] - tabla[ref]
sh = st.shapiro(dif)
asim = st.skew(dif)
ac1 = np.corrcoef(dif.values[:-1], dif.values[1:])[0, 1]
glue("sh_p", float(sh.pvalue), display=False); glue("asim_dif", float(asim), display=False); glue("ac1_dif", float(ac1), display=False)
n_tr = float(np.mean([len(p["train"]) for p in D.pliegues])); n_te = float(np.mean([len(p["test"]) for p in D.pliegues]))
glue("ratio_nb", n_te / n_tr, display=False)
pd.DataFrame([{"supuesto": "Normalidad de las diferencias por bloque (t corregida)", "verificación": f"Shapiro-Wilk p = {sh.pvalue:.3f}"},
              {"supuesto": "Simetría de las diferencias (Wilcoxon)", "verificación": f"asimetría = {asim:.2f}"},
              {"supuesto": "Independencia entre bloques", "verificación": f"autocorrelación de orden 1 de las diferencias = {ac1:.2f}"},
              {"supuesto": "Solapamiento de entrenamientos (Nadeau-Bengio)", "verificación": f"n_test/n_train medio = {n_te / n_tr:.3f}"},
              {"supuesto": "Independencia entre días (DeLong, McNemar)", "verificación": f"ACF a 1 día del objetivo ≈ 0,99: no se cumple"}])
```

Cada prueba descansa en supuestos que se verificaron antes de interpretarla. Las diferencias de
PR-AUC por bloque entre el modelo original y {glue:text}`ref_nombre` no se apartan de la
normalidad (Shapiro-Wilk, p = {glue:text}`sh_p:.3f`), condición de la prueba t, y su asimetría
({glue:text}`asim_dif:.2f`) es compatible con la simetría que supone Wilcoxon. Los bloques no son
independientes en sentido estricto, porque comparten días de entrenamiento: la ventana crece y
cada modelo se entrena con los bloques anteriores. La t de Nadeau y Bengio {cite}`nadeau2003`
corrige precisamente esa dependencia inflando la varianza por el factor
$(1/k + n_{\text{test}}/n_{\text{train}})$, con una razón media de {glue:text}`ratio_nb:.3f`. La
autocorrelación de las diferencias entre bloques consecutivos es {glue:text}`ac1_dif:.2f`. En
cambio, DeLong y McNemar suponen independencia entre días, que la autocorrelación del caudal
viola de forma grave; por eso la Tabla 3 de la guía los restringe en datos temporales, y aquí se
reportan como complemento sin contar para el requisito R3. Su lugar lo ocupa el bootstrap
estacionario sobre la prueba reservada, que respeta la dependencia.

## Pruebas de hipótesis

```{code-cell} ipython3
---
tags: [hide-input]
---
pr = pd.read_csv(E / "pruebas_original.csv")
pr
```

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Diagrama de diferencia crítica con el modelo original incluido en el universo (PR-AUC, 12 bloques).
    name: fig-cd-original
---
cd = json.loads((E / "cd.json").read_text(encoding="utf-8"))
rangos = pd.Series(cd["rangos"])
from src.estilo import dibujar_cd
fig, ax = plt.subplots(figsize=(10, 3.6)); dibujar_cd(rangos, cd["cd"], destacado="original", ax=ax, nombres=NM); plt.show()
glue("cd_val", cd["cd"], display=False); glue("rango_orig", float(rangos["original"]), display=False)
glue("rango_ref", float(rangos[ref]), display=False)
glue("fr_chi2", cd["friedman"]["chi2"], display=False); glue("fr_p", cd["friedman"]["p"], display=False)
glue("n_rechazos", ver["n_rechazos"], display=False)
glue("cliff", ver["delta_cliff"], display=False); glue("dcohen", ver["d_cohen"], display=False)
glue("veredicto", "superior" if ver["superior"] else "no superior", display=False)
```

INTERPRETACION_PRUEBAS

## Comparación con métodos distribucionales recientes

```{code-cell} ipython3
---
tags: [hide-input]
---
o = tabla_pliegues("original_calamar")
sota = o.pivot_table(index="pliegue", columns=["modelo", "optimizador"], values="pr_auc")
sota.columns = [f"{a} ({b})" for a, b in sota.columns]
nombres_sota = {"blgb (bayes)": "BLGB, TPE", "blgb (genetico)": "BLGB, genético", "ngboost (bayes)": "NGBoost, TPE",
                "dos_etapas (bayes)": "Dos etapas (XGBoost + gaussiana), TPE"}
sota = sota.rename(columns=nombres_sota)
filas = []
base = sota[[c for c in sota.columns if c.startswith("BLGB") and fo["optimizador"].replace("bayes", "TPE").replace("genetico", "genético") in c][0]]
for c in sota.columns:
    fila = {"Modelo": c, "PR-AUC externa": f"{sota[c].mean():.4f} ± {sota[c].std():.4f}"}
    if not c.startswith("BLGB"):
        w = st.wilcoxon(base, sota[c], alternative="greater", zero_method="zsplit")
        nb = nadeau_bengio(base, sota[c], n_tr, n_te)
        fila.update({"Δ frente a BLGB": sota[c].mean() - base.mean(), "p Wilcoxon": w.pvalue, "p Nadeau-Bengio": nb["p"],
                     "victorias de BLGB": int((base > sota[c]).sum())})
    filas.append(fila)
ts = pd.DataFrame(filas)
mask = ts["p Wilcoxon"].notna()
ts.loc[mask, "p Wilcoxon (Holm)"] = holm(ts.loc[mask, "p Wilcoxon"].values)
ts.loc[mask, "p Nadeau-Bengio (Holm)"] = holm(ts.loc[mask, "p Nadeau-Bengio"].values)
for c in sota.columns:
    glue("sota_" + c.split(",")[0].split(" ")[0].lower(), float(sota[c].mean()), display=False)
ts.round(4)
```

INTERPRETACION_SOTA

## Evaluación en tres tareas adicionales

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: PR-AUC media en los 12 bloques externos de cada tarea adicional (banda de descarga efectiva a 15 días en
      otra estación del cauce), para el modelo original y los siete benchmarks con igual presupuesto.
    name: fig-extra
---
import json as _j
par_all = _j.loads((RESULTADOS.parent / "data" / "processed" / "parametros.json").read_text(encoding="utf-8"))
filas, bloques_extra = [], []
for est in ("el_banco", "coyongal", "tres_cruces"):
    t = tabla_pliegues(f"extra_{est}")
    t["modelo"] = t.modelo.replace({"blgb": "original"})
    piv = t.pivot_table(index="pliegue", columns="modelo", values="pr_auc")
    fr_e = friedman(piv)
    w = wilcoxon_holm(piv, "original")
    mejor_b = piv.drop(columns="original").mean().idxmax()
    filas.append({"Estación objetivo": est.replace("_", " ").title(), "Q_ef (m³/s)": par_all[est]["Q_EF"],
                  "Prevalencia": par_all[est]["prevalencia_train"], "BLGB": piv["original"].mean(),
                  "Mejor benchmark": f"{NOMBRES_MODELO[mejor_b]} ({piv[mejor_b].mean():.4f})",
                  "Rango BLGB": fr_e["rangos"]["original"], "p Friedman": fr_e["p"],
                  "p Wilcoxon-Holm vs mejor": w.set_index("modelo").loc[mejor_b, "p_holm"],
                  "victorias vs mejor": f"{int((piv['original'] > piv[mejor_b]).sum())}/{len(piv)}"})
    piv.columns = [NM.get(c, c) for c in piv.columns]
    piv["estacion"] = est; bloques_extra.append(piv)
extra = pd.DataFrame(filas)
todo = pd.concat(bloques_extra).drop(columns="estacion")
fig, ax = plt.subplots(figsize=(10, 3.8))
medias = pd.concat(bloques_extra).groupby("estacion").mean().T
medias.columns = [c.replace("_", " ").title() for c in medias.columns]
medias = medias.loc[medias.mean(axis=1).sort_values().index]
yy = np.arange(len(medias))
for j, (c, col) in enumerate(zip(medias.columns, (RIO, CAU, SED))):
    ax.barh(yy + (j - 1) * .26, medias[c], height=.26, color=col, label=c)
ax.set_yticks(yy, medias.index); ax.set_xlabel("PR-AUC externa media"); ax.legend(loc="lower right")
ax.set_xlim(max(0, medias.values.min() - .1), 1); plt.show()
fr_global = friedman(todo)
glue("fr_global_p", fr_global["p"], display=False); glue("rango_global", float(fr_global["rangos"][NM["original"]]), display=False)
glue("n_bloques_global", fr_global["n_bloques"], display=False)
glue("extra_gana", int(sum(r["BLGB"] > float(r["Mejor benchmark"].split("(")[-1].rstrip(")")) for r in filas)), display=False)
extra.round(4)
```

INTERPRETACION_EXTRA

## Robustez, sensibilidad y límites

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: (a) Sensibilidad de la PR-AUC externa al peso λ de la supervisión continua. (b) PR-AUC bajo ruido
      multiplicativo en los predictores, desbalance inducido y menor tamaño de entrenamiento (seis bloques más
      recientes), para el modelo original y el mejor benchmark.
    name: fig-robustez
---
rob = pd.read_csv(RESULTADOS / "robustez" / "robustez.csv")
lam = rob[rob.tipo == "lambda"].groupby("valor").pr_auc.agg(["mean", "std"])
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9))
axs[0].errorbar(lam.index.map(lambda v: max(v, 0.03)), lam["mean"], yerr=lam["std"] / np.sqrt(12), fmt="-o", color=SED, capsize=3)
axs[0].set_xscale("log"); axs[0].set(xlabel="λ (0 representado en 0,03)", ylabel="PR-AUC externa media", title="(a) Sensibilidad a λ")
pert = rob[rob.tipo.isin(["ruido", "desbalance", "muestra"])]
g = pert.groupby(["tipo", "valor", "variante"]).pr_auc.mean().unstack()
etq = [f"{'ruido' if t == 'ruido' else 'prevalencia' if t == 'desbalance' else 'fracción n'} {v:g}" for t, v in g.index]
yy = np.arange(len(g))
axs[1].barh(yy - .2, g["original"], height=.4, color=SED, label=NM["original"])
axs[1].barh(yy + .2, g["benchmark"], height=.4, color=RIO, label=f"{NM[ref]} (mejor benchmark)")
axs[1].set_yticks(yy, etq); axs[1].set_xlim(max(0, g.values.min() - .1), 1); axs[1].legend(loc="lower right", fontsize=8)
axs[1].set(xlabel="PR-AUC media", title="(b) Perturbaciones")
plt.tight_layout(); plt.show()
glue("lam_mejor", float(lam["mean"].idxmax()), display=False)
glue("lam0", float(lam.loc[0.0, "mean"]), display=False); glue("lam_max", float(lam["mean"].max()), display=False)
gan = (g["original"] - g["benchmark"])
glue("pert_min_gan", float(gan.min()), display=False); glue("pert_max_gan", float(gan.max()), display=False)
glue("pert_peor", etq[int(np.argmin(gan.values))], display=False)
sem = rob[rob.tipo == "semilla"].groupby(["variante", "pliegue"]).pr_auc.std().groupby("variante").mean()
glue("sem_orig", float(sem["original"]), display=False); glue("sem_ref", float(sem["benchmark"]), display=False)
pd.DataFrame({"original": g["original"], "benchmark": g["benchmark"], "diferencia": gan}).round(4)
```

INTERPRETACION_ROBUSTEZ

## Análisis de errores: dónde no supera a los benchmarks

```{code-cell} ipython3
---
tags: [hide-input]
---
pr_o = fo["pred_test"]; pr_r = fin[ref]["pred_test"]
q0 = np.sqrt(D.X_test.calamar_cuad.values); qf = D.y_reg_test.values
LIMS = (8461.29, 11447.63)
tendencia = np.sign(D.X_test.calamar_d7.values)
regimen = np.select([q0 < LIMS[0] * .8, q0 < LIMS[0], q0 <= LIMS[1], q0 <= LIMS[1] * 1.2],
                    ["muy por debajo", "debajo, cerca del límite", "dentro de la banda", "encima, cerca del límite"], "muy por encima")
from sklearn.metrics import brier_score_loss
filas = []
for reg in ["muy por debajo", "debajo, cerca del límite", "dentro de la banda", "encima, cerca del límite", "muy por encima"]:
    for tnd, nom in ((1, "ascenso"), (-1, "descenso")):
        m = (regimen == reg) & (tendencia == tnd)
        if m.sum() < 40 or len(np.unique(y[m])) < 2:
            continue
        filas.append({"régimen actual": reg, "tendencia 7 d": nom, "días": int(m.sum()), "prevalencia": y[m].mean(),
                      "Brier BLGB": brier_score_loss(y[m], pr_o[m]), f"Brier {NM[ref]}": brier_score_loss(y[m], pr_r[m]),
                      "PR-AUC BLGB": average_precision_score(y[m], pr_o[m]), f"PR-AUC {NM[ref]}": average_precision_score(y[m], pr_r[m])})
err = pd.DataFrame(filas)
err["BLGB mejor (Brier)"] = err["Brier BLGB"] < err[f"Brier {NM[ref]}"]
glue("err_pierde", int((~err["BLGB mejor (Brier)"]).sum()), display=False); glue("err_total", len(err), display=False)
err.round(4)
```

INTERPRETACION_ERRORES

## Interpretabilidad del modelo original frente al mejor benchmark

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  figure:
    caption: Importancia global (media del valor absoluto de Shapley por permutaciones sobre 300 días de la
      prueba) del modelo original y del mejor benchmark, en la misma escala de probabilidad.
    name: fig-shap-orig
---
sh = pd.read_csv(RESULTADOS.parent / "data" / "dashboard" / "shap_global.csv")
from src.data import etiqueta
a = sh[sh.modelo == "original"].set_index("variable").importancia
b = sh[sh.modelo == ref].set_index("variable").importancia
top = (a + b).sort_values(ascending=False).head(12).index[::-1]
fig, ax = plt.subplots(figsize=(9.5, 5))
yy = np.arange(len(top))
ax.barh(yy - .2, a[top], height=.4, color=SED, label=NM["original"]); ax.barh(yy + .2, b[top], height=.4, color=RIO, label=NM[ref])
ax.set_yticks(yy, [etiqueta(c) for c in top], fontsize=8.5); ax.set_xlabel("Media de |φ| (probabilidad)"); ax.legend(loc="lower right")
plt.tight_layout(); plt.show()
rho = st.spearmanr(a, b.reindex(a.index))[0]
glue("rho_shap", float(rho), display=False)
glue("top_orig", etiqueta(a.idxmax()), display=False); glue("top_ref", etiqueta(b.idxmax()), display=False)
frac_arriba = lambda s: s[[c for c in s.index if not c.startswith(("calamar", "dist", "sen", "cos"))]].sum() / s.sum()
glue("arriba_orig", float(frac_arriba(a)), display=False); glue("arriba_ref", float(frac_arriba(b)), display=False)
```

INTERPRETACION_SHAP_ORIGINAL
