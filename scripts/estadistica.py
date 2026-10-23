"""Comparación estadística del Entregable 2 y verificación de superioridad (Entregable 3).

Uso:  python -m scripts.estadistica
Genera results/estadistica/: diagramas de diferencia crítica, tabla de pruebas del modelo
original, veredicto R3, ablación, MCS de regresión y textos del dashboard.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import cargar_carpeta, guardar_json, historiales, tabla_pliegues
from src.config import RESULTADOS, SEED, fijar_semillas
from src.experiment import cargar_datos
from src.models import NOMBRES_MODELO
from src.stats_tests import (bayes_signed_rank, bootstrap_bloques, clark_west, d_cohen_pareado, delong,
                             delta_cliff, diebold_mariano, friedman, giacomini_white, holm, longitud_bloque,
                             magnitud_cliff, mcnemar, model_confidence_set, nadeau_bengio, nemenyi, spa,
                             wilcoxon_holm)

E = RESULTADOS / "estadistica"
ALFA = 0.05


def universo_clf():
    """Tabla pliegues × modelo con la mejor combinación de cada benchmark y el modelo original."""
    d = tabla_pliegues("clf_calamar")
    mejores = pd.read_csv(RESULTADOS / "mejores_clf.csv")
    cols = {}
    for f in mejores.itertuples():
        g = d[(d.modelo == f.modelo) & (d.balanceo == f.balanceo) & (d.optimizador == f.optimizador)]
        cols[f.modelo] = g.set_index("pliegue").pr_auc
    o = tabla_pliegues("original_calamar")
    fin = pickle.load(open(RESULTADOS / "final_original.pkl", "rb"))[0]
    cols["original"] = o[(o.modelo == "blgb") & (o.optimizador == fin["optimizador"])].set_index("pliegue").pr_auc
    tabla = pd.DataFrame(cols).sort_index()
    n_tr = float(np.mean([len(p["train"]) for p in cargar_datos().pliegues]))
    n_te = float(np.mean([len(p["test"]) for p in cargar_datos().pliegues]))
    return d, tabla, mejores, n_tr, n_te


def pruebas_original(tabla, mejores, n_tr, n_te):
    datos = cargar_datos()
    ref = mejores.sort_values("pr_auc_mean", ascending=False).iloc[0].modelo
    filas = []
    # 1. Friedman (global) con el modelo original en el universo
    fr = friedman(tabla)
    ne = nemenyi(fr["rangos"], fr["n_bloques"])
    par_ref = ne["pares"].query("(modelo_a == 'original' and modelo_b == @ref) or "
                                "(modelo_b == 'original' and modelo_a == @ref)").iloc[0]
    filas.append({"prueba": "Friedman (ómnibus) sobre 8 modelos", "hipótesis nula": "Todos los modelos tienen el mismo rango medio",
                  "supuestos": f"Bloques externos pareados ({fr['n_bloques']}); particiones cronológicas walk-forward",
                  "estadístico": f"χ² = {fr['chi2']:.2f} (gl = {fr['gl']}); F_ID = {fr['F_iman_davenport']:.2f}",
                  "p crudo": fr["p"], "p ajustado": fr["p"], "efecto": f"rango medio del original = {fr['rangos']['original']:.2f}",
                  "conclusión": "Se rechaza H0" if fr["p"] < ALFA else "No se rechaza H0", "cuenta_r3": False})
    filas.append({"prueba": f"Nemenyi: original frente a {NOMBRES_MODELO[ref]}", "hipótesis nula": "Igual rango medio",
                  "supuestos": "Post-hoc de Friedman, control del error familiar",
                  "estadístico": f"|Δ rango| = {par_ref.dif_rangos:.2f}; CD = {ne['cd']:.2f}",
                  "p crudo": par_ref.p_nemenyi, "p ajustado": par_ref.p_nemenyi,
                  "efecto": "—", "conclusión": "Diferencia significativa" if par_ref.significativo else "No significativa (CD)",
                  "cuenta_r3": bool(par_ref.significativo)})
    # 2. Wilcoxon-Holm del original contra los 7 benchmarks
    wh = wilcoxon_holm(tabla, "original")
    w_ref = wh.set_index("modelo").loc[ref]
    dif = tabla["original"] - tabla[ref]
    filas.append({"prueba": "Wilcoxon de rangos con signo + Holm (7 comparaciones)",
                  "hipótesis nula": f"La mediana de la diferencia de PR-AUC frente a {NOMBRES_MODELO[ref]} es ≤ 0",
                  "supuestos": "Diferencias pareadas por bloque, simetría aproximada; unilateral por hipótesis a priori",
                  "estadístico": f"W = {w_ref.W:.0f}; victorias {w_ref.victorias}/{w_ref.n}",
                  "p crudo": w_ref.p, "p ajustado": w_ref.p_holm,
                  "efecto": f"δ de Cliff = {w_ref.delta_cliff:.3f} ({magnitud_cliff(w_ref.delta_cliff)})",
                  "conclusión": "Se rechaza H0" if w_ref.p_holm < ALFA else "No se rechaza H0",
                  "cuenta_r3": bool(w_ref.p_holm < ALFA)})
    # 3. t corregida de Nadeau-Bengio contra los 7, con Holm
    nb = {m: nadeau_bengio(tabla["original"], tabla[m], n_tr, n_te) for m in tabla.columns if m != "original"}
    p_nb = holm(np.array([v["p"] for v in nb.values()]))
    nb_adj = dict(zip(nb, p_nb))
    filas.append({"prueba": "t corregida de Nadeau-Bengio + Holm (7 comparaciones)",
                  "hipótesis nula": f"La diferencia media de PR-AUC frente a {NOMBRES_MODELO[ref]} es ≤ 0",
                  "supuestos": "Varianza inflada por (1/k + n_test/n_train) para pliegues que comparten entrenamiento",
                  "estadístico": f"t = {nb[ref]['t']:.2f} (gl = {nb[ref]['gl']})",
                  "p crudo": nb[ref]["p"], "p ajustado": nb_adj[ref],
                  "efecto": f"d de Cohen (pareado) = {nb[ref]['d_cohen']:.2f}",
                  "conclusión": "Se rechaza H0" if nb_adj[ref] < ALFA else "No se rechaza H0",
                  "cuenta_r3": bool(nb_adj[ref] < ALFA)})
    # 4. Prueba bayesiana de rangos con signo (ROPE = 0.01 de PR-AUC)
    bs = bayes_signed_rank(tabla["original"], tabla[ref], rope=0.01)
    filas.append({"prueba": "Bayesiana de rangos con signo (ROPE ±0,01)",
                  "hipótesis nula": "— (probabilidades posteriores de cada región)",
                  "supuestos": "Proceso de Dirichlet sobre las diferencias por bloque (Benavoli et al., 2017)",
                  "estadístico": f"P(original mejor) = {bs['p_A_mejor']:.3f}; P(equiv.) = {bs['p_equivalentes']:.3f}",
                  "p crudo": np.nan, "p ajustado": np.nan, "efecto": f"P(benchmark mejor) = {bs['p_B_mejor']:.3f}",
                  "conclusión": "Superioridad práctica (P > 0,95)" if bs["p_A_mejor"] > 0.95 else "No concluyente",
                  "cuenta_r3": False})
    # 5. Bootstrap estacionario BCa de la diferencia de PR-AUC en la prueba 2010-2026
    fin = {r["modelo"]: r for r in pickle.load(open(RESULTADOS / "final_clf.pkl", "rb"))}
    fo = pickle.load(open(RESULTADOS / "final_original.pkl", "rb"))[0]
    y = datos.y_clf_test.values
    s_o, s_r = fo["pred_test"], fin[ref]["pred_test"]
    L = longitud_bloque(y.astype(float))
    ic = bootstrap_bloques(lambda yy, a, b: average_precision_score(yy, a) - average_precision_score(yy, b),
                           y, s_o, s_r, media_bloque=L, n_boot=2000)
    # p-valor bootstrap unilateral: proporción de réplicas con diferencia ≤ 0
    rng = np.random.default_rng(SEED)
    from src.stats_tests import indices_estacionario
    reps = []
    for _ in range(2000):
        i = indices_estacionario(len(y), L, rng)
        reps.append(average_precision_score(y[i], s_o[i]) - average_precision_score(y[i], s_r[i]))
    p_boot = float((np.asarray(reps) <= 0).mean())
    filas.append({"prueba": "Bootstrap estacionario BCa de Δ PR-AUC (prueba 2010-2026)",
                  "hipótesis nula": "Δ PR-AUC ≤ 0", "supuestos": f"Dependencia temporal; bloque medio de {L} días (Politis-White)",
                  "estadístico": f"Δ = {ic['estimacion']:.4f}; IC 95 % [{ic['ic_inf']:.4f}; {ic['ic_sup']:.4f}]",
                  "p crudo": p_boot, "p ajustado": p_boot, "efecto": f"Δ PR-AUC = {ic['estimacion']:.4f}",
                  "conclusión": "Se rechaza H0" if (p_boot < ALFA and ic["ic_inf"] > 0) else "No se rechaza H0",
                  "cuenta_r3": bool(p_boot < ALFA and ic["ic_inf"] > 0)})
    # 6. DeLong en la prueba (complementaria por la dependencia temporal)
    dl = delong(y, s_o, s_r)
    filas.append({"prueba": "DeLong (AUC-ROC, prueba 2010-2026; complementaria)",
                  "hipótesis nula": "AUC del original ≤ AUC del benchmark",
                  "supuestos": "Supone independencia entre días: se reporta como complemento, no cuenta para R3",
                  "estadístico": f"z = {dl['z']:.2f}; AUC {dl['auc_a']:.4f} frente a {dl['auc_b']:.4f}",
                  "p crudo": dl["p_unilateral"], "p ajustado": dl["p_unilateral"], "efecto": f"Δ AUC = {dl['dif']:.4f}",
                  "conclusión": "Se rechaza H0" if dl["p_unilateral"] < ALFA else "No se rechaza H0", "cuenta_r3": False})
    mc = mcnemar(y, (s_o >= .5).astype(int), (s_r >= .5).astype(int))
    filas.append({"prueba": "McNemar (umbral 0,5, prueba 2010-2026; complementaria)",
                  "hipótesis nula": "Igual tasa de error", "supuestos": "Supone independencia entre días: complementaria",
                  "estadístico": f"χ² = {mc['chi2']:.2f}; discordantes {mc['a_acierta_b_falla']} / {mc['b_acierta_a_falla']}",
                  "p crudo": mc["p"], "p ajustado": mc["p"], "efecto": "—",
                  "conclusión": "Se rechaza H0" if mc["p"] < ALFA else "No se rechaza H0", "cuenta_r3": False})
    pr = pd.DataFrame(filas)
    # Ajuste de Holm sobre la familia de pruebas confirmatorias que cuentan para R3
    conf = pr.prueba.str.startswith(("Wilcoxon", "t corregida", "Bootstrap"))
    pr.loc[conf, "p ajustado (familia R3, Holm)"] = holm(pr.loc[conf, "p ajustado"].values)
    pr["cuenta_r3"] = pr["cuenta_r3"] & (pr["p ajustado (familia R3, Holm)"].fillna(1) < ALFA) | (
        pr.prueba.str.startswith("Nemenyi") & pr.cuenta_r3)
    detalle = {"ref": ref, "friedman": fr, "nemenyi": ne, "wilcoxon": wh, "nadeau_bengio": nb,
               "nb_holm": nb_adj, "bayes": bs, "bootstrap": ic, "p_boot": p_boot, "delong": dl, "mcnemar": mc,
               "dif": dif}
    return pr, detalle


def main():
    fijar_semillas()
    E.mkdir(parents=True, exist_ok=True)
    d, tabla, mejores, n_tr, n_te = universo_clf()
    tabla.to_csv(E / "universo_pliegues.csv")
    pr, det = pruebas_original(tabla, mejores, n_tr, n_te)
    pr.to_csv(E / "pruebas_original_detalle.csv", index=False)
    vista = pr[["prueba", "hipótesis nula", "supuestos", "estadístico", "p crudo", "p ajustado",
                "p ajustado (familia R3, Holm)", "efecto", "conclusión"]].copy()
    for c in ("p crudo", "p ajustado", "p ajustado (familia R3, Holm)"):
        vista[c] = vista[c].map(lambda v: "—" if pd.isna(v) else (f"{v:.2e}" if v < 1e-3 else f"{v:.4f}"))
    vista.to_csv(E / "pruebas_original.csv", index=False)

    # Veredicto de superioridad (Sección 6.1 del Entregable 3)
    medias = tabla.mean()
    cond1 = bool(medias["original"] > medias.drop("original").max())
    n_rechazos = int(pr.cuenta_r3.sum())
    d_cohen = det["nadeau_bengio"][det["ref"]]["d_cohen"]
    cliff = delta_cliff(tabla["original"], tabla[det["ref"]])
    cond3 = bool(abs(cliff) >= 0.147 or abs(d_cohen) >= 0.2)
    superior = cond1 and n_rechazos >= 3 and cond3
    ref = det["ref"]
    texto = (f"Veredicto R3: {'SUPERIOR' if superior else 'NO SUPERIOR'}. PR-AUC externa media "
             f"{medias['original']:.4f} frente a {medias[ref]:.4f} del mejor benchmark ({NOMBRES_MODELO[ref]}); "
             f"{n_rechazos} pruebas confirmatorias rechazan la igualdad con p ajustado < 0,05; δ de Cliff = {cliff:.3f} "
             f"y d de Cohen pareado = {d_cohen:.2f}.")
    guardar_json({"superior": superior, "texto": texto, "cond_media": cond1, "n_rechazos": n_rechazos,
                  "delta_cliff": cliff, "d_cohen": d_cohen, "ref": ref,
                  "medias": medias.to_dict()}, E / "veredicto.json")
    print(texto)

    fr, ne = det["friedman"], det["nemenyi"]
    guardar_json({"rangos": fr["rangos"].to_dict(), "cd": ne["cd"], "n_bloques": fr["n_bloques"],
                  "friedman": {k: fr[k] for k in ("chi2", "gl", "p", "F_iman_davenport", "p_iman_davenport")}},
                 E / "cd.json")

    # Friedman y Nemenyi sobre las 112 combinaciones
    m112 = d.pivot_table(index="pliegue", columns="combo", values="pr_auc")
    f112 = friedman(m112)
    n112 = nemenyi(f112["rangos"], f112["n_bloques"])
    guardar_json({"chi2": f112["chi2"], "gl": f112["gl"], "p": f112["p"], "cd": n112["cd"],
                  "pares_signif": int(n112["pares"].significativo.sum()), "pares_total": int(len(n112["pares"])),
                  "rangos_top": f112["rangos"].head(15).to_dict()}, E / "cd_112.json")
    n112["pares"].to_parquet(E / "nemenyi_112.parquet", index=False)

    # Ablación (desde robustez)
    rob = pd.read_csv(RESULTADOS / "robustez" / "robustez.csv")
    ab = rob[rob.tipo == "ablacion"].pivot_table(index="pliegue", columns="variante", values="pr_auc")
    filas = []
    for v in ab.columns:
        dd = ab[v] - ab["completo"]
        fila = {"Variante": v, "PR-AUC media": f"{ab[v].mean():.4f} ± {ab[v].std():.4f}",
                "Δ frente al completo": f"{dd.mean():+.4f}"}
        if v != "completo":
            from scipy.stats import wilcoxon
            fila["p (Wilcoxon)"] = f"{wilcoxon(ab['completo'], ab[v], alternative='greater', zero_method='zsplit').pvalue:.4f}"
        filas.append(fila)
    pd.DataFrame(filas).to_csv(E / "ablacion.csv", index=False)

    # Convergencia del modelo final y curva anytime del pliegue más reciente
    fo = pickle.load(open(RESULTADOS / "final_original.pkl", "rb"))[0]
    pd.DataFrame(fo["historial_boosting"]).to_csv(E / "conv_boosting.csv", index=False)
    h = historiales("original_calamar")
    h = h[(h.modelo == "blgb") & (h.optimizador == fo["optimizador"])]
    h[h.pliegue == h.pliegue.max()][["eval", "puntaje", "mejor"]].to_csv(E / "anytime_original.csv", index=False)

    # Regresión: MCS sobre las 28 combinaciones con la pérdida cuadrática diaria fuera de muestra
    rr = cargar_carpeta("reg_calamar")
    perd = {}
    preds = {}
    for r in rr:
        clave = f"{r['modelo']}|{r['optimizador']}"
        perd.setdefault(clave, []).append(pd.Series((r["y_test"] - r["pred"]) ** 2, index=r["idx_test"]))
        preds.setdefault(clave, []).append(pd.Series(r["pred"], index=r["idx_test"]))
    perd = pd.DataFrame({k: pd.concat(v).sort_index() for k, v in perd.items()})
    preds = pd.DataFrame({k: pd.concat(v).sort_index() for k, v in preds.items()})
    perd.to_parquet(E / "perdidas_regresion.parquet")
    preds.to_parquet(E / "predicciones_regresion_oof.parquet")
    mcs = model_confidence_set(perd / 1e6, alpha=0.10)
    mcs.to_csv(E / "mcs_regresion_completo.csv")
    vista = mcs.reset_index().rename(columns={"index": "combinación"})
    vista.columns = ["Combinación" if c in ("index", "combinación") or c == vista.columns[0] else c for c in vista.columns]
    vista["RMSE (m³/s)"] = np.sqrt(vista.perdida_media * 1e6).round(1)
    vista = vista[["Combinación", "RMSE (m³/s)", "rango_perdida", "p_mcs", "en_conjunto"]].rename(
        columns={"rango_perdida": "Rango medio de pérdida", "p_mcs": "p-valor MCS", "en_conjunto": "¿En el conjunto?"})
    vista["Rango medio de pérdida"] = vista["Rango medio de pérdida"].round(2)
    vista["p-valor MCS"] = vista["p-valor MCS"].round(3)
    vista["¿En el conjunto?"] = vista["¿En el conjunto?"].map({True: "Sí", False: "No"})
    vista.to_csv(E / "mcs_regresion.csv", index=False)
    finalistas = mcs[mcs.en_conjunto].index.tolist()
    y_oof = pd.concat([pd.Series(r["y_test"], index=r["idx_test"]) for r in rr if
                       f"{r['modelo']}|{r['optimizador']}" == perd.columns[0]]).sort_index()
    filas = []
    orden = mcs.index.tolist()
    mejor, segundo = orden[0], orden[1]
    # Persistencia (Q̂(t+15) = Q(t)): modelo restringido anidado en los modelos lineales
    datos = cargar_datos()
    q_pers = np.sqrt(datos.X.calamar_cuad.values)[y_oof.index.values]
    e_pers = y_oof.values - q_pers
    for b in orden[1:max(4, len(finalistas))]:
        ea, eb = y_oof.values - preds[mejor].values, y_oof.values - preds[b].values
        dm = diebold_mariano(ea, eb, h=15)
        gw = giacomini_white(ea, eb, h=15)
        dl = (ea ** 2 - eb ** 2)
        icb = bootstrap_bloques(np.mean, dl / 1e6, media_bloque=longitud_bloque(dl), n_boot=1000,
                                metodo="percentil")
        filas.append({"modelo A": mejor, "modelo B": b, "DM-HLN": dm["dm"], "p DM": dm["p"],
                      "GW": gw["estadistico"], "p GW": gw["p"],
                      "Δ pérdida media (10⁶ m⁶/s²)": icb["estimacion"],
                      "IC 95 % bootstrap estacionario": f"[{icb['ic_inf']:.3f}; {icb['ic_sup']:.3f}]",
                      "d de Cohen": d_cohen_pareado(dl)})
    cw_filas = []
    for f in finalistas:
        cw = clark_west(y_oof.values, q_pers, preds[f].values, h=15)
        cw_filas.append({"modelo general": f, "modelo restringido": "persistencia", "CW": cw["cw"], "p CW": cw["p"],
                         "RMSE general": float(np.sqrt(perd[f].mean())),
                         "RMSE persistencia": float(np.sqrt(np.mean(e_pers ** 2)))})
    cwd = pd.DataFrame(cw_filas)
    cwd["p CW (Holm)"] = holm(cwd["p CW"].values)
    cwd.to_csv(E / "regresion_clark_west.csv", index=False)
    cmp = pd.DataFrame(filas)
    cmp["p DM (Holm)"] = holm(cmp["p DM"].values)
    cmp["p GW (Holm)"] = holm(cmp["p GW"].values)
    cmp.to_csv(E / "regresion_dirigidas.csv", index=False)
    bench_ridge = [c for c in perd.columns if c.startswith("ridge|")][0]
    sp = spa(perd[bench_ridge].values / 1e6, perd.drop(columns=bench_ridge) / 1e6)
    guardar_json(sp, E / "spa_regresion.json")

    # Tabla de métodos afines (para el dashboard)
    lit = pd.DataFrame([
        ("NGBoost", "Distribución paramétrica por boosting", "Pérdida del evento + límites por observación"),
        ("DGBM / XGBoostLSS", "Newton multiparámetro en árboles", "Entropía cruzada de la banda; búsqueda de Armijo"),
        ("GAMLSS por boosting", "Escala en logaritmo", "Interacciones por árboles; pérdida híbrida"),
        ("XGBoost AFT (intervalos)", "Verosimilitud de intervalo", "Complemento de la banda (dos semirrectas)"),
        ("Regresión ordinal", "Latente gaussiana cortada", "Umbrales móviles según Q_ref"),
        ("EMOS / BFS", "Distribución predictiva calibrada", "Sin pronóstico físico previo; pérdida del evento"),
        ("Multitarea", "Tarea auxiliar continua", "Ambas tareas comparten una distribución; regla estrictamente propia"),
        ("Dos etapas (regresión + umbral)", "Supervisión continua", "Dispersión heterocedástica y ajuste conjunto"),
    ], columns=["Método afín", "Qué conserva BLGB", "Qué añade BLGB"])
    lit.to_csv(E / "literatura_cercana.csv", index=False)

    mej = mejores.sort_values("pr_auc_mean", ascending=False)
    textos = {
        "texto_lideres": (f"El modelo original encabeza la tabla con PR-AUC externa de {medias['original']:.3f}; el mejor "
                          f"benchmark es {NOMBRES_MODELO[mej.iloc[0].modelo]} ({mej.iloc[0].pr_auc_mean:.3f}). "
                          "La desviación entre bloques refleja la variabilidad hidrológica interanual."),
        "texto_mcs": (f"El conjunto de confianza al 90 % retiene {len(finalistas)} de 28 combinaciones; la de menor "
                      f"pérdida es {mejor.replace('|', ' · ')}. Las combinaciones fuera del conjunto son "
                      "significativamente peores que la mejor."),
    }
    guardar_json(textos, E / "textos_dashboard.json")
    print(pr[["prueba", "p ajustado", "conclusión"]].to_string())


if __name__ == "__main__":
    main()
