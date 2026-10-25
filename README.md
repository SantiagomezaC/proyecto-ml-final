# Descarga efectiva en el río Magdalena

Predicción, con quince días de anticipación, de si el caudal del río Magdalena en Calamar caerá
dentro de la banda de **descarga efectiva**: el rango de caudales que, a lo largo de las décadas,
transporta la mayor fracción del sedimento. Universidad del Norte, Barranquilla, 2026.

**Autores:** Kevin Clemente Rosario y Manuel Santiago Meza Castro

| Producto | Enlace |
|---|---|
| Aplicación interactiva (Render) | https://descarga-efectiva-magdalena.onrender.com |
| Documento del estudio (Jupyter Book) | https://santiagomezac.github.io/proyecto-ml-final/ |
| Repositorio | https://github.com/SantiagomezaC/proyecto-ml-final |

## Contenido

- **ETL y EDA** de siete series diarias del IDEAM (1940-2026), con auditoría de la serie de
  transporte de sedimentos, análisis magnitud-frecuencia y estudio del mecanismo de los faltantes.
- **140 modelos de referencia** con validación anidada cronológica: 7 clasificadores × 4 técnicas de
  balanceo × 4 optimizadores (rejilla, aleatorio, bayesiano con Optuna-TPE y genético con DEAP),
  más 7 regresores × 4 optimizadores.
- **Modelo propuesto, Band-Likelihood Gradient Boosting (BLGB)**: boosting de segundo orden
  programado en numpy que modela la distribución del caudal futuro y optimiza una pérdida híbrida
  (entropía cruzada de la banda + verosimilitud gaussiana), con búsqueda lineal de Armijo.
- **Métodos recientes de comparación**: NGBoost, un enfoque distribucional de dos etapas y la
  familia LTSF-Linear de pronóstico de largo plazo (Linear, NLinear y DLinear; Zeng et al., 2023),
  implementada en `src/dlinear.py`.
- **Comparación estadística**: Friedman-Nemenyi, Wilcoxon-Holm, t corregida de Nadeau-Bengio,
  prueba bayesiana de rangos con signo, bootstrap estacionario BCa, DeLong y McNemar; en
  regresión, MCS, Giacomini-White, Clark-West, Diebold-Mariano (HLN) y SPA.
- **Dashboard multipágina** en Dash con predicción en vivo, explicación local por valores de
  Shapley e intervalo de predicción del caudal.

## Estructura

```text
.
├── app.py                  punto de entrada del dashboard (expone server)
├── pages/                  contexto.py, eda.py, modelos.py
├── src/
│   ├── config.py           semilla global, rutas, partición y paleta
│   ├── etl.py              lectura DHIME, auditoría de transporte, descarga efectiva
│   ├── features.py         predictores y tareas de clasificación y regresión
│   ├── splits.py           particiones cronológicas de la validación anidada
│   ├── spaces.py           espacios de búsqueda comunes a los cuatro optimizadores
│   ├── models.py           catálogo de modelos, balanceo y pipelines
│   ├── optim.py            Grid, Random, TPE (Optuna) y genético (DEAP)
│   ├── experiment.py       motor de la validación anidada (resumible)
│   ├── original_model.py   modelo propuesto BLGB
│   ├── experiment_original.py  validación del modelo propuesto y métodos de comparación
│   ├── dlinear.py          LTSF-Linear: Linear, NLinear y DLinear
│   ├── stats_tests.py      pruebas estadísticas y bootstrap por bloques
│   ├── evaluation.py       calibración y valores de Shapley por permutaciones
│   ├── artifacts.py        serialización, metadata y prueba de ida y vuelta
│   ├── analysis.py         tablas de resultados para el libro
│   ├── data.py, plots.py   carga en caché y figuras del dashboard
│   └── estilo.py           estilo de las figuras del libro
├── scripts/                etl, run_benchmarks, run_original, run_dlinear, run_extra, final_models,
│                           robustez, computational, estadistica, exportar_dashboard,
│                           revision_literatura, ejecutar_libro
├── artifacts/              modelos finales (.joblib) y metadata.json
├── data/                   processed/ (tablas de modelado), dashboard/ (tablas livianas)
├── results/                tabla maestra, resultados por pliegue y análisis
├── tests/                  pruebas unitarias (pytest)
├── book/                   fuentes del Jupyter Book (_config.yml, _toc.yml, references.bib)
├── assets/                 hoja de estilo y logotipo del dashboard
├── requirements.txt        dependencias de producción (dashboard)
├── requirements-dev.txt    entorno completo de experimentación y del libro
├── environment.yml
└── render.yaml             despliegue en Render
```

## Instalación

```bash
conda env create -f environment.yml
conda activate ml-final
```

Semilla global: `SEED = 42` en `src/config.py`, propagada a numpy, scikit-learn, XGBoost,
Optuna y DEAP.

## Datos

Los archivos del IDEAM no se redistribuyen: la licencia del portal DHIME autoriza la descarga
para uso personal y no comercial. Descargue el caudal medio diario (`Q_MEDIA_D`) y el transporte
medio diario de sedimentos (`TR_KT_D_QS_D`) de las estaciones 29037020 (Calamar), 25027930
(Coyongal), 25027320 (San Roque), 25027020 (El Banco), 25027640 (Tres Cruces), 23037010
(Puerto Salgar) y 21237020 (Arrancaplumas) en http://dhime.ideam.gov.co y guarde los CSV en
`data/raw/` (o defina la variable de entorno `MAGDALENA_RAW`). Las tablas procesadas que
consume el dashboard sí se incluyen en `data/`.

## Reproducción

```bash
python -m scripts.etl                      # tablas de modelado y particiones
python -m scripts.run_benchmarks clf       # 112 combinaciones (validación anidada)
python -m scripts.run_benchmarks reg       # 28 combinaciones
python -m scripts.run_original             # modelo propuesto y métodos distribucionales
python -m scripts.run_dlinear              # Linear, NLinear y DLinear con el mismo protocolo
python -m scripts.final_models clf         # selección, reentrenamiento y artefactos
python -m scripts.final_models reg
python -m scripts.final_models original
python -m scripts.run_extra                # tres tareas adicionales
python -m scripts.robustez                 # ablación, semillas, ruido, desbalance
python -m scripts.computational            # complejidad, variantes eficientes, perfiles
python -m scripts.estadistica              # pruebas y veredicto de superioridad
python -m scripts.exportar_dashboard       # tablas de la aplicación
python -m pytest                           # pruebas unitarias
python -m scripts.ejecutar_libro           # ejecuta los cuadernos del libro
jupyter-book build book/
python -m scripts.formato_es              # cifras del libro en formato español
```

Los resultados por pliegue se guardan uno por archivo, de modo que una corrida interrumpida se
reanuda sin repetir trabajo. En un Intel Core i7-1255U con diez procesos, el bloque de
clasificación tarda unas cuatro horas.

## Aplicación interactiva

```bash
python app.py                 # local, http://127.0.0.1:8050
gunicorn app:server           # producción
```

Despliegue en Render: *New → Blueprint* sobre este repositorio (lee `render.yaml`), o un *Web
Service* con `pip install -r requirements.txt` y `gunicorn app:server`. Las versiones de
`requirements.txt` coinciden con las usadas para serializar los artefactos.
