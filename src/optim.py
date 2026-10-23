"""Optimización de hiperparámetros: Grid, Random, Bayesiana (Optuna-TPE) y Genética (DEAP).

Los cuatro métodos comparten la misma función objetivo, :func:`evaluar_config`, que
reconstruye el pipeline completo en cada pliegue interno. Así el preprocesamiento y el
sobremuestreo se ajustan solo con la porción de entrenamiento del pliegue, también en
Optuna y DEAP, que no integran esa protección de forma nativa.
"""
from __future__ import annotations

import time
import warnings
from typing import Callable

import numpy as np
import optuna
from deap import base, creator, tools
from sklearn.metrics import average_precision_score, mean_squared_error

from .config import SEED
from .spaces import Param, decodificar, muestrear, rejilla, sugerir_optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


def puntuar(tarea: str, y_true, y_score) -> float:
    """Métrica de selección: PR-AUC en clasificación y −RMSE en regresión (mayor es mejor)."""
    if tarea == "clf":
        return float(average_precision_score(y_true, y_score))
    return -float(np.sqrt(mean_squared_error(y_true, y_score)))


def evaluar_config(params: dict, fabrica: Callable, X, y, cv, tarea: str,
                   predecir: Callable) -> float:
    """Puntaje medio de validación interna para una configuración.

    Parameters
    ----------
    params : dict
        Hiperparámetros candidatos.
    fabrica : callable
        ``fabrica(params)`` devuelve un pipeline nuevo, sin ajustar.
    X, y : array-like
        Datos de entrenamiento del pliegue externo (nunca incluyen su bloque de prueba).
    cv : splitter
        Validación cruzada interna cronológica.
    predecir : callable
        ``predecir(estimador, X)`` devuelve el puntaje continuo o la predicción.
    """
    puntajes = []
    for tr, va in cv.split(X):
        modelo = fabrica(params).fit(X.iloc[tr], y.iloc[tr])
        puntajes.append(puntuar(tarea, y.iloc[va], predecir(modelo, X.iloc[va])))
    return float(np.mean(puntajes))


class Registro:
    """Historial de evaluaciones: insumo de las curvas *anytime* y del presupuesto."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.filas: list[dict] = []

    def anotar(self, params: dict, puntaje: float, t_eval: float, extra: dict | None = None):
        mejor = max([f["puntaje"] for f in self.filas] + [puntaje])
        self.filas.append({"eval": len(self.filas) + 1, "puntaje": puntaje, "mejor": mejor,
                           "t_eval": t_eval, "t_acum": time.perf_counter() - self.t0,
                           "params": dict(params), **(extra or {})})

    def mejor(self) -> tuple[dict, float]:
        f = max(self.filas, key=lambda r: r["puntaje"])
        return f["params"], f["puntaje"]


def _objetivo_cronometrado(objetivo, reg: Registro, extra=None):
    def f(params):
        t = time.perf_counter()
        s = objetivo(params)
        reg.anotar(params, s, time.perf_counter() - t, extra)
        return s
    return f


def grid_search(objetivo, espacio: list[Param], presupuesto: int, seed: int = SEED) -> Registro:
    """Búsqueda exhaustiva sobre una rejilla de a lo sumo ``presupuesto`` puntos."""
    reg = Registro()
    f = _objetivo_cronometrado(objetivo, reg)
    for params in rejilla(espacio, presupuesto):
        f(params)
    return reg


def random_search(objetivo, espacio: list[Param], presupuesto: int, seed: int = SEED) -> Registro:
    """Muestreo aleatorio (uniforme o log-uniforme) de ``presupuesto`` configuraciones."""
    reg = Registro()
    f = _objetivo_cronometrado(objetivo, reg)
    rng = np.random.default_rng(seed)
    for _ in range(presupuesto):
        f(muestrear(espacio, rng))
    return reg


def bayes_search(objetivo, espacio: list[Param], presupuesto: int, seed: int = SEED,
                 n_iniciales: int = 6) -> Registro:
    """Optimización bayesiana con Optuna.

    Sustituto: *Tree-structured Parzen Estimator* (TPE), que modela por separado la
    densidad de las configuraciones buenas, l(x), y malas, g(x), en lugar de p(y|x).
    Adquisición: Mejora Esperada; bajo TPE maximizar EI equivale a maximizar l(x)/g(x)
    (Bergstra et al., 2011). Las ``n_iniciales`` evaluaciones aleatorias aseguran
    exploración antes de que el cociente concentre la búsqueda (explotación).
    """
    reg = Registro()
    f = _objetivo_cronometrado(objetivo, reg)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_iniciales,
                                         multivariate=True)
    estudio = optuna.create_study(direction="maximize", sampler=sampler)
    estudio.optimize(lambda trial: f(sugerir_optuna(trial, espacio)), n_trials=presupuesto)
    return reg


# DEAP exige registrar las clases una sola vez por proceso
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individuo"):
    creator.create("Individuo", list, fitness=creator.FitnessMax)


def genetic_search(objetivo, espacio: list[Param], presupuesto: int, seed: int = SEED,
                   poblacion: int = 6, p_cruce: float = 0.7, p_mutacion: float = 0.3,
                   sigma: float = 0.2, torneo: int = 3) -> Registro:
    """Algoritmo genético con DEAP sobre genes en [0, 1].

    * Selección por torneo de tamaño 3: presión selectiva moderada y robusta al escalado
      del fitness, a diferencia de la ruleta.
    * Cruce de mezcla (``cxBlend``, α = 0.5) con probabilidad 0.7, adecuado para genes reales.
    * Mutación gaussiana (σ = 0.2, probabilidad por gen 0.3) con probabilidad 0.3, con
      recorte a [0, 1].
    * Elitismo de un individuo: el mejor pasa intacto a la generación siguiente.
    * Diversidad: distancia euclídea media entre pares de la población en cada generación,
      indicador de convergencia prematura.

    El número de generaciones se deduce del presupuesto: ``poblacion`` evaluaciones
    iniciales y ``poblacion − 1`` por generación (el élite no se reevalúa).
    """
    reg = Registro()
    rng = np.random.default_rng(seed)
    d = len(espacio)
    tb = base.Toolbox()
    tb.register("mate", tools.cxBlend, alpha=0.5)
    tb.register("mutate", tools.mutGaussian, mu=0.0, sigma=sigma, indpb=0.3)
    tb.register("select", tools.selTournament, tournsize=torneo)

    def diversidad(pop):
        g = np.array([list(i) for i in pop])
        if len(g) < 2:
            return 0.0
        dif = g[:, None, :] - g[None, :, :]
        return float(np.sqrt((dif ** 2).sum(-1))[np.triu_indices(len(g), 1)].mean())

    def evaluar(ind, gen, div):
        t = time.perf_counter()
        params = decodificar(ind, espacio)
        s = objetivo(params)
        reg.anotar(params, s, time.perf_counter() - t, {"generacion": gen, "diversidad": div})
        ind.fitness.values = (s,)

    import random as _random
    _random.seed(seed)   # DEAP usa el módulo random de la biblioteca estándar
    pop = [creator.Individuo(rng.random(d).tolist()) for _ in range(poblacion)]
    div = diversidad(pop)
    for ind in pop:
        evaluar(ind, 0, div)
    gen = 0
    while len(reg.filas) < presupuesto:
        gen += 1
        elite = tools.selBest(pop, 1)[0]
        hijos = [creator.Individuo(list(i)) for i in tb.select(pop, poblacion - 1)]
        for a, b in zip(hijos[::2], hijos[1::2]):
            if _random.random() < p_cruce:
                tb.mate(a, b)
        for h in hijos:
            if _random.random() < p_mutacion:
                tb.mutate(h)
            h[:] = np.clip(h, 0.0, 1.0).tolist()
        pop = [elite] + hijos
        div = diversidad(pop)
        for h in hijos[: presupuesto - len(reg.filas)]:
            evaluar(h, gen, div)
        pop = [i for i in pop if i.fitness.valid]
    return reg


OPTIMIZADORES = {"grid": grid_search, "random": random_search, "bayes": bayes_search,
                 "genetico": genetic_search}


def optimizar(metodo: str, objetivo, espacio: list[Param], presupuesto: int,
              seed: int = SEED) -> Registro:
    """Despacha al optimizador ``metodo`` con un presupuesto común de evaluaciones."""
    return OPTIMIZADORES[metodo](objetivo, espacio, presupuesto, seed=seed)
