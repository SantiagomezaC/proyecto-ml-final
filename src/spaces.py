"""Espacios de búsqueda declarados una sola vez y traducidos a los cuatro optimizadores.

Cada hiperparámetro se describe con su tipo, rango y escala. A partir de esa única
declaración se derivan la rejilla (Grid Search), las distribuciones (Random Search), las
sugerencias de Optuna y la decodificación de los genes (DEAP), lo que garantiza que los
cuatro métodos exploren exactamente el mismo dominio.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Param:
    """Hiperparámetro de un espacio de búsqueda.

    Attributes
    ----------
    nombre : str
        Nombre del argumento del estimador.
    tipo : {"float", "int", "cat"}
    bajo, alto : float
        Límites del rango (ignorados en categóricos).
    log : bool
        Escala logarítmica, recomendada para C, λ y tasas de aprendizaje.
    opciones : tuple
        Valores posibles de un categórico.
    """
    nombre: str
    tipo: str
    bajo: float = 0.0
    alto: float = 1.0
    log: bool = False
    opciones: tuple = field(default_factory=tuple)

    def desde_unitario(self, u: float):
        """Mapea u ∈ [0, 1] al dominio del parámetro (decodificación de genes y muestreo)."""
        u = float(np.clip(u, 0.0, 1.0 - 1e-12))
        if self.tipo == "cat":
            return self.opciones[int(u * len(self.opciones))]
        if self.log:
            v = np.exp(np.log(self.bajo) + u * (np.log(self.alto) - np.log(self.bajo)))
        else:
            v = self.bajo + u * (self.alto - self.bajo)
        return int(round(v)) if self.tipo == "int" else float(v)

    def niveles(self, k: int) -> list:
        """``k`` valores equiespaciados (en la escala del parámetro) para la rejilla."""
        if self.tipo == "cat":
            return list(self.opciones)
        if k == 1:
            return [self.desde_unitario(0.5)]
        vals = [self.desde_unitario(u) for u in np.linspace(0, 1 - 1e-9, k)]
        return list(dict.fromkeys(vals))  # elimina duplicados de enteros


def rejilla(espacio: list[Param], presupuesto: int) -> list[dict]:
    """Rejilla cartesiana con a lo sumo ``presupuesto`` combinaciones.

    Los categóricos entran con todas sus opciones; los numéricos reciben niveles de forma
    rotativa mientras el producto no exceda el presupuesto.
    """
    num = [p for p in espacio if p.tipo != "cat"]
    base = int(np.prod([len(p.opciones) for p in espacio if p.tipo == "cat"])) or 1
    k = {p.nombre: 1 for p in num}

    def tam(kk):
        return base * int(np.prod(list(kk.values()) or [1]))

    cambio = True
    while cambio and num:
        cambio = False
        for p in num:
            prueba = dict(k)
            prueba[p.nombre] += 1
            if tam(prueba) <= presupuesto:
                k = prueba
                cambio = True
    ejes = [(p.nombre, p.niveles(k[p.nombre]) if p.tipo != "cat" else list(p.opciones)) for p in espacio]
    nombres = [n for n, _ in ejes]
    return [dict(zip(nombres, combo)) for combo in itertools.product(*[v for _, v in ejes])]


def muestrear(espacio: list[Param], rng: np.random.Generator) -> dict:
    """Muestra uniforme (o log-uniforme) del espacio, para Random Search."""
    return {p.nombre: p.desde_unitario(rng.random()) for p in espacio}


def sugerir_optuna(trial, espacio: list[Param]) -> dict:
    """Traduce el espacio a las sugerencias de un ``optuna.Trial``."""
    out = {}
    for p in espacio:
        if p.tipo == "cat":
            out[p.nombre] = trial.suggest_categorical(p.nombre, list(p.opciones))
        elif p.tipo == "int":
            out[p.nombre] = trial.suggest_int(p.nombre, int(p.bajo), int(p.alto), log=p.log)
        else:
            out[p.nombre] = trial.suggest_float(p.nombre, p.bajo, p.alto, log=p.log)
    return out


def decodificar(genes, espacio: list[Param]) -> dict:
    """Convierte un individuo con genes en [0, 1] en un diccionario de hiperparámetros."""
    return {p.nombre: p.desde_unitario(g) for p, g in zip(espacio, genes)}
