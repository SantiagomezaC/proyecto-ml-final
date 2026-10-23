"""Estilo de las figuras estáticas del libro (matplotlib), coherente con el dashboard."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

from .config import PALETA

RIO, SED, CAU, ALE, NEU = (PALETA[k] for k in ("rio", "sedimento", "caudal", "alerta", "neutro"))
COLORES = [RIO, SED, CAU, ALE, NEU]


def aplicar():
    """Aplica la hoja de estilo del proyecto a todas las figuras siguientes."""
    mpl.rcParams.update({
        "figure.figsize": (9.5, 4.2), "figure.dpi": 110, "savefig.dpi": 150,
        "savefig.bbox": "tight", "font.size": 10.5, "axes.titlesize": 11.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 10.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9aa8ae",
        "axes.grid": True, "grid.color": "#e6ecef", "grid.linewidth": 0.8,
        "axes.prop_cycle": cycler(color=COLORES), "legend.frameon": False,
        "legend.fontsize": 9.5, "xtick.color": "#4b5960", "ytick.color": "#4b5960",
        "axes.labelcolor": "#13232c", "text.color": "#13232c", "lines.linewidth": 1.8,
        "image.cmap": "viridis",
    })


def mapa_div():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div", [ALE, "#f7f7f7", RIO])


def mapa_seq():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq", ["#f2f7f9", CAU, RIO])


aplicar()


def dibujar_cd(rangos, cd, destacado=None, ax=None, nombres=None):
    """Diagrama de diferencia crítica al estilo de Demšar (2006).

    Parameters
    ----------
    rangos : pandas.Series
        Rango promedio de cada modelo (1 = mejor).
    cd : float
        Diferencia crítica de Nemenyi.
    destacado : str, optional
        Modelo que se resalta en ocre.
    nombres : dict, optional
        Nombres legibles de los modelos.
    """
    nombres = nombres or {}
    r = rangos.sort_values()
    k = len(r)
    ax = ax or plt.gca()
    ax.set_xlim(0.2, k + 0.8)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.hlines(.85, 1, k, color="#13232c", lw=1)
    for t in range(1, k + 1):
        ax.vlines(t, .85, .88, color="#13232c", lw=1)
        ax.text(t, .9, str(t), ha="center", fontsize=9)
    ax.hlines(.97, 1, 1 + cd, color=ALE, lw=2.5)
    ax.text(1 + cd / 2, .99, f"CD = {cd:.2f}", ha="center", color=ALE, fontsize=9)
    mitad = (k + 1) // 2
    for i, (n, v) in enumerate(r.items()):
        izq = i < mitad
        y = .75 - .09 * (i if izq else k - 1 - i)
        xt = 0.9 if izq else k + .1
        color = SED if n == destacado else RIO
        ax.vlines(v, y, .85, color=color, lw=1)
        ax.hlines(y, v, xt, color=color, lw=1)
        ax.text(xt + (-.05 if izq else .05), y, f"{nombres.get(n, n)} ({v:.2f})", ha="right" if izq else "left",
                va="center", fontsize=9, color=SED if n == destacado else "#13232c",
                fontweight="bold" if n == destacado else None)
    vals = r.values
    grupos = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in grupos):
            grupos.append((i, j))
    for g, (i, j) in enumerate(grupos):
        ax.hlines(.81 - .025 * g, vals[i] - .03, vals[j] + .03, color=NEU, lw=4)
    return ax
