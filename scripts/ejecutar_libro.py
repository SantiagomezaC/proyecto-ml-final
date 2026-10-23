"""Convierte los capítulos MyST (book/fuentes/*.md) en cuadernos .ipynb (book/) y los ejecuta.

Uso:  python -m scripts.ejecutar_libro [capitulo ...]
Los cuadernos ejecutados quedan en book/ con sus salidas; ``jupyter-book build book/``
los compila sin volver a ejecutarlos (execute_notebooks: off).
"""
import sys
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

LIBRO = Path(__file__).resolve().parent.parent / "book"
FUENTES = LIBRO / "fuentes"


def ejecutar(md: Path, timeout: int = 3600) -> Path:
    nb = jupytext.read(md)
    nb.metadata["kernelspec"] = {"display_name": "Python (ml-final)", "language": "python", "name": "ml-final"}
    NotebookClient(nb, timeout=timeout, kernel_name="ml-final",
                   resources={"metadata": {"path": str(LIBRO)}}).execute()
    salida = LIBRO / md.with_suffix(".ipynb").name
    nbformat.write(nb, salida)
    return salida


if __name__ == "__main__":
    nombres = sys.argv[1:]
    archivos = [FUENTES / f"{n}.md" for n in nombres] if nombres else sorted(FUENTES.glob("0*.md"))
    for md in archivos:
        if md.read_text(encoding="utf-8").lstrip().startswith("---\njupytext"):
            print("Ejecutando", md.name, flush=True)
            print("  ->", ejecutar(md).name, flush=True)
