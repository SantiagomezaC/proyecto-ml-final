"""Convierte al formato numérico español las cifras insertadas con glue en el HTML del libro.

Los valores que los cuadernos insertan en la prosa ({glue:text}) se formatean con la
convención de Python (0.910; 28,558). Este paso, que se ejecuta después de
``jupyter-book build``, los lleva a la convención del texto (0,910; 28.558) y escribe la
notación científica como potencia de diez. Solo modifica los fragmentos marcados por glue,
nunca la prosa ni las tablas.

Uso:  python -m scripts.formato_es [carpeta_html]
"""
import re
import sys
from pathlib import Path

HTML = Path(__file__).resolve().parent.parent / "book" / "_build" / "html"
FRAGMENTO = re.compile(r'(<span class="pasted-text">)([^<]*)(</span>)')
NUMERO = re.compile(r"^[-+−]?\d[\d,]*(\.\d+)?(e[-+]\d+)?%?$")
SUPER = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def a_espanol(texto: str) -> str:
    if not NUMERO.match(texto):
        return texto
    base, _, exp = texto.partition("e")
    base = base.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    if exp:
        return f"{base} × 10{str(int(exp)).translate(SUPER)}"
    return base[:-1] + " %" if base.endswith("%") else base


def convertir(carpeta: Path = HTML) -> int:
    n = 0
    for f in carpeta.glob("*.html"):
        t = f.read_text(encoding="utf-8")
        nuevo = FRAGMENTO.sub(lambda m: m.group(1) + a_espanol(m.group(2)) + m.group(3), t)
        if nuevo != t:
            f.write_text(nuevo, encoding="utf-8")
            n += 1
    return n


if __name__ == "__main__":
    carpeta = Path(sys.argv[1]) if len(sys.argv) > 1 else HTML
    print(f"{convertir(carpeta)} páginas convertidas")
