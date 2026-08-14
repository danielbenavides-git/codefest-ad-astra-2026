#!/usr/bin/env python3
"""Reconocimiento del corpus antes de indexar. No ejecuta OCR: lo estima.

Abre cada PDF y cuenta las páginas sin capa de texto. Es rápido —abrir un PDF
y pedir su texto no cuesta casi nada— y responde la pregunta que decide el
cronograma: cuántas horas de OCR hay por delante.

Medido en este proyecto: 4,8 s por página escaneada a 150 dpi en un núcleo.
Con 759 PDFs, la diferencia entre un corpus 20% escaneado y uno 80% escaneado
es la diferencia entre una tarde y tres días.

Uso:

    python scripts/reconocimiento.py
    python scripts/reconocimiento.py --corpus data/raw --nucleos 8
"""

import argparse
import csv
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.extraction import MIN_PALABRAS_PAGINA, formato_desde_extension

SEGUNDOS_POR_PAGINA_OCR = 4.8      # medido a 150 dpi, un núcleo


def revisar_pdf(ruta: Path) -> dict:
    """Cuenta páginas totales y páginas sin capa de texto. Sin OCR."""
    import fitz

    registro = {"archivo": str(ruta), "paginas": 0, "sin_texto": 0, "error": ""}
    try:
        doc = fitz.open(ruta)
    except Exception as exc:  # noqa: BLE001
        registro["error"] = f"{type(exc).__name__}: {exc}"[:120]
        return registro

    try:
        for pagina in doc:
            registro["paginas"] += 1
            texto = pagina.get_text("text")
            if len(texto.split()) < MIN_PALABRAS_PAGINA:
                registro["sin_texto"] += 1
    except Exception as exc:  # noqa: BLE001
        registro["error"] = f"{type(exc).__name__}: {exc}"[:120]
    finally:
        doc.close()

    return registro


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Estima el coste de OCR del corpus.")
    parser.add_argument("--corpus", type=Path, default=RAIZ / "data" / "raw")
    parser.add_argument("--nucleos", type=int, default=4)
    parser.add_argument("--salida", type=Path,
                        default=RAIZ / "data" / "interim" / "reconocimiento_ocr.csv")
    args = parser.parse_args(argv)

    if not args.corpus.exists():
        print(f"No existe {args.corpus}", file=sys.stderr)
        return 1

    archivos = sorted(p for p in args.corpus.rglob("*") if p.is_file() and not p.name.startswith("."))
    formatos = Counter(formato_desde_extension(p) for p in archivos)
    pdfs = [p for p in archivos if formato_desde_extension(p) == "pdf"]

    print(f"{len(archivos)} archivos en {args.corpus}")
    for fmt, n in formatos.most_common():
        print(f"  {fmt:12} {n:>5}")
    print(f"\nRevisando {len(pdfs)} PDFs con {args.nucleos} núcleos...")

    inicio = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.nucleos) as pool:
        registros = list(pool.map(revisar_pdf, pdfs, chunksize=8))
    print(f"  hecho en {time.perf_counter() - inicio:.0f}s\n")

    paginas = sum(r["paginas"] for r in registros)
    sin_texto = sum(r["sin_texto"] for r in registros)
    rotos = [r for r in registros if r["error"]]
    escaneados = [r for r in registros if r["paginas"] and r["sin_texto"] == r["paginas"]]
    mixtos = [r for r in registros if 0 < r["sin_texto"] < r["paginas"]]

    print(f"páginas totales:        {paginas:>7}")
    print(f"páginas sin capa texto: {sin_texto:>7}  ({100 * sin_texto / paginas:.0f}%)"
          if paginas else "sin páginas")
    print(f"PDFs escaneados enteros:{len(escaneados):>7}")
    print(f"PDFs mixtos:            {len(mixtos):>7}")
    print(f"PDFs ilegibles:         {len(rotos):>7}")

    horas = sin_texto * SEGUNDOS_POR_PAGINA_OCR / 3600
    print(f"\nOCR estimado: {horas:.1f} h en 1 núcleo, "
          f"{horas / max(args.nucleos, 1):.1f} h con {args.nucleos}")
    if horas / max(args.nucleos, 1) > 6:
        print("  -> No cabe en una sesión de trabajo. Opciones: más núcleos, "
              "bajar DPI_OCR, o indexar primero sin OCR y añadir esos documentos después.")

    if rotos:
        print("\nilegibles (revisar a mano):")
        for r in rotos[:10]:
            print(f"  {Path(r['archivo']).name}: {r['error']}")

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with args.salida.open("w", encoding="utf-8", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=["archivo", "paginas", "sin_texto", "error"])
        escritor.writeheader()
        escritor.writerows(registros)
    print(f"\ndetalle -> {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
