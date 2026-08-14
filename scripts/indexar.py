#!/usr/bin/env python3
"""Construye la base vectorial: corpus -> index.faiss + metadata.jsonl.

Encadena las fases 1 a 5: extraer texto, fragmentar, codificar, indexar y
guardar. Es el script que hay que correr una vez sobre el corpus completo
antes de poder generar los resultados.

Estructura de corpus que espera:

    data/raw/
        fenomeno_1/  archivos de IA en defensa
        fenomeno_2/  archivos de seguridad espacial
        fenomeno_3/  archivos de dinámicas territoriales

El número del fenómeno sale del nombre de la carpeta de primer nivel, porque
es un campo obligatorio de la metadata y no viene dentro de los archivos.

Uso:

    python scripts/indexar.py
    python scripts/indexar.py --limite 20        # prueba con 20 archivos
    python scripts/indexar.py --reusar-vectores  # no vuelve a codificar

Los documentos que no producen texto (PDF escaneado sin OCR, imagen
ilegible) se anotan en un CSV de errores y se sigue sin interrumpir la corrida.
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import config
from src.chunking import Chunk, DocumentoSinTexto, documento_a_chunks
from src.encoding import codificar_chunks, guardar_embeddings
from src.extraction import extraer, formato_desde_extension, titulo
from src.indexing import construir_indice, guardar_base

CORPUS_POR_DEFECTO = RAIZ / "data" / "raw"
VECTORES_POR_DEFECTO = RAIZ / "data" / "processed"
ERRORES_POR_DEFECTO = RAIZ / "data" / "interim" / "errores_extraccion.csv"

_FENOMENO = re.compile(r"[123]")
_NO_ALFANUMERICO = re.compile(r"[^A-Za-z0-9]+")


def carpeta_de_salida(base: Path | None = None) -> Path:
    """`entrega/base_vectorial/encoder_<nombre>/`, como pide la spec (§1.4)."""
    nombre = config.ENCODER.split("/")[-1]
    raiz = base or (RAIZ / "entrega" / "base_vectorial")
    return raiz / f"encoder_{nombre}"


def recolectar_archivos(corpus: Path) -> list[Path]:
    """Todos los archivos del corpus, en orden alfabético.

    El orden fijo importa: si dos corridas recorren el corpus en distinto
    orden, los chunk_id cambian de documento y la base deja de ser
    reproducible.
    """
    if not corpus.exists():
        raise FileNotFoundError(
            f"No existe la carpeta del corpus: {corpus}. Descarga los "
            f"archivos de ADL y ponlos en subcarpetas fenomeno_1, fenomeno_2 "
            f"y fenomeno_3."
        )

    archivos = [
        p
        for p in sorted(corpus.rglob("*"))
        if p.is_file() and not p.name.startswith(".")
    ]

    if not archivos:
        raise FileNotFoundError(f"La carpeta del corpus está vacía: {corpus}")

    return archivos


def fenomeno_de(ruta: Path, corpus: Path) -> int:
    """Deduce el número de fenómeno de la carpeta de primer nivel."""
    relativa = ruta.relative_to(corpus)

    if len(relativa.parts) < 2:
        raise ValueError(
            f"{relativa} está suelto en la raíz del corpus. Cada archivo debe "
            f"estar dentro de una carpeta de fenómeno (fenomeno_1, "
            f"fenomeno_2 o fenomeno_3)."
        )

    encontrado = _FENOMENO.search(relativa.parts[0])

    if not encontrado:
        raise ValueError(
            f"La carpeta {relativa.parts[0]!r} no dice a qué fenómeno "
            f"pertenece. Debe contener un 1, un 2 o un 3."
        )

    return int(encontrado.group())


def doc_id_de(ruta: Path, fenomeno: int, usados: set[str]) -> str:
    """Identificador interno, legible y único.

    La organización empareja los documentos por el campo `fuente` (el nombre
    del archivo original), no por este identificador, así que acá lo único
    que importa es que sea único y que permita rastrear el archivo a mano.
    """
    limpio = _NO_ALFANUMERICO.sub("_", ruta.stem).strip("_")[:60] or "doc"
    base = f"F{fenomeno}-{limpio}"

    doc_id = base
    sufijo = 2
    while doc_id in usados:
        doc_id = f"{base}_{sufijo}"
        sufijo += 1

    usados.add(doc_id)
    return doc_id


def procesar_archivo(ruta: Path, corpus: Path, usados: set[str]) -> list[Chunk]:
    """Un archivo del corpus -> sus chunks. Lanza DocumentoSinTexto si falla."""
    fenomeno = fenomeno_de(ruta, corpus)
    formato = formato_desde_extension(ruta)
    texto = extraer(ruta, formato)

    try:
        titulo_doc = titulo(ruta, formato)
    except Exception:
        # El título es opcional: enriquece lo que se codifica, pero un fallo
        # acá no justifica dejar el documento fuera del índice.
        titulo_doc = None

    return documento_a_chunks(
        texto,
        doc_id=doc_id_de(ruta, fenomeno, usados),
        fuente=ruta.name,
        formato=formato,
        fenomeno=fenomeno,
        titulo_doc=titulo_doc,
    )


def construir_chunks(
    archivos: list[Path],
    corpus: Path,
    verboso: bool = True,
) -> tuple[list[Chunk], list[dict]]:
    """Recorre el corpus y devuelve (chunks, errores)."""
    chunks: list[Chunk] = []
    errores: list[dict] = []
    usados: set[str] = set()

    for n, ruta in enumerate(archivos, start=1):
        try:
            nuevos = procesar_archivo(ruta, corpus, usados)
        except (DocumentoSinTexto, ValueError, AssertionError) as exc:
            errores.append(
                {
                    "archivo": str(ruta.relative_to(corpus)),
                    "error": type(exc).__name__,
                    "detalle": str(exc)[:300],
                }
            )
            continue
        except Exception as exc:  # noqa: BLE001
            errores.append(
                {
                    "archivo": str(ruta.relative_to(corpus)),
                    "error": type(exc).__name__,
                    "detalle": str(exc)[:300],
                }
            )
            continue

        chunks.extend(nuevos)

        if verboso and n % 25 == 0:
            print(
                f"  {n}/{len(archivos)} archivos, {len(chunks)} fragmentos, "
                f"{len(errores)} con error"
            )

    return chunks, errores


def guardar_errores(errores: list[dict], destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)

    with destino.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=["archivo", "error", "detalle"])
        escritor.writeheader()
        escritor.writerows(errores)


def main(argv=None, modelo=None) -> int:
    """`modelo` solo se usa en los tests, para no descargar el encoder real."""
    parser = argparse.ArgumentParser(description="Construye la base vectorial.")
    parser.add_argument("--corpus", type=Path, default=CORPUS_POR_DEFECTO)
    parser.add_argument("--salida", type=Path, default=None,
                        help="Por defecto entrega/base_vectorial/encoder_<nombre>/")
    parser.add_argument("--vectores", type=Path, default=VECTORES_POR_DEFECTO,
                        help="Dónde guardar los embeddings para no recodificar.")
    parser.add_argument("--errores", type=Path, default=ERRORES_POR_DEFECTO)
    parser.add_argument("--lote", type=int, default=8)
    parser.add_argument("--limite", type=int, default=None,
                        help="Procesa solo los primeros N archivos (para probar).")
    parser.add_argument("--reusar-vectores", action="store_true",
                        help="Carga los embeddings guardados en vez de codificar.")
    parser.add_argument("--grafo", action="store_true",
                        help="Construye también grafo/grafo.graphml (bonus, §8.5).")
    args = parser.parse_args(argv)

    salida = args.salida or carpeta_de_salida()
    inicio = time.perf_counter()

    print(f"Corpus:  {args.corpus}")
    print(f"Encoder: {config.ENCODER}")
    print(f"Salida:  {salida}")
    print()

    archivos = recolectar_archivos(args.corpus)
    if args.limite:
        archivos = archivos[: args.limite]
    print(f"1-3. Extrayendo y fragmentando {len(archivos)} archivos...")

    chunks, errores = construir_chunks(archivos, args.corpus)

    if not chunks:
        print("No se obtuvo ningún fragmento. Revisa el corpus.", file=sys.stderr)
        return 1

    documentos = len({c.doc_id for c in chunks})
    print(f"     {len(chunks)} fragmentos de {documentos} documentos")

    if errores:
        guardar_errores(errores, args.errores)
        print(f"     {len(errores)} archivos sin texto -> {args.errores}")
        print("     Revísalos: esos documentos no se pueden recuperar.")

    if args.reusar_vectores:
        from src.encoding import cargar_embeddings

        print("4.   Reusando vectores guardados...")
        embeddings = cargar_embeddings(args.vectores, chunks)
    else:
        print(f"4.   Codificando {len(chunks)} fragmentos (esto tarda horas)...")
        embeddings, chunks = codificar_chunks(chunks, modelo=modelo, lote=args.lote)
        guardar_embeddings(embeddings, chunks, args.vectores)
        print(f"     Vectores guardados en {args.vectores}")

    print("5.   Construyendo el índice FAISS...")
    index = construir_indice(embeddings, chunks)
    guardar_base(index, chunks, salida)

    if args.grafo:
        print("bonus. Construyendo el grafo de conocimiento...")
        try:
            from src.knowledge_graph import construir_desde_chunks

            construir_desde_chunks(chunks, salida)
        except Exception as exc:  # noqa: BLE001
            # El grafo es opcional: un fallo acá no puede tumbar la base
            # vectorial, que sí es obligatoria.
            print(f"     grafo omitido ({type(exc).__name__}: {exc})", file=sys.stderr)

    minutos = (time.perf_counter() - inicio) / 60
    print()
    print(f"Listo en {minutos:.1f} min.")
    print(f"  {salida / 'index.faiss'}")
    print(f"  {salida / 'metadata.jsonl'} ({len(chunks)} líneas)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
