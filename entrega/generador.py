#!/usr/bin/env python3
"""Entregable obligatorio: lee las consultas y produce `resultados.jsonl`.

Es el script con el que la organización reproduce los resultados. Si no
corre, el equipo queda fuera de la evaluación (§1.4).

No construye nada: usa el índice que ya dejó `scripts/indexar.py`. Para cada
consulta convierte la pregunta en vector con el mismo encoder, busca los
fragmentos más parecidos, agrupa por documento y escribe una línea.

Uso:

    python entrega/generador.py --consultas consultas.json

El archivo de consultas puede ser .json, .jsonl o .csv. Se aceptan varios
nombres de campo (`id`/`query_id`, `text`/`texto`/`consulta`/`query`) porque
la organización todavía no publicó el formato exacto.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# `src/` puede estar junto a este archivo (entrega empaquetada por
# scripts/empaquetar.py) o un nivel arriba (repo de desarrollo). Se prueban
# los dos para que la entrega funcione por sí sola en la máquina del jurado.
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI if (AQUI / "src").is_dir() else AQUI.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

try:
    from src import config
    from src.aggregation import ESTRATEGIAS, ESTRATEGIA_POR_DEFECTO, agregar_documentos
    from src.encoding import codificar_consulta
    from src.indexing import cargar_base
    from src.output import armar_linea, escribir_resultados
    from src.retrieval import buscar_vector
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        f"No se pudo importar el paquete src ({exc}). Este script necesita "
        f"el repositorio completo: se esperaba encontrar src/ en {RAIZ}."
    ) from exc

N_FRAGMENTOS = 10
N_DOCUMENTOS = 3

# Se recuperan más fragmentos de los 10 que se entregan porque los 3
# documentos se deducen de los fragmentos: con solo 10 podría no haber
# fragmentos de 3 documentos distintos.
K_POR_DEFECTO = 50

_CAMPOS_ID = ("id", "query_id", "consulta_id", "qid")
_CAMPOS_TEXTO = ("text", "texto", "consulta", "query", "pregunta")


def _extraer_par(registro: dict) -> tuple[str, str]:
    id_consulta = next((registro[c] for c in _CAMPOS_ID if registro.get(c)), None)
    texto = next((registro[c] for c in _CAMPOS_TEXTO if registro.get(c)), None)

    if not id_consulta or not texto:
        raise ValueError(
            f"No se encontró el id o el texto en {registro!r}. Campos "
            f"aceptados para el id: {_CAMPOS_ID}. Para el texto: "
            f"{_CAMPOS_TEXTO}."
        )

    return str(id_consulta).strip(), str(texto).strip()


def cargar_consultas(path: str | Path) -> list[tuple[str, str]]:
    """Lee el archivo de consultas y devuelve pares (id, texto) ordenados."""
    ruta = Path(path)

    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo de consultas: {ruta}")

    if ruta.suffix == ".jsonl":
        registros = [
            json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
    elif ruta.suffix == ".json":
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        if isinstance(datos, dict):
            # Puede venir envuelto: {"consultas": [...]} o {"q001": "texto"}.
            for clave in ("consultas", "queries", "data"):
                if isinstance(datos.get(clave), list):
                    datos = datos[clave]
                    break
            else:
                datos = [{"id": k, "text": v} for k, v in datos.items()]
        registros = datos
    elif ruta.suffix == ".csv":
        with ruta.open(encoding="utf-8-sig", newline="") as archivo:
            registros = list(csv.DictReader(archivo))
    else:
        raise ValueError(
            f"Extensión {ruta.suffix!r} no soportada. Usa .json, .jsonl o .csv."
        )

    consultas = [_extraer_par(r) for r in registros]

    vistos = [c[0] for c in consultas]
    if len(vistos) != len(set(vistos)):
        raise ValueError("Hay id de consulta repetidos en el archivo.")

    return sorted(consultas, key=lambda c: c[0])


def responder(
    texto: str,
    index,
    metadata,
    *,
    k: int = K_POR_DEFECTO,
    estrategia: str = ESTRATEGIA_POR_DEFECTO,
    modelo=None,
) -> tuple[list[dict], list[dict]]:
    """Una consulta -> (documentos, fragmentos) listos para `output`.

    Si con `k` fragmentos no aparecen 3 documentos distintos, se reintenta con
    una k mayor hasta agotar el índice. La entrega exige exactamente 3
    documentos por consulta (§9.3.2) y `output.armar_documentos` lanza
    ValueError si recibe menos: sin este reintento, una sola consulta pobre
    aborta la generación completa y deja al equipo sin `resultados.jsonl`.
    """
    vector = codificar_consulta(texto, modelo=modelo)

    fragmentos = buscar_vector(vector, index, metadata, k=k)
    documentos = agregar_documentos(fragmentos, estrategia=estrategia)

    k_actual = k
    while len(documentos) < N_DOCUMENTOS and k_actual < index.ntotal:
        k_actual = min(k_actual * 4, index.ntotal)
        fragmentos = buscar_vector(vector, index, metadata, k=k_actual)
        documentos = agregar_documentos(fragmentos, estrategia=estrategia)

    if len(documentos) < N_DOCUMENTOS:
        # El índice entero tiene menos de 3 documentos: no es un problema de
        # búsqueda sino de corpus, y hay que verlo ahora y no al validar.
        raise ValueError(
            f"Solo hay {len(documentos)} documento(s) distinto(s) en todo el "
            f"índice ({index.ntotal} fragmentos). La entrega exige 3."
        )

    return documentos, fragmentos[:N_FRAGMENTOS]


def main(argv=None, modelo=None) -> int:
    """`modelo` solo se usa en los tests, para no descargar el encoder real."""
    parser = argparse.ArgumentParser(description="Genera resultados.jsonl.")
    parser.add_argument("--consultas", type=Path, default=None)
    parser.add_argument("--ayuda-imports", action="store_true",
                        help="Solo comprueba que el paquete src se importa. Lo usa "
                             "scripts/empaquetar.py para verificar la entrega.")
    parser.add_argument("--base", type=Path, default=None,
                        help="Por defecto entrega/base_vectorial/encoder_<nombre>/")
    parser.add_argument("--salida", type=Path, default=RAIZ / "entrega" / "resultados.jsonl")
    parser.add_argument("--k", type=int, default=K_POR_DEFECTO)
    parser.add_argument("--estrategia", choices=ESTRATEGIAS, default=ESTRATEGIA_POR_DEFECTO)
    args = parser.parse_args(argv)

    if args.ayuda_imports:
        print(f"src importado desde {RAIZ}; encoder {config.ENCODER}")
        return 0

    if args.consultas is None:
        parser.error("hace falta --consultas (o --ayuda-imports)")

    base = args.base or (
        RAIZ / "entrega" / "base_vectorial" / f"encoder_{config.ENCODER.split('/')[-1]}"
    )
    inicio = time.perf_counter()

    print(f"Base:    {base}")
    print(f"Encoder: {config.ENCODER}")

    index, metadata = cargar_base(base)
    print(f"         {index.ntotal} fragmentos indexados")

    consultas = cargar_consultas(args.consultas)
    print(f"Consultas: {len(consultas)} desde {args.consultas}")
    print()

    lineas = []
    for n, (id_consulta, texto) in enumerate(consultas, start=1):
        documentos, fragmentos = responder(
            texto,
            index,
            metadata,
            k=args.k,
            estrategia=args.estrategia,
            modelo=modelo,
        )
        lineas.append(armar_linea(id_consulta, documentos, fragmentos, metadata))

        if n % 10 == 0:
            print(f"  {n}/{len(consultas)} consultas")

    escribir_resultados(lineas, args.salida)

    segundos = time.perf_counter() - inicio
    print()
    print(f"Listo en {segundos:.1f} s: {args.salida} ({len(lineas)} líneas)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
