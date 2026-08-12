import json
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np

from .chunking import Chunk


def validar_entradas(
    embeddings: np.ndarray,
    chunks: Sequence[Chunk],
) -> np.ndarray:
    """
    Valida y prepara los embeddings antes de construir el índice FAISS.

    Cada fila de embeddings debe corresponder exactamente al chunk
    ubicado en la misma posición de la lista `chunks`.

    Retorna:
        np.ndarray: matriz 2D, float32 y contigua en memoria.
    """

    vectores = np.asarray(embeddings, dtype=np.float32)

    if vectores.ndim != 2:
        raise ValueError(
            f"Los embeddings deben formar una matriz 2D. "
            f"Forma recibida: {vectores.shape}"
        )

    numero_vectores, dimension = vectores.shape

    if numero_vectores == 0:
        raise ValueError("No se recibieron embeddings para indexar.")

    if dimension == 0:
        raise ValueError("Los embeddings no pueden tener dimensión 0.")

    if numero_vectores != len(chunks):
        raise ValueError(
            f"Cantidad de embeddings ({numero_vectores}) distinta "
            f"a la cantidad de chunks ({len(chunks)})."
        )

    if not np.isfinite(vectores).all():
        raise ValueError(
            "Los embeddings contienen valores NaN o infinitos."
        )

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Se encontraron chunk_id duplicados.")

    return np.ascontiguousarray(vectores, dtype=np.float32)

def construir_indice(
    embeddings: np.ndarray,
    chunks: Sequence[Chunk],
) -> faiss.Index:
    """
    Construye un índice FAISS usando similitud coseno.

    """

    vectores = validar_entradas(embeddings, chunks)

    faiss.normalize_L2(vectores)

    dimension = vectores.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(vectores)

    if index.ntotal != len(chunks):
        raise RuntimeError(
            f"FAISS contiene {index.ntotal} vectores, "
            f"pero existen {len(chunks)} chunks."
        )

    return index

def guardar_base(
    index: faiss.Index,
    chunks: Sequence[Chunk],
    directorio: str | Path,
) -> None:
    """
    Guarda el índice FAISS y la metadata de los chunks en disco.
    """

    if index.ntotal != len(chunks):
        raise ValueError(
            f"FAISS contiene {index.ntotal} vectores, "
            f"pero se recibieron {len(chunks)} chunks."
        )

    ruta = Path(directorio)
    ruta.mkdir(parents=True, exist_ok=True)

    ruta_indice = ruta / "index.faiss"
    ruta_metadata = ruta / "metadata.jsonl"

    faiss.write_index(index, str(ruta_indice))

    with ruta_metadata.open("w", encoding="utf-8") as archivo:
        for chunk in chunks:
            registro = chunk.to_dict()
            archivo.write(
                json.dumps(registro, ensure_ascii=False) + "\n"
            )

def cargar_base(
    directorio: str | Path,
) -> tuple[faiss.Index, list[dict]]:
    """
    Carga desde disco un índice FAISS y su metadata asociada.

    Verifica que exista la misma cantidad de vectores en FAISS
    que registros en metadata.jsonl.
    """

    ruta = Path(directorio)

    ruta_indice = ruta / "index.faiss"
    ruta_metadata = ruta / "metadata.jsonl"

    if not ruta_indice.exists():
        raise FileNotFoundError(
            f"No existe el índice FAISS: {ruta_indice}"
        )

    if not ruta_metadata.exists():
        raise FileNotFoundError(
            f"No existe el archivo de metadata: {ruta_metadata}"
        )

    index = faiss.read_index(str(ruta_indice))

    metadata = []

    with ruta_metadata.open("r", encoding="utf-8") as archivo:
        for numero_linea, linea in enumerate(archivo, start=1):
            linea = linea.strip()

            if not linea:
                raise ValueError(
                    f"Línea vacía encontrada en metadata.jsonl: "
                    f"línea {numero_linea}"
                )

            metadata.append(json.loads(linea))

    if index.ntotal != len(metadata):
        raise ValueError(
            f"FAISS contiene {index.ntotal} vectores, "
            f"pero metadata.jsonl contiene {len(metadata)} registros."
        )

    return index, metadata