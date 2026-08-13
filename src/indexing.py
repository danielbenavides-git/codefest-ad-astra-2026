import json
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np

from src.chunking import Chunk


def validar_entradas(
    embeddings: np.ndarray,
    chunks: Sequence[Chunk],
) -> np.ndarray:
    """
    Valida y prepara los embeddings antes de construir el índice FAISS.

    Cada fila de embeddings debe corresponder exactamente al chunk
    ubicado en la misma posición de la lista `chunks`.

    Siempre devuelve una copia: usa `np.array(..., copy=True)`, nunca
    `np.asarray`, para no modificar el arreglo que pasó quien llama. Con
    `np.asarray`, si `embeddings` ya era float32 y contiguo (lo típico de
    `sentence-transformers`), no se copiaba y `construir_indice` terminaba
    normalizando en el sitio el arreglo original del caller sin avisar.

    Retorna:
        np.ndarray: matriz 2D, float32 y contigua en memoria, independiente
        del arreglo de entrada.
    """

    vectores = np.array(embeddings, dtype=np.float32, copy=True)

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

_TOLERANCIA_NORMA = 1e-3


def construir_indice(
    embeddings: np.ndarray,
    chunks: Sequence[Chunk],
) -> faiss.Index:
    """
    Construye un índice FAISS usando similitud coseno.

    Requiere que `embeddings` llegue ya normalizado a norma unitaria por
    fila. La normalización es responsabilidad de la Fase 4 (encoding.py,
    gobernada por `config.NORMALIZAR`), no de esta función (D5): si
    `retrieval.py` normaliza el vector de consulta con la misma regla,
    tiene que ser la regla de un solo sitio, y `encoding.py` es ese sitio
    porque `indexing.py` no interviene en la consulta. Por eso aquí solo
    se VERIFICA la norma y se falla con ValueError si no es ~1 — no se
    corrige en silencio, para no ocultar un bug real de la Fase 4.
    """

    vectores = validar_entradas(embeddings, chunks)

    normas = np.linalg.norm(vectores, axis=1)
    desviacion = np.abs(normas - 1.0)
    fuera_de_tolerancia = desviacion > _TOLERANCIA_NORMA
    if np.any(fuera_de_tolerancia):
        peor = int(np.argmax(desviacion))
        raise ValueError(
            "Los embeddings deben llegar normalizados a norma unitaria "
            "(L2 = 1) desde la Fase 4; indexing.py ya no normaliza (D5). "
            f"Revisa que `config.NORMALIZAR` esté en True y que encoding.py "
            f"lo respete. {int(fuera_de_tolerancia.sum())} de {len(normas)} "
            f"vector(es) fuera de tolerancia (±{_TOLERANCIA_NORMA}); peor "
            f"caso: fila {peor}, norma {normas[peor]:.6f}, "
            f"chunk_id {chunks[peor].chunk_id!r}."
        )

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