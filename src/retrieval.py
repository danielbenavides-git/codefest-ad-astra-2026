from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np

from src.indexing import cargar_base


_TOLERANCIA_NORMA = 1e-3


def buscar_vector(
    vector_consulta: np.ndarray,
    index: faiss.Index,
    metadata: Sequence[dict],
    k: int = 10,
) -> list[dict]:
    """
    Busca en FAISS los fragmentos más similares a un vector de consulta.
    """

    vector = np.array(vector_consulta, dtype=np.float32, copy=True)

    if vector.ndim == 1:
        vector = vector.reshape(1, -1)

    if vector.ndim != 2 or vector.shape[0] != 1:
        raise ValueError(
            "El vector de consulta debe representar una sola consulta."
        )

    if vector.shape[1] != index.d:
        raise ValueError(
            f"Dimensión de consulta {vector.shape[1]} distinta "
            f"a la dimensión del índice {index.d}."
        )

    if not np.isfinite(vector).all():
        raise ValueError(
            "El vector de consulta contiene valores NaN o infinitos."
        )

    if index.ntotal != len(metadata):
        raise ValueError(
            f"FAISS contiene {index.ntotal} vectores, "
            f"pero hay {len(metadata)} registros de metadata."
        )

    if k <= 0:
        raise ValueError("k debe ser mayor que cero.")

    norma = np.linalg.norm(vector)

    if abs(norma - 1.0) > _TOLERANCIA_NORMA:
        raise ValueError(
            f"El vector de consulta debe estar normalizado. "
            f"Norma recibida: {norma:.6f}."
        )

    k_real = min(k, index.ntotal)

    puntuaciones, indices = index.search(vector, k_real)

    resultados = []

    for rank, (indice, puntuacion) in enumerate(
        zip(indices[0], puntuaciones[0]),
        start=1,
    ):
        fragmento = dict(metadata[int(indice)])
        fragmento["rank"] = rank
        fragmento["puntuacion"] = float(puntuacion)
        resultados.append(fragmento)

    return resultados

def buscar_en_base(
    vector_consulta: np.ndarray,
    directorio: str | Path,
    k: int = 10,
) -> list[dict]:
    """
    Carga una base vectorial desde disco y busca los fragmentos
    más similares al vector de consulta.
    """

    index, metadata = cargar_base(directorio)

    return buscar_vector(
        vector_consulta=vector_consulta,
        index=index,
        metadata=metadata,
        k=k,
    )