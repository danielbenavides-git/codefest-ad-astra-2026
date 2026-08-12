import numpy as np
import pytest

from src.chunking import Chunk
from src.indexing import cargar_base, construir_indice, guardar_base, validar_entradas


def _chunk(doc_id, chunk_id, fenomeno, posicion, texto):
    return Chunk(
        doc_id=doc_id,
        chunk_id=chunk_id,
        fuente=f"{doc_id.lower()}.pdf",
        formato="pdf",
        fenomeno=fenomeno,
        posicion=posicion,
        num_tokens=10,
        texto=texto,
    )


@pytest.fixture
def chunks():
    return [
        _chunk("DOC-001", "DOC-001-chunk-0000", 1, 0, "Primer fragmento de prueba."),
        _chunk("DOC-001", "DOC-001-chunk-0001", 1, 1, "Segundo fragmento de prueba."),
        _chunk("DOC-002", "DOC-002-chunk-0000", 2, 0, "Tercer fragmento de prueba."),
    ]


@pytest.fixture
def embeddings_normalizados():
    """3 vectores de dimensión 3, cada uno con norma unitaria.

    construir_indice ya no normaliza (Tarea 3, D5): los fixtures del
    camino feliz tienen que llegar ya normalizados a norma 1.
    """
    vectores = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        dtype=np.float32,
    )
    normas = np.linalg.norm(vectores, axis=1, keepdims=True)
    return (vectores / normas).astype(np.float32)


@pytest.fixture
def embeddings_sin_normalizar():
    return np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=np.float32,
    )


def test_validar_entradas_forma_y_dtype(chunks, embeddings_normalizados):
    vectores = validar_entradas(embeddings_normalizados, chunks)
    assert vectores.shape == (3, 3)
    assert vectores.dtype == np.float32


def test_construir_indice(chunks, embeddings_normalizados):
    index = construir_indice(embeddings_normalizados, chunks)
    assert index.ntotal == 3
    assert index.d == 3


def test_guardar_y_cargar_base(tmp_path, chunks, embeddings_normalizados):
    index = construir_indice(embeddings_normalizados, chunks)
    directorio = tmp_path / "base_vectorial" / "encoder_prueba"

    guardar_base(index=index, chunks=chunks, directorio=directorio)

    index_cargado, metadata_cargada = cargar_base(directorio)

    assert index_cargado.ntotal == 3
    assert len(metadata_cargada) == 3
    assert metadata_cargada[0]["chunk_id"] == "DOC-001-chunk-0000"


# --- Tarea 3: la normalización L2 es responsabilidad de la Fase 4 (D5) ---


def test_validar_entradas_no_muta_el_arreglo_original(chunks, embeddings_normalizados):
    copia = embeddings_normalizados.copy()
    validar_entradas(embeddings_normalizados, chunks)
    assert np.array_equal(embeddings_normalizados, copia)


def test_construir_indice_acepta_vectores_normalizados(chunks, embeddings_normalizados):
    index = construir_indice(embeddings_normalizados, chunks)
    assert index.ntotal == len(chunks)


def test_construir_indice_rechaza_vectores_sin_normalizar(chunks, embeddings_sin_normalizar):
    with pytest.raises(ValueError, match="normalizados"):
        construir_indice(embeddings_sin_normalizar, chunks)


def test_construir_indice_no_muta_el_arreglo_aunque_falle(chunks, embeddings_sin_normalizar):
    copia = embeddings_sin_normalizar.copy()
    with pytest.raises(ValueError):
        construir_indice(embeddings_sin_normalizar, chunks)
    assert np.array_equal(embeddings_sin_normalizar, copia)
