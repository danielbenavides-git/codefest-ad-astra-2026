import numpy as np

from src.chunking import Chunk
from src.indexing import construir_indice, guardar_base
from src.retrieval import buscar_en_base, buscar_vector

def crear_datos_prueba():
    chunks = [
        Chunk(
            doc_id="DOC-001",
            chunk_id="DOC-001-chunk-0000",
            fuente="doc1.pdf",
            formato="pdf",
            fenomeno=1,
            posicion=0,
            num_tokens=5,
            texto="Fragmento uno.",
        ),
        Chunk(
            doc_id="DOC-002",
            chunk_id="DOC-002-chunk-0000",
            fuente="doc2.pdf",
            formato="pdf",
            fenomeno=2,
            posicion=0,
            num_tokens=5,
            texto="Fragmento dos.",
        ),
        Chunk(
            doc_id="DOC-003",
            chunk_id="DOC-003-chunk-0000",
            fuente="doc3.pdf",
            formato="pdf",
            fenomeno=3,
            posicion=0,
            num_tokens=5,
            texto="Fragmento tres.",
        ),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    metadata = [chunk.to_dict() for chunk in chunks]

    return chunks, embeddings, metadata


def test_busqueda_devuelve_fragmento_correcto():
    chunks, embeddings, metadata = crear_datos_prueba()

    index = construir_indice(embeddings, chunks)

    consulta = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    resultados = buscar_vector(
        consulta,
        index,
        metadata,
        k=2,
    )

    assert len(resultados) == 2
    assert resultados[0]["chunk_id"] == "DOC-001-chunk-0000"
    assert resultados[0]["rank"] == 1
    assert resultados[0]["puntuacion"] == 1.0

import pytest


def test_error_si_dimension_no_coincide():
    chunks, embeddings, metadata = crear_datos_prueba()
    index = construir_indice(embeddings, chunks)

    consulta = np.array([1.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError):
        buscar_vector(
            consulta,
            index,
            metadata,
            k=2,
        )


def test_error_si_consulta_no_esta_normalizada():
    chunks, embeddings, metadata = crear_datos_prueba()
    index = construir_indice(embeddings, chunks)

    consulta = np.array([2.0, 0.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError):
        buscar_vector(
            consulta,
            index,
            metadata,
            k=2,
        )


def test_k_mayor_que_indice_devuelve_solo_disponibles():
    chunks, embeddings, metadata = crear_datos_prueba()
    index = construir_indice(embeddings, chunks)

    consulta = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    resultados = buscar_vector(
        consulta,
        index,
        metadata,
        k=10,
    )

    assert len(resultados) == 3


def test_error_si_metadata_no_coincide_con_faiss():
    chunks, embeddings, metadata = crear_datos_prueba()
    index = construir_indice(embeddings, chunks)

    metadata_incompleta = metadata[:2]

    consulta = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError):
        buscar_vector(
            consulta,
            index,
            metadata_incompleta,
            k=2,
        )

def test_busqueda_desde_base_guardada(tmp_path):
    chunks, embeddings, _ = crear_datos_prueba()

    index = construir_indice(embeddings, chunks)

    guardar_base(
        index=index,
        chunks=chunks,
        directorio=tmp_path,
    )

    consulta = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    resultados = buscar_en_base(
        consulta,
        tmp_path,
        k=2,
    )

    assert len(resultados) == 2
    assert resultados[0]["chunk_id"] == "DOC-001-chunk-0000"
    assert resultados[0]["rank"] == 1