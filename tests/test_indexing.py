import numpy as np

from src.chunking import Chunk
from src.indexing import cargar_base, construir_indice, guardar_base, validar_entradas

chunks = [
    Chunk(
        doc_id="DOC-001",
        chunk_id="DOC-001-chunk-0000",
        fuente="documento1.pdf",
        formato="pdf",
        fenomeno=1,
        posicion=0,
        num_tokens=10,
        texto="Primer fragmento de prueba.",
    ),
    Chunk(
        doc_id="DOC-001",
        chunk_id="DOC-001-chunk-0001",
        fuente="documento1.pdf",
        formato="pdf",
        fenomeno=1,
        posicion=1,
        num_tokens=10,
        texto="Segundo fragmento de prueba.",
    ),
    Chunk(
        doc_id="DOC-002",
        chunk_id="DOC-002-chunk-0000",
        fuente="documento2.pdf",
        formato="pdf",
        fenomeno=2,
        posicion=0,
        num_tokens=10,
        texto="Tercer fragmento de prueba.",
    ),
]


embeddings = np.array(
    [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]
)

vectores = validar_entradas(embeddings, chunks)

assert vectores.shape == (3, 3)
assert vectores.dtype == np.float32

index = construir_indice(embeddings, chunks)

assert index.ntotal == 3
assert index.d == 3

ruta_prueba = "entrega/base_vectorial/encoder_prueba"

guardar_base(
    index=index,
    chunks=chunks,
    directorio=ruta_prueba,
)

index_cargado, metadata_cargada = cargar_base(ruta_prueba)

assert index_cargado.ntotal == 3
assert len(metadata_cargada) == 3
assert metadata_cargada[0]["chunk_id"] == "DOC-001-chunk-0000"
