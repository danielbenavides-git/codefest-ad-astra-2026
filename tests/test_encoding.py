import numpy as np
import pytest

from src import config
from src.chunking import Chunk
from src.encoding import (
    cargar_embeddings,
    codificar_chunks,
    codificar_consulta,
    codificar_textos,
    guardar_embeddings,
)


class ModeloFalso:
    """Encoder de mentira: mismo texto -> mismo vector, sin descargar nada.

    Devuelve vectores SIN normalizar a propósito, para comprobar que la
    normalización la hace encoding.py y no la librería.
    """

    def __init__(self, dim: int = config.ENCODER_DIM):
        self.dim = dim
        self.llamadas: list[list[str]] = []

    def encode(self, textos, batch_size=8, show_progress_bar=False, convert_to_numpy=True):
        self.llamadas.append(list(textos))
        filas = []
        for texto in textos:
            semilla = abs(hash(texto)) % (2**32)
            generador = np.random.default_rng(semilla)
            filas.append(generador.normal(size=self.dim) * 7.5)
        return np.array(filas, dtype=np.float32)


def chunk(indice: int, texto: str, titulo=None) -> Chunk:
    return Chunk(
        doc_id="doc_1",
        chunk_id=f"doc_1_c{indice}",
        fuente="informe_orbital.pdf",
        formato="pdf",
        fenomeno=2,
        posicion=indice,
        num_tokens=len(texto.split()),
        texto=texto,
        idioma="es",
        titulo_doc=titulo,
    )


TEXTOS = [
    "La congestión en órbita baja terrestre preocupa a los operadores.",
    "Artificial intelligence is reshaping defense procurement.",
    "As dinâmicas territoriais na Amazônia mudaram na última década.",
]


def test_forma_dtype_y_norma():
    matriz = codificar_textos(TEXTOS, modelo=ModeloFalso())

    assert matriz.shape == (3, config.ENCODER_DIM)
    assert matriz.dtype == np.float32

    normas = np.linalg.norm(matriz, axis=1)
    assert np.allclose(normas, 1.0, atol=1e-3)


def test_el_orden_de_salida_es_el_de_entrada():
    modelo = ModeloFalso()

    completo = codificar_textos(TEXTOS, modelo=modelo)
    suelto = codificar_textos([TEXTOS[1]], modelo=modelo)

    assert np.allclose(completo[1], suelto[0])


def test_es_determinista():
    modelo = ModeloFalso()

    primera = codificar_textos(TEXTOS, modelo=modelo)
    segunda = codificar_textos(TEXTOS, modelo=modelo)

    assert np.array_equal(primera, segunda)


def test_dimension_distinta_a_la_config_falla():
    with pytest.raises(ValueError, match="ENCODER_DIM"):
        codificar_textos(TEXTOS, modelo=ModeloFalso(dim=config.ENCODER_DIM + 1))


def test_texto_vacio_falla():
    with pytest.raises(ValueError, match="posición 1"):
        codificar_textos(["algo", "   "], modelo=ModeloFalso())


def test_lista_vacia_falla():
    with pytest.raises(ValueError, match="ningún texto"):
        codificar_textos([], modelo=ModeloFalso())


def test_prefijo_se_antepone():
    modelo = ModeloFalso()

    codificar_textos(["hola"], prefijo="query: ", modelo=modelo)

    assert modelo.llamadas[-1] == ["query: hola"]


def test_codificar_chunks_devuelve_el_par_alineado():
    chunks = [chunk(i, t) for i, t in enumerate(TEXTOS)]

    embeddings, devueltos = codificar_chunks(chunks, modelo=ModeloFalso())

    assert embeddings.shape[0] == len(devueltos) == 3
    assert [c.chunk_id for c in devueltos] == [c.chunk_id for c in chunks]


def test_codificar_chunks_usa_texto_embed_con_titulo():
    modelo = ModeloFalso()
    con_titulo = chunk(0, "Cuerpo del fragmento.", titulo="Informe orbital 2026")

    codificar_chunks([con_titulo], modelo=modelo)

    assert modelo.llamadas[-1] == ["Informe orbital 2026 | Cuerpo del fragmento."]


def test_chunk_id_duplicado_falla():
    repetidos = [chunk(0, TEXTOS[0]), chunk(0, TEXTOS[1])]

    with pytest.raises(ValueError, match="duplicados"):
        codificar_chunks(repetidos, modelo=ModeloFalso())


def test_consulta_devuelve_vector_1d_normalizado():
    vector = codificar_consulta("¿Qué es la congestión orbital?", modelo=ModeloFalso())

    assert vector.shape == (config.ENCODER_DIM,)
    assert vector.dtype == np.float32
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-3


def test_guardar_y_cargar_conserva_los_vectores(tmp_path):
    chunks = [chunk(i, t) for i, t in enumerate(TEXTOS)]
    embeddings, _ = codificar_chunks(chunks, modelo=ModeloFalso())

    guardar_embeddings(embeddings, chunks, tmp_path)
    recuperados = cargar_embeddings(tmp_path, chunks)

    assert np.array_equal(embeddings, recuperados)


def test_cargar_con_otros_chunks_falla(tmp_path):
    chunks = [chunk(i, t) for i, t in enumerate(TEXTOS)]
    embeddings, _ = codificar_chunks(chunks, modelo=ModeloFalso())
    guardar_embeddings(embeddings, chunks, tmp_path)

    otros = chunks[:2]

    with pytest.raises(ValueError, match="no corresponden"):
        cargar_embeddings(tmp_path, otros)


def test_normalizar_false_falla_temprano(monkeypatch):
    monkeypatch.setattr(config, "NORMALIZAR", False)

    with pytest.raises(ValueError, match="NORMALIZAR"):
        codificar_textos(TEXTOS, modelo=ModeloFalso())