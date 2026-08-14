import pytest

from src.chunking import Chunk
from src.knowledge_graph import (
    NOMBRE_ARCHIVO,
    _normalizar_entidad,
    construir_desde_chunks,
    construir_grafo,
    extraer_entidades,
    guardar_grafo,
)

nx = pytest.importorskip("networkx")


class NerFalso:
    """NER de mentira: busca nombres fijos en el texto, sin descargar el modelo.

    Devuelve una lista por texto, igual que el pipeline de transformers con
    `aggregation_strategy="simple"`.
    """

    CONOCIDAS = {
        "Colombia": "LOC",
        "Brasil": "LOC",
        "ONU": "ORG",
        "Agencia Espacial Europea": "ORG",
        "ruidito": "MISC",
    }

    def __init__(self, score: float = 0.99):
        self.score = score

    def __call__(self, textos):
        salida = []
        for texto in textos:
            encontradas = [
                {"entity_group": tipo, "word": nombre, "score": self.score}
                for nombre, tipo in self.CONOCIDAS.items()
                if nombre in texto
            ]
            salida.append(encontradas)
        return salida


def chunk(indice: int, texto: str, doc_id: str = "doc_1") -> Chunk:
    return Chunk(
        doc_id=doc_id,
        chunk_id=f"{doc_id}_c{indice}",
        fuente=f"{doc_id}.pdf",
        formato="pdf",
        fenomeno=3,
        posicion=indice,
        num_tokens=len(texto.split()),
        texto=texto,
        idioma="es",
    )


# Colombia y ONU coinciden en dos chunks: la arista supera MIN_COOCURRENCIAS.
# Brasil sale una sola vez: no llega a MIN_APARICIONES y queda fuera.
CHUNKS = [
    chunk(0, "Colombia presentó el informe ante la ONU."),
    chunk(1, "La ONU respondió a Colombia con nuevas medidas.", doc_id="doc_2"),
    chunk(2, "Brasil no participó en la sesión."),
]


def test_normaliza_puntuacion_y_espacios():
    assert _normalizar_entidad("  la  ONU,  ") == "la ONU"
    assert _normalizar_entidad("«Colombia»") == "Colombia"


def test_extrae_entidades_por_chunk():
    entidades = extraer_entidades(CHUNKS, ner=NerFalso())

    assert set(entidades) == {c.chunk_id for c in CHUNKS}
    assert ("Colombia", "LOC") in entidades["doc_1_c0"]
    assert ("ONU", "ORG") in entidades["doc_1_c0"]


def test_descarta_el_tipo_misc():
    entidades = extraer_entidades([chunk(0, "Un ruidito cualquiera.")], ner=NerFalso())

    assert entidades["doc_1_c0"] == []


def test_descarta_entidades_por_debajo_del_umbral():
    entidades = extraer_entidades(CHUNKS, ner=NerFalso(score=0.5), umbral=0.85)

    assert all(v == [] for v in entidades.values())


def test_procesa_en_lotes_sin_perder_chunks():
    muchos = [chunk(i, "Colombia y la ONU.") for i in range(35)]

    entidades = extraer_entidades(muchos, ner=NerFalso(), lote=16)

    assert len(entidades) == 35


def test_construye_nodos_y_aristas():
    grafo = construir_grafo(extraer_entidades(CHUNKS, ner=NerFalso()), CHUNKS)

    assert set(grafo.nodes) == {"Colombia", "ONU"}
    assert grafo.has_edge("Colombia", "ONU")


def test_descarta_entidades_de_una_sola_aparicion():
    grafo = construir_grafo(extraer_entidades(CHUNKS, ner=NerFalso()), CHUNKS)

    assert "Brasil" not in grafo.nodes


def test_los_nodos_guardan_tipo_y_documentos():
    grafo = construir_grafo(extraer_entidades(CHUNKS, ner=NerFalso()), CHUNKS)

    nodo = grafo.nodes["Colombia"]
    assert nodo["tipo"] == "LOC"
    assert nodo["apariciones"] == 2
    assert sorted(nodo["documentos"].split("|")) == ["doc_1", "doc_2"]


def test_las_aristas_son_trazables_a_sus_chunks():
    grafo = construir_grafo(extraer_entidades(CHUNKS, ner=NerFalso()), CHUNKS)

    arista = grafo.edges["Colombia", "ONU"]
    assert arista["peso"] == 2
    assert sorted(arista["chunk_ids"].split("|")) == ["doc_1_c0", "doc_2_c1"]


def test_una_sola_coocurrencia_no_crea_arista():
    solos = [
        chunk(0, "Colombia y la Agencia Espacial Europea firmaron."),
        chunk(1, "Colombia insistió."),
        chunk(2, "La Agencia Espacial Europea confirmó."),
    ]

    grafo = construir_grafo(extraer_entidades(solos, ner=NerFalso()), solos)

    assert grafo.number_of_edges() == 0


def test_guarda_en_la_subcarpeta_y_el_nombre_de_la_spec(tmp_path):
    grafo = construir_grafo(extraer_entidades(CHUNKS, ner=NerFalso()), CHUNKS)

    destino = guardar_grafo(grafo, tmp_path)

    assert destino == tmp_path / "grafo" / NOMBRE_ARCHIVO
    assert destino.exists()


def test_el_graphml_se_puede_volver_a_leer(tmp_path):
    grafo = construir_grafo(extraer_entidades(CHUNKS, ner=NerFalso()), CHUNKS)
    destino = guardar_grafo(grafo, tmp_path)

    leido = nx.read_graphml(destino)

    assert set(leido.nodes) == {"Colombia", "ONU"}
    assert leido.nodes["Colombia"]["tipo"] == "LOC"


def test_atajo_desde_chunks(tmp_path):
    destino = construir_desde_chunks(CHUNKS, tmp_path, ner=NerFalso())

    assert destino.exists()
    assert destino.name == NOMBRE_ARCHIVO


def test_grafo_vacio_no_revienta(tmp_path):
    sin_entidades = [chunk(0, "Texto sin nombres propios de ningún tipo.")]

    destino = construir_desde_chunks(sin_entidades, tmp_path, ner=NerFalso())

    assert nx.read_graphml(destino).number_of_nodes() == 0
