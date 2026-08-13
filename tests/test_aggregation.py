import pytest

from src.aggregation import (
    agregar_documentos,
    agrupar_por_documento,
    puntuar_documento,
)


def frag(chunk_id: str, doc_id: str, puntuacion: float, rank: int) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "texto": f"texto de {chunk_id}",
        "puntuacion": puntuacion,
        "rank": rank,
    }


# doc_A: un fragmento muy bueno. doc_B: tres fragmentos medianos.
# doc_C: uno solo, mediano. Cada estrategia elige un ganador distinto.
FRAGMENTOS = [
    frag("A1", "doc_A", 0.90, 1),
    frag("B1", "doc_B", 0.70, 2),
    frag("B2", "doc_B", 0.65, 3),
    frag("B3", "doc_B", 0.60, 4),
    frag("C1", "doc_C", 0.68, 5),
]


def test_agrupa_conservando_el_orden():
    grupos = agrupar_por_documento(FRAGMENTOS)

    assert list(grupos) == ["doc_A", "doc_B", "doc_C"]
    assert [f["chunk_id"] for f in grupos["doc_B"]] == ["B1", "B2", "B3"]


def test_fragmento_sin_doc_id_falla():
    with pytest.raises(ValueError, match="no tiene doc_id"):
        agrupar_por_documento([{"chunk_id": "X", "puntuacion": 0.5}])


def test_fragmento_sin_puntuacion_falla():
    with pytest.raises(ValueError, match="no tiene puntuacion"):
        agrupar_por_documento([{"chunk_id": "X", "doc_id": "doc_A"}])


def test_las_tres_estrategias_dan_valores_distintos():
    del_b = [f for f in FRAGMENTOS if f["doc_id"] == "doc_B"]

    assert puntuar_documento(del_b, "max") == pytest.approx(0.70)
    assert puntuar_documento(del_b, "suma") == pytest.approx(1.95)
    assert puntuar_documento(del_b, "media") == pytest.approx(0.65)


def test_estrategia_desconocida_falla():
    with pytest.raises(ValueError, match="desconocida"):
        puntuar_documento(FRAGMENTOS, "mediana")


def test_max_gana_el_del_mejor_fragmento():
    docs = agregar_documentos(FRAGMENTOS, estrategia="max")

    assert [d["doc_id"] for d in docs] == ["doc_A", "doc_B", "doc_C"]


def test_suma_gana_el_que_aporta_mas_fragmentos():
    docs = agregar_documentos(FRAGMENTOS, estrategia="suma")

    assert [d["doc_id"] for d in docs] == ["doc_B", "doc_A", "doc_C"]


def test_media_castiga_al_documento_con_fragmentos_flojos():
    docs = agregar_documentos(FRAGMENTOS, estrategia="media")

    assert [d["doc_id"] for d in docs] == ["doc_A", "doc_C", "doc_B"]


def test_asigna_rank_consecutivo_desde_uno():
    docs = agregar_documentos(FRAGMENTOS)

    assert [d["rank"] for d in docs] == [1, 2, 3]


def test_devuelve_solo_tres_aunque_haya_mas_documentos():
    extra = FRAGMENTOS + [frag("D1", "doc_D", 0.99, 0)]

    docs = agregar_documentos(extra)

    assert len(docs) == 3
    assert docs[0]["doc_id"] == "doc_D"


def test_devuelve_menos_de_tres_si_no_hay_mas_documentos():
    docs = agregar_documentos([frag("A1", "doc_A", 0.9, 1)])

    assert len(docs) == 1


def test_empate_lo_resuelve_el_mejor_rank():
    empatados = [
        frag("Z1", "doc_Z", 0.5, 4),
        frag("Y1", "doc_Y", 0.5, 2),
    ]

    docs = agregar_documentos(empatados)

    assert [d["doc_id"] for d in docs] == ["doc_Y", "doc_Z"]


def test_empate_total_lo_resuelve_el_doc_id():
    empatados = [
        frag("Z1", "doc_Z", 0.5, 1),
        frag("Y1", "doc_Y", 0.5, 1),
    ]

    docs = agregar_documentos(empatados)

    assert [d["doc_id"] for d in docs] == ["doc_Y", "doc_Z"]


def test_reporta_cuantos_fragmentos_aporto_cada_documento():
    docs = agregar_documentos(FRAGMENTOS, estrategia="suma")

    por_id = {d["doc_id"]: d for d in docs}
    assert por_id["doc_B"]["n_fragmentos"] == 3
    assert por_id["doc_B"]["mejor_rank"] == 2


def test_lista_vacia_falla():
    with pytest.raises(ValueError, match="ningún fragmento"):
        agregar_documentos([])