import json

import pytest

from src.output import (
    armar_documentos,
    armar_fragmentos,
    armar_linea,
    escribir_resultados,
)


def documentos(n: int = 3) -> list[dict]:
    return [
        {"doc_id": f"doc_{i}", "puntuacion": 1.0 - i / 10, "rank": i + 1}
        for i in range(n)
    ]


def fragmentos(n: int = 10, texto: str = "contenido del fragmento") -> list[dict]:
    return [
        {
            "chunk_id": f"doc_0_c{i}",
            "doc_id": "doc_0",
            "fuente": "informe.pdf",
            "texto": texto,
            "puntuacion": 1.0 - i / 100,
            "rank": i + 1,
        }
        for i in range(n)
    ]


def test_traduce_texto_a_text_y_deja_los_otros_nombres():
    salida = armar_fragmentos(fragmentos())

    assert salida[0]["text"] == "contenido del fragmento"
    assert "texto" not in salida[0]
    assert set(salida[0]) == {"rank", "chunk_id", "doc_id", "text"}


def test_no_arrastra_campos_de_metadata_a_la_entrega():
    salida = armar_fragmentos(fragmentos())

    assert "fuente" not in salida[0]
    assert "puntuacion" not in salida[0]


def test_reasigna_el_rank_por_posicion():
    desordenados = fragmentos()
    for f in desordenados:
        f["rank"] = 99

    salida = armar_fragmentos(desordenados)

    assert [f["rank"] for f in salida] == list(range(1, 11))


def test_documentos_solo_llevan_rank_y_doc_id():
    salida = armar_documentos(documentos())

    assert salida == [
        {"rank": 1, "doc_id": "doc_0"},
        {"rank": 2, "doc_id": "doc_1"},
        {"rank": 3, "doc_id": "doc_2"},
    ]


def test_menos_de_tres_documentos_falla():
    with pytest.raises(ValueError, match="sube la k"):
        armar_documentos(documentos(2))


def test_distinto_de_diez_fragmentos_falla():
    with pytest.raises(ValueError, match="exactamente 10"):
        armar_fragmentos(fragmentos(9))


def test_fragmento_de_mas_de_250_palabras_falla():
    largos = fragmentos(texto="palabra " * 300)

    with pytest.raises(ValueError, match="251|300 palabras"):
        armar_fragmentos(largos)


def test_fragmento_sin_texto_falla():
    sin_texto = fragmentos()
    sin_texto[3]["texto"] = ""

    with pytest.raises(ValueError, match="fragmento 4"):
        armar_fragmentos(sin_texto)


def test_query_id_fuera_de_rango_falla():
    with pytest.raises(ValueError, match="q001"):
        armar_linea("q051", documentos(), fragmentos())


def test_query_id_mal_formado_falla():
    with pytest.raises(ValueError, match="inválido"):
        armar_linea("1", documentos(), fragmentos())


def test_linea_completa_tiene_el_esquema_de_la_tabla_2():
    linea = armar_linea("q007", documentos(), fragmentos())

    assert set(linea) == {"query_id", "documents", "fragments"}
    assert linea["query_id"] == "q007"
    assert len(linea["documents"]) == 3
    assert len(linea["fragments"]) == 10


def cincuenta_lineas() -> list[dict]:
    return [
        armar_linea(f"q{i:03d}", documentos(), fragmentos())
        for i in range(1, 51)
    ]


def test_escribe_cincuenta_lineas_validas(tmp_path):
    destino = escribir_resultados(cincuenta_lineas(), tmp_path / "resultados.jsonl")

    contenido = destino.read_text(encoding="utf-8").splitlines()
    assert len(contenido) == 50
    assert json.loads(contenido[0])["query_id"] == "q001"
    assert json.loads(contenido[-1])["query_id"] == "q050"


def test_conserva_los_acentos_sin_escapar(tmp_path):
    lineas = cincuenta_lineas()
    lineas[0]["fragments"][0]["text"] = "congestión en órbita baja"

    destino = escribir_resultados(lineas, tmp_path / "resultados.jsonl")

    assert "congestión en órbita baja" in destino.read_text(encoding="utf-8")


def test_menos_de_cincuenta_consultas_falla(tmp_path):
    with pytest.raises(ValueError, match="exactamente 50"):
        escribir_resultados(cincuenta_lineas()[:49], tmp_path / "resultados.jsonl")


def test_consultas_desordenadas_fallan(tmp_path):
    lineas = cincuenta_lineas()
    lineas[3], lineas[9] = lineas[9], lineas[3]

    with pytest.raises(ValueError, match="posición 4"):
        escribir_resultados(lineas, tmp_path / "resultados.jsonl")