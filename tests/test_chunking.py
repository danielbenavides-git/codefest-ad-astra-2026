"""Tests de fragmentación (Fase 3).

Cada test corresponde a una invariante que, si se rompe, cuesta puntos o
descalifica: fragmentos por encima de 250 palabras, oraciones cortadas,
chunks huérfanos que ocupan un vector inútil, o `posicion` desalineada
respecto al orden del índice FAISS.
"""

import pytest

from src.chunking import (
    IDIOMAS_PYSBD,
    LIMITE_DURO,
    PRESUPUESTO_SALIDA,
    Chunk,
    DocumentoSinTexto,
    agrupar,
    contar_palabras,
    detectar_idioma,
    documento_a_chunks,
    expandir,
    expandir_desde,
    segmentar,
    trocear_oracion_larga,
)

FRASE = ("La densidad de objetos en la órbita terrestre baja aumentó de forma "
         "sostenida durante la última década analizada. ")


def texto_largo(n: int = 60) -> str:
    return " ".join(f"El apartado {i} del informe analiza la densidad orbital "
                    f"y las dinámicas territoriales de la región estudiada." for i in range(n))


# ------------------------------------------------------------ segmentación

@pytest.mark.parametrize(
    "idioma,texto,esperado",
    [
        ("es", "El Dr. Gómez publicó en 2024. Analizó 3.5 millones de objetos. ¿Y ahora?", 3),
        ("en", "Dr. Smith et al. reported 3.5 million objects in 2024. It is clear. What now?", 3),
        ("pt", "O Dr. Silva publicou em 2024. Analisou 3,5 milhões de objetos. E agora?", 3),
    ],
)
def test_segmenta_sin_partir_abreviaturas_ni_decimales(idioma, texto, esperado):
    assert len(segmentar(texto, idioma)) == esperado


def test_segmentar_no_pierde_caracteres():
    texto = "Primera oración. Segunda oración con datos: 3,5 y 1.200. Tercera."
    unido = "".join("".join(segmentar(texto, "es")).split())
    assert unido == "".join(texto.split())


def test_segmentar_texto_vacio():
    assert segmentar("", "es") == []
    assert segmentar("   \n  ", "es") == []


def test_portugues_se_mapea_a_espanol():
    """pysbd no soporta 'pt'. El mapeo es explícito para que no falle callando."""
    assert IDIOMAS_PYSBD["pt"] == "es"


def test_idioma_no_previsto_lanza_error_explicito():
    with pytest.raises(ValueError, match="no previsto"):
        segmentar("Bonjour le monde.", "fr")


@pytest.mark.parametrize(
    "idioma,texto",
    [
        ("es", "La densidad de objetos en la órbita terrestre baja aumentó según los informes."),
        ("pt", "A densidade de objetos na órbita terrestre baixa aumentou segundo os relatórios."),
        ("en", "The density of objects in low Earth orbit increased according to the reports."),
    ],
)
def test_detecta_idioma_a_nivel_documento(idioma, texto):
    assert detectar_idioma(texto) == idioma


def test_detectar_idioma_texto_vacio_no_lanza():
    assert detectar_idioma("") in IDIOMAS_PYSBD


# --------------------------------------------------------------- troceado

def test_trocea_oracion_que_sola_supera_el_presupuesto():
    fila = "; ".join(f"Pais: X{i}, Anio: 2024, Indice: 0.{i}" for i in range(120))
    piezas = trocear_oracion_larga(fila)
    assert len(piezas) > 1
    assert all(contar_palabras(p) <= PRESUPUESTO_SALIDA for p in piezas)


def test_trocear_conserva_los_separadores():
    """`texto` va a metadata sin modificaciones: no se pueden perder los ';'."""
    fila = "; ".join(f"campo{i}: valor {i}" for i in range(200))
    piezas = trocear_oracion_larga(fila)
    assert "".join("".join(piezas).split()) == "".join(fila.split())


def test_oracion_corta_no_se_trocea():
    assert trocear_oracion_larga("Una frase corta.") == ["Una frase corta."]


# --------------------------------------------------------------- agrupado

def test_chunks_respetan_el_objetivo():
    chunks = agrupar(segmentar(texto_largo(), "es"), objetivo=140)
    assert len(chunks) > 3
    assert all(contar_palabras(c) <= 200 for c in chunks)


def test_agrupar_no_pierde_ni_duplica_texto():
    texto = texto_largo()
    chunks = agrupar(segmentar(texto, "es"))
    assert "".join("".join(chunks).split()) == "".join(texto.split())


def test_ningun_chunk_termina_a_mitad_de_oracion():
    for chunk in agrupar(segmentar(texto_largo(), "es")):
        assert chunk.strip().endswith((".", "?", "!"))


def test_sin_chunks_huerfanos():
    """Un chunk de una palabra ocupa un vector del índice y no empareja nada.

    El caso real: los extractores tabulares ponen el nombre del archivo como
    primera línea ('ancho.'), que quedaba como chunk propio porque `agrupar`
    solo absorbía restos por la cola.
    """
    piezas = agrupar(["ancho.", *[FRASE.strip()] * 8])
    assert all(contar_palabras(p) >= 25 for p in piezas), [contar_palabras(p) for p in piezas]


def test_documento_de_una_frase_da_un_chunk():
    assert agrupar(["Corto."]) == ["Corto."]


def test_agrupar_lista_vacia():
    assert agrupar([]) == []


# --------------------------------------------------------------- expansión

def test_expandir_crece_hasta_el_presupuesto_sin_pasarse():
    chunks = agrupar(segmentar(texto_largo(120), "es"))
    ampliado = expandir(chunks[2], chunks[1], chunks[3], idioma="es")
    assert contar_palabras(chunks[2]) < contar_palabras(ampliado) <= PRESUPUESTO_SALIDA


def test_expandir_conserva_el_fragmento_central():
    chunks = agrupar(segmentar(texto_largo(120), "es"))
    assert chunks[2] in expandir(chunks[2], chunks[1], chunks[3], idioma="es")


def test_expandir_sin_vecinos_devuelve_el_original():
    assert expandir("Una frase.", "", "", idioma="es") == "Una frase."


def test_expandir_aprovecha_la_mayor_parte_del_presupuesto():
    """Si expande poco, se entrega menos evidencia de la permitida."""
    chunks = agrupar(segmentar(texto_largo(120), "es"))
    ampliados = [expandir(c, chunks[max(i - 1, 0)], chunks[min(i + 1, len(chunks) - 1)], idioma="es")
                 for i, c in enumerate(chunks)]
    media = sum(contar_palabras(t) for t in ampliados) / len(ampliados)
    assert media > 0.85 * PRESUPUESTO_SALIDA, media


def test_expandir_desde_solo_usa_vecinos_inmediatos():
    """§9.2.1 permite el fragmento anterior o posterior, no cualquiera."""
    chunks = documento_a_chunks(texto_largo(150), doc_id="D", fuente="d.pdf",
                                formato="pdf", fenomeno=1)
    ampliado = expandir_desde(chunks, 5)
    assert chunks[5].texto in ampliado
    assert chunks[8].texto not in ampliado


# ------------------------------------------------------ documento_a_chunks

def test_posicion_es_contigua_desde_cero():
    """Debe coincidir con el orden en que los vectores entran a FAISS."""
    chunks = documento_a_chunks(texto_largo(), doc_id="D", fuente="d.pdf",
                                formato="pdf", fenomeno=2)
    assert [c.posicion for c in chunks] == list(range(len(chunks)))


def test_chunk_id_es_cadena_y_unico():
    chunks = documento_a_chunks(texto_largo(), doc_id="D", fuente="d.pdf",
                                formato="pdf", fenomeno=2)
    assert all(isinstance(c.chunk_id, str) for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_esquema_de_metadata_tabla_1():
    chunks = documento_a_chunks(texto_largo(), doc_id="D", fuente="informe 2024.pdf",
                                formato="pdf", fenomeno=3, titulo_doc="Informe")
    d = chunks[0].to_dict()
    assert list(d)[:8] == ["doc_id", "chunk_id", "fuente", "formato",
                           "fenomeno", "posicion", "num_tokens", "texto"]
    assert "text" not in d, "en metadata el campo es `texto`, no `text`"
    assert d["fuente"] == "informe 2024.pdf", "`fuente` no se puede alterar"
    assert isinstance(d["fenomeno"], int) and isinstance(d["posicion"], int)


def test_texto_embed_antepone_el_titulo_sin_tocar_texto():
    chunks = documento_a_chunks(texto_largo(), doc_id="D", fuente="d.pdf",
                                formato="pdf", fenomeno=1, titulo_doc="Informe UNOOSA")
    assert chunks[0].texto_embed.startswith("Informe UNOOSA | ")
    assert chunks[0].to_dict()["texto"] == chunks[0].texto


def test_documento_vacio_lanza_documento_sin_texto():
    with pytest.raises(DocumentoSinTexto, match="d.pdf"):
        documento_a_chunks("   ", doc_id="D", fuente="d.pdf", formato="pdf", fenomeno=1)


def test_limite_duro_incluso_con_texto_anomalo():
    """Filas de CSV serializadas: una 'oración' de cientos de palabras."""
    texto = "Columna: " + " ".join(f"valor{i}" for i in range(600))
    chunks = documento_a_chunks(texto, doc_id="D", fuente="t.csv",
                                formato="csv", fenomeno=1)
    assert all(contar_palabras(c.texto) <= LIMITE_DURO for c in chunks)
    assert all(contar_palabras(expandir_desde(chunks, i)) <= LIMITE_DURO
               for i in range(len(chunks)))


def test_contar_tokens_inyectable():
    chunks = documento_a_chunks(texto_largo(), doc_id="D", fuente="d.pdf", formato="pdf",
                                fenomeno=1, contar_tokens=lambda t: 7)
    assert all(c.num_tokens == 7 for c in chunks)
