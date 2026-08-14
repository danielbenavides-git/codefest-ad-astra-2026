"""Tests de limpieza (Fase 2).

Dos versiones anteriores de este filtro borraron párrafos de contenido en
PDFs reales. El riesgo asimétrico manda: dejar una cabecera dentro de un
chunk cuesta ruido; borrar un párrafo cuesta evidencia irrecuperable. Por eso
la mayoría de estos tests comprueban que el filtro NO actúa.
"""

import fitz
import pytest

from src.cleaning import (
    MAX_FRACCION_BORRADA,
    MIN_PAGINAS,
    ZONA_CABECERA,
    ZONA_PIE,
    bloques_por_pagina,
    quitar_numeros_sueltos,
    quitar_repetidos,
)

CUERPO = ("La densidad de objetos en la órbita terrestre baja aumentó de forma "
          "sostenida durante la última década según los informes de seguimiento.")


def pdf(tmp_path, n_paginas, *, cabecera=None, pie=None, cuerpo_variable=True):
    doc = fitz.open()
    for i in range(n_paginas):
        pagina = doc.new_page()
        if cabecera:
            pagina.insert_text((50, 40), cabecera, fontsize=9)
        texto = f"{CUERPO} Apartado {i}." if cuerpo_variable else CUERPO
        pagina.insert_textbox(fitz.Rect(50, 150, 545, 650), texto * 3, fontsize=10)
        if pie:
            pagina.insert_text((50, 780), f"{pie} {i + 1}", fontsize=8)
    ruta = tmp_path / "d.pdf"
    doc.save(ruta)
    doc.close()
    return ruta


def paginas_de(ruta):
    doc = fitz.open(ruta)
    try:
        return bloques_por_pagina(doc)
    finally:
        doc.close()


# ------------------------------------------------- el filtro sí debe actuar

def test_elimina_cabecera_y_pie_repetidos(tmp_path):
    ruta = pdf(tmp_path, 10, cabecera="Observatorio de Seguridad Espacial",
               pie="Todos los derechos reservados. Página")
    limpias, informe = quitar_repetidos(paginas_de(ruta))

    assert len(informe.lineas_eliminadas) == 2, informe
    assert not any("Observatorio" in p for p in limpias)
    assert not any("derechos reservados" in p for p in limpias)


def test_no_toca_el_cuerpo_al_eliminar_bordes(tmp_path):
    ruta = pdf(tmp_path, 10, cabecera="Cabecera fija", pie="Pie fijo")
    limpias, _ = quitar_repetidos(paginas_de(ruta))
    assert all("densidad de objetos" in p for p in limpias)


def test_los_digitos_no_impiden_reconocer_la_misma_cabecera(tmp_path):
    """'Página 3 de 12' y 'Página 4 de 12' son la misma línea estructural."""
    ruta = pdf(tmp_path, 8, pie="Informe 2024 — Página")
    _, informe = quitar_repetidos(paginas_de(ruta))
    assert informe.lineas_eliminadas


# ----------------------------------------------- el filtro NO debe actuar

def test_no_elimina_nada_si_no_hay_repeticiones(tmp_path):
    ruta = pdf(tmp_path, 8)
    limpias, informe = quitar_repetidos(paginas_de(ruta))
    assert informe.lineas_eliminadas == []
    assert all("densidad de objetos" in p for p in limpias)


def test_se_abstiene_con_pocas_paginas(tmp_path):
    """Con 2 o 3 páginas la estadística no distingue cabecera de coincidencia."""
    ruta = pdf(tmp_path, MIN_PAGINAS - 1, cabecera="Cabecera")
    limpias, informe = quitar_repetidos(paginas_de(ruta))
    assert informe.nota is not None
    assert any("Cabecera" in p for p in limpias)


def test_se_abstiene_si_borraria_demasiado(tmp_path):
    """Un filtro que se lleva un cuarto del documento está equivocado."""
    doc = fitz.open()
    for _ in range(8):
        pagina = doc.new_page()
        pagina.insert_text((50, 40), "Linea repetida arriba", fontsize=9)
        pagina.insert_text((50, 780), "Linea repetida abajo", fontsize=9)
    ruta = tmp_path / "patologico.pdf"
    doc.save(ruta)
    doc.close()

    limpias, informe = quitar_repetidos(paginas_de(ruta))
    assert informe.nota is not None and "tope" in informe.nota
    assert informe.lineas_eliminadas == []
    assert "".join(limpias).strip(), "no puede dejar el documento vacío"


def test_no_elimina_una_linea_larga_aunque_se_repita(tmp_path):
    """Una frase larga repetida es contenido (un estribillo legal, una nota)."""
    larga = ("Este documento forma parte del programa institucional de seguimiento "
             "orbital y su reproducción total o parcial se rige por la licencia vigente")
    doc = fitz.open()
    for i in range(10):
        pagina = doc.new_page()
        pagina.insert_textbox(fitz.Rect(50, 20, 545, 90), larga, fontsize=8)
        pagina.insert_textbox(fitz.Rect(50, 150, 545, 650), f"{CUERPO} Apartado {i}.", fontsize=10)
    ruta = tmp_path / "larga.pdf"
    doc.save(ruta)
    doc.close()

    limpias, _ = quitar_repetidos(paginas_de(ruta))
    assert any("programa institucional" in p for p in limpias)


def test_la_fraccion_borrada_se_mantiene_baja(tmp_path):
    ruta = pdf(tmp_path, 12, cabecera="Observatorio", pie="Página")
    _, informe = quitar_repetidos(paginas_de(ruta))
    assert 0 < informe.fraccion_borrada <= MAX_FRACCION_BORRADA


# ---------------------------------------------------------- números sueltos

@pytest.mark.parametrize("linea", ["7", "  12  ", "Página 3", "página 3 de 45", "Page 7 of 9", "3 / 45"])
def test_quita_lineas_que_son_solo_numero_de_pagina(linea):
    assert quitar_numeros_sueltos(f"Texto real.\n{linea}\nMás texto.") == "Texto real.\nMás texto."


@pytest.mark.parametrize("linea", ["Total: 7 objetos", "2024 fue el peor año", "7 satélites activos"])
def test_no_quita_numeros_con_contexto(linea):
    assert linea in quitar_numeros_sueltos(f"Antes.\n{linea}\nDespués.")


# ------------------------------------------------------------------ informe

def test_el_informe_es_legible(tmp_path):
    ruta = pdf(tmp_path, 10, cabecera="Observatorio de Seguridad Espacial")
    _, informe = quitar_repetidos(paginas_de(ruta))
    texto = str(informe)
    assert "palabras" in texto and "observatorio" in texto.lower()


def test_zonas_no_se_solapan():
    assert ZONA_CABECERA < ZONA_PIE
