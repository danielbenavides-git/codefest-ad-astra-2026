from pathlib import Path

import fitz
import pytest

from src.extraction import extraer_pdf

FIXTURE_CABECERA_REPETIDA = Path(__file__).parent / "fixtures" / "pdf_cabecera_repetida.pdf"


def test_extraer_pdf_quita_cabecera_y_pie_repetidos():
    """Fixture de 6 páginas con la misma cabecera y un pie de página numerado.

    Cuerpos largos (~80-90 palabras/página) para que cabecera+pie queden
    bajo el tope de seguridad del 25% (ver el otro test de este archivo
    para el caso en que SÍ lo superan).
    """
    texto, informe = extraer_pdf(FIXTURE_CABECERA_REPETIDA, devolver_informe=True)
    plano = " ".join(texto.split())  # normaliza saltos de línea de párrafos envueltos

    assert "INFORME ANUAL DE PRUEBA 2024" not in texto
    assert not any(f"Pagina {i} de 6" in texto for i in range(1, 7))
    assert "inteligencia artificial en el sector defensa" in plano
    assert "cambio climatico y desplazamiento forzado" in plano
    assert informe.fraccion_borrada <= 0.25
    assert informe.lineas_eliminadas


def test_extraer_pdf_sin_devolver_informe_regresa_solo_texto():
    """El contrato por defecto (`Callable[[Path], str]` de EXTRACTORES) no cambia."""
    resultado = extraer_pdf(FIXTURE_CABECERA_REPETIDA)
    assert isinstance(resultado, str)


def test_extraer_pdf_preserva_tope_de_seguridad_25_por_ciento(tmp_path):
    """Cuerpos muy cortos: cabecera + pie superan el 25% de palabras por página.

    `quitar_repetidos` no debe tocar nada (cleaning.MAX_FRACCION_BORRADA);
    confirma que integrar la limpieza en extraer_pdf no rompió ese tope.
    """
    doc = fitz.open()
    for i in range(1, 7):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 40), "INFORME ANUAL DE PRUEBA 2024", fontsize=10)
        page.insert_textbox(fitz.Rect(72, 120, 540, 650), "Texto breve de la pagina.", fontsize=11)
        page.insert_text((72, 760), f"Pagina {i} de 6", fontsize=9)
    ruta = tmp_path / "cabecera_domina.pdf"
    doc.save(ruta)
    doc.close()

    texto, informe = extraer_pdf(ruta, devolver_informe=True)

    assert informe.nota is not None
    assert "INFORME ANUAL DE PRUEBA 2024" in texto
