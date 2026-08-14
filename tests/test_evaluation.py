"""Tests de las métricas (§10.2). Todos los valores están calculados a mano.

Un error en este módulo no rompe nada visible: solo hace que cada experimento
de la rejilla se decida sobre números falsos. Por eso los casos son a mano y
no comparados contra otra implementación, que podría traer el mismo error.
"""

import math

import pytest

from src.evaluation import (
    InformeValidacion,
    dcg_at_k,
    f1_at_3,
    ndcg_at_10,
    promedio,
    validar_resultados,
)


# --------------------------------------------------------------------- DCG

def test_dcg_primera_posicion_no_se_descuenta():
    # log2(1+1) = 1, así que el primer elemento entra con su valor íntegro.
    assert dcg_at_k([5]) == 5.0


def test_dcg_segunda_posicion_se_descuenta_por_log2_de_3():
    assert math.isclose(dcg_at_k([0, 3]), 3 / math.log2(3))


def test_dcg_respeta_el_corte_en_k():
    # El elemento 11 no debe contar en DCG@10.
    assert dcg_at_k([0] * 10 + [99], 10) == 0.0


# -------------------------------------------------------------------- NDCG

def test_ndcg_orden_ideal_es_uno():
    assert math.isclose(ndcg_at_10([3, 2, 1] + [0] * 7, [3, 2, 1]), 1.0)


def test_ndcg_ejemplo_graduado_verificado_a_mano():
    # DCG  = 3 + 2/log2(3) + 3/2 + 0 + 1/log2(6) + 2/log2(7) = 6.86112
    # IDCG = 3 + 3/log2(3) + 1 + 2/log2(5) + 1/log2(6) + 0   = 7.14105
    devuelto = [3, 2, 3, 0, 1, 2]
    ideal = [3, 3, 2, 2, 1, 0]
    assert math.isclose(ndcg_at_10(devuelto, ideal), 0.9608, abs_tol=1e-4)


def test_ndcg_sin_relevantes_es_cero_y_no_lanza():
    assert ndcg_at_10([0] * 10, []) == 0.0
    assert ndcg_at_10([0] * 10, [0, 0]) == 0.0


def test_ndcg_penaliza_la_cobertura_incompleta():
    """El IDCG sale del ground truth, no de lo devuelto.

    Es el bug que hacía que recuperar 1 de 5 relevantes puntuara 1.000, con
    lo cual ninguna mejora de cobertura era medible.
    """
    ideal = [3, 3, 3, 3, 3]
    uno = ndcg_at_10([3] + [0] * 9, ideal)
    tres = ndcg_at_10([3, 3, 3] + [0] * 7, ideal)
    cinco = ndcg_at_10(ideal + [0] * 5, ideal)

    assert uno < tres < cinco
    assert math.isclose(cinco, 1.0)
    assert uno < 0.35, "recuperar 1 de 5 no puede parecerse a un resultado perfecto"


def test_ndcg_premia_colocar_lo_relevante_arriba():
    ideal = [3, 3]
    arriba = ndcg_at_10([3, 3] + [0] * 8, ideal)
    abajo = ndcg_at_10([0] * 8 + [3, 3], ideal)
    assert arriba > abajo


# ---------------------------------------------------------------------- F1

def test_f1_acierto_total():
    assert math.isclose(f1_at_3(["A", "B", "C"], {"A", "B", "C"}), 1.0)


def test_f1_sin_aciertos():
    assert f1_at_3(["A", "B", "C"], {"X", "Y"}) == 0.0


def test_f1_denominador_acotado_con_un_solo_relevante():
    # P = 1/3, R = 1/min(1,3) = 1 -> F1 = 0.5
    assert math.isclose(f1_at_3(["A", "B", "C"], {"A"}), 0.5)


def test_f1_denominador_acotado_difiere_del_recall_estandar():
    """Con 5 relevantes y 2 aciertos: spec F1=2/3, recall estándar daría 0.5."""
    obtenido = f1_at_3(["A", "B", "X"], {"A", "B", "C", "D", "E"})
    assert math.isclose(obtenido, 2 / 3)

    p, r_estandar = 2 / 3, 2 / 5
    f1_estandar = 2 * p * r_estandar / (p + r_estandar)
    assert not math.isclose(obtenido, f1_estandar)


def test_f1_es_metrica_de_conjunto():
    """El orden de los 3 documentos no cambia el puntaje (§10.2)."""
    rel = {"A", "B", "Z"}
    assert f1_at_3(["A", "B", "C"], rel) == f1_at_3(["C", "B", "A"], rel)


def test_f1_ignora_documentos_mas_alla_del_tercero():
    assert f1_at_3(["X", "Y", "Z", "A"], {"A"}) == 0.0


def test_f1_sin_relevantes_no_divide_por_cero():
    assert f1_at_3(["A", "B", "C"], set()) == 0.0


def test_promedio_de_lista_vacia():
    assert promedio([]) == 0.0


# -------------------------------------------------------------- validador

def _linea(i: int) -> dict:
    return {
        "query_id": f"q{i:03d}",
        "documents": [{"rank": r, "doc_id": f"DOC-{r}"} for r in (1, 2, 3)],
        "fragments": [
            {"rank": r, "chunk_id": f"DOC-1-chunk-{r}", "doc_id": "DOC-1",
             "text": "texto de prueba " * 5}
            for r in range(1, 11)
        ],
    }


def _escribir(tmp_path, lineas):
    import json

    destino = tmp_path / "resultados.jsonl"
    destino.write_text("\n".join(json.dumps(l) for l in lineas), encoding="utf-8")
    return destino


def test_validador_acepta_archivo_correcto(tmp_path):
    informe = validar_resultados(_escribir(tmp_path, [_linea(i) for i in range(1, 51)]))
    assert informe.valido, informe


@pytest.mark.parametrize(
    "romper,fragmento_esperado",
    [
        (lambda ls: ls.pop(), "49 líneas"),
        (lambda ls: ls[0].__setitem__("query_id", "q999"), "q001"),
        (lambda ls: ls[3]["documents"].pop(), "documents"),
        (lambda ls: ls[3]["fragments"].pop(), "fragments"),
        (lambda ls: ls[5]["fragments"][2].__setitem__("rank", 9), "rank"),
        (lambda ls: ls[7]["fragments"][0].__setitem__("text", "palabra " * 300), "palabras"),
        (lambda ls: ls[9]["fragments"][0].__setitem__("text", ""), "text"),
        (lambda ls: ls[2]["documents"][1].__setitem__("doc_id", ""), "doc_id"),
    ],
)
def test_validador_detecta_cada_fallo(tmp_path, romper, fragmento_esperado):
    lineas = [_linea(i) for i in range(1, 51)]
    romper(lineas)
    informe = validar_resultados(_escribir(tmp_path, lineas))
    assert not informe.valido
    assert any(fragmento_esperado in e for e in informe.errores), informe.errores[:5]


def test_validador_recorre_todas_las_lineas_aunque_falle_la_primera(tmp_path):
    """Se quiere un reporte completo, no fallar en la línea 1."""
    lineas = [_linea(i) for i in range(1, 51)]
    lineas[0]["documents"].pop()
    lineas[49]["fragments"].pop()
    informe = validar_resultados(_escribir(tmp_path, lineas))
    assert any("q001" in e for e in informe.errores)
    assert any("q050" in e for e in informe.errores)


def test_informe_vacio_es_valido():
    assert InformeValidacion().valido
