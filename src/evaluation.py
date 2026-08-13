"""Fase 7 — Evaluación.

Implementa las dos métricas del reto (NDCG@10 para fragmentos, F1@3 para
documentos, spec §10.2) y el validador de esquema de `resultados.jsonl`
(spec §9.3.2, §10.3). La organización calcula estas mismas métricas con su
propio ground truth oculto; este módulo es la herramienta interna del equipo
para medir el pipeline antes de entregar.

El emparejamiento con el ground truth —¿qué fragmento/documento devuelto
corresponde a cuál del ground truth?— es responsabilidad de quien arme el
conjunto de consultas de prueba, no de este módulo: los fragmentos se
comparan por su campo `text` y los documentos por `fuente` (§10.2.1), nunca
por `chunk_id`/`doc_id`, que son identificadores internos de cada equipo.
Las funciones de aquí solo consumen listas ya resueltas: relevancias en el
caso de NDCG, conjuntos de identificadores en el caso de F1.
"""

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from src.chunking import LIMITE_DURO, contar_palabras

N_CONSULTAS = 50
N_DOCUMENTOS = 3
N_FRAGMENTOS = 10


# Métricas (§10.2)

def dcg_at_k(relevancias: Sequence[float], k: int = 10) -> float:
    """DCG@k = sum r_i / log2(i+1), con i la posición 1-indexada (ec. 8)."""
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevancias[:k]))


def ndcg_at_10(relevancias: Sequence[float]) -> float:
    """NDCG@10 de una lista de 10 relevancias, en el orden devuelto por el sistema.

    `relevancias[i]` es la relevancia según el ground truth del fragmento que
    el sistema puso en la posición i+1. Si el ground truth no marca ningún
    fragmento como relevante para la consulta, IDCG es 0 y se define NDCG=0
    en vez de indeterminado, para que la media sobre las 50 consultas (ec. 10)
    no falle con una división por cero.
    """
    idcg = dcg_at_k(sorted(relevancias, reverse=True), 10)
    if idcg == 0:
        return 0.0
    return dcg_at_k(relevancias, 10) / idcg


def f1_at_3(devueltos: Sequence[str], relevantes: Sequence[str]) -> float:
    """F1@3 de los documentos devueltos contra el conjunto relevante (ec. 11-13).

    El denominador del recall es min(|relevantes|, 3): una consulta con menos
    de 3 documentos relevantes en el ground truth no debe penalizar al equipo
    por devolver 3, como exige la spec.
    """
    devueltos3 = list(devueltos)[:N_DOCUMENTOS]
    relevantes_set = set(relevantes)
    interseccion = len(set(devueltos3) & relevantes_set)

    precision = interseccion / N_DOCUMENTOS
    denom_recall = min(len(relevantes_set), N_DOCUMENTOS)
    recall = interseccion / denom_recall if denom_recall else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def promedio(valores: Sequence[float]) -> float:
    """Media sobre las 50 consultas (ec. 10 y 14): mismo cálculo para ambas métricas."""
    return sum(valores) / len(valores) if valores else 0.0


# Validador de esquema de resultados.jsonl (§9.3, §9.3.2, §10.3)

@dataclass
class InformeValidacion:
    """Qué falla en resultados.jsonl y en qué línea. Se revisa antes de entregar."""

    errores: list[str] = field(default_factory=list)
    n_lineas: int = 0

    @property
    def valido(self) -> bool:
        return not self.errores

    def __str__(self) -> str:
        if self.valido:
            return f"OK: {self.n_lineas} líneas, esquema válido"
        detalle = "\n".join(f"  - {e}" for e in self.errores[:20])
        extra = f"\n  (+{len(self.errores) - 20} más)" if len(self.errores) > 20 else ""
        return f"INVÁLIDO: {len(self.errores)} error(es)\n{detalle}{extra}"


def _verificar_documentos(query_id: str, documentos) -> list[str]:
    if not isinstance(documentos, list) or len(documentos) != N_DOCUMENTOS:
        n = len(documentos) if isinstance(documentos, list) else "no es lista"
        return [f"{query_id}: 'documents' debe tener {N_DOCUMENTOS} elementos, tiene {n}"]
    errores = []
    for i, doc in enumerate(documentos):
        if not isinstance(doc, dict):
            errores.append(f"{query_id}: documents[{i}] no es un objeto")
            continue
        if doc.get("rank") != i + 1:
            errores.append(f"{query_id}: documents[{i}].rank debería ser {i + 1}, es {doc.get('rank')!r}")
        if not doc.get("doc_id"):
            errores.append(f"{query_id}: documents[{i}].doc_id vacío o ausente")
    return errores


def _verificar_fragmentos(query_id: str, fragmentos) -> list[str]:
    if not isinstance(fragmentos, list) or len(fragmentos) != N_FRAGMENTOS:
        n = len(fragmentos) if isinstance(fragmentos, list) else "no es lista"
        return [f"{query_id}: 'fragments' debe tener {N_FRAGMENTOS} elementos, tiene {n}"]
    errores = []
    for i, frag in enumerate(fragmentos):
        if not isinstance(frag, dict):
            errores.append(f"{query_id}: fragments[{i}] no es un objeto")
            continue
        if frag.get("rank") != i + 1:
            errores.append(f"{query_id}: fragments[{i}].rank debería ser {i + 1}, es {frag.get('rank')!r}")
        for campo in ("chunk_id", "doc_id", "text"):
            if not frag.get(campo):
                errores.append(f"{query_id}: fragments[{i}].{campo} vacío o ausente")
        texto = frag.get("text", "")
        if isinstance(texto, str) and contar_palabras(texto) > LIMITE_DURO:
            errores.append(
                f"{query_id}: fragments[{i}].text tiene {contar_palabras(texto)} palabras, "
                f"máximo {LIMITE_DURO}"
            )
    return errores


def validar_resultados(path: "Path | str") -> InformeValidacion:
    """Valida resultados.jsonl contra el esquema de la Tabla 2 (§9.3.2, §10.3).

    Recorre las 50 líneas SIEMPRE, incluso tras el primer error: el objetivo
    es un reporte completo para corregir de una vez, no fallar en la línea 1.
    """
    path = Path(path)
    informe = InformeValidacion()
    lineas = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    informe.n_lineas = len(lineas)

    if len(lineas) != N_CONSULTAS:
        informe.errores.append(f"el archivo tiene {len(lineas)} líneas, se esperaban {N_CONSULTAS}")

    for n, linea in enumerate(lineas, start=1):
        try:
            obj = json.loads(linea)
        except json.JSONDecodeError as exc:
            informe.errores.append(f"línea {n}: JSON inválido ({exc.msg})")
            continue

        query_id = obj.get("query_id")
        esperado = f"q{n:03d}"
        if query_id != esperado:
            informe.errores.append(f"línea {n}: query_id es {query_id!r}, se esperaba {esperado!r} (orden §10.3)")
            query_id = query_id or f"línea {n}"

        if "documents" not in obj:
            informe.errores.append(f"{query_id}: falta 'documents'")
        else:
            informe.errores.extend(_verificar_documentos(query_id, obj["documents"]))

        if "fragments" not in obj:
            informe.errores.append(f"{query_id}: falta 'fragments'")
        else:
            informe.errores.extend(_verificar_fragmentos(query_id, obj["fragments"]))

    return informe


# Autoverificación: ejemplos calculados a mano, para confiar en las fórmulas
# antes de que exista un índice real contra el cual correrlas (ver conversación
# de diseño: "verificar contra ejemplos de juguete calculados a mano").

def _autoverificar_metricas() -> None:
    # Un fragmento relevante ya en la posición 1: el orden es ideal, NDCG=1.
    assert math.isclose(ndcg_at_10([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 1.0)

    # El mismo fragmento relevante pero en la posición 2:
    # DCG = 1/log2(3), IDCG = 1/log2(2) = 1.
    esperado = (1 / math.log2(3)) / 1.0
    assert math.isclose(ndcg_at_10([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), esperado)

    # Ningún fragmento relevante en el ground truth: IDCG=0, se define NDCG=0.
    assert ndcg_at_10([0] * 10) == 0.0

    # Match perfecto de documentos.
    assert math.isclose(f1_at_3(["A", "B", "C"], {"A", "B", "C"}), 1.0)

    # 1 de 3 relevantes encontrado, y hay 3 relevantes en el ground truth:
    # precision=1/3, recall=1/min(3,3)=1/3, F1=1/3.
    assert math.isclose(f1_at_3(["A", "B", "C"], {"A", "D", "E"}), 1 / 3)

    # Solo existe 1 documento relevante en el ground truth: el denominador del
    # recall se limita a min(1,3)=1, no a 3 -> precision=1/3, recall=1, F1=0.5.
    assert math.isclose(f1_at_3(["A", "B", "C"], {"A"}), 0.5)

    print("métricas: OK (6 casos verificados a mano)")


def _autoverificar_validador() -> None:
    import tempfile

    def linea_valida(i: int) -> dict:
        return {
            "query_id": f"q{i:03d}",
            "documents": [{"rank": r, "doc_id": f"DOC-{i:03d}-{r}"} for r in (1, 2, 3)],
            "fragments": [
                {
                    "rank": r,
                    "chunk_id": f"DOC-{i:03d}-chunk-{r}",
                    "doc_id": f"DOC-{i:03d}-{r}",
                    "text": "texto de prueba " * 5,
                }
                for r in range(1, 11)
            ],
        }

    with tempfile.TemporaryDirectory() as tmp:
        valido = Path(tmp) / "valido.jsonl"
        valido.write_text("\n".join(json.dumps(linea_valida(i)) for i in range(1, 51)), encoding="utf-8")
        informe = validar_resultados(valido)
        assert informe.valido, informe

        lineas = [linea_valida(i) for i in range(1, 51)]
        lineas[4]["fragments"] = lineas[4]["fragments"][:9]    # q005: solo 9 fragmentos
        lineas[10]["fragments"][0]["text"] = "palabra " * 300  # q011: excede 250 palabras
        roto = Path(tmp) / "roto.jsonl"
        roto.write_text("\n".join(json.dumps(l) for l in lineas), encoding="utf-8")
        informe = validar_resultados(roto)
        assert not informe.valido
        assert any("q005" in e and "fragments" in e for e in informe.errores)
        assert any("q011" in e and "palabras" in e for e in informe.errores)

    print("validador: OK (detecta archivo válido, conteo incorrecto y exceso de palabras)")


if __name__ == "__main__":
    _autoverificar_metricas()
    _autoverificar_validador()

    if len(sys.argv) > 1:
        print()
        print(validar_resultados(sys.argv[1]))
