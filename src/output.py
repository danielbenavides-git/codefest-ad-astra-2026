"""Fase de salida — arma `resultados.jsonl` con el esquema exacto de la spec.

Este es el único módulo donde los nombres internos en español se traducen a
los nombres en inglés que exige la entrega (Tabla 2, §9.3.2). Si esa
traducción apareciera en dos sitios, la entrega saldría a medias en cada
idioma y nadie lo notaría hasta que fallara `evaluation.validar_resultados`.

Solo cambia un nombre de campo: `texto` -> `text`. `chunk_id` y `doc_id` se
llaman igual en los dos lados. `fuente`, `formato`, `fenomeno`, `posicion` y
`num_tokens` son campos de `metadata.jsonl` y no aparecen en la entrega.

Este módulo es la última puerta antes de escribir el archivo, así que es
estricto: exige 3 documentos, 10 fragmentos y textos de máximo 250 palabras
(§9.3.2). Prefiere fallar acá, donde el error se puede corregir, que
entregar un archivo que la evaluación automática descarta.
"""

import json
import re
from collections.abc import Sequence
from pathlib import Path

from src.chunking import LIMITE_DURO, contar_palabras

N_CONSULTAS = 50
N_DOCUMENTOS = 3
N_FRAGMENTOS = 10

_QUERY_ID = re.compile(r"^q0(?:0[1-9]|[1-4]\d|50)$")


def _validar_query_id(query_id: str) -> str:
    if not isinstance(query_id, str) or not _QUERY_ID.match(query_id):
        raise ValueError(
            f"query_id inválido: {query_id!r}. Debe ir de 'q001' a 'q050' "
            f"(§10.1)."
        )
    return query_id


def armar_documentos(documentos: Sequence[dict]) -> list[dict]:
    """Salida de `aggregation.agregar_documentos` -> array `documents`.

    El `rank` se reasigna acá según la posición en la lista, no se copia del
    diccionario de entrada: así el archivo entregado siempre tiene rangos
    1, 2, 3 consecutivos, que es lo que valida la organización.
    """
    if len(documentos) != N_DOCUMENTOS:
        raise ValueError(
            f"Se recibieron {len(documentos)} documentos y la entrega exige "
            f"exactamente {N_DOCUMENTOS} (§9.3.2). Si son menos, la búsqueda "
            f"no trajo fragmentos de suficientes documentos distintos: sube "
            f"la k de retrieval."
        )

    salida = []
    for posicion, documento in enumerate(documentos, start=1):
        doc_id = documento.get("doc_id")
        if not doc_id:
            raise ValueError(f"El documento en la posición {posicion} no tiene doc_id.")
        salida.append({"rank": posicion, "doc_id": str(doc_id)})

    return salida


def armar_fragmentos(fragmentos: Sequence[dict]) -> list[dict]:
    """Salida de `retrieval.buscar_vector` -> array `fragments`.

    Recibe los fragmentos ya ordenados y ya recortados a los 10 primeros.
    Este módulo no ordena ni recorta la lista: solo traduce y verifica.
    """
    if len(fragmentos) != N_FRAGMENTOS:
        raise ValueError(
            f"Se recibieron {len(fragmentos)} fragmentos y la entrega exige "
            f"exactamente {N_FRAGMENTOS} (§9.3.2)."
        )

    salida = []
    for posicion, fragmento in enumerate(fragmentos, start=1):
        chunk_id = fragmento.get("chunk_id")
        doc_id = fragmento.get("doc_id")
        texto = fragmento.get("texto")

        if not chunk_id:
            raise ValueError(f"El fragmento {posicion} no tiene chunk_id.")
        if not doc_id:
            raise ValueError(f"El fragmento {posicion} no tiene doc_id.")
        if not isinstance(texto, str) or not texto.strip():
            raise ValueError(
                f"El fragmento {posicion} (chunk_id {chunk_id!r}) no tiene "
                f"campo 'texto'. Viene de metadata.jsonl, donde el campo se "
                f"llama así; acá se traduce a 'text'."
            )

        palabras = contar_palabras(texto)
        if palabras > LIMITE_DURO:
            raise ValueError(
                f"El fragmento {posicion} (chunk_id {chunk_id!r}) tiene "
                f"{palabras} palabras y el máximo es {LIMITE_DURO} (§9.2). "
                f"chunking.py ya garantiza ese límite, así que esto indica "
                f"que el texto se modificó después de fragmentar."
            )

        salida.append(
            {
                "rank": posicion,
                "chunk_id": str(chunk_id),
                "doc_id": str(doc_id),
                "text": texto,
            }
        )

    return salida


def armar_linea(
    query_id: str,
    documentos: Sequence[dict],
    fragmentos: Sequence[dict],
) -> dict:
    """Una consulta -> el objeto JSON de una línea de `resultados.jsonl`."""
    return {
        "query_id": _validar_query_id(query_id),
        "documents": armar_documentos(documentos),
        "fragments": armar_fragmentos(fragmentos),
    }


def escribir_resultados(
    lineas: Sequence[dict],
    path: str | Path,
    *,
    validar: bool = True,
) -> Path:
    """Escribe `resultados.jsonl`: 50 líneas, en orden q001 a q050 (§10.3).

    Con `validar=True` vuelve a leer el archivo escrito y lo pasa por
    `evaluation.validar_resultados`, que es el mismo esquema que revisa la
    organización. Comprobar el archivo en disco y no la lista en memoria es
    a propósito: así también se detecta un problema de codificación o de
    saltos de línea al escribir.
    """
    if len(lineas) != N_CONSULTAS:
        raise ValueError(
            f"Se recibieron {len(lineas)} consultas y la entrega exige "
            f"exactamente {N_CONSULTAS} (§10.3)."
        )

    esperados = [f"q{i:03d}" for i in range(1, N_CONSULTAS + 1)]
    recibidos = [linea.get("query_id") for linea in lineas]

    if recibidos != esperados:
        primera = next(
            (i for i, (r, e) in enumerate(zip(recibidos, esperados)) if r != e),
            0,
        )
        raise ValueError(
            f"Las líneas deben ir en orden q001 a q050 (§10.3). La primera "
            f"discrepancia está en la posición {primera + 1}: "
            f"{recibidos[primera]!r} en vez de {esperados[primera]!r}."
        )

    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)

    with destino.open("w", encoding="utf-8") as archivo:
        for linea in lineas:
            archivo.write(json.dumps(linea, ensure_ascii=False) + "\n")

    if validar:
        from src.evaluation import validar_resultados

        informe = validar_resultados(destino)
        if not informe.valido:
            raise ValueError(f"El archivo escrito no cumple el esquema.\n{informe}")

    return destino