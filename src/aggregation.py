"""Fase 6 — agrupa los fragmentos recuperados y arma el ranking de documentos.

Recibe la lista de fragmentos ya ordenada que produce `retrieval.py` (o
`fusion.py` si hubo más de un encoder) y devuelve los documentos ordenados
de mayor a menor relevancia agregada (§8.6).

Este paso siempre corre, use el equipo uno o varios encoders: toda consulta
tiene que devolver 3 documentos (§9.2).

Solo hace aritmética sobre las puntuaciones de FAISS. No reordena con
modelos generativos (§8.3) ni toca el texto.
"""

from collections.abc import Sequence

N_DOCUMENTOS = 3

ESTRATEGIAS = ("max", "suma", "media")

# Por defecto max pooling: un documento vale lo que vale su mejor fragmento.
# Es la primera opción que nombra la spec y la más estable de las tres.
# "suma" premia a los documentos largos, que aportan más fragmentos al top-k
# aunque ninguno sea muy bueno. "media" hace lo contrario: castiga al
# documento que aportó un fragmento excelente y varios mediocres, y favorece
# a los documentos con un solo fragmento recuperado.
ESTRATEGIA_POR_DEFECTO = "max"


def agrupar_por_documento(fragmentos: Sequence[dict]) -> dict[str, list[dict]]:
    """Agrupa los fragmentos recuperados por su `doc_id`, conservando el orden."""
    grupos: dict[str, list[dict]] = {}

    for posicion, fragmento in enumerate(fragmentos):
        doc_id = fragmento.get("doc_id")

        if not doc_id:
            raise ValueError(
                f"El fragmento en la posición {posicion} no tiene doc_id. "
                f"Sin él no se puede saber de qué documento viene y ese "
                f"documento nunca entraría al ranking."
            )

        if "puntuacion" not in fragmento:
            raise ValueError(
                f"El fragmento en la posición {posicion} (chunk_id "
                f"{fragmento.get('chunk_id')!r}) no tiene puntuacion. La "
                f"asigna retrieval.buscar_vector."
            )

        grupos.setdefault(doc_id, []).append(fragmento)

    return grupos


def puntuar_documento(
    fragmentos: Sequence[dict],
    estrategia: str = ESTRATEGIA_POR_DEFECTO,
) -> float:
    """Puntuación agregada de un documento a partir de sus fragmentos (§8.6)."""
    if estrategia not in ESTRATEGIAS:
        raise ValueError(
            f"Estrategia {estrategia!r} desconocida. Opciones: "
            f"{', '.join(ESTRATEGIAS)}."
        )

    puntuaciones = [float(f["puntuacion"]) for f in fragmentos]

    if estrategia == "max":
        return max(puntuaciones)
    if estrategia == "suma":
        return sum(puntuaciones)
    return sum(puntuaciones) / len(puntuaciones)


def agregar_documentos(
    fragmentos: Sequence[dict],
    *,
    estrategia: str = ESTRATEGIA_POR_DEFECTO,
    n: int = N_DOCUMENTOS,
) -> list[dict]:
    """Fragmentos ordenados -> los `n` documentos más relevantes.

    Devuelve una lista de diccionarios con `doc_id`, `puntuacion`, `rank`,
    `n_fragmentos` (cuántos fragmentos de ese documento entraron al top-k) y
    `mejor_rank` (la mejor posición que alcanzó uno de sus fragmentos). Los
    dos últimos no van en la entrega: sirven para depurar y para el informe.

    Si hay menos de `n` documentos distintos entre los fragmentos recibidos,
    devuelve los que haya. Completar hasta 3 no es tarea de este módulo:
    quien llama debe recuperar más fragmentos (subir `k` en la búsqueda).
    `output.py` es el que exige los 3 antes de escribir la entrega.
    """
    if not fragmentos:
        raise ValueError("No se recibió ningún fragmento para agregar.")

    if n <= 0:
        raise ValueError("n debe ser mayor que cero.")

    grupos = agrupar_por_documento(fragmentos)

    documentos = []
    for doc_id, del_documento in grupos.items():
        documentos.append(
            {
                "doc_id": doc_id,
                "puntuacion": puntuar_documento(del_documento, estrategia),
                "n_fragmentos": len(del_documento),
                "mejor_rank": min(
                    int(f.get("rank", i + 1))
                    for i, f in enumerate(del_documento)
                ),
            }
        )

    # Desempate explícito, en este orden: mejor puntuación, luego el documento
    # cuyo mejor fragmento quedó más arriba, luego el doc_id alfabético. Sin
    # esta regla, dos corridas idénticas podrían devolver documentos distintos
    # cuando las puntuaciones empatan, y el F1@3 cambiaría sin razón aparente.
    documentos.sort(key=lambda d: (-d["puntuacion"], d["mejor_rank"], d["doc_id"]))

    for rank, documento in enumerate(documentos[:n], start=1):
        documento["rank"] = rank

    return documentos[:n]