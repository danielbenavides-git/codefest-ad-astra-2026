"""Fase 6 — combina los rankings de fragmentos de varios índices FAISS.

Recibe: para una misma consulta, una lista de fragmentos con puntuación
por cada índice/encoder usado (uno si el equipo usa un solo encoder, más
de uno si usa varios, §4.4). Devuelve: una única lista de fragmentos con
puntuación combinada, sin modelos generativos (§8.4: CombSUM, CombMNZ o
Reciprocal Rank Fusion), lista para pasar a `aggregation.py`.

Solo hace algo si hay más de un índice; con un único encoder es un
passthrough. No se solapa con `aggregation.py`: este módulo combina el
mismo fragmento visto por *distintos encoders*, `aggregation.py` agrupa
*distintos fragmentos* del mismo documento dentro de un único ranking ya
resuelto.
"""
