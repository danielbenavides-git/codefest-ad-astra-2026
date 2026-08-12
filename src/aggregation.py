"""Fase 6 — agrega fragmentos rankeados al nivel de documento.

Recibe: la lista de fragmentos ya resuelta (de un único índice, o ya
fusionada por `fusion.py` si hubo más de un encoder), cada uno con su
puntuación y su `doc_id`. Devuelve: los 3 `doc_id` de mayor puntuación
agregada, ordenados de mayor a menor (§8.6: max pooling, suma o media de
las puntuaciones de los fragmentos de cada documento).

A diferencia de `fusion.py` (que solo aplica con más de un encoder), este
paso siempre se ejecuta: toda consulta debe devolver exactamente 3
documentos (§9.2), sin importar cuántos encoders se usaron para llegar
ahí.
"""
