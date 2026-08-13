"""Fase de salida — arma `resultados.jsonl` con el esquema exacto de la spec.

Responsabilidad (D4): este es el **único** módulo donde los resultados
internos de la recuperación, con nombres de campo en español, se traducen
a los nombres en inglés que exige la spec para la entrega. Ni
`retrieval.py` ni ningún otro módulo deben hacer esta traducción — si
aparece en dos sitios, va a haber una entrega con nombres a medias en
español y a medias en inglés sin que nadie lo note hasta que falle
`evaluacion.validar_resultados`.

Recibe: la salida ya resuelta de la Fase 6 (`retrieval.py`) para una
consulta — típicamente algo como una lista de hasta 3 `doc_id` ya
ordenados por relevancia agregada (§8.6) y una lista de hasta 10 objetos
de fragmento ya recortados a 250 palabras (§9.2.1), cada uno con los
campos en español que trae `Chunk.to_dict()` (más el `rank` que le asigna
la propia recuperación, que no viene de ningún chunk).

Devuelve/escribe: un objeto JSON por consulta con el esquema de la Tabla 2
(§9.3.2), y el archivo completo con las 50 líneas de `resultados.jsonl`
(§9.3, §10.3) — nunca toca `metadata.jsonl`.

Correspondencia de nombres, sacada de la spec (no de suposiciones):

| Interno (Tabla 1, §3.4 — `Chunk.to_dict()` / `metadata.jsonl`, español) | Entrega (Tabla 2, §9.3.2 — `resultados.jsonl`, inglés)                              |
|---------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| — (no es un campo de metadata; lo asigna la consulta)                     | `query_id`: identificador de la consulta, `q001`–`q050`                              |
| — (lista de `doc_id`, ordenada por §8.6, no es un campo de metadata)      | `documents`: array de exactamente 3 objetos, orden de mayor a menor relevancia        |
| — (posición en esa lista, la asigna la recuperación, no un chunk)         | `documents[i].rank`: entero, posición en el ranking (1, 2, 3)                        |
| `doc_id`                                                                   | `documents[i].doc_id` — **mismo nombre, sin traducir**                                |
| — (lista de fragmentos recuperados, ordenada, no es un campo de metadata) | `fragments`: array de exactamente 10 objetos, orden de mayor a menor relevancia       |
| — (posición en esa lista, la asigna la recuperación)                      | `fragments[i].rank`: entero, posición en el ranking (1 a 10)                          |
| `chunk_id`                                                                 | `fragments[i].chunk_id` — **mismo nombre, sin traducir**                              |
| `doc_id`                                                                   | `fragments[i].doc_id` — **mismo nombre, sin traducir**                                |
| `texto`                                                                    | `fragments[i].text` — **el único campo que cambia de nombre: `texto` → `text`**       |

`fuente`, `formato`, `fenomeno`, `posicion`, `num_tokens` de la Tabla 1
son campos de `metadata.jsonl` (la base vectorial) y no aparecen en el
esquema de `resultados.jsonl` (Tabla 2) — no hay que traducirlos porque
no van en el archivo de entrega, solo en la base.

`metadata.jsonl` (Tabla 1, §3.4) sí usa nombres en español (`texto`
incluido) y lo escribe directamente `indexing.guardar_base` — ese archivo
no pasa por este módulo.

No hay lógica implementada todavía: este archivo es solo el contrato,
pendiente de que exista `retrieval.py` para tener algo real que traducir.
"""
