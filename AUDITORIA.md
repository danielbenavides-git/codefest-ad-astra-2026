# Auditoría del repositorio — CODEFEST Ad Astra 2026

Diagnóstico. No se ha tocado ningún archivo de `src/`, `scripts/`, `tests/`, `README.md` ni `entrega/`. El único archivo nuevo es este.

## Comandos de partida y su salida

```
$ git log --oneline -15
9d6d485 Merge pull request #2 from cmolina12/feat/benchmark-encoders
cd8cd99 Minor changes
39e48a6 Add benchmark encoder CSV files and update config parameters
de93684 Refactor benchmark encoders notebook and update mini corpus relevance scale
5f6ae77 Update mini_corpus.json to enhance relevance scoring and add new texts for improved testing
e1e62e1 Refactor benchmark encoders notebook and update mini_corpus.json with relevant queries and texts
7b3e1eb Merge branch 'feature/Avances_Andrew' of .../codefest-ad-astra-2026 into feat/benchmark-encoders
ac509bd Avances Andrew
50df7c5 Add mini_corpus.json fixture for testing purposes
673a557 Merge branch 'main' of .../codefest-ad-astra-2026 into feat/benchmark-encoders
fb48dfb Merge pull request #1 from cmolina12:Camilo
728f55f Implementación de métricas NDCG@10 y F1@3, y validador de esquema para resultados.jsonl preliminar
760353e 1ra version limpieza
680be07 Merge branch 'main' of .../codefest-ad-astra-2026 into feat/benchmark-encoders
fb2a7ac k

$ git branch -a
* claude/instrucciones-archivo-md-qhpqq2
  main
  remotes/origin/claude/instrucciones-archivo-md-qhpqq2
  remotes/origin/main
```

`git branch -a` local solo muestra `main`, pero `git ls-remote --heads origin` expone tres ramas adicionales que no aparecen en un `fetch` normal por no estar trackeadas localmente: **`Camilo`**, **`feat/benchmark-encoders`** y **`feature/Avances_Andrew`**. Las tres ya están completamente mergeadas en `main` (`git log origin/main..origin/<rama>` no devuelve commits) pero nadie las borró — son las "tres ramas abiertas" del enunciado. No las toqué.

```
$ find . -path ./.git -prune -o -type f -name "*.py" -print | xargs wc -l | sort -n
    0 ./src/__init__.py
    0 ./src/aggregation.py
    0 ./src/bono_graph.py
    0 ./src/cleaning.py
    0 ./src/encoding.py
    0 ./src/fusion.py
    0 ./src/retrieval.py
   13 ./src/config.py
   70 ./tests/test_indexing.py
  138 ./src/limpieza.py
  162 ./src/indexing.py
  252 ./src/evaluacion.py
  339 ./src/chunking.py
  569 ./src/extractor.py
 1543 total

$ python -c "import src.chunking, src.evaluacion, src.extractor, src.indexing, src.config, src.limpieza"
Traceback (most recent call last):
  ...
  File "/home/user/codefest-ad-astra-2026/src/evaluacion.py", line 25, in <module>
    from chunking import LIMITE_DURO, contar_palabras
ModuleNotFoundError: No module named 'chunking'

$ grep -rn "TODO\|FIXME\|XXX\|HACK\|NotImplementedError" src/ tests/ scripts/
(sin resultados)

$ grep -rn "^from \|^import " src/*.py | grep -v "^\s*#"
src/chunking.py:1:import re
src/chunking.py:2:from dataclasses import dataclass
src/chunking.py:3:from typing import Callable, List, Optional, Sequence
src/evaluacion.py:18:import json
src/evaluacion.py:19:import math
src/evaluacion.py:20:import sys
src/evaluacion.py:21:from dataclasses import dataclass, field
src/evaluacion.py:22:from pathlib import Path
src/evaluacion.py:23:from typing import Sequence
src/evaluacion.py:25:from chunking import LIMITE_DURO, contar_palabras
src/extractor.py:1:import json
src/extractor.py:2:import re
src/extractor.py:3:import unicodedata
src/extractor.py:4:from pathlib import Path
src/extractor.py:5:from typing import Any, Callable, Iterable
src/extractor.py:7:from chunking import DocumentoSinTexto, contar_palabras, detectar_idioma
src/indexing.py:1:import json
src/indexing.py:2:from collections.abc import Sequence
src/indexing.py:3:from pathlib import Path
src/indexing.py:5:import faiss
src/indexing.py:6:import numpy as np
src/indexing.py:8:from .chunking import Chunk
src/limpieza.py:2:import re
src/limpieza.py:3:from collections import Counter
src/limpieza.py:4:from dataclasses import dataclass, field
src/limpieza.py:5:from typing import Sequence
```

Nota sobre el entorno: este sandbox partió **sin ninguna dependencia de `requirements.txt` instalada** (`pip freeze` inicial: 33 paquetes, todos de tooling del sistema, cero de la lista del proyecto). Para poder "verificar ejecutando" instalé `faiss-cpu` (que trajo `numpy` como dependencia) — es lo único que agregué al entorno, y solo en este sandbox desechable, no en el repositorio. Con eso pude ejecutar `chunking`, `indexing`, `evaluacion` y `test_indexing`, y reproducir el hallazgo E con un caso mínimo. **No pude ejecutar** nada que dependa de `torch`, `transformers`, `sentence-transformers`, `pymupdf`, `beautifulsoup4`, `pytesseract`, `easyocr`, `osmium` u `openpyxl` (no instalados, y varios son pesados o requieren binarios del sistema) — por tanto no verifiqué por ejecución el comportamiento interno de `extractor.py` sobre documentos reales, solo su import, su dispatcher, y su historial de git. Lo dejo explícito para no reportar como "verificado" algo que solo leí.

Leí `documentation/CODEFEST_2026-1.pdf` completo (23 páginas) sin modificarlo, en particular §1.3–1.4, §3.4, §8.3, §9.2–9.3 y §10.2, citadas abajo.

---

## 1. Resumen

**7 hallazgos con severidad asignada** (sección 2-3): 1 bloqueante, 4 altos, 2 bajos. A eso se suman 9 incompletos (módulos y entregables vacíos, sección 4, sin severidad porque no son errores) y 7 sugerencias (sección 5, confirmadas por ejecución donde fue posible). El corpus real no está en el repo (`data/raw/` solo tiene `.gitkeep`), así que todo lo verificado es a nivel de código, no de resultados sobre datos reales.

Lo más urgente: los imports absolutos en `evaluacion.py:25` y `extractor.py:7` impiden usar esos dos módulos desde cualquier script que no esté parado dentro de `src/` — confirmado, es lo primero que bloquea armar un pipeline real. Justo detrás, en severidad: el dispatcher de `extractor.py` solo llega al 43% de los formatos que la propia spec (§1.3) dice que entrega ADL (excluye JSON, XLSX, imágenes y PBF, con funciones completas y ya escritas para los cuatro). Ningún cambio de código se aplicó; todo queda en la lista de abajo a la espera de aprobación.

---

## 2. Bloqueantes

### B1. Imports absolutos rompen `evaluacion.py` y `extractor.py` fuera de `src/`

**Severidad: Bloqueante. Arreglo: coordinado** (toca un patrón usado en 2 archivos activos, hay que fijar convención antes — ver Pregunta 1).

Conviven tres estilos de import para el mismo paquete `src/`:

```python
src/evaluacion.py:25   from chunking import LIMITE_DURO, contar_palabras   # absoluto sin paquete
src/extractor.py:7     from chunking import DocumentoSinTexto, ...         # absoluto sin paquete
src/indexing.py:8      from .chunking import Chunk                        # relativo
tests/test_indexing.py from src.chunking import Chunk                     # absoluto con paquete, desde la raíz
```

Verificado ejecutando cada módulo por separado desde la raíz del repo:

```
$ python3 -c "import src.evaluacion"
ModuleNotFoundError: No module named 'chunking'
$ python3 -c "import src.extractor"
ModuleNotFoundError: No module named 'chunking'
$ python3 -c "import src.indexing"      # (funciona, una vez instalado faiss)
$ python3 -c "import src.chunking, src.config, src.limpieza"   # funcionan, sin imports internos entre sí
```

Y confirmado que la única forma de que `evaluacion.py`/`extractor.py` funcionen hoy es pararse dentro de `src/`:

```
$ cd src && python3 -c "import evaluacion"      # OK
$ cd src && python3 evaluacion.py               # OK, corre la autoverificación
```

**Qué se rompe:** cualquier script de orquestación en `scripts/` o cualquier test en `tests/` que importe `src.evaluacion` o `src.extractor` desde la raíz del repo (el modo normal de trabajar en un paquete Python) falla. Fase 1 y Fase 7 quedan inutilizables desde fuera de `src/`.

**Arreglo propuesto:** cambiar `src/evaluacion.py:25` y `src/extractor.py:7` a `from .chunking import ...` (relativo, como ya hace `indexing.py:8`), y ejecutar todo como paquete (`python -m ...` o con `src/` instalado/en el path como paquete). Es el cambio de una línea por archivo, pero antes hay que decidir la convención de ejecución del proyecto completo — por eso "coordinado", ver Pregunta 1.

---

## 3. Incongruencias

### I1. `extractor.py` implementa 8 extractores, el dispatcher solo expone 3

**Severidad: Alto. Arreglo: coordinado** (toca el punto de entrada único de Fase 1, que cualquiera que esté ingiriendo corpus ahora mismo depende de él).

```python
# src/extractor.py:542
EXTRACTORES: dict[str, Callable[[Path], str]] = {
    "pdf": extraer_pdf,
    "html": extraer_html,
    "csv": extraer_csv,
}
```

Las funciones `extraer_xlsx`, `extraer_json`, `extraer_imagen`, `extraer_pbf` y `extraer_texto_plano` están **completas** (no son stubs: tienen manejo de errores, umbrales mínimos de palabras, y docstrings que documentan decisiones de diseño no triviales — p. ej. `extraer_json` explica por qué se separan claves de cuerpo vs. metadatos, `extraer_pbf` por qué se usa `osmium` y no `pyrosm`). No hay ningún `TODO`/`pass`/`NotImplementedError` en ellas — confirmado con `grep` (sin resultados) y con lectura completa del archivo.

`TITULADORES` tiene el mismo problema: registra `pdf` y `html` (`extractor.py:548`), pero `titulo_json` (línea 392) existe y no está registrado.

**Por qué importa, con la spec en mano:** §1.3 de `CODEFEST_2026-1.pdf` dice que ADL entrega el corpus en **PDF, HTML, JSON, CSV, XLSX, Imágenes y PBF** — 7 formatos. El dispatcher actual cubre PDF, HTML y CSV: **3 de 7**. Con el registro tal como está, `extraer(path, "json")` (o `"xlsx"`, `"imagen"`, `"pbf"`) lanza `DocumentoSinTexto` con el mensaje "sin extractor" aunque la función que lo procesaría ya esté escrita y probada a nivel de lectura de código.

**Historial de git (`git blame`):** las 5 funciones y el dispatcher recortado llegaron juntos en el mismo commit (`1a71e55 avances fases 1 y 2`, autor Rozen14, cuando el archivo vivía en `src/extraction/extractor.py`) y no se han tocado desde. No hay commits posteriores que reduzcan el dispatcher a propósito ni issues/TODOs alrededor. Mi lectura: **es un descuido**, no una decisión de dejarlas a medias — las funciones no registradas están al mismo nivel de terminación que las que sí están registradas (mismo estilo de manejo de errores, mismos docstrings extensos, mismos umbrales `MIN_PALABRAS_*` definidos para cada una). Pero es un juicio, no un hecho verificado; lo dejo también en Preguntas por si Rozen14 recuerda algo que el historial no muestra.

**Arreglo propuesto:** añadir las 5 entradas faltantes a `EXTRACTORES` (línea 542-546) y `titulo_json` a `TITULADORES` (línea 548-551). Cambio mecánico, pero "coordinado" porque cambia qué formatos entran al índice — quien esté escribiendo el script de ingesta debe saberlo.

### I2. Nombres de campo: `texto` (interno) vs. `text` (entrega) — confirmado como diseño correcto por la spec, pero la traducción no existe en ningún módulo

**Severidad: Alto (bloquea la entrega si no se resuelve, aunque no bloquea el desarrollo hoy). Arreglo: coordinado** (depende de qué módulo se escriba primero, `retrieval.py` o `generador.py`).

`Chunk.to_dict()` (`chunking.py:254`) produce, en español: `chunk_id, doc_id, fenomeno, formato, fuente, num_tokens, posicion, texto`. El validador de `evaluacion.py` (`_verificar_fragmentos`, línea 127) exige en inglés: `chunk_id, doc_id, text`.

Contrastando contra la spec, **esto no es una contradicción del equipo — son dos archivos distintos con dos esquemas distintos, y ambos coinciden exactamente con lo que exige la spec:**

- **Tabla 1 (§3.4, metadata por fragmento — lo que va en `metadata.jsonl` de la base vectorial):** `doc_id, chunk_id, fuente, formato, fenomeno, posicion, num_tokens, texto`. Es una coincidencia campo por campo, en español, con `Chunk.to_dict()`.
- **Tabla 2 (§9.3.2, esquema de `resultados.jsonl` — el archivo de entrega):** `fragments[i].chunk_id, fragments[i].doc_id, fragments[i].text`. En inglés, y solo un subconjunto de campos.

O sea: `metadata.jsonl` usa `texto` (correcto, así lo pide la spec) y `resultados.jsonl` usa `text` (correcto, así lo pide la spec). **Alguien tiene que traducir entre ambos.** Ese "alguien" es, como ya intuía el hallazgo original, el módulo que arma la respuesta final — hoy `retrieval.py` (vacío) o el futuro `generador.py` del entregable (que tampoco existe aún, ver I5/Incompletos). Ninguno de los dos está escrito, así que la traducción hoy no está en ningún lado, pero **no hace falta decidir nada nuevo**: la spec ya fija ambos nombres, solo falta escribir el paso intermedio.

**Arreglo propuesto (para cuando se implemente, no ahora):** en el módulo que construya `resultados.jsonl`, mapear explícitamente `chunk["texto"] → "text"` al armar cada objeto de `fragments`. No renombrar `texto` en `Chunk`/`metadata.jsonl` — eso rompería la Tabla 1.

### I3. Fase 1 → Fase 2: `extraer_pdf` nunca invoca `bloques_por_pagina` / `quitar_repetidos`

**Severidad: Alto. Arreglo: coordinado** (depende de decidir dónde vive esta responsabilidad, ver Pregunta 3).

`limpieza.py` define `bloques_por_pagina(doc)` (línea 61) que espera un `fitz.Document` crudo, y `quitar_repetidos(paginas, ...)` (línea 77) que consume la lista `[(texto, y_relativa), ...]` que produce `bloques_por_pagina`. Verificado con `grep -rn "quitar_repetidos\|bloques_por_pagina"` en todo `src/`, `tests/`, `scripts/`, `notebooks/`: **las únicas apariciones son las definiciones mismas.** Nadie las llama.

`extraer_pdf` (`extractor.py:73`) abre el PDF con `fitz.open(path)`, extrae bloques con `pagina.get_text("blocks", sort=True)` **directamente** (línea 93), los une en un string por página, y al final llama solo a `normalizar()` (línea 119) — que quita caracteres de control y números de página sueltos, pero no cabeceras/pies repetidos. El objeto `fitz.Document` se cierra (`doc.close()`, línea 117) antes de que `limpieza.py` pueda usarlo, y de todas formas nunca se le pasa.

**Qué falla en la cadena:** el texto que sale de `extraer_pdf()` (Fase 1) es un `str` plano ya normalizado; `quitar_repetidos()` (Fase 2) espera la estructura `paginas: Sequence[Sequence[tuple[str, float]]]` con coordenadas, que solo existe dentro de `extraer_pdf` como variable local `bloques` (línea 93) y nunca sale de esa función. **La Fase 2, para PDFs, es inalcanzable tal como está cableado hoy** — no hay forma de invocarla sin reescribir `extraer_pdf` para que exponga los bloques con coordenadas antes de aplanarlos a texto.

**Arreglo propuesto:** no lo propongo todavía — depende de una decisión de diseño (¿`extraer_pdf` debe devolver algo más que `str` para que `limpieza.py` pueda operar, o `quitar_repetidos` debe integrarse dentro de `extraer_pdf` antes de que se pierdan las coordenadas?). Ver Pregunta 3.

### I4. `README.md` describe una estructura de `src/` que no es la actual — parcialmente, porque alguna vez sí lo fue

**Severidad: Bajo (no bloquea nada, es documentación). Arreglo: seguro** (es solo texto, no afecta código de nadie).

| README dice | Existe realmente |
|---|---|
| `src/extraction/` (carpeta) | `src/extractor.py` (archivo) |
| `src/encoders.py` | `src/encoding.py` |
| `src/indexado.py` | `src/indexing.py` |
| `src/recuperacion.py` | `src/retrieval.py` |

Dato que no estaba en el hallazgo original: **`src/extraction/` sí existió.** `git blame -L 540,551 src/extractor.py` muestra que hasta el commit `1a71e55` el archivo vivía en `src/extraction/extractor.py`, y se movió a `src/extractor.py` en el commit `760353e ("1ra version limpieza")`, sin que nadie actualizara el README. No es una descripción inventada desde el principio, es documentación que quedó desactualizada tras un rename real.

Además el README no menciona `src/config.py`, `src/fusion.py`, `src/aggregation.py` ni `src/bono_graph.py` (los cuatro existen en el repo).

**Arreglo propuesto:** corregir el README para que refleje los nombres reales (`extractor.py`, `encoding.py`, `indexing.py`, `retrieval.py`) y añadir una fila para `config.py`, `fusion.py`, `aggregation.py` y `bono_graph.py`. Es más barato que mover archivos, y no toca código — por eso "seguro". Pendiente de aprobar la convención de nombres (Pregunta 2) antes de tocar el README, para no corregirlo dos veces.

### I5. Nomenclatura mezclada español/inglés entre módulos

**Severidad: Bajo. Arreglo: coordinado** (renombrar rompe imports de otros).

`chunking`, `indexing`, `encoding`, `retrieval`, `fusion`, `aggregation`, `config` en inglés; `limpieza`, `evaluacion`, `extractor` en español. No propongo convención todavía — ver Pregunta 2, con pros/contras.

### I6. `construir_indice` muta los embeddings de quien la llama

**Severidad: Alto. Arreglo: seguro** (cambio localizado en `indexing.py`, no afecta la interfaz pública ni a quien ya la usa).

Reproducido con un caso mínimo (embeddings `float32` contiguos, como los que devuelve `sentence-transformers`):

```python
import numpy as np
from src.indexing import construir_indice
from src.chunking import Chunk

emb = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
copia = emb.copy()
construir_indice(emb, chunks)   # chunks: 2 Chunk mínimos, uno por fila
```

Salida real:
```
embeddings originales modificados?: True
emb ahora (normalizado in-place):
[[0.26726124 0.5345225  0.8017837 ]
 [0.45584232 0.5698029  0.6837635 ]]
copia original (antes de llamar):
[[1. 2. 3.]
 [4. 5. 6.]]
```

`validar_entradas` (`indexing.py:25`) hace `np.asarray(embeddings, dtype=np.float32)`: si el arreglo que entra ya es `float32`, `np.asarray` **no copia**, devuelve el mismo objeto (y `np.ascontiguousarray` en la línea siguiente tampoco copia si ya es contiguo, que es el caso típico de salida de un encoder). `faiss.normalize_L2(vectores)` en `construir_indice` (línea 70) opera in-place sobre ese mismo objeto, así que el arreglo que el caller pasó queda normalizado sin que lo pida.

Nótese que `tests/test_indexing.py` **no detecta este bug**: construye `embeddings = np.array([[0.1, 0.2, 0.3], ...])` sin especificar dtype (por defecto `float64`), así que ahí `np.asarray(..., dtype=np.float32)` sí copia, y el test pasa sin rozar el problema — es un ejemplo de cómo el propio test oculta el bug que debería poder atrapar.

**Por qué importa:** si quien llama a `construir_indice` reutiliza su arreglo `embeddings` después (para guardarlo en caché, para pasarlo a un segundo índice, para depurar), lo encuentra modificado sin ningún error ni aviso.

**Arreglo propuesto:** en `validar_entradas` (`indexing.py:25`), usar `np.array(embeddings, dtype=np.float32)` en vez de `np.asarray(...)` — `np.array` copia siempre — o documentar explícitamente en el docstring que la función muta su argumento, si se prefiere mantener por rendimiento con corpus grandes. Cuál de las dos es una decisión de una línea de docstring o una palabra de código; la dejo para aprobación porque toca una función que ya usa el test existente.

Hay además una redundancia relacionada, no un bug: `config.NORMALIZAR = True` (comentario: "indica normalizar al codificar") sugiere que la Fase 4 normalizaría los vectores al generarlos, y `construir_indice` normaliza otra vez, sin mirar `config.NORMALIZAR` en absoluto (`grep` de `NORMALIZAR` fuera de `config.py`: sin resultados). Normalizar dos veces no cambia el resultado matemáticamente. Pero como `encoding.py` todavía no existe, hoy es una responsabilidad que solo está cableada en un sitio (`indexing.py`) y declarada como intención en otro (`config.py`) sin que nada los conecte — ver Pregunta 5.

---

## 4. Incompletos

Todo lo que sigue está vacío o parcial. Los ordeno por dependencia (qué debe existir antes de qué), no por archivo.

1. **Arreglar I1 (imports)** — sin esto, nada que dependa de `evaluacion.py` o `extractor.py` desde fuera de `src/` es alcanzable, incluidos los pasos siguientes.

2. **Registrar los extractores/tituladores que faltan (I1)** — bloquea que el corpus real (JSON, XLSX, imágenes, PBF) entre al pipeline. Depende solo de decidir el punto 1 de las Preguntas está resuelto; el cambio en sí es mecánico.

3. **Decidir y cablear la Fase 1→2 (I3)** — antes de escribir `encoding.py`/`retrieval.py`, porque si `quitar_repetidos` se integra dentro de `extractor.py`, cambia qué produce Fase 1; si se deja como paso aparte, alguien tiene que orquestarlo.

4. **`src/encoding.py` (Fase 4, vacío)** — depende de `config.py` (ya tiene `ENCODER`, `ENCODER_DIM`, `PREFIJO_*`, `NORMALIZAR` listos y sin usar) y de `Chunk.texto_embed` (`chunking.py:272`, ya escrito y sin consumidores — confirmado con `grep -rn "texto_embed"`, solo aparece la definición). Debe: cargar el encoder de `config.ENCODER`, tokenizar respetando `config.ENCODER_MAX_TOKENS`, aplicar `config.PREFIJO_TEXTO`/`PREFIJO_CONSULTA`, y devolver vectores en el orden de entrada (contrato que espera `indexing.construir_indice`, que ya valida `len(chunks) == numero_vectores`).

5. **`src/retrieval.py` (Fase 6, vacío)** — depende de `encoding.py` (para vectorizar la consulta con el mismo encoder) y de `indexing.cargar_base` (ya escrito: devuelve `(index, metadata)`, `indexing.py:116`). Debe además resolver I2 (traducir `texto`→`text` al armar la respuesta) y aplicar `chunking.expandir_desde` (`chunking.py:227`, ya escrito, cero consumidores confirmados por grep) para respetar el límite de 250 palabras por fragmento de §9.2.1.

6. **`src/fusion.py` y `src/aggregation.py` (vacíos, no mencionados en el README)** — la spec da la respuesta exacta a "qué harían por su nombre": son dos etapas distintas de recuperación, no un duplicado.
   - **`fusion.py`** — combina los rankings de **múltiples índices FAISS** (uno por encoder, si el equipo usa más de uno) en una sola lista, sin LLM. La spec (§8.4) da tres estrategias nombradas: CombSUM, CombMNZ, Reciprocal Rank Fusion. Solo tiene sentido si el equipo termina usando >1 encoder; con uno solo, este módulo no hace nada.
   - **`aggregation.py`** — agrega los fragmentos recuperados de **un solo índice ya fusionado** al nivel de documento, para obtener los 3 `doc_id` finales. Spec §8.6: max pooling, suma o media de las puntuaciones de los fragmentos de cada `doc_id`.
   
   No se solapan: `fusion` opera *entre* índices (mismo fragmento, distintos encoders), `aggregation` opera *dentro* de un índice ya resuelto (fragmentos distintos, mismo `doc_id`). `fusion` es opcional (solo si hay >1 encoder); `aggregation` es obligatoria siempre (la spec exige devolver 3 documentos en toda consulta, §9.2).

7. **`src/bono_graph.py` (Fase bonus, vacío)** — spec §7: NER + extracción de relaciones + grafo con NetworkX/Neo4j/RDFLib, exportado como `grafo.graphml`. Prioridad más baja: es opcional (puntaje adicional) y depende de que Fases 1-6 funcionen primero, porque el grafo referencia `doc_id`/`chunk_id` de los chunks ya indexados.

8. **`scripts/` (vacío por completo, ni siquiera un `.gitkeep` con contenido)** — no existe nada que corra el pipeline de punta a punta. El mínimo, una vez resueltos los puntos 1-5: un script que recorra `data/raw/`, llame `extraer()` → `documento_a_chunks()` → encoder → `construir_indice()` → `guardar_base()`, con manejo de errores por documento (ver Sugerencia S2). Este es distinto de `generador.py`.

9. **`entrega/generador.py` (exigido por la spec, §1.4 punto 4, no existe en ningún lado del repo)** — no es lo mismo que el script de orquestación del punto 8. Según la spec: "Script Python que utilice el índice, lea el archivo de consultas y genere el archivo de resultados `resultados.jsonl`... Si no es posible reproducir los resultados, se excluirá de la evaluación." Es decir, es un entregable obligatorio y con una condición de descalificación explícita si no reproduce resultados. Verificado con `find`/`git ls-files`: no existe ni como esqueleto. Depende de que `retrieval.py` (punto 5) exista.

---

## 5. Sugerencias

**S1. `src/config.py` no lo importa nadie todavía.** Confirmado: `grep -rn "config" src/ scripts/ tests/ notebooks/` solo encuentra la palabra "configura" dentro de texto de prueba en `tests/fixtures/mini_corpus.json`, ninguna importación real. Cuando se escriba `encoding.py` debería ser el primer consumidor (`ENCODER`, `ENCODER_DIM`, `ENCODER_MAX_TOKENS`, `PREFIJO_*`, `NORMALIZAR` ya están definidos y listos, `config.py` es CRLF — ver S5).

**S2. Manejo de errores en la ingesta.** `extraer()` lanza `DocumentoSinTexto` (heredada de `Exception`) para cualquier fallo — formato sin registrar, PDF protegido, CSV vacío, JSON mal formado, etc. Hoy no existe ningún script que la capture: si se escribe un bucle ingiriendo cientos de documentos sin un `try/except DocumentoSinTexto` por documento, el primer documento problemático tumba la corrida completa. Con un corpus de cientos de archivos de fuentes heterogéneas (16 observatorios solo para JSON, según el propio docstring de `extraer_json`), algún documento va a fallar. Sugerencia: el script de orquestación (Incompleto 8) debe capturar `DocumentoSinTexto` por documento, loguearlo con su causa (el mensaje ya viene descriptivo, p. ej. "3 páginas, 2 palabras extraídas. ¿Escaneado con OCR fallido?") y seguir con el siguiente, en vez de abortar.

**S3. Cobertura de portugués.** Confirmado en `chunking.py:22`: `IDIOMAS_PYSBD = {"es": "es", "pt": "es", "en": "en"}`, documentado explícitamente ("pysbd NO soporta portugués"). El resto del código sí trata portugués aparte donde importa: `PACK_TESSERACT` en `extractor.py:22` tiene `"pt": "por"` (Tesseract sí soporta portugués), `_CLAVES_CUERPO`/`_CLAVES_RUIDO` en `extractor.py:314-322` incluyen variantes portuguesas (`descrição`, `resumo`), y `easyocr` se inicializa con `["es", "pt", "en"]` (`extractor.py:439`). No encontré otro sitio donde portugués se trate distinto o se olvide — el único punto de mezcla deliberada es el de `pysbd`, y está comentado.

**S4. Reproducibilidad.** No encontré rutas absolutas ni dependencias del directorio de ejecución en `src/` (`grep` de `C:\`, `/home/`, `/Users/`, `os.getcwd`, `os.chdir`: sin resultados). El único problema de "depende de dónde se ejecuta" es el de I1 (imports) y el de `tests/test_indexing.py` (ver hallazgo F original, confirmado abajo).

**S5. Higiene de git — CRLF y espacios finales.** Verificado con `file` y `grep -P ' +$'` sobre los archivos trackeados:
- **CRLF** (en vez de LF) en 2 archivos: `requirements.txt` y `src/config.py`.
- **Líneas con espacios finales**: `src/chunking.py` (30 líneas), `src/extractor.py` (69 líneas), `src/limpieza.py` (29 líneas). `evaluacion.py`, `indexing.py` y `config.py` están limpios de esto (aunque `config.py` tiene el problema de CRLF).
- `.gitignore` cubre lo necesario: `__pycache__/`, `.venv/`, `data/*` (con excepciones a los `.gitkeep`), `entrega/base_vectorial/`, `corpus/`, `.ipynb_checkpoints/`. Verificado con `git ls-files`: no hay `.pyc` ni artefactos versionados por error.
- El notebook `benchmark_encoders.ipynb` **sí guarda outputs** (11 de 13 celdas de código tienen `outputs` no vacíos) — coherente con la instrucción de no tocarlo porque ya está corrido y es evidencia.

**S6. Confirmado: `tests/test_indexing.py` no es un test de pytest, con reproducción de los tres problemas.**
- No hay ninguna función `test_*`: `ast.walk` sobre el archivo no encuentra ninguna `FunctionDef` — son asserts sueltos a nivel de módulo.
- Escribe efectivamente en `entrega/base_vectorial/encoder_prueba/` — reproducido: tras `python3 -m tests.test_indexing` desde la raíz, aparecen `entrega/base_vectorial/encoder_prueba/{index.faiss,metadata.jsonl}` (los borré después de confirmarlo; `entrega/base_vectorial/` está en `.gitignore`, así que no se cuela en un commit, pero si alguien empaqueta `entrega/` a mano para la entrega sin fijarse, el ruido queda ahí).
- Ruta relativa dependiente del cwd: `python3 tests/test_indexing.py` (parado en la raíz) falla con `ModuleNotFoundError: No module named 'src'` porque el intérprete solo añade el directorio del script (`tests/`) al `sys.path`, no la raíz. Solo funciona como `python3 -m tests.test_indexing` desde la raíz.
- `pytest` no está en `requirements.txt` (confirmado, `pip show pytest`: not found).

**S7. `openpyxl` falta en `requirements.txt`.** `extraer_xlsx` (`extractor.py:264`) hace `import openpyxl`, pero no está listado. Es el único paquete usado en `src/` y ausente de la lista (confirmado comparando todos los `import`/`from` dentro de funciones de `extractor.py`, `chunking.py`, `indexing.py` contra las 16 líneas de `requirements.txt`). `pysbd`, `numpy`, `pandas`, `beautifulsoup4`+`lxml`, `pytesseract`, `pillow`, `osmium`, `easyocr`, `py3langid`, `pymupdf` (import `fitz`), `faiss-cpu` (import `faiss`) sí están todos listados y sí se usan. `easyocr` y `pytesseract` **se usan los dos**, no es redundancia: `extraer_imagen` acepta un parámetro `motor` ("tesseract" por defecto, "easyocr" alternativo), documentado con una comparación de CER medida a mano en el docstring.
`sentence-transformers`, `transformers`, `torch` están listados y se usan en `notebooks/benchmark_encoders.ipynb`, pero todavía no en `src/` (porque `encoding.py` está vacío) — es esperable, no un problema. Ninguna versión está fijada (`==`) en todo el archivo, pese a que dos módulos (`chunking.py:49`, `chunking.py:72`) mencionan textualmente "la versión pinneada de requirements.txt" en sus mensajes de error — esa versión pinneada no existe hoy.

---

## 6. Preguntas

**P1. Convención de ejecución/import del paquete `src/`.** Hay tres estilos conviviendo (absoluto sin paquete, relativo, absoluto con paquete desde la raíz). Antes de tocar I1 hace falta decidir: ¿se ejecuta siempre como paquete desde la raíz (`python -m scripts.pipeline`, imports relativos `from .chunking import ...` en todos los módulos de `src/`) o se instala `src/` en modo editable (`pip install -e .`, lo que pediría un `pyproject.toml`/`setup.py` que hoy no existe)? Cualquiera de las dos resuelve I1, pero cambian cómo todo el equipo corre sus scripts.

**P2. Convención de nombres de archivo.** ¿Todo en español (`indexado.py`, `recuperacion.py`, `codificacion.py`) para ser consistente con `limpieza.py`/`evaluacion.py`, o todo en inglés (`cleaning.py`, `evaluation.py`, `extraction.py`) para ser consistente con `chunking.py`/`indexing.py`/`config.py`? Hay un archivo vacío `src/cleaning.py` (0 bytes) que ya sugiere que alguien empezó a moverse hacia inglés y no terminó — o que fue un archivo creado por error al lado de `limpieza.py`. Sea cual sea la convención, faltaría decidir qué hacer con el duplicado vacío (no lo borro sin aprobación, regla 3).

**P3. ¿Quién es responsable de invocar `quitar_repetidos`/`bloques_por_pagina` (I3)?** Tres opciones con costos distintos: (a) `extraer_pdf` se reescribe para exponer los bloques con coordenadas y llamar a `limpieza.py` internamente antes de aplanar a texto — cambia la firma interna de Fase 1; (b) se agrega una función nueva en `limpieza.py` que reciba directamente el `path` del PDF, abra el documento con `fitz` una segunda vez y aplique la limpieza como paso independiente después de `extraer()` — duplica la apertura del PDF; (c) se acepta que, para PDFs, la limpieza de cabeceras/pies no se aplica y solo queda `normalizar()` — hoy es de facto la opción que está corriendo, pero no está decidida, solo no está implementada la alternativa.

**P4. ¿Los campos internos (`Chunk`, `metadata.jsonl`) se quedan en español permanentemente?** La spec ya responde esto por partida doble (I2): sí, para `metadata.jsonl` (Tabla 1, campo `texto`); no, para `resultados.jsonl` (Tabla 2, campo `text`). La pregunta real ya no es "español o inglés" sino **quién escribe la traducción** — ¿vive dentro de `retrieval.py` (como parte de construir la respuesta) o dentro de `generador.py` (como paso final antes de escribir el JSON Lines de entrega)? Ninguno de los dos existe todavía, así que es una decisión libre hoy, pero hay que tomarla antes de escribir cualquiera de los dos.

**P5. ¿Dónde vive la normalización L2, `encoding.py` o `indexing.py`?** (contexto completo en I6) `config.NORMALIZAR = True` sugiere que Fase 4 normalizaría; `indexing.construir_indice` normaliza otra vez, sin mirar `config.NORMALIZAR`. Hoy no hay contradicción porque `encoding.py` no existe, pero si se implementa sin normalizar (leyendo y respetando `config.NORMALIZAR`) y alguien simplifica `indexing.py` asumiendo que ya no hace falta normalizar ahí "porque es responsabilidad de Fase 4", el índice quedaría con vectores sin normalizar y el producto interno de `IndexFlatIP` dejaría de equivaler a similitud coseno **sin ningún error visible**. Sugiero fijar la responsabilidad en un solo sitio antes de escribir `encoding.py`.
