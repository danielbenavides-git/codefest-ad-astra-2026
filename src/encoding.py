"""Fase 4 - convierte texto en vectores.

Único sitio del pipeline donde se carga el encoder y donde se normalizan
los vectores. `indexing.py` solo verifica la norma (D5) y `retrieval.py`
recibe el vector de consulta ya hecho, así que la regla vive acá y en
ningún otro lado.

Dos rutas separadas a propósito:

- `codificar_chunks` aplica `config.PREFIJO_TEXTO` sobre `Chunk.texto_embed`.
- `codificar_consulta` aplica `config.PREFIJO_CONSULTA`.

Con granite los dos prefijos son cadena vacía y las rutas hacen lo mismo.
La separación existe para que cambiar a e5 sea cambiar `config.py` y nada
más: si estuvieran fusionadas, la consulta se codificaría como si fuera un
pasaje y la búsqueda se degradaría sin lanzar ningún error.
"""

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

from src import config
from src.chunking import Chunk

# Misma tolerancia que indexing.py y retrieval.py: si acá se normaliza más
# flojo que allá, el índice rechaza vectores que este módulo dio por buenos.
TOLERANCIA_NORMA = 1e-3

# En CPU los lotes grandes no aceleran: la ganancia se la come la memoria.
# 8 es el valor con el que se midieron las 4,6 horas del benchmark.
LOTE_POR_DEFECTO = 8

NOMBRE_EMBEDDINGS = "embeddings.npz"


@lru_cache(maxsize=1)
def cargar_modelo(nombre: str = config.ENCODER):
    """Carga el encoder una sola vez por proceso.

    El caché no es un lujo: `generador.py` codifica 50 consultas y sin él
    serían 50 cargas del modelo desde disco.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(nombre, device="cpu")


def codificar_textos(
    textos: Sequence[str],
    *,
    prefijo: str = "",
    modelo=None,
    lote: int = LOTE_POR_DEFECTO,
    progreso: bool = False,
) -> np.ndarray:
    """Lista de textos -> matriz (n, ENCODER_DIM) float32, normalizada.

    La fila `i` corresponde siempre al texto `i`. `modelo` permite inyectar
    un doble en los tests para no descargar el encoder real.
    """
    if not config.NORMALIZAR:
        raise ValueError(
            "config.NORMALIZAR está en False. El pipeline solo soporta "
            "coseno: indexing.py construye un IndexFlatIP y verifica que las "
            "normas sean 1, así que sin normalizar el índice falla igual, "
            "unos pasos más tarde. Si de verdad se quiere distancia "
            "euclídea, hay que cambiar también el tipo de índice."
        )

    if not textos:
        raise ValueError("No se recibió ningún texto para codificar.")

    for posicion, texto in enumerate(textos):
        if not isinstance(texto, str) or not texto.strip():
            raise ValueError(
                f"El texto en la posición {posicion} está vacío o no es una "
                f"cadena. Un texto vacío produce un vector sin significado "
                f"que igual entraría al índice."
            )

    if lote <= 0:
        raise ValueError("El tamaño de lote debe ser mayor que cero.")

    if modelo is None:
        modelo = cargar_modelo()

    entradas = [prefijo + texto for texto in textos]

    crudos = modelo.encode(
        entradas,
        batch_size=lote,
        show_progress_bar=progreso,
        convert_to_numpy=True,
    )

    vectores = np.array(crudos, dtype=np.float32, copy=True)

    if vectores.ndim != 2:
        raise ValueError(
            f"El encoder devolvió una matriz de forma {vectores.shape}; "
            f"se esperaba 2D."
        )

    if vectores.shape[0] != len(textos):
        raise ValueError(
            f"El encoder devolvió {vectores.shape[0]} vectores para "
            f"{len(textos)} textos. Sin correspondencia uno a uno, cada "
            f"resultado de búsqueda apuntaría al fragmento equivocado."
        )

    if vectores.shape[1] != config.ENCODER_DIM:
        raise ValueError(
            f"El encoder devolvió vectores de dimensión {vectores.shape[1]} "
            f"y config.ENCODER_DIM dice {config.ENCODER_DIM}. Revisa que "
            f"config.ENCODER y config.ENCODER_DIM sean del mismo modelo."
        )

    if not np.isfinite(vectores).all():
        raise ValueError("El encoder devolvió valores NaN o infinitos.")

    return _normalizar(vectores)


def _normalizar(vectores: np.ndarray) -> np.ndarray:
    """Lleva cada fila a norma 1.

    Se hace acá en numpy y no con `normalize_embeddings=True` de
    sentence-transformers para que la regla no dependa de la librería: el
    día que se cambie de librería, esto sigue siendo verdad.
    """
    normas = np.linalg.norm(vectores, axis=1, keepdims=True)

    if np.any(normas < TOLERANCIA_NORMA):
        fila = int(np.argmin(normas))
        raise ValueError(
            f"La fila {fila} tiene norma {float(normas[fila][0]):.6f}: no se "
            f"puede normalizar un vector nulo. Suele ser texto que quedó "
            f"vacío después de tokenizar."
        )

    return np.ascontiguousarray(vectores / normas, dtype=np.float32)


def codificar_chunks(
    chunks: Sequence[Chunk],
    *,
    modelo=None,
    lote: int = LOTE_POR_DEFECTO,
    progreso: bool = True,
) -> tuple[np.ndarray, list[Chunk]]:
    """Chunks -> (embeddings, chunks) para pasarle a `indexing`.

    Devuelve el par y no solo la matriz a propósito: la fila `i` vale
    únicamente junto al chunk `i`. Si el script de indexación filtrara u
    ordenara la lista después de codificar, cada búsqueda devolvería el
    texto de otro fragmento y no saltaría ningún error. Devolviendo los dos
    juntos, separarlos cuesta trabajo.

    Usa `Chunk.texto_embed`, que antepone el título del documento cuando
    existe; `Chunk.texto` se guarda en la metadata sin tocar.
    """
    if not chunks:
        raise ValueError("No se recibió ningún chunk para codificar.")

    lista = list(chunks)

    ids = [chunk.chunk_id for chunk in lista]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Hay chunk_id duplicados. indexing.validar_entradas los rechaza "
            "igual, pero acá se detecta antes de gastar horas de CPU."
        )

    embeddings = codificar_textos(
        [chunk.texto_embed for chunk in lista],
        prefijo=config.PREFIJO_TEXTO,
        modelo=modelo,
        lote=lote,
        progreso=progreso,
    )

    return embeddings, lista


def codificar_consulta(texto: str, *, modelo=None) -> np.ndarray:
    """Una consulta -> un vector 1D listo para `retrieval.buscar_vector`."""
    matriz = codificar_textos(
        [texto],
        prefijo=config.PREFIJO_CONSULTA,
        modelo=modelo,
        lote=1,
        progreso=False,
    )

    return matriz[0]


def guardar_embeddings(
    embeddings: np.ndarray,
    chunks: Sequence[Chunk],
    directorio: str | Path,
) -> Path:
    """Guarda los vectores junto a los chunk_id que les corresponden.

    Codificar el corpus toma horas en CPU. Con esto se puede reconstruir el
    índice, probar otro tipo de índice o depurar sin volver a codificar.

    Los chunk_id van en el mismo archivo para que `cargar_embeddings` pueda
    comprobar que los vectores guardados son de estos chunks y no de una
    corrida anterior con otro corpus.
    """
    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            f"{embeddings.shape[0]} vectores para {len(chunks)} chunks."
        )

    ruta = Path(directorio)
    ruta.mkdir(parents=True, exist_ok=True)
    archivo = ruta / NOMBRE_EMBEDDINGS

    np.savez(
        archivo,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        chunk_ids=np.array([c.chunk_id for c in chunks], dtype=object),
        encoder=np.array(config.ENCODER),
    )

    return archivo


def cargar_embeddings(
    directorio: str | Path,
    chunks: Sequence[Chunk],
) -> np.ndarray:
    """Recupera los vectores guardados, comprobando que sean de estos chunks."""
    archivo = Path(directorio) / NOMBRE_EMBEDDINGS

    if not archivo.exists():
        raise FileNotFoundError(f"No existe el archivo de vectores: {archivo}")

    datos = np.load(archivo, allow_pickle=True)

    guardados = [str(x) for x in datos["chunk_ids"]]
    esperados = [chunk.chunk_id for chunk in chunks]

    if guardados != esperados:
        raise ValueError(
            f"Los vectores guardados no corresponden a estos chunks "
            f"({len(guardados)} guardados, {len(esperados)} esperados, o "
            f"cambió el orden). Vuelve a codificar en vez de reutilizarlos: "
            f"un desajuste acá no produce ningún error visible, solo "
            f"resultados equivocados."
        )

    encoder_guardado = str(datos["encoder"])
    if encoder_guardado != config.ENCODER:
        raise ValueError(
            f"Los vectores se generaron con {encoder_guardado!r} y "
            f"config.ENCODER dice {config.ENCODER!r}."
        )

    return np.ascontiguousarray(datos["embeddings"], dtype=np.float32)