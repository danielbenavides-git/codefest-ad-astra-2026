"""Parámetros compartidos del pipeline. Único sitio donde se define el encoder."""

ENCODER = "ibm-granite/granite-embedding-97m-multilingual-r2"
ENCODER_DIM = 384          # dimensión del índice FAISS
ENCODER_MAX_TOKENS = 32768

# granite no usa prefijos. Si algún día se cambia a e5, acá van
# "query: " y "passage: " y el resto del código no se toca.
PREFIJO_CONSULTA = ""
PREFIJO_TEXTO = ""

# El índice usa producto interno; con vectores normalizados eso ES el coseno.
# Si esto queda en False, la búsqueda deja de ser coseno sin avisar.
#
# La normalización es responsabilidad exclusiva de la Fase 4 (encoding.py),
# que debe leer este flag y normalizar antes de que los vectores lleguen a
# construir_indice() (indexing.py). indexing.py NO normaliza: solo verifica
# que las normas ya sean ~1 y falla con ValueError si no (D5). El motivo de
# no partir la responsabilidad entre dos módulos es que retrieval.py tiene
# que aplicar la misma normalización al vector de consulta de todas formas,
# así que encoding.py es el único sitio donde hace falta escribir la regla.
NORMALIZAR = True