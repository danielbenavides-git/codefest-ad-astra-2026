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
NORMALIZAR = True