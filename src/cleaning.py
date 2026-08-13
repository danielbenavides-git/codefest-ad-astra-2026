import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence
 
#: Franjas de la página donde viven cabeceras y pies, como fracción de altura.
ZONA_CABECERA = 0.10
ZONA_PIE = 0.88
 
UMBRAL_REPETICION = 0.6      # fracción de páginas en que debe aparecer la línea
MIN_PAGINAS = 4              # con menos, la estadística no distingue nada
MAX_PALABRAS_LINEA = 12      # una línea larga es contenido aunque se repita
MAX_FRACCION_BORRADA = 0.25  # tope de seguridad: por encima, no se toca nada
 
_DIGITOS = re.compile(r"\d+")
 
#: Línea que solo contiene un número de página. Se elimina siempre: no puede
#: confundirse con contenido porque la línea entera es eso y nada más.
_SOLO_NUMERO = re.compile(
    r"^\s*(?:p[áa]g(?:ina)?\.?\s*|page\s+)?\d{1,4}(?:\s*(?:de|of|/)\s*\d{1,4})?\s*$",
    re.IGNORECASE,
)
 
 
@dataclass
class InformeLimpieza:
    """Qué se eliminó y por qué. Se revisa antes de confiar en el filtro."""
 
    lineas_eliminadas: list[tuple[str, int]] = field(default_factory=list)
    palabras_antes: int = 0
    palabras_despues: int = 0
    nota: str | None = None
 
    @property
    def fraccion_borrada(self) -> float:
        if not self.palabras_antes:
            return 0.0
        return 1 - self.palabras_despues / self.palabras_antes
 
    def __str__(self) -> str:
        if self.nota:
            return f"sin filtrar: {self.nota}"
        if not self.lineas_eliminadas:
            return "sin repeticiones detectadas"
        detalle = ", ".join(f"{t!r}×{n}" for t, n in self.lineas_eliminadas[:4])
        extra = f" (+{len(self.lineas_eliminadas) - 4} más)" if len(self.lineas_eliminadas) > 4 else ""
        return (f"-{self.fraccion_borrada:.0%} ({self.palabras_antes}→"
                f"{self.palabras_despues} palabras): {detalle}{extra}")
 
 
def _clave(linea: str) -> str:
    """Normaliza para comparar. Los dígitos varían de página a página.
 
    'Informe 2024 — página 3' y '... página 4' son la misma cabecera; sin
    sustituir los números nunca coincidirían.
    """
    return _DIGITOS.sub("#", linea.strip().lower())
 
 
def bloques_por_pagina(doc) -> list[list[tuple[str, float]]]:
    """Extrae `(texto, y_relativa)` por bloque. `doc` es un `fitz.Document`.
 
    La `y` relativa (0 arriba, 1 abajo) es lo que permite distinguir una
    cabecera de la primera línea del cuerpo.
    """
    paginas = []
    for pagina in doc:
        alto = pagina.rect.height or 1
        bloques = [(b[4].strip(), b[1] / alto)
                   for b in pagina.get_text("blocks", sort=True)
                   if isinstance(b[4], str) and b[4].strip()]
        paginas.append(bloques)
    return paginas
 
 
def quitar_repetidos(
    paginas: Sequence[Sequence[tuple[str, float]]],
    umbral: float = UMBRAL_REPETICION,
    max_fraccion: float = MAX_FRACCION_BORRADA,
) -> tuple[list[str], InformeLimpieza]:
    """Elimina cabeceras y pies repetidos. Devuelve `(paginas, informe)`.
 
    Solo son candidatas las líneas que están en la zona de cabecera o de pie
    Y son cortas Y se repiten en la mayoría de las páginas. Las tres
    condiciones a la vez: cualquiera por separado borra contenido.
    """
    informe = InformeLimpieza()
    informe.palabras_antes = sum(len(t.split()) for pg in paginas for t, _ in pg)
 
    utiles = [pg for pg in paginas if pg]
    if len(utiles) < MIN_PAGINAS:
        informe.nota = f"{len(utiles)} páginas con texto, se necesitan {MIN_PAGINAS}"
        return _unir(paginas, set(), informe)
 
    conteo: Counter = Counter()
    for pg in utiles:
        candidatas = {
            _clave(t) for t, y in pg
            if (y <= ZONA_CABECERA or y >= ZONA_PIE) and len(t.split()) <= MAX_PALABRAS_LINEA
        }
        for k in candidatas:
            conteo[k] += 1
 
    minimo = max(3, int(len(utiles) * umbral))
    repetidas = {k for k, c in conteo.items() if c >= minimo}
 
    resultado, informe = _unir(paginas, repetidas, informe)
    informe.lineas_eliminadas = sorted(((k, conteo[k]) for k in repetidas), key=lambda kv: -kv[1])
 
    # Un filtro que se lleva un cuarto del documento está equivocado, por muy
    # convincente que sea la estadística.
    if informe.fraccion_borrada > max_fraccion:
        informe.nota = (f"habría borrado el {informe.fraccion_borrada:.0%} "
                        f"(tope {max_fraccion:.0%}); documento intacto")
        informe.lineas_eliminadas = []
        return _unir(paginas, set(), informe)
 
    return resultado, informe
 
 
def _unir(paginas, repetidas: set, informe: InformeLimpieza) -> tuple[list[str], InformeLimpieza]:
    salida = []
    for pg in paginas:
        lineas = [
            t for t, y in pg
            if not ((y <= ZONA_CABECERA or y >= ZONA_PIE) and _clave(t) in repetidas)
            and not _SOLO_NUMERO.match(t)
        ]
        salida.append("\n".join(lineas))
    informe.palabras_despues = sum(len(p.split()) for p in salida)
    return salida, informe
 
 
def quitar_numeros_sueltos(texto: str) -> str:
    """Para texto sin coordenadas (páginas OCR). Lo único seguro sin posición."""
    return "\n".join(l for l in texto.split("\n") if not _SOLO_NUMERO.match(l))
 
