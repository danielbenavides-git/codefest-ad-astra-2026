#!/usr/bin/env python3
"""Arma la carpeta de entrega y comprueba que corra por sí sola.

`generador.py` importa `src/`. Si la organización recibe solo la carpeta
`entrega/`, el script no arranca y el equipo queda fuera de la evaluación
(§1.4). Este script copia `src/` dentro de la entrega y verifica el arranque
en un proceso aparte, que es la única forma de saber que funciona.

Lo que produce:

    dist/entrega_talon_systems/
        generador.py
        resultados.jsonl
        informe_tecnico.pdf
        requirements.txt
        base_vectorial/encoder_<nombre>/
            index.faiss
            metadata.jsonl
            grafo/grafo.graphml      (si se construyó el bonus)
        src/*.py

Uso:

    python scripts/empaquetar.py                # entrega completa
    python scripts/empaquetar.py --sin-base     # rápido, para probar
    python scripts/empaquetar.py --zip          # además comprime

Si falta un entregable obligatorio, se detiene. Con `--forzar` continúa y
solo avisa, que sirve mientras el corpus todavía no está indexado.
"""

import argparse
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import config

NOMBRE_ENTREGA = "entrega_talon_systems"
DESTINO_POR_DEFECTO = RAIZ / "dist" / NOMBRE_ENTREGA

# Los cuatro entregables de §1.4. `generador.py` está aparte porque no se
# copia desde la misma carpeta que los demás.
OBLIGATORIOS = ("resultados.jsonl", "informe_tecnico.pdf")


def carpeta_base() -> str:
    return f"encoder_{config.ENCODER.split('/')[-1]}"


def revisar_entregables(entrega: Path, con_base: bool) -> list[str]:
    """Devuelve la lista de entregables que faltan, con su ruta esperada."""
    faltantes = []

    if not (entrega / "generador.py").exists():
        faltantes.append(str(entrega / "generador.py"))

    for nombre in OBLIGATORIOS:
        if not (entrega / nombre).exists():
            faltantes.append(str(entrega / nombre))

    if con_base:
        base = entrega / "base_vectorial" / carpeta_base()
        for nombre in ("index.faiss", "metadata.jsonl"):
            if not (base / nombre).exists():
                faltantes.append(str(base / nombre))

    return faltantes


def copiar_src(destino: Path) -> int:
    """Copia los .py de `src/`. Devuelve cuántos archivos copió.

    Solo el código: nada de __pycache__ ni de archivos sueltos que hayan
    quedado en la carpeta durante el desarrollo.
    """
    origen = RAIZ / "src"
    carpeta = destino / "src"
    carpeta.mkdir(parents=True, exist_ok=True)

    copiados = 0
    for archivo in sorted(origen.glob("*.py")):
        shutil.copy2(archivo, carpeta / archivo.name)
        copiados += 1

    if not (carpeta / "__init__.py").exists():
        # Sin esto, `from src import config` falla en la máquina del jurado.
        (carpeta / "__init__.py").write_text("", encoding="utf-8")
        copiados += 1

    return copiados


def copiar_entrega(destino: Path, con_base: bool) -> list[str]:
    """Copia lo que haya en `entrega/`. Devuelve los nombres copiados."""
    origen = RAIZ / "entrega"
    copiados = []

    for archivo in sorted(origen.iterdir()):
        if archivo.name == "base_vectorial":
            continue
        if archivo.is_file():
            shutil.copy2(archivo, destino / archivo.name)
            copiados.append(archivo.name)

    base = origen / "base_vectorial"
    if con_base and base.is_dir():
        shutil.copytree(base, destino / "base_vectorial", dirs_exist_ok=True)
        copiados.append("base_vectorial/")

    requisitos = RAIZ / "requirements.txt"
    if requisitos.exists():
        shutil.copy2(requisitos, destino / "requirements.txt")
        copiados.append("requirements.txt")

    return copiados


def verificar_imports(destino: Path) -> tuple[bool, str]:
    """Corre `generador.py --ayuda-imports` en un proceso limpio.

    Se lanza desde otro directorio de trabajo a propósito: si se corriera
    desde la raíz del repositorio, Python encontraría el `src/` de
    desarrollo y la prueba no diría nada sobre la carpeta empaquetada.
    """
    guion = destino / "generador.py"
    if not guion.exists():
        return False, "no hay generador.py que verificar"

    proceso = subprocess.run(
        [sys.executable, str(guion), "--ayuda-imports"],
        cwd=destino.parent,
        capture_output=True,
        text=True,
        timeout=300,
    )

    salida = (proceso.stdout + proceso.stderr).strip()
    return proceso.returncode == 0, salida


def limpiar_pycache(destino: Path) -> None:
    """Borra los __pycache__ que deja la verificación al importar src."""
    for carpeta in sorted(destino.rglob("__pycache__"), reverse=True):
        shutil.rmtree(carpeta, ignore_errors=True)


def comprimir(destino: Path) -> Path:
    """Comprime la carpeta en un .zip hermano."""
    archivo = destino.with_suffix(".zip")

    with zipfile.ZipFile(archivo, "w", zipfile.ZIP_DEFLATED) as zf:
        for ruta in sorted(destino.rglob("*")):
            if ruta.is_file():
                zf.write(ruta, Path(destino.name) / ruta.relative_to(destino))

    return archivo


def tamano_legible(destino: Path) -> str:
    total = sum(f.stat().st_size for f in destino.rglob("*") if f.is_file())
    for unidad in ("B", "KB", "MB", "GB"):
        if total < 1024:
            return f"{total:.0f} {unidad}"
        total /= 1024
    return f"{total:.1f} TB"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Arma la carpeta de entrega.")
    parser.add_argument("--destino", type=Path, default=DESTINO_POR_DEFECTO)
    parser.add_argument("--sin-base", action="store_true",
                        help="No copia la base vectorial (pesa y tarda).")
    parser.add_argument("--zip", action="store_true", dest="comprimir")
    parser.add_argument("--forzar", action="store_true",
                        help="Empaqueta aunque falten entregables.")
    args = parser.parse_args(argv)

    con_base = not args.sin_base
    inicio = time.perf_counter()

    if args.destino.exists():
        shutil.rmtree(args.destino)
    args.destino.mkdir(parents=True)

    print(f"Destino: {args.destino}")
    print()

    copiados = copiar_entrega(args.destino, con_base)
    print(f"1. Entregables: {', '.join(copiados) or 'ninguno'}")

    modulos = copiar_src(args.destino)
    print(f"2. Código: {modulos} módulos de src/")

    faltantes = revisar_entregables(args.destino, con_base)
    if faltantes:
        print()
        print("Faltan entregables obligatorios:")
        for ruta in faltantes:
            print(f"  - {ruta}")
        if not args.forzar:
            print()
            print("Corre scripts/indexar.py y entrega/generador.py, o usa --forzar.",
                  file=sys.stderr)
            return 1
        print("  (--forzar: se continúa igual)")

    print()
    print("3. Verificando que generador.py arranque solo...")
    bien, salida = verificar_imports(args.destino)
    if salida:
        print(f"   {salida.splitlines()[0]}")

    if not bien:
        print()
        print("generador.py no arranca desde la carpeta empaquetada. La entrega "
              "no es reproducible: revisa qué import falla.", file=sys.stderr)
        return 1

    limpiar_pycache(destino=args.destino)

    if args.comprimir:
        archivo = comprimir(args.destino)
        print(f"4. Comprimido: {archivo}")

    print()
    print(f"Listo en {time.perf_counter() - inicio:.1f} s. "
          f"{tamano_legible(args.destino)} en {args.destino}")

    if faltantes:
        print("Ojo: la entrega está incompleta (ver arriba).")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
