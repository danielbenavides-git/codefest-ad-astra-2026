import importlib.util
import zipfile
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent


def _cargar(nombre: str, ruta: Path):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


empaquetar = _cargar("empaquetar", RAIZ / "scripts" / "empaquetar.py")


@pytest.fixture
def entrega_falsa(tmp_path):
    """Carpeta empaquetada con todos los entregables, sin correr el pipeline."""
    destino = tmp_path / "entrega"
    base = destino / "base_vectorial" / empaquetar.carpeta_base()
    base.mkdir(parents=True)

    (destino / "generador.py").write_text("print('ok')", encoding="utf-8")
    (destino / "resultados.jsonl").write_text("{}\n", encoding="utf-8")
    (destino / "informe_tecnico.pdf").write_bytes(b"%PDF-1.4")
    (base / "index.faiss").write_bytes(b"faiss")
    (base / "metadata.jsonl").write_text("{}\n", encoding="utf-8")

    return destino


def test_no_falta_nada_cuando_esta_todo(entrega_falsa):
    assert empaquetar.revisar_entregables(entrega_falsa, con_base=True) == []


def test_detecta_el_informe_faltante(entrega_falsa):
    (entrega_falsa / "informe_tecnico.pdf").unlink()

    faltantes = empaquetar.revisar_entregables(entrega_falsa, con_base=True)

    assert len(faltantes) == 1
    assert faltantes[0].endswith("informe_tecnico.pdf")


def test_detecta_el_indice_faltante(entrega_falsa):
    base = entrega_falsa / "base_vectorial" / empaquetar.carpeta_base()
    base.rmdir() if not any(base.iterdir()) else [f.unlink() for f in base.iterdir()]

    faltantes = empaquetar.revisar_entregables(entrega_falsa, con_base=True)

    assert any(f.endswith("index.faiss") for f in faltantes)
    assert any(f.endswith("metadata.jsonl") for f in faltantes)


def test_sin_base_no_exige_el_indice(entrega_falsa):
    for archivo in (entrega_falsa / "base_vectorial").rglob("*"):
        if archivo.is_file():
            archivo.unlink()

    assert empaquetar.revisar_entregables(entrega_falsa, con_base=False) == []


def test_la_carpeta_de_la_base_lleva_el_nombre_del_encoder():
    from src import config

    assert empaquetar.carpeta_base() == f"encoder_{config.ENCODER.split('/')[-1]}"


def test_copia_todos_los_modulos_de_src(tmp_path):
    copiados = empaquetar.copiar_src(tmp_path)

    esperados = len(list((RAIZ / "src").glob("*.py")))
    assert copiados == esperados
    assert (tmp_path / "src" / "encoding.py").exists()
    assert (tmp_path / "src" / "__init__.py").exists()


def test_no_copia_pycache(tmp_path):
    empaquetar.copiar_src(tmp_path)

    assert not (tmp_path / "src" / "__pycache__").exists()


def test_comprime_con_la_carpeta_dentro(entrega_falsa):
    archivo = empaquetar.comprimir(entrega_falsa)

    with zipfile.ZipFile(archivo) as zf:
        nombres = zf.namelist()

    assert all(n.startswith("entrega/") for n in nombres)
    assert "entrega/resultados.jsonl" in nombres


# Integración: empaqueta el repositorio de verdad.


def test_se_detiene_si_faltan_entregables(tmp_path):
    codigo = empaquetar.main(["--destino", str(tmp_path / "salida"), "--sin-base"])

    assert codigo == 1


def test_forzar_empaqueta_igual_y_generador_arranca_solo(tmp_path):
    destino = tmp_path / "salida"

    codigo = empaquetar.main(
        ["--destino", str(destino), "--sin-base", "--forzar"]
    )

    # Devuelve 1 porque la entrega está incompleta, pero lo importante es que
    # generador.py arrancó desde la carpeta empaquetada: si el import fallara,
    # main habría salido antes de copiar nada.
    assert codigo == 1
    assert (destino / "src" / "config.py").exists()
    assert (destino / "generador.py").exists()
    assert not list(destino.rglob("__pycache__")), "la verificación dejó basura"

    bien, salida = empaquetar.verificar_imports(destino)
    assert bien, salida
    assert str(destino) in salida


def test_el_generador_empaquetado_no_usa_el_src_del_repo(tmp_path):
    destino = tmp_path / "salida"
    empaquetar.main(["--destino", str(destino), "--sin-base", "--forzar"])

    (destino / "src" / "config.py").unlink()

    bien, _ = empaquetar.verificar_imports(destino)

    assert not bien, "debería fallar: sin config.py la entrega no es autocontenida"
