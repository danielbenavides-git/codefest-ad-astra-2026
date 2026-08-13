import importlib.util
import json
from pathlib import Path

import pytest

from tests.test_encoding import ModeloFalso

RAIZ = Path(__file__).resolve().parent.parent


def _cargar(nombre: str, ruta: Path):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


indexar = _cargar("indexar", RAIZ / "scripts" / "indexar.py")
generador = _cargar("generador", RAIZ / "entrega" / "generador.py")


PARRAFO = (
    "La congestión de la órbita baja terrestre plantea riesgos crecientes "
    "para los operadores satelitales. Los desechos orbitales aumentan cada "
    "año y las maniobras de evasión se han vuelto rutinarias. "
)


@pytest.fixture
def corpus(tmp_path):
    """Corpus de juguete: 6 archivos de texto en tres fenómenos."""
    raiz = tmp_path / "raw"
    for fenomeno in (1, 2, 3):
        carpeta = raiz / f"fenomeno_{fenomeno}"
        carpeta.mkdir(parents=True)
        for i in range(2):
            # Largos a propósito: cada archivo debe dar varios fragmentos,
            # porque la entrega exige 10 fragmentos por consulta.
            (carpeta / f"informe_{i}.txt").write_text(
                f"Documento {fenomeno}-{i}. " + PARRAFO * 20,
                encoding="utf-8",
            )
    return raiz


# scripts/indexar.py


def test_recolecta_en_orden_alfabetico(corpus):
    archivos = indexar.recolectar_archivos(corpus)

    assert len(archivos) == 6
    assert archivos == sorted(archivos)


def test_ignora_archivos_ocultos(corpus):
    (corpus / "fenomeno_1" / ".gitkeep").write_text("", encoding="utf-8")

    assert len(indexar.recolectar_archivos(corpus)) == 6


def test_corpus_inexistente_falla(tmp_path):
    with pytest.raises(FileNotFoundError, match="No existe la carpeta"):
        indexar.recolectar_archivos(tmp_path / "no_existe")


def test_deduce_el_fenomeno_de_la_carpeta(corpus):
    ruta = corpus / "fenomeno_2" / "informe_0.txt"

    assert indexar.fenomeno_de(ruta, corpus) == 2


def test_carpeta_sin_numero_de_fenomeno_falla(tmp_path):
    (tmp_path / "otros").mkdir()
    ruta = tmp_path / "otros" / "x.txt"
    ruta.write_text("hola", encoding="utf-8")

    with pytest.raises(ValueError, match="no dice a qué fenómeno"):
        indexar.fenomeno_de(ruta, tmp_path)


def test_archivo_suelto_en_la_raiz_falla(tmp_path):
    ruta = tmp_path / "x.txt"
    ruta.write_text("hola", encoding="utf-8")

    with pytest.raises(ValueError, match="suelto en la raíz"):
        indexar.fenomeno_de(ruta, tmp_path)


def test_doc_id_no_se_repite():
    usados = set()

    primero = indexar.doc_id_de(Path("a/informe.pdf"), 1, usados)
    segundo = indexar.doc_id_de(Path("b/informe.pdf"), 1, usados)

    assert primero == "F1-informe"
    assert segundo == "F1-informe_2"


def test_archivo_ilegible_no_detiene_la_corrida(corpus):
    (corpus / "fenomeno_1" / "roto.txt").write_text("   ", encoding="utf-8")

    chunks, errores = indexar.construir_chunks(
        indexar.recolectar_archivos(corpus), corpus, verboso=False
    )

    assert len(errores) == 1
    assert errores[0]["archivo"].endswith("roto.txt")
    assert chunks


def test_indexar_deja_index_y_metadata(corpus, tmp_path):
    salida = tmp_path / "base"

    codigo = indexar.main(
        [
            "--corpus", str(corpus),
            "--salida", str(salida),
            "--vectores", str(tmp_path / "vectores"),
            "--errores", str(tmp_path / "errores.csv"),
        ],
        modelo=ModeloFalso(),
    )

    assert codigo == 0
    assert (salida / "index.faiss").exists()

    lineas = (salida / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    assert lineas
    primero = json.loads(lineas[0])
    assert primero["fuente"] == "informe_0.txt"
    assert primero["fenomeno"] == 1


def test_reusar_vectores_no_vuelve_a_codificar(corpus, tmp_path):
    comunes = [
        "--corpus", str(corpus),
        "--salida", str(tmp_path / "base"),
        "--vectores", str(tmp_path / "vectores"),
        "--errores", str(tmp_path / "errores.csv"),
    ]
    modelo = ModeloFalso()
    indexar.main(comunes, modelo=modelo)
    llamadas = len(modelo.llamadas)

    indexar.main(comunes + ["--reusar-vectores"], modelo=modelo)

    assert len(modelo.llamadas) == llamadas


# entrega/generador.py


def test_lee_consultas_en_json(tmp_path):
    ruta = tmp_path / "c.json"
    ruta.write_text(
        json.dumps([{"id": "q002", "text": "segunda"}, {"id": "q001", "text": "primera"}]),
        encoding="utf-8",
    )

    assert generador.cargar_consultas(ruta) == [("q001", "primera"), ("q002", "segunda")]


def test_lee_consultas_envueltas_en_una_llave(tmp_path):
    ruta = tmp_path / "c.json"
    ruta.write_text(
        json.dumps({"consultas": [{"id": "q001", "text": "primera"}]}), encoding="utf-8"
    )

    assert generador.cargar_consultas(ruta) == [("q001", "primera")]


def test_lee_consultas_en_jsonl_con_otros_nombres_de_campo(tmp_path):
    ruta = tmp_path / "c.jsonl"
    ruta.write_text(
        '{"query_id": "q001", "consulta": "primera"}\n'
        '{"query_id": "q002", "consulta": "segunda"}\n',
        encoding="utf-8",
    )

    assert generador.cargar_consultas(ruta) == [("q001", "primera"), ("q002", "segunda")]


def test_lee_consultas_en_csv(tmp_path):
    ruta = tmp_path / "c.csv"
    ruta.write_text("id,texto\nq001,primera\nq002,segunda\n", encoding="utf-8")

    assert generador.cargar_consultas(ruta) == [("q001", "primera"), ("q002", "segunda")]


def test_consultas_con_id_repetido_fallan(tmp_path):
    ruta = tmp_path / "c.jsonl"
    ruta.write_text(
        '{"id": "q001", "text": "a"}\n{"id": "q001", "text": "b"}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="repetidos"):
        generador.cargar_consultas(ruta)


def test_extension_desconocida_falla(tmp_path):
    ruta = tmp_path / "c.txt"
    ruta.write_text("q001,primera", encoding="utf-8")

    with pytest.raises(ValueError, match="no soportada"):
        generador.cargar_consultas(ruta)


def test_genera_resultados_validos(corpus, tmp_path):
    modelo = ModeloFalso()
    base = tmp_path / "base"
    indexar.main(
        [
            "--corpus", str(corpus),
            "--salida", str(base),
            "--vectores", str(tmp_path / "vectores"),
            "--errores", str(tmp_path / "errores.csv"),
        ],
        modelo=modelo,
    )

    consultas = tmp_path / "consultas.json"
    consultas.write_text(
        json.dumps(
            [{"id": f"q{i:03d}", "text": f"{PARRAFO} pregunta {i}"} for i in range(1, 51)]
        ),
        encoding="utf-8",
    )
    salida = tmp_path / "resultados.jsonl"

    codigo = generador.main(
        [
            "--consultas", str(consultas),
            "--base", str(base),
            "--salida", str(salida),
        ],
        modelo=modelo,
    )

    assert codigo == 0

    from src.evaluation import validar_resultados

    informe = validar_resultados(salida)
    assert informe.valido, informe