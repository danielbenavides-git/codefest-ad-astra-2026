# Estado de la entrega

**Equipo Talon Systems** · CODEFEST Ad Astra 2026, Etapa 1

## Qué contiene esta carpeta

| Archivo | Estado |
|---|---|
| `generador.py` | Completo |
| `informe_tecnico.pdf` | Completo |
| `src/` | Completo, 12 módulos |
| `requirements.txt` | Completo |
| `resultados.jsonl` | **No incluido** |
| `base_vectorial/` | **No incluido** |

## Por qué faltan dos

El pipeline está implementado y probado, pero no alcanzó el tiempo para indexar
el corpus completo: la codificación de los fragmentos toma unas 4,6 horas en CPU
con el encoder seleccionado.

Los dos archivos faltantes son la salida de ese proceso, no código pendiente.

## Cómo reproducirlos

Con el corpus en `data/raw/`, en subcarpetas `fenomeno_1`, `fenomeno_2` y
`fenomeno_3`:

```bash
pip install -r requirements.txt
python scripts/indexar.py                            # produce base_vectorial/
python generador.py --consultas <archivo_consultas>  # produce resultados.jsonl
```

`indexar.py` está en el repositorio, no en esta carpeta.

## Verificación sin corpus

El sistema tiene 158 pruebas que no requieren el corpus ni descargar modelos:

```bash
pytest
```

## Repositorio

https://github.com/danielbenavides-git/codefest-ad-astra-2026