# Entity Linker

Servicio REST de Reconocimiento de Entidades Nombradas (NER) y Desambiguación / Enlace de Entidades (*Entity Linking*) a **Wikidata**, desarrollado como prueba técnica para la posición de Desarrollador de IA.

---

## Arquitectura del Pipeline

El sistema implementa un flujo modular y asíncrono diseñado para alta velocidad, resiliencia y escalabilidad:

```text
[Texto de entrada] ──► [1. NER Engine (SpaCy)] ──► [2. Async Candidate Retrieval (Wikidata API)] ──► [3. Disambiguator & Confidence Scoring] ──► [Respuesta JSON]
```

1. **NER (Named Entity Recognition):** Extracción de *spans* (posiciones exactas en caracteres) y etiquetas (`PER`, `ORG`, `LOC`, `MISC`) utilizando SpaCy (`es_core_news_lg`).
2. **Candidate Retrieval:** Búsqueda asíncrona concurrente (`httpx` + `asyncio.gather`) en la API oficial de Wikidata (`wbsearchentities`) con sistema de caché en memoria para minimizar la latencia.
3. **Entity Disambiguation:** Algoritmo de puntuación de confianza basado en coincidencia de etiquetas, análisis de contexto circundante y alineación semántica de categorías. Si ningún candidato supera el umbral de confianza (`0.35`), el sistema asigna automáticamente la entidad como `NIL`.

---

## Requisitos e Instalación

Este proyecto utiliza **`uv`** como gestor de entorno y paquetes para garantizar ejecuciones 100% reproducibles y ultrarrápidas.

### 1. Clonar el repositorio
```bash
git clone https://github.com/MCarmenCampos/entity-linker.git
cd entity-linker
```

### 2. Sincronizar el entorno virtual con `uv`
Ejecuta el siguiente comando para crear el entorno virtual e instalar automáticamente todas las dependencias exactas desde `uv.lock`:

```bash
uv sync
```

### 3. Descargar el modelo de lenguaje de SpaCy
```bash
uv run python -m spacy download es_core_news_lg
```

---

## Ejecución del Servicio API

Para levantar el servidor web en modo desarrollo con recarga automática:

```bash
uv run uvicorn src.entity_linker.main:app --reload
```

El servidor estará disponible en `http://127.0.0.1:8000`.

### Documentación Interactiva (Swagger UI / OpenAPI)
Puedes probar la API directamente desde tu navegador accediendo a:
 **`http://127.0.0.1:8000/docs`**

---

## Ejecución de Pruebas Unitarias

Para ejecutar la suite completa de pruebas unitarias e integración con `pytest`:

```bash
uv run pytest -v
```

---

## Estructura del Proyecto

```text
entity-linker/
├── .python-version         # Versión fija de Python
├── pyproject.toml          # Configuración de dependencias y metadatos con uv
├── uv.lock                 # Lockfile estricto de dependencias
├── README.md               # Documentación principal
├── .github/
│   └── workflows/
│       └── ci.yml          # Pipeline de Integración Continua (GitHub Actions)
├── src/
│   └── entity_linker/
│       ├── main.py         # Punto de entrada FastAPI y definición de endpoints
│       ├── models/
│       │   └── schemas.py  # Modelos de datos Pydantic v2 (Request/Response)
│       └── services/
│           ├── ner.py          # Extracción de entidades con SpaCy
│           ├── wikidata.py     # Búsqueda asíncrona en Wikidata + Caché
│           ├── disambiguator.py# Algoritmo de desambiguación y reglas NIL
│           └── pipeline.py     # Orquestador del flujo completo
└── tests/
    └── test_pipeline.py    # Pruebas unitarias e integración con pytest
```

---

## Ejemplo de Petición y Respuesta

### Petición (`POST /api/v1/link`):
```json
{
  "text": "La Universidad Politécnica de Madrid está ubicada en España."
}
```

### Respuesta:
```json
{
  "original_text": "La Universidad Politécnica de Madrid está ubicada en España.",
  "entities": [
    {
      "text": "Universidad Politécnica de Madrid",
      "start_char": 3,
      "end_char": 36,
      "label": "ORG",
      "qid": "Q1546411",
      "wikidata_label": "Universidad Politécnica de Madrid",
      "wikidata_description": "universidad pública española",
      "confidence": 0.8,
      "is_nil": false,
      "candidates": [
        {
          "qid": "Q1546411",
          "label": "Universidad Politécnica de Madrid",
          "description": "universidad pública española",
          "concept_uri": "http://www.wikidata.org/entity/Q1546411"
        }
      ]
    },
    {
      "text": "España",
      "start_char": 53,
      "end_char": 59,
      "label": "LOC",
      "qid": "Q29",
      "wikidata_label": "España",
      "wikidata_description": "país de Europa",
      "confidence": 0.8,
      "is_nil": false,
      "candidates": [
        {
          "qid": "Q29",
          "label": "España",
          "description": "país de Europa",
          "concept_uri": "http://www.wikidata.org/entity/Q29"
        }
      ]
    }
  ],
  "processing_time_ms": 142.5
}
```
