# Entity Linker

Servicio REST de Reconocimiento de Entidades Nombradas (NER) y Enlace de Entidades (*Entity Linking*) a **Wikidata**, desarrollado como prueba técnica para la posición de Desarrollador de IA.

Dado un texto, detecta las entidades nombradas y enlaza cada una con su QID de Wikidata (o `NIL`), resolviendo los casos en que varias entidades comparten nombre. Idioma principal: inglés; español con `"language": "es"`.

---

## Arquitectura del Pipeline

```text
[Texto] ──► [1. NER (GLiNER)] ──► [2. Candidatos (Wikidata)] ──► [3. Desambiguación probabilística (+ LLM local)] ──► [JSON]
```

1. **NER:** [GLiNER](https://github.com/urchade/GLiNER) (`urchade/gliner_multi-v2.1`), modelo multilingüe *zero-shot*. Se le piden 18 **tipos finos** (*country*, *sports team*, *university*, *award*…) que se agrupan en `PER`, `ORG`, `LOC`, `MISC`; el tipo fino se conserva para la desambiguación. Los textos largos se procesan por trozos de frases completas.
2. **Candidatos:** búsqueda asíncrona en la API de Wikidata (`wbsearchentities`, 20 candidatos por mención, en el idioma del texto), con caché persistente en SQLite, reintentos que respetan `Retry-After` y un máximo de 4 peticiones simultáneas.
3. **Desambiguación:** cada uno de los 10 primeros candidatos recibe señales en [0, 1] que se combinan linealmente; un softmax da la probabilidad de cada uno (logit condicional):

   | Señal | Qué mide |
   |---|---|
   | `prior` | posición en la búsqueda de Wikidata (popularidad) |
   | `context` | palabras de la descripción del candidato alrededor de la mención (sin contar la propia mención) |
   | `type` | compatibilidad con el tipo fino del NER (clases P31/P279*, vía SPARQL) |
   | `coherence` | clases compartidas con las entidades ya resueltas del documento |
   | `llm` | candidato elegido por un LLM local (Ollama, `qwen2.5:3b`) |

   Carga diferida: las clases de Wikidata solo se piden para las menciones inciertas (probabilidad < 0,7) y el LLM solo actúa si siguen inciertas. Los pesos se aprenden con `scripts/tune_weights.py`.

**`NIL`:** si Wikidata no devuelve candidatos o el LLM responde que ninguno encaja. **Revisión humana:** las entidades con confianza menor que `review_threshold` (o `NIL`) se marcan con `needs_review` y llevan sus candidatos alternativos ordenados por probabilidad.

---

## Requisitos e Instalación

El proyecto usa **`uv`** como gestor de entorno y paquetes (Python fijado en `.python-version`).

```bash
git clone https://github.com/MCarmenCampos/entity-linker.git
cd entity-linker
uv sync
```

- **GLiNER:** el modelo (~1 GB) se descarga de Hugging Face la primera vez que arranca el servicio. PyTorch se instala en su versión solo CPU.
- **LLM local (opcional):** solo se necesita con `use_llm: true`. Instala [Ollama](https://ollama.com) y descarga el modelo:

  ```bash
  ollama pull qwen2.5:3b
  ```

### Configuración

Los valores ajustables (modelos, URLs, umbrales, pesos) están en [`src/entity_linker/config.py`](src/entity_linker/config.py). Los de despliegue se pueden sobrescribir con variables de entorno `ENTITY_LINKER_<NOMBRE>`:

| Variable | Por defecto |
|---|---|
| `ENTITY_LINKER_GLINER_MODEL` | `urchade/gliner_multi-v2.1` |
| `ENTITY_LINKER_LLM_MODEL` | `qwen2.5:3b` |
| `ENTITY_LINKER_OLLAMA_URL` | `http://localhost:11434/v1/chat/completions` |
| `ENTITY_LINKER_CACHE_PATH` | `.cache/wikidata.sqlite` |
| `ENTITY_LINKER_USER_AGENT` | `EntityLinker/1.0 (...)` |

---

## Ejecución del Servicio API

```bash
uv run uvicorn entity_linker.main:app --reload
```

El servidor estará disponible en `http://127.0.0.1:8000`, con la documentación interactiva (Swagger UI) en **`http://127.0.0.1:8000/docs`**.

### Petición (`POST /api/v1/link`)

| Campo | Por defecto | Descripción |
|---|---|---|
| `text` | — | Texto a analizar |
| `language` | `en` | Idioma del texto (para buscar en Wikidata) |
| `use_llm` | `true` | El LLM local resuelve las menciones inciertas |
| `review_threshold` | `0.7` | Por debajo, la entidad se marca con `needs_review` |
| `max_candidates` | `5` | Candidatos devueltos por entidad (máximo 10) |

```json
{
  "text": "Marie Curie was born in Warsaw and studied at the University of Paris.",
  "use_llm": false,
  "max_candidates": 2
}
```

### Respuesta (real, recortada a una entidad)

```json
{
  "original_text": "Marie Curie was born in Warsaw and studied at the University of Paris.",
  "entities": [
    {
      "text": "University of Paris",
      "start_char": 50,
      "end_char": 69,
      "label": "ORG",
      "fine_label": "university",
      "qid": "Q209842",
      "wikidata_label": "University of Paris",
      "wikidata_description": "French university (c. 1150–1970)",
      "confidence": 0.445,
      "is_nil": false,
      "reasoning": "p=0.45 sobre 10 candidatos",
      "needs_review": true,
      "candidates": [
        {
          "qid": "Q209842",
          "label": "University of Paris",
          "description": "French university (c. 1150–1970)",
          "concept_uri": "http://www.wikidata.org/entity/Q209842"
        },
        {
          "qid": "Q55849612",
          "label": "Paris Cité University",
          "description": "French public university (2019-)",
          "concept_uri": "http://www.wikidata.org/entity/Q55849612"
        }
      ]
    }
  ],
  "processing_time_ms": 283.15
}
```

Las otras dos entidades se enlazan con `Q7186` (Marie Curie, confianza 0,83, sin revisión) y `Q270` (Warsaw, 0,55). "University of Paris" muestra un caso ambiguo real: la universidad histórica y sus sucesoras compiten, por lo que se marca para revisión. El tiempo es con la caché de Wikidata caliente; la primera vez, unos segundos.

---

## Evaluación

Datos en `src/entity_linker/evaluation/data/` (AIDA CoNLL-YAGO con QIDs de Wikidata y una biografía anotada a mano). Métricas con `nervaluate` (NER y enlace, estilo GERBIL) y `ranx` (retrieval).

```bash
# NER, retrieval, enlace completo o desambiguación sobre AIDA (split test)
uv run python -m entity_linker.evaluation.evaluate --stage ner --limit 10
uv run python -m entity_linker.evaluation.evaluate --stage retrieval --limit 10
uv run python -m entity_linker.evaluation.evaluate --stage linking --limit 10
uv run python -m entity_linker.evaluation.evaluate --stage disambiguation [--llm]

# Aprender los pesos de la desambiguación (train) y medirlos (test)
uv run python scripts/tune_weights.py
```

Los informes se guardan en `src/entity_linker/evaluation/results/`. El notebook `Untitled-1.ipynb` recorre las decisiones del pipeline paso a paso.

| Fase | Métrica | Resultado |
|---|---|---|
| NER (10 documentos AIDA) | F1 estricto | 0,843 (etiquetas genéricas: 0,667) |
| Retrieval (20 candidatos) | recall@1 / @10 / @20 | 0,46 / 0,63 / 0,70 |
| Desambiguación (333 menciones de test, sin LLM) | accuracy | 0,778 (primer candidato: 0,664) |
| End-to-end sin LLM (10 documentos) | F1 `mention` / `linked` / `nil` | 0,855 / 0,440 / 0,313 |

---

## Pruebas

```bash
uv run pytest -v
```

---

## Estructura del Proyecto

```text
entity-linker/
├── pyproject.toml / uv.lock / .python-version
├── Untitled-1.ipynb               # Justificación de las decisiones, paso a paso
├── scripts/
│   ├── build_disambiguation_sample.py  # Muestra de desambiguación (AIDA + candidatos)
│   ├── tune_weights.py                 # Aprende los pesos de las señales
│   └── benchmark_gliner_labels.py      # Compara conjuntos de etiquetas de GLiNER
├── src/entity_linker/
│   ├── main.py                    # FastAPI: lifespan y endpoints
│   ├── config.py                  # Valores ajustables (ENTITY_LINKER_*)
│   ├── models/schemas.py          # Modelos Pydantic v2 (petición/respuesta)
│   ├── services/
│   │   ├── pipeline.py            # Orquestador: NER → candidatos → desambiguación
│   │   ├── entity_types.py        # Tipos finos de GLiNER y sus clases de Wikidata
│   │   ├── clients/
│   │   │   ├── wikidata.py        # Búsqueda, clases (SPARQL), caché y reintentos
│   │   │   └── ollama.py          # Petición y respuesta JSON del LLM local
│   │   ├── ner/
│   │   │   ├── gliner.py          # NER del servicio
│   │   │   └── llm_ner.py         # NER con LLM (solo para comparar)
│   │   └── linking/
│   │       ├── scoring.py         # Señales y probabilidad de cada candidato
│   │       └── disambiguator.py   # Flujo diferido + LLM en casos inciertos
│   └── evaluation/                # Datasets, métricas y evaluación
└── tests/                         # Pruebas con pytest (red simulada)
```
