"""Configuración ajustable del servicio.

Solo valores que una evaluación o un despliegue querría cambiar sin tocar la
lógica (modelos, URLs, umbrales, pesos). Las constantes intrínsecas a un módulo
(expresiones regulares, prompts, códigos HTTP reintentables...) se quedan junto
a su código.

Los valores de despliegue se pueden sobrescribir con variables de entorno
``ENTITY_LINKER_<NOMBRE>``, p. ej. ``ENTITY_LINKER_LLM_MODEL=qwen2.5:7b``.
"""

import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(f"ENTITY_LINKER_{name}", default)


# --- NER (GLiNER) ---------------------------------------------------------
GLINER_MODEL = _env("GLINER_MODEL", "urchade/gliner_multi-v2.1")
NER_THRESHOLD = 0.5

# --- Wikidata --------------------------------------------------------------
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
USER_AGENT = _env(
    "USER_AGENT", "EntityLinker/1.0 (technical_test@example.com) httpx/Python"
)
CACHE_PATH = Path(_env("CACHE_PATH", ".cache/wikidata.sqlite"))
WIKIDATA_TIMEOUT = 5.0  # segundos por petición
WIKIDATA_CONCURRENCY = 4  # peticiones simultáneas (límite de uso de Wikidata)
SEARCH_LIMIT = 20  # candidatos por búsqueda

# --- LLM local (Ollama) ----------------------------------------------------
OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434/v1/chat/completions")
# 3B cuantizado a 4 bits (~1,9 GB): cabe entero en una GPU de 4 GB. El 7B
# (~4,7 GB) no cabe, se reparte con la CPU y supera los 100 s por documento
LLM_MODEL = _env("LLM_MODEL", "qwen2.5:3b")
LLM_TIMEOUT = 300.0

# --- Desambiguación ---------------------------------------------------------
MAX_CANDIDATES = 10  # candidatos que se puntúan y se muestran al LLM
CONTEXT_CHARS = 300  # ventana de contexto a cada lado de la mención
CONFIDENT_PROB = 0.7  # por encima, la mención no necesita más señales
# Pesos de las señales: aprendidos en 13 documentos de AIDA y validados en
# otros 17 (scripts/tune_weights.py). El de llm se fija a mano
WEIGHTS = {
    "prior": 2.59,
    "context": 5.84,
    "type": 1.63,
    "coherence": 7.95,
    "llm": 4.0,
}

# --- Salida de la API -------------------------------------------------------
REVIEW_THRESHOLD = 0.7  # por debajo (o NIL), la entidad se marca para revisión
OUTPUT_CANDIDATES = 5  # candidatos devueltos por entidad (máximo: MAX_CANDIDATES)
