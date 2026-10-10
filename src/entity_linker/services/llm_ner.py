"""NER con un LLM local (Ollama) para compararlo con GLiNER.

El LLM devuelve el texto y el tipo de cada entidad; las posiciones se calculan
buscando ese texto en el documento (los LLM no dan offsets fiables). Misma
interfaz que ``NERService`` para poder compararlos con la misma evaluación.

Requisitos: ``curl -fsSL https://ollama.com/install.sh | sh`` y
``ollama pull qwen2.5:3b``.
"""

import json
import re

import httpx

from entity_linker.models.schemas import EntitySpan
from entity_linker.services.entity_types import FINE_LABELS

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
# 3B cuantizado a 4 bits (~1,9 GB): cabe entero en una GPU de 4 GB. El 7B
# (~4,7 GB) no cabe, se reparte con la CPU y supera los 100 s por documento
DEFAULT_MODEL = "qwen2.5:3b"

# Guía de anotación alineada con CoNLL-2003, el esquema contra el que se evalúa
PROMPT = """Extract all named entities from the text below.
Allowed entity types: {types}.
Rules:
- Only proper names. Do not extract dates, numbers, job titles or common nouns.
- Nationalities and adjectives derived from names (e.g. "Russian") are named entities.
- Copy each entity exactly as it appears in the text.
- List every distinct surface form: if "Boris Yeltsin" and later "Yeltsin" appear,
  list both.
Answer only with JSON: {{"entities": [{{"text": "...", "type": "..."}}]}}

Text:
{text}"""


class LLMNERError(Exception):
    """El LLM no respondió o devolvió algo que no es JSON válido."""


class LLMNERService:
    def __init__(self, model: str = DEFAULT_MODEL, timeout: float = 300.0):
        # Timeout amplio: en CPU un 7B genera ~6 tokens/s
        self.model = model
        self._client = httpx.Client(timeout=timeout)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        # Entidades que no aparecen en el texto y tipos no pedidos (última llamada)
        self.last_diagnostics: dict[str, list[str]] = {}

    def extract_entities(
        self, text: str, labels: dict[str, str] | None = None
    ) -> list[EntitySpan]:
        """Detecta entidades; ``labels`` (tipo pedido -> etiqueta) como en GLiNER.

        Raises:
            LLMNERError: Si Ollama falla o la respuesta no es JSON válido.
        """
        labels = labels or FINE_LABELS
        prompt = PROMPT.format(types=", ".join(labels), text=text)
        try:
            response = self._client.post(
                OLLAMA_URL,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            data = response.json()
            entities = json.loads(data["choices"][0]["message"]["content"])["entities"]
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
            raise LLMNERError(str(exc)) from exc

        usage = data.get("usage", {})
        self.total_input_tokens += usage.get("prompt_tokens", 0)
        self.total_output_tokens += usage.get("completion_tokens", 0)
        return self._locate(entities, text, labels)

    def _locate(
        self, entities: list[dict[str, str]], text: str, labels: dict[str, str]
    ) -> list[EntitySpan]:
        """Busca cada entidad devuelta en el texto y anota todas sus apariciones."""
        self.last_diagnostics = {"malformed_spans": [], "invalid_types": []}
        found: list[EntitySpan] = []
        # Las más largas primero: "Paris" no se anota dentro de "University of Paris"
        for item in sorted(entities, key=lambda e: -len(str(e.get("text", "")))):
            surface, kind = str(item.get("text", "")), str(item.get("type", ""))
            if kind not in labels:
                self.last_diagnostics["invalid_types"].append(f"{surface} [{kind}]")
                continue
            # Palabra completa: ni letras ni guion pegados ("U.S." no casa dentro
            # de "Russian-U.S.", ni "Curie" dentro de "Joliot-Curie")
            pattern = rf"(?<![\w-]){re.escape(surface)}(?![\w-])"
            matches = list(re.finditer(pattern, text)) if surface else []
            if not matches:
                self.last_diagnostics["malformed_spans"].append(surface)
            for m in matches:
                overlaps = any(
                    m.start() < e.end_char and e.start_char < m.end() for e in found
                )
                if not overlaps:
                    found.append(
                        EntitySpan(
                            text=surface,
                            start_char=m.start(),
                            end_char=m.end(),
                            label=labels[kind],
                            fine_label=kind,
                        )
                    )
        return sorted(found, key=lambda e: e.start_char)
