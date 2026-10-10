"""LLM local con Ollama (API compatible con OpenAI): formato de las peticiones
y respuestas, compartido por el NER con LLM y el desambiguador. La URL y el
modelo están en ``config`` (``OLLAMA_URL``, ``LLM_MODEL``).

Requisitos: ``curl -fsSL https://ollama.com/install.sh | sh`` y
``ollama pull qwen2.5:3b``.
"""

import json
from typing import Any


def json_request(
    model: str, prompt: str, max_tokens: int | None = None
) -> dict[str, Any]:
    """Cuerpo de la petición: respuesta en JSON y determinista."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    if max_tokens:
        body["max_tokens"] = max_tokens
    return body


def parse_json_reply(data: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    """(JSON de la respuesta, tokens de entrada, tokens de salida).

    Raises:
        KeyError, ValueError: Si la respuesta no tiene el formato esperado.
    """
    usage = data.get("usage", {})
    content = json.loads(data["choices"][0]["message"]["content"])
    return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
