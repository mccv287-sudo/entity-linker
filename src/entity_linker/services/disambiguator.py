"""Desambiguación: primer candidato de Wikidata + LLM local solo para casos dudosos.

1. Sin candidatos -> NIL.
2. Criterio de incertidumbre (``is_ambiguous``): si la mención no es dudosa, se
   acepta el primer candidato (ya reordenado por tipo fino, ver entity_types).
3. Si es dudosa, un LLM local (Ollama) elige entre los primeros candidatos con
   una ventana de contexto y las entidades ya resueltas del documento
   (coherencia barata). Solo devuelve el número del candidato: en CPU el coste
   lo marcan los tokens. Con ``use_llm=False`` se queda con el primero tras
   reordenar por coherencia.
4. Si el LLM falla, se vuelve al primer candidato.
"""

import asyncio
import json
import logging

import httpx

from entity_linker.models.schemas import EntitySpan, LinkedEntity, WikidataCandidate
from entity_linker.services.entity_types import Classes, rerank_by_coherence
from entity_linker.services.llm_ner import DEFAULT_MODEL, OLLAMA_URL
from entity_linker.services.wikidata import normalize_mention

logger = logging.getLogger(__name__)

CONTEXT_CHARS = 300  # caracteres a cada lado de la mención
MAX_CANDIDATES = 10  # candidatos que se muestran al LLM
MAX_DESCRIPTION = 100

# Confianza fija por vía de decisión (no hay una probabilidad calibrada)
CONFIDENCE = {"top": 0.9, "llm": 0.7, "coherence": 0.6, "fallback": 0.5}

PROMPT = """Which Wikidata entity does the mention refer to in this text?

Text: "...{context}..."
Other entities in the document: {document_entities}
Mention: "{mention}" (type: {label})

Candidates:
{candidates}

Answer only with JSON: {{"choice": <candidate number, or 0 if none fits>}}"""


def is_ambiguous(mention: str, candidates: list[WikidataCandidate]) -> bool:
    """Mención dudosa: el primer candidato no se llama como la mención, o hay
    varios candidatos que se llaman igual (homónimos).

    No usa el gold: es aplicable en producción.
    """
    name = normalize_mention(mention).lower()
    same_name = [c for c in candidates if c.label.lower() == name]
    return candidates[0].label.lower() != name or len(same_name) > 1


class LLMDisambiguatorService:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        timeout: float = 300.0,
        use_gate: bool = True,
        use_llm: bool = True,
    ):
        """
        Args:
            model: Modelo de Ollama.
            timeout: Segundos por llamada (en CPU un 7B procesa ~17 tokens/s).
            use_gate: Si es False, el LLM decide todas las menciones (para
                medir qué aporta el criterio de incertidumbre).
            use_llm: Si es False, las dudosas se resuelven solo con tipo y
                coherencia (evaluación end-to-end barata).
        """
        self.model = model
        self.timeout = timeout
        self.use_gate = use_gate
        self.use_llm = use_llm
        # Ollama procesa una petición a la vez: se encolan aquí, no en su cola
        self._semaphore = asyncio.Semaphore(1)
        self.llm_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def disambiguate(
        self,
        context_text: str,
        entity_text: str,
        start_char: int,
        end_char: int,
        entity_label: str,
        candidates: list[WikidataCandidate],
        document_entities: list[str] | None = None,
        fine_label: str | None = None,
    ) -> LinkedEntity:
        """Elige el QID de la mención entre los candidatos, o NIL.

        ``document_entities``: entidades ya resueltas del documento, que el LLM
        usa como contexto (coherencia). ``fine_label``: tipo fino del NER, más
        informativo para el LLM que la etiqueta CoNLL ("sports team" frente a
        "ORG"); se devuelve también en la salida.
        """

        def linked(
            cand: WikidataCandidate | None, confidence: float, reason: str
        ) -> LinkedEntity:
            return LinkedEntity(
                text=entity_text,
                start_char=start_char,
                end_char=end_char,
                label=entity_label,
                fine_label=fine_label,
                qid=cand.qid if cand else "NIL",
                wikidata_label=cand.label if cand else None,
                wikidata_description=cand.description if cand else None,
                confidence=confidence,
                is_nil=cand is None,
                reasoning=reason,
                candidates=candidates,
            )

        if not candidates:
            return linked(None, 0.0, "Sin candidatos en Wikidata")
        if self.use_gate and not is_ambiguous(entity_text, candidates):
            return linked(candidates[0], CONFIDENCE["top"], "No ambigua: 1er candidato")
        if not self.use_llm:
            return linked(candidates[0], CONFIDENCE["coherence"], "Dudosa: sin LLM")

        shown = candidates[:MAX_CANDIDATES]
        context = context_text[
            max(0, start_char - CONTEXT_CHARS) : end_char + CONTEXT_CHARS
        ]
        try:
            choice = await self._ask_llm(
                context,
                entity_text,
                fine_label or entity_label,
                shown,
                document_entities or [],
            )
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
            logger.warning("LLM falló para %r: %s", entity_text, exc)
            return linked(candidates[0], CONFIDENCE["fallback"], "LLM falló")
        if not 1 <= choice <= len(shown):
            return linked(None, 0.0, "LLM: ningún candidato encaja")
        return linked(shown[choice - 1], CONFIDENCE["llm"], f"LLM: candidato {choice}")

    async def disambiguate_document(
        self,
        text: str,
        spans: list[EntitySpan],
        candidates_list: list[list[WikidataCandidate]],
        classes: Classes,
    ) -> list[LinkedEntity]:
        """Desambigua todas las menciones de un documento en dos pasadas.

        1. Menciones no dudosas: primer candidato.
        2. Menciones dudosas: candidatos reordenados por coherencia con las ya
           resueltas, que además se pasan al LLM como contexto.
        """
        dubious = [
            bool(c) and is_ambiguous(s.text, c) for s, c in zip(spans, candidates_list)
        ]
        results: list[LinkedEntity | None] = [None] * len(spans)
        for i, (span, cands) in enumerate(zip(spans, candidates_list)):
            if not dubious[i]:
                results[i] = await self.disambiguate(
                    text, span.text, span.start_char, span.end_char, span.label,
                    cands, fine_label=span.fine_label,
                )

        resolved = [r for r in results if r is not None and not r.is_nil]
        resolved_qids = [r.qid for r in resolved]
        names = list(dict.fromkeys(r.wikidata_label or r.text for r in resolved))
        for i, (span, cands) in enumerate(zip(spans, candidates_list)):
            if dubious[i]:
                cands = rerank_by_coherence(
                    span.fine_label, cands, resolved_qids, classes
                )
                results[i] = await self.disambiguate(
                    text, span.text, span.start_char, span.end_char, span.label,
                    cands, document_entities=names[:MAX_CANDIDATES],
                    fine_label=span.fine_label,
                )
        return [r for r in results if r is not None]

    async def _ask_llm(
        self,
        context: str,
        mention: str,
        label: str,
        candidates: list[WikidataCandidate],
        document_entities: list[str],
    ) -> int:
        listing = "\n".join(
            f"{i}. {c.label}: {(c.description or 'no description')[:MAX_DESCRIPTION]}"
            for i, c in enumerate(candidates, 1)
        )
        prompt = PROMPT.format(
            context=" ".join(context.split()),
            document_entities=", ".join(document_entities) or "none",
            mention=mention,
            label=label,
            candidates=listing,
        )
        async with self._semaphore, httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0,
                    "max_tokens": 20,  # solo {"choice": n}
                },
            )
        response.raise_for_status()
        data = response.json()
        self.llm_calls += 1
        usage = data.get("usage", {})
        self.total_input_tokens += usage.get("prompt_tokens", 0)
        self.total_output_tokens += usage.get("completion_tokens", 0)
        return int(json.loads(data["choices"][0]["message"]["content"])["choice"])
