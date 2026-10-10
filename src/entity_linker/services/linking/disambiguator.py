"""Desambiguación probabilística + LLM solo en los casos inciertos.

Las señales y la probabilidad de cada candidato están en ``scoring``. Flujo por
documento, con carga diferida:

1. Señales sin red (prior, context). Si el mejor candidato supera
   ``CONFIDENT_PROB``, se acepta.
2. Solo las menciones inciertas piden a Wikidata las clases de sus candidatos
   (y de las ya resueltas, para la coherencia) y suman type y coherence.
3. Si siguen inciertas y ``use_llm``, el LLM elige entre los candidatos
   (elección múltiple) y su elección entra como señal. Si responde que ninguno
   encaja, NIL.
"""

import asyncio
import logging

import httpx

from entity_linker.config import (
    CONFIDENT_PROB,
    CONTEXT_CHARS,
    LLM_MODEL,
    LLM_TIMEOUT,
    MAX_CANDIDATES,
    OLLAMA_URL,
    REVIEW_THRESHOLD,
)
from entity_linker.models.schemas import EntitySpan, LinkedEntity, WikidataCandidate
from entity_linker.services.clients.ollama import json_request, parse_json_reply
from entity_linker.services.clients.wikidata import WikidataService
from entity_linker.services.entity_types import ROOT_CLASSES, Classes
from entity_linker.services.linking.scoring import (
    argmax,
    coherence_score,
    context_score,
    probabilities,
    type_score,
)

logger = logging.getLogger(__name__)

MAX_DESCRIPTION = 100  # caracteres de cada descripción en el prompt

PROMPT = """Which Wikidata entity does the mention refer to in this text?

Text: "...{context}..."
Other entities in the document: {document_entities}
Mention: "{mention}" (type: {label})

Candidates:
{candidates}

Answer only with JSON: {{"choice": <candidate number, or 0 if none fits>}}"""


class LLMDisambiguatorService:
    def __init__(
        self,
        wikidata: WikidataService | None = None,
        model: str = LLM_MODEL,
        timeout: float = LLM_TIMEOUT,
        use_llm: bool = True,
        use_types: bool = True,
    ):
        """
        Args:
            wikidata: Para pedir las clases de los candidatos inciertos (None:
                sin señales de tipo ni coherencia).
            model: Modelo de Ollama.
            timeout: Segundos por llamada al LLM.
            use_llm: Consultar al LLM en las menciones que siguen inciertas.
            use_types: Usar las señales de tipo y coherencia (para comparar).
        """
        self.wikidata = wikidata
        self.model = model
        self.timeout = timeout
        self.use_llm = use_llm
        self.use_types = use_types
        # Ollama procesa una petición a la vez: se encolan aquí, no en su cola
        self._semaphore = asyncio.Semaphore(1)
        self.llm_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def disambiguate_document(
        self,
        text: str,
        spans: list[EntitySpan],
        candidates_list: list[list[WikidataCandidate]],
        use_llm: bool | None = None,
    ) -> list[LinkedEntity]:
        """Elige el QID de cada mención del documento, o NIL.

        ``use_llm`` anula para esta llamada la opción del servicio.
        """
        use_llm = self.use_llm if use_llm is None else use_llm
        candidates_list = [c[:MAX_CANDIDATES] for c in candidates_list]
        contexts = [
            text[max(0, s.start_char - CONTEXT_CHARS) : s.end_char + CONTEXT_CHARS]
            for s in spans
        ]

        # 1. Señales sin red
        signals = [
            [{"context": context_score(c, ctx)} for c in cands]
            for cands, ctx in zip(candidates_list, contexts)
        ]
        probs = [probabilities(sig) for sig in signals]
        uncertain = [bool(p) and max(p) < CONFIDENT_PROB for p in probs]
        winners = [
            cands[argmax(p)]
            for cands, p, u in zip(candidates_list, probs, uncertain)
            if p and not u
        ]
        resolved = [c.qid for c in winners]  # para la coherencia
        resolved_names = list(dict.fromkeys(c.label for c in winners))  # para el LLM

        # 2. Carga diferida: clases solo de los candidatos inciertos
        classes: Classes = {}
        if self.use_types and self.wikidata and any(uncertain):
            qids = resolved + [
                c.qid
                for cands, u in zip(candidates_list, uncertain)
                if u
                for c in cands
            ]
            classes = await self.wikidata.get_classes(qids, ROOT_CLASSES)

        results = []
        for i, (span, cands) in enumerate(zip(spans, candidates_list)):
            nil = False
            if uncertain[i] and classes:
                for sig, cand in zip(signals[i], cands):
                    sig["type"] = type_score(span.fine_label, cand.qid, classes)
                    sig["coherence"] = coherence_score(cand.qid, resolved, classes)
                probs[i] = probabilities(signals[i])

            # 3. LLM solo si sigue incierta
            if uncertain[i] and use_llm and max(probs[i]) < CONFIDENT_PROB:
                choice = await self._choose_with_llm(
                    span, cands, probs[i], contexts[i], resolved_names
                )
                if choice is None:
                    nil = True
                elif choice >= 0:
                    signals[i][choice]["llm"] = 1.0
                    probs[i] = probabilities(signals[i])
            results.append(_linked(span, cands, probs[i], nil))
        return results

    async def disambiguate(
        self, text: str, span: EntitySpan, candidates: list[WikidataCandidate]
    ) -> LinkedEntity:
        """Desambigua una mención suelta (sin coherencia con otras)."""
        return (await self.disambiguate_document(text, [span], [candidates]))[0]

    async def _choose_with_llm(
        self,
        span: EntitySpan,
        cands: list[WikidataCandidate],
        probs: list[float],
        context: str,
        resolved_names: list[str],
    ) -> int | None:
        """Índice del candidato elegido, None si ninguno encaja, -1 si falla."""
        order = sorted(range(len(cands)), key=lambda j: -probs[j])  # mejores primero
        listing = "\n".join(
            f"{n}. {cands[j].label}: "
            f"{(cands[j].description or 'no description')[:MAX_DESCRIPTION]}"
            for n, j in enumerate(order, 1)
        )
        prompt = PROMPT.format(
            context=" ".join(context.split()),
            document_entities=", ".join(resolved_names[:MAX_CANDIDATES]) or "none",
            mention=span.text,
            label=span.fine_label or span.label,
            candidates=listing,
        )
        try:
            choice = await self._ask_llm(prompt)
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
            logger.warning("LLM falló para %r: %s", span.text, exc)
            return -1
        if choice == 0:
            return None
        return order[choice - 1] if 1 <= choice <= len(order) else -1

    async def _ask_llm(self, prompt: str) -> int:
        async with self._semaphore, httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                OLLAMA_URL,
                json=json_request(self.model, prompt, max_tokens=20),  # {"choice": n}
            )
        response.raise_for_status()
        reply, tokens_in, tokens_out = parse_json_reply(response.json())
        self.llm_calls += 1
        self.total_input_tokens += tokens_in
        self.total_output_tokens += tokens_out
        return int(reply["choice"])


def _linked(
    span: EntitySpan,
    cands: list[WikidataCandidate],
    probs: list[float],
    nil: bool,
) -> LinkedEntity:
    best = None if nil or not cands else cands[argmax(probs)]
    confidence = round(max(probs), 3) if best else 0.0
    return LinkedEntity(
        **span.model_dump(),
        qid=best.qid if best else "NIL",
        wikidata_label=best.label if best else None,
        wikidata_description=best.description if best else None,
        confidence=confidence,
        is_nil=best is None,
        reasoning=(
            "Sin candidatos en Wikidata" if not cands
            else "LLM: ningún candidato encaja" if nil
            else f"p={confidence:.2f} sobre {len(cands)} candidatos"
        ),
        needs_review=best is None or confidence < REVIEW_THRESHOLD,
        # ordenados por probabilidad, para la revisión humana
        candidates=[c for _, c in sorted(zip(probs, cands), key=lambda x: -x[0])],
    )
