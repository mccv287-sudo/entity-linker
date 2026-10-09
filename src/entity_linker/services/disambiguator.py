"""
Service for entity disambiguation using a hybrid Fast-Path + LLM approach.
Integrates fine-grained GLiNER entities with Wikidata candidates and LLM reasoning.
"""

import logging

import httpx

from entity_linker.models.schemas import (
    LinkedEntity,
    LLMDisambiguationResult,
    WikidataCandidate,
)

logger = logging.getLogger(__name__)


class LLMDisambiguatorService:
    """Hybrid Entity Disambiguator combining heuristic fast-paths with LLM reasoning."""

    def __init__(
        self,
        llm_api_url: str = "http://localhost:11434/v1/chat/completions",
        model_name: str = "llama3.2:3b",
        api_key: str = "not-needed",
        confidence_threshold: float = 0.35,
        timeout_seconds: float = 3.0,
    ) -> None:
        self.llm_api_url = llm_api_url
        self.model_name = model_name
        self.api_key = api_key
        self.confidence_threshold = confidence_threshold
        self.timeout_seconds = timeout_seconds

    async def disambiguate(
        self,
        context_text: str,
        entity_text: str,
        start_char: int,
        end_char: int,
        entity_label: str,
        candidates: list[WikidataCandidate],
    ) -> LinkedEntity:
        """Disambiguates an entity span against Wikidata candidates using Fast-Path or LLM."""

        # 1. Edge case: No candidates found in Wikidata
        if not candidates:
            return LinkedEntity(
                text=entity_text,
                start_char=start_char,
                end_char=end_char,
                label=entity_label,
                qid="NIL",
                confidence=0.0,
                is_nil=True,
                reasoning="No candidates returned from Wikidata search API.",
            )

        # 2. Fast-Path: Single candidate with exact match
        if len(candidates) == 1:
            candidate = candidates[0]
            if candidate.label.lower() == entity_text.lower():
                return LinkedEntity(
                    text=entity_text,
                    start_char=start_char,
                    end_char=end_char,
                    label=entity_label,
                    qid=candidate.qid,
                    wikidata_label=candidate.label,
                    wikidata_description=candidate.description,
                    confidence=0.95,
                    is_nil=False,
                    reasoning="Fast-path: Single candidate with exact name match.",
                    candidates=candidates,
                )

        # 3. LLM Disambiguation Path
        try:
            llm_result = await self._call_llm_disambiguator(
                context_text=context_text,
                entity_text=entity_text,
                entity_label=entity_label,
                candidates=candidates,
            )

            chosen_candidate = next(
                (c for c in candidates if c.qid == llm_result.selected_qid), None
            )

            if (
                llm_result.is_nil
                or not chosen_candidate
                or llm_result.confidence < self.confidence_threshold
            ):
                return LinkedEntity(
                    text=entity_text,
                    start_char=start_char,
                    end_char=end_char,
                    label=entity_label,
                    qid="NIL",
                    confidence=llm_result.confidence,
                    is_nil=True,
                    reasoning=llm_result.reasoning,
                    candidates=candidates,
                )

            return LinkedEntity(
                text=entity_text,
                start_char=start_char,
                end_char=end_char,
                label=entity_label,
                qid=chosen_candidate.qid,
                wikidata_label=chosen_candidate.label,
                wikidata_description=chosen_candidate.description,
                confidence=llm_result.confidence,
                is_nil=False,
                reasoning=llm_result.reasoning,
                candidates=candidates,
            )

        except Exception as exc:
            logger.warning(
                f"LLM disambiguation failed/timed out for '{entity_text}': {exc}. Falling back to top candidate."
            )
            top_cand = candidates[0]
            return LinkedEntity(
                text=entity_text,
                start_char=start_char,
                end_char=end_char,
                label=entity_label,
                qid=top_cand.qid,
                wikidata_label=top_cand.label,
                wikidata_description=top_cand.description,
                confidence=0.5,
                is_nil=False,
                reasoning="Fallback: Top candidate chosen due to LLM timeout/error.",
                candidates=candidates,
            )

    async def _call_llm_disambiguator(
        self,
        context_text: str,
        entity_text: str,
        entity_label: str,
        candidates: list[WikidataCandidate],
    ) -> LLMDisambiguationResult:
        """Sends prompt to OpenAI/Ollama compatible API requesting JSON output."""

        candidates_formatted = "\n".join(
            [
                f"- QID: {c.qid} | Label: '{c.label}' | Description: '{c.description or 'N/A'}'"
                for c in candidates
            ]
        )

        system_prompt = (
            "You are an expert Entity Linking system. Your task is to select the exact Wikidata QID "
            "that matches the target entity in the given text, or assign 'NIL' if none match."
        )

        user_prompt = f"""Full Text Context: "{context_text}"
Target Entity: "{entity_text}" (Type: {entity_label})

Wikidata Candidates:
{candidates_formatted}

Instructions:
1. Compare the target entity and its context against each candidate's label and description.
2. Select the matching QID or "NIL" if none fits.
3. Output MUST be a valid JSON object matching this schema:
{{
  "selected_qid": "<QID or NIL>",
  "confidence": <float between 0.0 and 1.0>,
  "is_nil": <boolean>,
  "reasoning": "<short explanation>"
}}"""

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                self.llm_api_url, json=payload, headers=headers
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            return LLMDisambiguationResult.model_validate_json(content)