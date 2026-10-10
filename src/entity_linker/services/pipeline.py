import asyncio
import time

from entity_linker.config import OUTPUT_CANDIDATES, REVIEW_THRESHOLD
from entity_linker.models.schemas import LinkedEntity, TextLinkingResponse
from entity_linker.services.clients.wikidata import WikidataService
from entity_linker.services.linking.disambiguator import LLMDisambiguatorService
from entity_linker.services.ner.gliner import NERService


class EntityLinkingPipeline:
    """NER (tipos finos) -> candidatos de Wikidata -> desambiguación."""

    def __init__(self, use_llm: bool = True, use_types: bool = True):
        """
        Args:
            use_llm: El LLM resuelve las menciones que siguen inciertas.
            use_types: Señales de tipo fino y coherencia (False: sirve para
                comparar).
        """
        self.ner_service = NERService()
        self.wikidata_service = WikidataService()
        self.disambiguator_service = LLMDisambiguatorService(
            self.wikidata_service, use_llm=use_llm, use_types=use_types
        )

    async def link(
        self, text: str, language: str = "en", use_llm: bool | None = None
    ) -> list[LinkedEntity]:
        # 1. NER con tipos finos (fine_label) agrupados en PER/ORG/LOC/MISC
        spans = self.ner_service.extract_entities(text)

        # 2. Candidatos de Wikidata para cada mención (en el idioma del texto)
        candidates_list = await asyncio.gather(
            *(
                self.wikidata_service.get_candidates(span.text, language=language)
                for span in spans
            )
        )

        # 3. Desambiguación probabilística (pide las clases solo si hacen falta)
        return await self.disambiguator_service.disambiguate_document(
            text, spans, list(candidates_list), use_llm=use_llm
        )

    async def process_text(
        self,
        text: str,
        language: str = "en",
        use_llm: bool | None = None,
        review_threshold: float = REVIEW_THRESHOLD,
        max_candidates: int = OUTPUT_CANDIDATES,
    ) -> TextLinkingResponse:
        """Respuesta de la API: entidades enlazadas con la marca de revisión
        (``needs_review``) según ``review_threshold`` y, como mucho,
        ``max_candidates`` candidatos por entidad."""
        start_time = time.perf_counter()
        entities = [
            e.model_copy(
                update={
                    "needs_review": e.is_nil or e.confidence < review_threshold,
                    "candidates": e.candidates[:max_candidates],
                }
            )
            for e in await self.link(text, language, use_llm)
        ]
        return TextLinkingResponse(
            original_text=text,
            entities=entities,
            processing_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
        )
