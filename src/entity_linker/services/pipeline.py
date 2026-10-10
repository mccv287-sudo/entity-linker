import asyncio
import time

from entity_linker.models.schemas import LinkedEntity, TextLinkingResponse
from entity_linker.services.disambiguator import MAX_CANDIDATES, LLMDisambiguatorService
from entity_linker.services.entity_types import rerank_by_type
from entity_linker.services.ner import NERService
from entity_linker.services.wikidata import WikidataService


class EntityLinkingPipeline:
    """NER (tipos finos) -> candidatos -> reordenación por tipo -> desambiguación."""

    def __init__(self, use_llm: bool = True, use_types: bool = True):
        """
        Args:
            use_llm: El LLM resuelve las menciones dudosas.
            use_types: Reordenar candidatos por tipo fino y coherencia (si es
                False, se mantiene el orden de Wikidata; sirve para comparar).
        """
        self.ner_service = NERService()
        self.wikidata_service = WikidataService()
        self.disambiguator_service = LLMDisambiguatorService(use_llm=use_llm)
        self.use_types = use_types

    async def link(self, text: str, language: str = "en") -> list[LinkedEntity]:
        # 1. NER con tipos finos (fine_label) agrupados en PER/ORG/LOC/MISC
        spans = self.ner_service.extract_entities(text)

        # 2. Candidatos de Wikidata para cada mención (en el idioma del texto)
        candidates_list = await asyncio.gather(
            *(
                self.wikidata_service.get_candidates(span.text, language=language)
                for span in spans
            )
        )

        # 3. Clases de los candidatos y reordenación por compatibilidad de tipo
        # (sin clases, ni el tipo ni la coherencia cambian el orden de Wikidata)
        qids = [c.qid for cands in candidates_list for c in cands[:MAX_CANDIDATES]]
        classes = (
            await self.wikidata_service.get_classes(qids) if self.use_types else {}
        )
        candidates_list = [
            rerank_by_type(span.fine_label, cands, classes)
            for span, cands in zip(spans, candidates_list)
        ]

        # 4. Desambiguación con coherencia entre las entidades del documento
        return await self.disambiguator_service.disambiguate_document(
            text, spans, candidates_list, classes
        )

    async def process_text(self, text: str) -> TextLinkingResponse:
        start_time = time.perf_counter()
        entities = await self.link(text)
        return TextLinkingResponse(
            original_text=text,
            entities=entities,
            processing_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
        )
