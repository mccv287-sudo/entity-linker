import asyncio
import time

from entity_linker.models.schemas import LinkedEntity, TextLinkingResponse
from entity_linker.services.disambiguator import LLMDisambiguatorService
from entity_linker.services.ner import NERBackend, create_ner_service
from entity_linker.services.wikidata import WikidataService


class EntityLinkingPipeline:
    def __init__(self, ner_backend: NERBackend = "gliner"):
        self.ner_service = create_ner_service(ner_backend)
        self.wikidata_service = WikidataService()
        self.disambiguator_service = LLMDisambiguatorService()

    async def process_text(self, text: str) -> TextLinkingResponse:
        start_time = time.perf_counter()

        # 1. Extracción de entidades
        spans = self.ner_service.extract_entities(text)

        # 2. Búsqueda paralela de candidatos en Wikidata
        candidates_list = await asyncio.gather(
            *(self.wikidata_service.get_candidates(span.text) for span in spans)
        )

        # 3. Desambiguación en paralelo (cada mención puede llamar al LLM)
        linked_entities: list[LinkedEntity] = await asyncio.gather(
            *(
                self.disambiguator_service.disambiguate(
                    context_text=text,
                    entity_text=span.text,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    entity_label=span.label,
                    candidates=candidates,
                )
                for span, candidates in zip(spans, candidates_list)
            )
        )

        processing_time = (time.perf_counter() - start_time) * 1000
        return TextLinkingResponse(
            original_text=text,
            entities=linked_entities,
            processing_time_ms=round(processing_time, 2),
        )
