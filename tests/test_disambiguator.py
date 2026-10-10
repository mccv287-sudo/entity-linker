"""Desambiguador sin red: sin cliente de Wikidata (sin type ni coherence) y
con el LLM simulado."""

from unittest.mock import AsyncMock

import pytest

from entity_linker.models.schemas import EntitySpan, WikidataCandidate
from entity_linker.services.linking import disambiguator as module
from entity_linker.services.linking.disambiguator import LLMDisambiguatorService

TEXT = "Kuwait national football team beat Japan 2-0 in the Asian Cup."
SPAN = EntitySpan(text="Kuwait", start_char=0, end_char=6, label="LOC")


def candidate(qid: str, description: str | None = None) -> WikidataCandidate:
    return WikidataCandidate(
        qid=qid,
        label="Kuwait",
        description=description,
        concept_uri=f"http://www.wikidata.org/entity/{qid}",
    )


COUNTRY = candidate("Q817", "country in Western Asia")
TEAM = candidate("Q195851", "national association football team")


async def test_context_overrides_search_order():
    dis = LLMDisambiguatorService(use_llm=False)
    linked = await dis.disambiguate(TEXT, SPAN, [COUNTRY, TEAM])
    assert linked.qid == TEAM.qid
    assert linked.candidates[0].qid == TEAM.qid  # ordenados por probabilidad


async def test_no_candidates_is_nil_and_needs_review():
    dis = LLMDisambiguatorService(use_llm=False)
    linked = await dis.disambiguate(TEXT, SPAN, [])
    assert linked.is_nil and linked.qid == "NIL"
    assert linked.needs_review


@pytest.fixture
def always_uncertain(monkeypatch):
    """Todas las menciones inciertas, para que siempre se consulte al LLM."""
    monkeypatch.setattr(module, "CONFIDENT_PROB", 1.01)


@pytest.mark.usefixtures("always_uncertain")
async def test_llm_choice_wins():
    dis = LLMDisambiguatorService()
    dis._ask_llm = AsyncMock(return_value=2)  # el 2.º más probable
    linked = await dis.disambiguate("Kuwait", SPAN, [COUNTRY, candidate("Q2")])
    assert linked.qid == "Q2"


@pytest.mark.usefixtures("always_uncertain")
async def test_llm_none_fits_is_nil():
    dis = LLMDisambiguatorService()
    dis._ask_llm = AsyncMock(return_value=0)
    linked = await dis.disambiguate(TEXT, SPAN, [COUNTRY, TEAM])
    assert linked.is_nil and linked.needs_review


@pytest.mark.usefixtures("always_uncertain")
async def test_llm_failure_keeps_most_probable():
    dis = LLMDisambiguatorService()
    dis._ask_llm = AsyncMock(side_effect=ValueError("respuesta no JSON"))
    linked = await dis.disambiguate(TEXT, SPAN, [COUNTRY, TEAM])
    assert linked.qid == TEAM.qid
