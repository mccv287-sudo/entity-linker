"""Cliente de Wikidata con la red simulada (``httpx.MockTransport``)."""

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from entity_linker.services.clients.wikidata import WikidataService, normalize_mention

Handler = Callable[[httpx.Request], httpx.Response]


def service(handler: Handler) -> WikidataService:
    """Servicio sin caché en disco cuyas peticiones responde ``handler``."""
    wikidata = WikidataService(cache_path=None)
    wikidata._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return wikidata


def search_result(*qids: str) -> dict[str, Any]:
    return {
        "search": [
            {
                "id": q,
                "label": f"label {q}",
                "description": f"description {q}",
                "concepturi": f"http://www.wikidata.org/entity/{q}",
            }
            for q in qids
        ]
    }


@pytest.mark.parametrize(
    ("mention", "expected"),
    [("Obama's", "Obama"), (' "Reuters". ', "Reuters"), ("New  York", "New York")],
)
def test_normalize_mention(mention, expected):
    assert normalize_mention(mention) == expected


async def test_get_candidates_parses_search_in_requested_language():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=search_result("Q90", "Q167646"))

    wikidata = service(handler)
    candidates = await wikidata.get_candidates("Paris", language="es")
    await wikidata.aclose()

    assert [c.qid for c in candidates] == ["Q90", "Q167646"]
    assert candidates[0].description == "description Q90"
    assert requests[0].url.params["uselang"] == "es"


async def test_get_candidates_retries_without_leading_article():
    def handler(request: httpx.Request) -> httpx.Response:
        found = request.url.params["search"] == "Beatles"
        return httpx.Response(200, json=search_result("Q1299") if found else {})

    wikidata = service(handler)
    candidates = await wikidata.get_candidates("the Beatles")
    await wikidata.aclose()

    assert [c.qid for c in candidates] == ["Q1299"]


async def test_get_candidates_waits_and_retries_after_429():
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.Response(200, json=search_result("Q90")),
        ]
    )
    wikidata = service(lambda request: next(responses))
    candidates = await wikidata.get_candidates("Paris")
    await wikidata.aclose()

    assert [c.qid for c in candidates] == ["Q90"]


async def test_get_classes_parses_sparql_bindings():
    entity = "http://www.wikidata.org/entity/"

    def handler(request: httpx.Request) -> httpx.Response:
        row = {
            "item": {"value": entity + "Q90"},
            "cls": {"value": entity + "Q515"},
            "root": {"value": entity + "Q515"},
        }
        return httpx.Response(200, json={"results": {"bindings": [row]}})

    wikidata = service(handler)
    classes = await wikidata.get_classes(["Q90", "Q2"], roots={"Q515"})
    await wikidata.aclose()

    assert classes["Q90"] == ({"Q515"}, {"Q515"})
    assert classes["Q2"] == (set(), set())  # sin P31 en la respuesta
