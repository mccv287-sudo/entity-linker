"""Endpoints con un pipeline simulado (sin GLiNER ni Wikidata)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from entity_linker import main
from entity_linker.models.schemas import TextLinkingResponse


@pytest.fixture
def pipeline(monkeypatch):
    fake = MagicMock()
    fake.wikidata_service.aclose = AsyncMock()
    fake.process_text = AsyncMock(
        return_value=TextLinkingResponse(
            original_text="Paris", entities=[], processing_time_ms=1.0
        )
    )
    monkeypatch.setattr(main, "EntityLinkingPipeline", lambda: fake)
    return fake


@pytest.fixture
def client(pipeline):
    with TestClient(main.app) as client:
        yield client


def test_health_check(client):
    assert client.get("/").json()["status"] == "ok"


def test_link_passes_request_options_to_pipeline(client, pipeline):
    body = {"text": "Paris", "language": "es", "use_llm": False}
    response = client.post("/api/v1/link", json=body)
    assert response.status_code == 200
    pipeline.process_text.assert_awaited_once_with(
        "Paris",
        language="es",
        use_llm=False,
        review_threshold=0.7,
        max_candidates=5,
    )


def test_blank_text_is_400(client):
    assert client.post("/api/v1/link", json={"text": "   "}).status_code == 400


def test_invalid_options_are_422(client):
    body = {"text": "Paris", "review_threshold": 2}
    assert client.post("/api/v1/link", json=body).status_code == 422
