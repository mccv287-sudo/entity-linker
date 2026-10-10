from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from entity_linker.models.schemas import TextLinkingRequest, TextLinkingResponse
from entity_linker.services.pipeline import EntityLinkingPipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Crea los recursos al arrancar (en el bucle de eventos que los usará) y los
    cierra al apagar: modelo, cliente HTTP y caché de Wikidata."""
    app.state.pipeline = EntityLinkingPipeline()
    yield
    await app.state.pipeline.wikidata_service.aclose()


app = FastAPI(
    title="Entity Linker API",
    description=(
        "Servicio REST de Reconocimiento y Enlace de Entidades a Wikidata "
        "(NER + Entity Linking)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "Argos Entity Linker"}


@app.post("/api/v1/link", response_model=TextLinkingResponse)
async def link_entities(
    request: TextLinkingRequest, http_request: Request
) -> TextLinkingResponse:
    """Detecta las entidades del texto y las enlaza con Wikidata.

    Las entidades con ``needs_review`` (NIL o confianza baja) incluyen los
    principales candidatos para que una persona elija.

    Raises:
        HTTPException: 400 si el texto está vacío.
    """
    if not request.text.strip():
        raise HTTPException(
            status_code=400, detail="El texto proporcionado no puede estar vacío."
        )
    pipeline: EntityLinkingPipeline = http_request.app.state.pipeline
    return await pipeline.process_text(
        request.text,
        language=request.language,
        use_llm=request.use_llm,
        review_threshold=request.review_threshold,
        max_candidates=request.max_candidates,
    )
