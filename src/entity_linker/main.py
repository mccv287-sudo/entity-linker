import os

from fastapi import FastAPI, HTTPException

from entity_linker.models.schemas import TextLinkingRequest, TextLinkingResponse
from entity_linker.services.pipeline import EntityLinkingPipeline

app = FastAPI(
    title="Entity Linker API",
    description="Servicio REST de Reconocimiento y Enlace de Entidades a Wikidata (NER + Entity Linking)",
    version="1.0.0"
)

# NER_BACKEND=spacy para usar spaCy en lugar de GLiNER
pipeline = EntityLinkingPipeline(ner_backend=os.getenv("NER_BACKEND", "gliner"))

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Argos Entity Linker"}

@app.post("/api/v1/link", response_model=TextLinkingResponse)
async def link_entities(request: TextLinkingRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="El texto proporcionado no puede estar vacío.")

    return await pipeline.process_text(request.text)
