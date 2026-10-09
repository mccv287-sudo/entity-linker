"""Reconocimiento de entidades con dos implementaciones intercambiables.

Ambas devuelven ``EntitySpan`` con las mismas cuatro etiquetas (PER, ORG, LOC,
MISC), así que el resto del pipeline no depende de cuál se use. Se elige con
``create_ner_service("gliner" | "spacy")``.
"""

import re
from typing import Literal, Protocol

from entity_linker.models.schemas import EntitySpan

NERBackend = Literal["gliner", "spacy"]


class NERService(Protocol):
    def extract_entities(self, text: str) -> list[EntitySpan]: ...


# --- GLiNER --------------------------------------------------------------------

GLINER_MODEL = "urchade/gliner_multi-v2.1"

# GLiNER reconoce las etiquetas que se le pidan (zero-shot). Se le piden en
# inglés, con las que da puntuaciones más altas también en textos en español,
# y se agrupan en las cuatro categorías que usa el resto del pipeline.
GLINER_LABELS = {
    "person": "PER",
    "organization": "ORG",
    "location": "LOC",
    "work of art": "MISC",
    "event": "MISC",
}

# El modelo procesa como máximo ~384 palabras por pasada y descarta el resto,
# así que los textos largos se trocean por frases
MAX_CHUNK_CHARS = 1500
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class GLiNERService:
    def __init__(self, model_name: str = GLINER_MODEL, threshold: float = 0.5):
        # Import diferido: carga PyTorch, innecesario si se usa spaCy
        from gliner import GLiNER

        self.model = GLiNER.from_pretrained(model_name)
        self.threshold = threshold

    def extract_entities(self, text: str) -> list[EntitySpan]:
        entities = []
        for offset, chunk in _chunks(text):
            predictions = self.model.predict_entities(
                chunk, list(GLINER_LABELS), threshold=self.threshold
            )
            for pred in predictions:
                entities.append(
                    EntitySpan(
                        text=pred["text"],
                        start_char=offset + int(pred["start"]),
                        end_char=offset + int(pred["end"]),
                        label=GLINER_LABELS[pred["label"]],
                    )
                )
        return entities


def _chunks(text: str) -> list[tuple[int, str]]:
    """Divide el texto en trozos de frases completas con su posición inicial."""
    chunks: list[tuple[int, str]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        if match.start() - start > MAX_CHUNK_CHARS:
            chunks.append((start, text[start : match.start()]))
            start = match.end()
    chunks.append((start, text[start:]))
    return chunks


# --- spaCy ---------------------------------------------------------------------

# Se usa el primer modelo instalado de la lista
SPACY_MODELS = ("es_core_news_lg", "es_core_news_md", "es_core_news_sm")
SPACY_LABELS = {"PER", "ORG", "LOC", "MISC"}


class SpacyNERService:
    def __init__(self, model_names: tuple[str, ...] = SPACY_MODELS):
        import spacy

        for name in model_names:
            try:
                self.nlp = spacy.load(name)
                break
            except OSError:
                continue
        else:
            raise OSError(f"No hay ningún modelo de spaCy instalado: {model_names}")

    def extract_entities(self, text: str) -> list[EntitySpan]:
        return [
            EntitySpan(
                text=ent.text,
                start_char=ent.start_char,
                end_char=ent.end_char,
                label=ent.label_,
            )
            for ent in self.nlp(text).ents
            if ent.label_ in SPACY_LABELS
        ]


def create_ner_service(backend: NERBackend = "gliner") -> NERService:
    if backend == "spacy":
        return SpacyNERService()
    return GLiNERService()
