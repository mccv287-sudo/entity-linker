"""Reconocimiento de entidades con GLiNER (zero-shot)."""

import re

from entity_linker.models.schemas import EntitySpan
from entity_linker.services.entity_types import FINE_LABELS

MODEL_NAME = "urchade/gliner_multi-v2.1"

# Etiquetas genéricas: solo como referencia para comparar con los tipos finos
# (FINE_LABELS, el valor por defecto)
LABELS = {
    "person": "PER",
    "organization": "ORG",
    "location": "LOC",
    "miscellaneous entity": "MISC",
}

# GLiNER procesa ~384 palabras por pasada y descarta el resto: los textos
# largos se trocean por frases
MAX_CHUNK_CHARS = 1500
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class NERService:
    def __init__(self, model_name: str = MODEL_NAME, threshold: float = 0.5):
        # Import diferido: PyTorch solo se carga si se usa GLiNER (no al
        # importar este módulo, p. ej. desde el NER con LLM)
        from gliner import GLiNER

        self.model = GLiNER.from_pretrained(model_name)
        self.threshold = threshold

    def extract_entities(
        self, text: str, labels: dict[str, str] | None = None
    ) -> list[EntitySpan]:
        """Detecta entidades; ``labels`` (tipo pedido -> etiqueta CoNLL) permite
        probar otros prompts. Por defecto, los tipos finos (``FINE_LABELS``)."""
        labels = labels or FINE_LABELS
        entities = []
        for offset, chunk in _chunks(text):
            for pred in self.model.predict_entities(
                chunk, list(labels), threshold=self.threshold
            ):
                start, end = offset + pred["start"], offset + pred["end"]
                entities.append(
                    EntitySpan(
                        text=text[start:end],
                        start_char=start,
                        end_char=end,
                        label=labels[pred["label"]],
                        fine_label=pred["label"],
                    )
                )
        return entities


def _chunks(text: str) -> list[tuple[int, str]]:
    """Trozos de frases completas, con su posición en el texto original."""
    chunks = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        if match.start() - start > MAX_CHUNK_CHARS:
            chunks.append((start, text[start : match.start()]))
            start = match.end()
    chunks.append((start, text[start:]))
    return chunks
