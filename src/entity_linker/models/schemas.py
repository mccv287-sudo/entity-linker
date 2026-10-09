from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Tipos reutilizables con sus restricciones
Qid = Annotated[str, Field(pattern=r"^Q\d+$", description="Identificador de Wikidata")]
QidOrNil = Annotated[
    str, Field(pattern=r"^(Q\d+|NIL)$", description="QID de Wikidata o NIL")
]
Confidence = Annotated[
    float, Field(ge=0.0, le=1.0, description="Puntuación de confianza (0.0 a 1.0)")
]


class StrictModel(BaseModel):
    """Base de los modelos internos: sin conversiones implícitas de tipos."""

    model_config = ConfigDict(strict=True)


class EntitySpan(StrictModel):
    text: str = Field(..., description="Texto de la entidad extraída")
    start_char: int = Field(
        ..., ge=0, description="Índice de inicio en el texto original"
    )
    end_char: int = Field(..., ge=0, description="Índice de fin en el texto original")
    label: str = Field(..., description="Categoría NER (PER, ORG, LOC, MISC)")


class WikidataCandidate(StrictModel):
    """Candidato recuperado de la API de Wikidata."""

    qid: Qid
    label: str = Field(..., description="Etiqueta u nombre oficial en Wikidata")
    description: str | None = Field(
        default=None, description="Descripción corta de la entidad"
    )
    concept_uri: str = Field(..., description="URI completa en Wikidata")


class LLMDisambiguationResult(StrictModel):
    """Salida estructurada que se exige al LLM al desambiguar.

    Se valida desde el JSON del LLM con ``model_validate_json``: un campo con
    el tipo equivocado (p. ej. ``"0.9"`` como texto) es un error, y el
    desambiguador recurre a su alternativa.
    """

    # El LLM puede añadir campos que no se piden; se ignoran
    model_config = ConfigDict(strict=True, extra="ignore")

    selected_qid: QidOrNil
    confidence: Confidence
    is_nil: bool = Field(
        ..., description="True si la entidad no está en Wikidata o no hay confianza"
    )
    reasoning: str = Field(
        ..., description="Breve explicación del candidato o NIL elegido"
    )

    @model_validator(mode="after")
    def _nil_is_consistent(self) -> "LLMDisambiguationResult":
        # Los LLM pequeños a veces responden selected_qid="NIL" con
        # is_nil=false: se corrige en lugar de descartar la respuesta
        if self.selected_qid == "NIL":
            self.is_nil = True
        return self


class LinkedEntity(EntitySpan):
    qid: QidOrNil = "NIL"
    wikidata_label: str | None = Field(
        default=None, description="Nombre oficial en Wikidata"
    )
    wikidata_description: str | None = Field(
        default=None, description="Descripción oficial en Wikidata"
    )
    confidence: Confidence
    is_nil: bool = Field(
        default=False, description="Indica si la entidad no pudo ser vinculada"
    )
    reasoning: str | None = Field(
        default=None, description="Motivo de la decisión (LLM o regla aplicada)"
    )
    candidates: list[WikidataCandidate] = Field(
        default_factory=list, description="Candidatos evaluados"
    )


class TextLinkingRequest(BaseModel):
    text: str = Field(
        ...,
        json_schema_extra={
            "example": "La Universidad Politécnica de Madrid está ubicada en España."
        },
    )


class TextLinkingResponse(BaseModel):
    original_text: str
    entities: list[LinkedEntity]
    processing_time_ms: float
