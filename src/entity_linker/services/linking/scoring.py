"""Señales de desambiguación y probabilidad de cada candidato.

Cada señal vale entre 0 y 1; se combinan linealmente (``WEIGHTS``) y un softmax
sobre los candidatos de una mención da su probabilidad (un logit condicional):

- ``prior``: relevancia según la búsqueda de Wikidata (1 / posición), que
  aproxima la popularidad del sentido. Se calcula a partir del orden.
- ``context``: palabras de la descripción del candidato cerca de la mención
  (sin contar las de la propia mención: la similitud de nombre ya está en
  ``prior``, porque Wikidata busca por etiqueta y alias).
- ``type``: compatibilidad con el tipo fino del NER (clases P31/P279*).
- ``coherence``: clases compartidas con las entidades ya resueltas del documento.
- ``llm``: 1 para el candidato que elige el LLM (si se le consulta).

Los pesos se aprenden por máxima verosimilitud sobre 13 documentos de la
muestra de AIDA y se validan en otros 17 (``scripts/tune_weights.py``). La
similitud de nombre se descartó: con homónimos que se llaman igual, su peso
óptimo era 0.
"""

import re

from scipy.special import softmax

from entity_linker.config import CONTEXT_CHARS, WEIGHTS
from entity_linker.models.schemas import WikidataCandidate
from entity_linker.services.entity_types import FINE_TYPE_CLASSES, Classes

_WORD = re.compile(r"[a-záéíóúüñ]{4,}")
_NO_CLASSES: tuple[set[str], set[str]] = (set(), set())


def surroundings(text: str, start: int, end: int) -> str:
    """Texto alrededor de la mención (``CONTEXT_CHARS`` a cada lado), sin la
    propia mención."""
    return text[max(0, start - CONTEXT_CHARS) : start] + text[end : end + CONTEXT_CHARS]


def context_score(cand: WikidataCandidate, context: str, mention: str) -> float:
    """Fracción de palabras de la descripción del candidato que aparecen cerca
    de la mención ("football club" en una noticia de fútbol).

    Las palabras de la mención no cuentan ni en el contexto ni en la
    descripción: que la descripción repita el nombre ("Nobel Prize in Physics
    controversies") es similitud de nombre, no de contexto."""
    name = _words(mention)
    words = _words(cand.description or "") - name
    return len(words & (_words(context) - name)) / len(words) if words else 0.0


def type_score(fine_label: str | None, qid: str, classes: Classes) -> float:
    """1 si el candidato desciende de la clase del tipo fino, 0 si no, 0,5 si no
    se sabe (tipo genérico o sin clases consultadas)."""
    root = FINE_TYPE_CLASSES.get(fine_label or "")
    if root is None or qid not in classes:
        return 0.5
    return 1.0 if root in classes[qid][1] else 0.0


def coherence_score(qid: str, resolved: list[str], classes: Classes) -> float:
    """Fracción de entidades ya resueltas del documento que comparten clase
    directa (P31) con el candidato: en una noticia con selecciones de fútbol,
    "Kuwait" se acerca a la selección y no al país."""
    if not resolved:
        return 0.0
    direct = classes.get(qid, _NO_CLASSES)[0]
    shared = sum(bool(direct & classes.get(q, _NO_CLASSES)[0]) for q in resolved)
    return shared / len(resolved)


def probabilities(signals: list[dict[str, float]]) -> list[float]:
    """Probabilidad de cada candidato (en el orden de Wikidata): softmax de la
    combinación lineal de sus señales, más ``prior`` según la posición."""
    if not signals:
        return []
    scores = [
        WEIGHTS["prior"] / (rank + 1)
        + sum(WEIGHTS[name] * value for name, value in sig.items())
        for rank, sig in enumerate(signals)
    ]
    return softmax(scores).tolist()


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))
