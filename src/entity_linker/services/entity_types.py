"""Tipos finos de GLiNER y su compatibilidad con las clases de Wikidata.

GLiNER detecta tipos finos ("sports team", "city"...), que se agrupan en el
esquema CoNLL (PER, ORG, LOC, MISC) para la salida y la evaluación del NER,
pero se conservan en ``EntitySpan.fine_label`` para filtrar candidatos: un
"sports team" encaja con un candidato que desciende (P31/P279*) de la clase
Q12973014 y no con una ciudad.
"""

from entity_linker.models.schemas import WikidataCandidate

# Tipo fino pedido a GLiNER -> etiqueta CoNLL
FINE_LABELS = {
    "person": "PER",
    "sports team": "ORG",
    "company": "ORG",
    "political party": "ORG",
    "news agency": "ORG",
    "international organization": "ORG",
    "government agency": "ORG",
    "university": "ORG",
    "organization": "ORG",
    "country": "LOC",
    "city": "LOC",
    "region": "LOC",
    "location": "LOC",
    "nationality": "MISC",
    "sports competition": "MISC",
    "event": "MISC",
    "award": "MISC",
    "miscellaneous entity": "MISC",
}

# Tipo fino -> clase raíz de Wikidata. Sin entrada: no se filtra (tipo genérico)
FINE_TYPE_CLASSES = {
    "person": "Q5",  # human
    "sports team": "Q12973014",
    "company": "Q4830453",  # business
    "political party": "Q7278",
    "news agency": "Q192283",
    "international organization": "Q484652",
    "government agency": "Q327333",
    "university": "Q3918",
    "organization": "Q43229",
    "country": "Q6256",
    "nationality": "Q6256",  # AIDA enlaza los gentilicios al país ("Chinese" -> China)
    "city": "Q515",
    "region": "Q82794",
    "location": "Q2221906",  # geographic location
    "sports competition": "Q13406554",
    "event": "Q1190554",  # occurrence
    "award": "Q618779",
}

Classes = dict[str, tuple[set[str], set[str]]]  # QID -> (P31 directas, ancestros)


def fits_type(fine_label: str | None, qid: str, classes: Classes) -> bool:
    """El candidato desciende de la clase del tipo fino (o el tipo es genérico)."""
    root = FINE_TYPE_CLASSES.get(fine_label or "")
    return root is None or root in classes.get(qid, (set(), set()))[1]


def rerank_by_type(
    fine_label: str | None, candidates: list[WikidataCandidate], classes: Classes
) -> list[WikidataCandidate]:
    """Pone primero los candidatos compatibles con el tipo fino.

    Reordena sin descartar (si el NER se equivoca de tipo, el correcto sigue en
    la lista) y respeta el orden de Wikidata dentro de cada grupo.
    """
    return sorted(candidates, key=lambda c: not fits_type(fine_label, c.qid, classes))


def rerank_by_coherence(
    fine_label: str | None,
    candidates: list[WikidataCandidate],
    resolved: list[str],
    classes: Classes,
) -> list[WikidataCandidate]:
    """Coherencia barata: dentro de los compatibles con el tipo, primero los que
    comparten clase directa (P31) con entidades ya resueltas del documento.

    En una noticia con selecciones de fútbol ya resueltas, "Kuwait" (tipo
    "sports team") sube la selección por encima de un club. La coherencia solo
    desempata: nunca pasa por delante de la compatibilidad de tipo (si no, en
    un documento lleno de personas cualquier candidato persona subiría).
    """
    context = [classes.get(q, (set(), set()))[0] for q in resolved]

    def key(cand: WikidataCandidate) -> tuple[bool, int]:
        direct = classes.get(cand.qid, (set(), set()))[0]
        shared = sum(bool(direct & other) for other in context)
        return fits_type(fine_label, cand.qid, classes), shared

    return sorted(candidates, key=key, reverse=True)  # estable: empates en orden
