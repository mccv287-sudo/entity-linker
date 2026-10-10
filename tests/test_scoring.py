from entity_linker.models.schemas import WikidataCandidate
from entity_linker.services.entity_types import FINE_TYPE_CLASSES
from entity_linker.services.linking.scoring import (
    argmax,
    coherence_score,
    context_score,
    probabilities,
    surroundings,
    type_score,
)


def candidate(qid: str, description: str | None = None) -> WikidataCandidate:
    return WikidataCandidate(
        qid=qid,
        label=qid,
        description=description,
        concept_uri=f"http://www.wikidata.org/entity/{qid}",
    )


def test_context_score_counts_description_words_in_context():
    cand = candidate("Q1", "French football club")
    assert context_score(cand, "the football club won in Paris", "PSG") == 2 / 3
    assert context_score(candidate("Q2"), "any text", "PSG") == 0.0


def test_context_ignores_the_mention():
    text = "Curie won the Nobel Prize in Physics in 1903."
    mention = "Nobel Prize in Physics"
    start = text.index(mention)
    context = surroundings(text, start, start + len(mention))
    assert context == "Curie won the  in 1903."
    # La descripción que repite el nombre no gana contexto (ni aunque el
    # nombre aparezca otra vez cerca)
    repeats_name = candidate("Q1", "controversies around the Nobel Prize in Physics")
    assert context_score(repeats_name, text, mention) == 0.0


def test_type_score():
    person = FINE_TYPE_CLASSES["person"]
    classes = {"Q1": ({"Q5"}, {person}), "Q2": ({"Q515"}, set())}
    assert type_score("person", "Q1", classes) == 1.0
    assert type_score("person", "Q2", classes) == 0.0
    # Tipo genérico o candidato sin clases consultadas: no se sabe
    assert type_score(None, "Q1", classes) == 0.5
    assert type_score("person", "Q3", classes) == 0.5


def test_coherence_score_shares_direct_classes_with_resolved():
    classes = {
        "Q1": ({"Q6979593"}, set()),  # selección nacional de fútbol
        "Q2": ({"Q6256"}, set()),  # país
        "Q3": ({"Q6979593"}, set()),
    }
    assert coherence_score("Q1", ["Q3"], classes) == 1.0
    assert coherence_score("Q2", ["Q3"], classes) == 0.0
    assert coherence_score("Q1", [], classes) == 0.0


def test_probabilities_sum_to_one_and_favor_first_rank():
    probs = probabilities([{"context": 0.0}, {"context": 0.0}, {"context": 0.0}])
    assert abs(sum(probs) - 1) < 1e-9
    assert argmax(probs) == 0
    assert probabilities([]) == []


def test_probabilities_signal_can_beat_rank():
    probs = probabilities([{"context": 0.0}, {"context": 1.0}])
    assert argmax(probs) == 1
