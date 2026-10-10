"""Adaptadores a librerías estándar de métricas.

- NER: ``nervaluate`` (esquema de SemEval-2013 / MUC). ``strict`` (límites y
  tipo exactos) equivale a la métrica de CoNLL; ``exact`` (solo límites) se
  expone como ``boundaries``; ``partial`` da crédito parcial por solapamiento.
- Retrieval: ``ranx``, con cada mención como consulta y su QID como único
  documento relevante: ``hit_rate@k`` es la fracción de menciones con el QID
  correcto entre los k primeros candidatos.

Los nombres y el formato de salida se mantienen para no cambiar a quien llama.
"""

from typing import cast

from nervaluate import Evaluator
from ranx import Qrels, Run, evaluate

Span = tuple[int, int, str]  # (inicio, fin exclusivo, etiqueta)

# Esquema de nervaluate -> nombre que se expone
NER_SCHEMAS = {"strict": "strict", "exact": "boundaries", "partial": "partial"}


def ner_metrics(
    gold_docs: list[set[Span]], pred_docs: list[set[Span]]
) -> dict[str, dict[str, float | int]]:
    """P/R/F1 globales (``strict``, ``boundaries``, ``partial``) y por etiqueta.

    Las etiquetas por tipo usan el esquema ``strict``.
    """
    tags = sorted({s[2] for spans in (*gold_docs, *pred_docs) for s in spans})
    results = Evaluator(
        [_to_nervaluate(spans) for spans in gold_docs],
        [_to_nervaluate(spans) for spans in pred_docs],
        tags=tags,
        loader="dict",  # explícito: deducirlo falla si el 1er doc no tiene entidades
    ).evaluate()
    metrics = {
        name: _prf(results["overall"][schema]) for schema, name in NER_SCHEMAS.items()
    }
    for tag in tags:
        metrics[tag] = _prf(results["entities"][tag]["strict"])
    return metrics


def linking_metrics(
    gold_docs: list[set[Span]], pred_docs: list[set[Span]]
) -> dict[str, dict[str, float | int]]:
    """Entity linking end-to-end; cada span lleva como etiqueta su QID o "NIL".

    Equivalentes a las métricas estándar de GERBIL / TAC-KBP:
    - ``mention``: límites correctos (calidad del NER).
    - ``all``: límites y QID correctos (*strong annotation match*).
    - ``linked``: solo menciones enlazadas a un QID (sin NIL en gold ni pred).
    - ``nil``: solo menciones NIL: ¿se detecta que la entidad no está en Wikidata?
    """

    def only(docs: list[set[Span]], nil: bool) -> list[set[Span]]:
        return [{s for s in spans if (s[2] == "NIL") == nil} for spans in docs]

    overall = ner_metrics(gold_docs, pred_docs)
    return {
        "mention": overall["boundaries"],
        "all": overall["strict"],
        "linked": ner_metrics(only(gold_docs, False), only(pred_docs, False))["strict"],
        "nil": ner_metrics(only(gold_docs, True), only(pred_docs, True))["strict"],
    }


def retrieval_metrics(
    gold_qids: list[str], candidates: list[list[str]], ks: tuple[int, ...] = (1, 5, 10)
) -> dict[str, float]:
    """Calidad de la lista de candidatos de cada mención (un QID correcto).

    ``recall@k``: QID correcto entre los k primeros; ``recall@all``: en
    cualquier posición (techo del desambiguador); ``mrr``: inverso de la
    posición media; ``no_candidates``: menciones sin ningún candidato.
    """
    n = len(gold_qids)
    qrels = Qrels({f"m{i}": {qid: 1} for i, qid in enumerate(gold_qids)})
    # ranx ordena por puntuación: el primer candidato recibe la más alta
    run = Run(
        {
            f"m{i}": {qid: float(len(cands) - rank) for rank, qid in enumerate(cands)}
            for i, cands in enumerate(candidates)
        }
    )
    names = [f"hit_rate@{k}" for k in ks] + ["hit_rate", "mrr"]
    # ranx devuelve un float si se pide una sola métrica y un dict si son varias
    scores = cast(dict[str, float], evaluate(qrels, run, names, make_comparable=True))
    metrics = {f"recall@{k}": scores[f"hit_rate@{k}"] for k in ks}
    metrics["recall@all"] = scores["hit_rate"]
    metrics["mrr"] = scores["mrr"]
    metrics["no_candidates"] = sum(not c for c in candidates) / n
    return {k: round(float(v), 4) for k, v in metrics.items()}


def _to_nervaluate(spans: set[Span]) -> list[dict[str, int | str]]:
    # nervaluate usa el fin inclusivo para calcular solapamientos
    return [
        {"label": label, "start": start, "end": end - 1}
        for start, end, label in sorted(spans)
    ]


def _prf(result) -> dict[str, float | int]:
    return {
        "precision": round(result.precision, 4),
        "recall": round(result.recall, 4),
        "f1": round(result.f1, 4),
        "tp": result.correct,
        "fp": result.actual - result.correct,
        "fn": result.possible - result.correct,
    }
