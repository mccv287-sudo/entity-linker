"""Evalúa el NER, el retrieval o el entity linking completo sobre AIDA.

Uso:
  uv run python -m entity_linker.evaluation.evaluate
  uv run python -m entity_linker.evaluation.evaluate --provider both --limit 5
  uv run python -m entity_linker.evaluation.evaluate --stage retrieval --limit 20
  uv run python -m entity_linker.evaluation.evaluate --stage linking --limit 10
  uv run python -m entity_linker.evaluation.evaluate --stage linking --llm --limit 1

El NER con LLM usa Ollama en local (``ollama pull qwen2.5:3b``); ``--llm-model``
permite probar otro modelo descargado en Ollama.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from entity_linker.config import LLM_MODEL, REVIEW_THRESHOLD
from entity_linker.evaluation.datasets import (
    Document,
    load_aida,
    load_disambiguation_sample,
)
from entity_linker.evaluation.metrics import (
    linking_metrics,
    ner_metrics,
    retrieval_metrics,
)
from entity_linker.models.schemas import EntitySpan
from entity_linker.services.clients.wikidata import WikidataService
from entity_linker.services.linking.disambiguator import LLMDisambiguatorService
from entity_linker.services.ner.llm_ner import LLMNERError, LLMNERService

RESULTS_DIR = Path(__file__).parent / "results"


def evaluate_ner(
    ner: Any, docs: list[Document], labels: dict[str, str] | None = None
) -> dict[str, Any]:
    """Métricas, ms por documento, errores por documento y tokens (si es un LLM).

    ``ner`` es cualquier servicio con ``extract_entities(text, labels)``.
    """
    gold_docs, pred_docs, errors = [], [], []
    start = time.perf_counter()
    for doc in docs:
        gold = {(m.start, m.end, m.label) for m in doc.mentions if m.label}
        error: dict[str, Any] = {}
        try:
            predicted = ner.extract_entities(doc.text, labels)
            pred = {(e.start_char, e.end_char, e.label) for e in predicted}
        except LLMNERError as exc:
            pred, error["request_error"] = set(), str(exc)
        gold_docs.append(gold)
        pred_docs.append(pred)

        # Entidades inventadas o con tipos no pedidos (solo el LLM las reporta)
        diagnostics = getattr(ner, "last_diagnostics", {})
        error.update({k: v for k, v in diagnostics.items() if v})
        if gold != pred or error:
            errors.append(
                {
                    "doc_id": doc.doc_id,
                    "missed": _describe(doc.text, gold - pred),
                    "spurious": _describe(doc.text, pred - gold),
                    **error,
                }
            )

    result: dict[str, Any] = {
        "metrics": ner_metrics(gold_docs, pred_docs),
        "ms_per_doc": round((time.perf_counter() - start) * 1000 / len(docs), 1),
        "errors": errors,
    }
    if isinstance(ner, LLMNERService):
        result["tokens"] = {
            "input": ner.total_input_tokens,
            "output": ner.total_output_tokens,
        }
    return result


async def evaluate_retrieval(docs: list[Document], limit: int) -> dict[str, Any]:
    """Busca en Wikidata cada mención anotada con QID (sin pasar por el NER)."""
    wikidata = WikidataService()
    mentions = [
        (doc.text[m.start : m.end], m.qid)
        for doc in docs
        for m in doc.mentions
        if m.qid
    ]
    # Cada texto distinto se busca una sola vez
    texts = list({text for text, _ in mentions})
    results = await asyncio.gather(
        *(wikidata.get_candidates(text, limit=limit) for text in texts)
    )
    found = {text: [c.qid for c in cands] for text, cands in zip(texts, results)}
    candidates = [found[text] for text, _ in mentions]
    return {
        "metrics": retrieval_metrics([qid for _, qid in mentions], candidates),
        "errors": [
            {"mention": text, "gold": qid, "candidates": cands[:5]}
            for (text, qid), cands in zip(mentions, candidates)
            if qid not in cands
        ],
    }


async def evaluate_linking(docs: list[Document], pipeline: Any) -> dict[str, Any]:
    """Pipeline completo frente al gold de AIDA (QID o NIL por mención)."""
    calls_before = pipeline.disambiguator_service.llm_calls
    gold_docs, pred_docs, errors = [], [], []
    start = time.perf_counter()
    for doc in docs:
        gold = {(m.start, m.end, m.qid or "NIL") for m in doc.mentions}
        linked = await pipeline.link(doc.text)
        pred = {(e.start_char, e.end_char, e.qid or "NIL") for e in linked}
        gold_docs.append(gold)
        pred_docs.append(pred)
        if gold != pred:
            errors.append(
                {
                    "doc_id": doc.doc_id,
                    "missed": _describe(doc.text, gold - pred),
                    "spurious": _describe(doc.text, pred - gold),
                }
            )
    return {
        "metrics": linking_metrics(gold_docs, pred_docs),
        "ms_per_doc": round((time.perf_counter() - start) * 1000 / len(docs), 1),
        "llm_calls": pipeline.disambiguator_service.llm_calls - calls_before,
        "errors": errors,
    }


def _bucket(gold_rank: int) -> str:
    return "rank 1" if gold_rank == 1 else "rank 2-10" if gold_rank <= 10 else ">10"


async def evaluate_disambiguation(
    rows: list[dict[str, Any]], disambiguator: Any, use_llm: bool
) -> dict[str, Any]:
    """Accuracy del desambiguador por tramo de dificultad (posición del QID
    correcto entre los candidatos de Wikidata) y validez de su probabilidad.

    ``confident``/``uncertain``: accuracy de las menciones con confianza por
    encima / por debajo de ``REVIEW_THRESHOLD``; si la probabilidad es útil, la
    primera debe ser mucho mayor. Agrupa por documento, como el pipeline.
    """
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], []).append(r)

    calls_before = disambiguator.llm_calls
    hits: dict[str, list[bool]] = {}
    start = time.perf_counter()
    for doc_rows in by_doc.values():
        spans = [
            EntitySpan(
                text=r["mention"],
                start_char=r["start"],
                end_char=r["end"],
                label=r["label"],
                fine_label=r.get("fine_label"),
            )
            for r in doc_rows
        ]
        linked = await disambiguator.disambiguate_document(
            doc_rows[0]["text"],
            spans,
            [r["candidates"] for r in doc_rows],
            use_llm=use_llm,
        )
        for r, e in zip(doc_rows, linked):
            hit = e.qid == r["gold"]
            hits.setdefault(_bucket(r["gold_rank"]), []).append(hit)
            confident = e.confidence >= REVIEW_THRESHOLD
            hits.setdefault("confident" if confident else "uncertain", []).append(hit)

    all_hits = [
        h for b, bucket in hits.items() if b.startswith(("rank", ">")) for h in bucket
    ]
    return {
        "accuracy": {
            "all": round(sum(all_hits) / len(all_hits), 3),
            **{b: round(sum(h) / len(h), 3) for b, h in sorted(hits.items())},
        },
        "mentions": {b: len(h) for b, h in sorted(hits.items())},
        "llm_calls": disambiguator.llm_calls - calls_before,
        "seconds": round(time.perf_counter() - start, 1),
    }


async def _compare_disambiguation(limit: int | None, llm: bool) -> dict[str, Any]:
    """Primer candidato vs señales sin red vs +tipo/coherencia vs +LLM."""
    rows = load_disambiguation_sample()[:limit]
    first = sum(r["gold_rank"] == 1 for r in rows) / len(rows)
    print(f"{'1er candidato':26} accuracy {first:.3f}")
    wikidata = WikidataService()
    disambiguator = LLMDisambiguatorService(wikidata)
    configs = {"señales sin red": (False, False), "+tipo+coherencia": (True, False)}
    if llm:
        configs["+tipo+coherencia+LLM"] = (True, True)
    report: dict[str, Any] = {"1er candidato": first}
    for name, (use_types, use_llm) in configs.items():
        disambiguator.use_types = use_types
        report[name] = await evaluate_disambiguation(rows, disambiguator, use_llm)
        print(f"{name:26} {report[name]}")
    await wikidata.aclose()
    return report


def print_metrics(metrics: dict[str, dict[str, float | int]]) -> None:
    print(f"  {'':12}{'P':>7}{'R':>7}{'F1':>7}")
    for name, m in metrics.items():
        print(f"  {name:12}{m['precision']:7.3f}{m['recall']:7.3f}{m['f1']:7.3f}")


def _describe(text: str, spans: set[tuple[int, int, str]]) -> list[str]:
    return [f"{text[s:e]} [{label}]" for s, e, label in sorted(spans)]


def _create_ner(provider: str, args: argparse.Namespace) -> Any:
    if provider == "gliner":
        from entity_linker.services.ner.gliner import NERService  # carga PyTorch

        return NERService(threshold=args.threshold)
    return LLMNERService(model=args.llm_model)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalúa NER o retrieval sobre AIDA")
    parser.add_argument("--split", default="test", help="train, validation o test")
    parser.add_argument("--limit", type=int, help="solo los N primeros documentos")
    parser.add_argument(
        "--stage",
        choices=["ner", "retrieval", "linking", "disambiguation"],
        default="ner",
    )
    parser.add_argument(
        "--llm", action="store_true", help="linking: el LLM resuelve las dudosas"
    )
    parser.add_argument(
        "--provider", choices=["gliner", "llm", "both"], default="gliner"
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="umbral GLiNER")
    parser.add_argument("--candidates", type=int, default=20, help="límite de búsqueda")
    parser.add_argument("--llm-model", default=LLM_MODEL, help="modelo de Ollama")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    docs = load_aida(args.split)[: args.limit]
    print(f"AIDA [{args.split}]: {len(docs)} documentos")

    if args.stage == "retrieval":
        report = asyncio.run(evaluate_retrieval(docs, args.candidates))
        print(report["metrics"])
        name = "retrieval"
    elif args.stage == "disambiguation":
        report = asyncio.run(_compare_disambiguation(args.limit, args.llm))
        name = "disambiguation"
    elif args.stage == "linking":
        from entity_linker.services.pipeline import EntityLinkingPipeline  # GLiNER

        report = asyncio.run(
            evaluate_linking(docs, EntityLinkingPipeline(use_llm=args.llm))
        )
        print(f"{report['ms_per_doc']} ms/doc; {report['llm_calls']} llamadas LLM")
        print_metrics(report["metrics"])
        name = "linking_llm" if args.llm else "linking"
    else:
        providers = ["gliner", "llm"] if args.provider == "both" else [args.provider]
        report = {}
        for provider in providers:
            report[provider] = evaluate_ner(_create_ner(provider, args), docs)
            print(f"\n== {provider} ({report[provider]['ms_per_doc']} ms/documento)")
            print_metrics(report[provider]["metrics"])
        name = f"ner_{args.provider}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}_aida_{args.split}.json"
    path.write_text(
        json.dumps({"settings": vars(args), **report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nInforme con los errores en {path}")


if __name__ == "__main__":
    main()
