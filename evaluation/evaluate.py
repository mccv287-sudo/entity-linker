"""Evaluación del entity linker sobre WikiAnc (Wikipedia en español).

Dataset: https://huggingface.co/datasets/cyanic-selkie/wikianc (config "es").
Cada fila es un párrafo de Wikipedia con sus enlaces internos ("anclas") y el
QID de Wikidata al que apuntan. Los enlaces rojos (a artículos que no existen)
no tienen QID y se usan como casos NIL. Solo se evalúan las anclas que empiezan
por mayúscula, como aproximación a "entidad nombrada" (WikiAnc también enlaza
conceptos comunes como "guitarra").

Se miden dos cosas:
  linking  Con las menciones anotadas como entrada: calidad de la búsqueda de
           candidatos y de la desambiguación, sin errores del NER.
  e2e      Pipeline completo (NER + linking) comparado con las anclas.

La primera ejecución descarga la muestra y la guarda en evaluation/data/; las
siguientes la leen de disco, así que todas evalúan los mismos párrafos.

Uso:
  uv run python evaluation/evaluate.py
  uv run python evaluation/evaluate.py --docs 50 --seed 7
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from entity_linker.models.schemas import EntitySpan, LinkedEntity
from entity_linker.services.ner import NERBackend
from entity_linker.services.pipeline import EntityLinkingPipeline

DATASET = "cyanic-selkie/wikianc"
CONFIG = "es"
SPLIT = "validation"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
EVAL_DIR = Path(__file__).parent

# Se leen bloques de filas contiguas (del mismo artículo) y se toman pocos
# párrafos de cada uno, de artículos distintos, para que la muestra sea variada
BLOCK_SIZE = 100
DOCS_PER_BLOCK = 3
MIN_CHARS, MAX_CHARS = 100, 1500
# Hugging Face limita las peticiones anónimas (429) y no envía Retry-After
MAX_RETRIES = 6


@dataclass(frozen=True)
class GoldMention:
    start: int
    end: int
    text: str
    qid: str  # "NIL" para enlaces rojos


@dataclass(frozen=True)
class Document:
    title: str
    text: str
    mentions: list[GoldMention]


# --- Descarga de la muestra --------------------------------------------------


def _fetch_rows(client: httpx.Client, offset: int, length: int) -> dict[str, Any]:
    params = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "offset": offset,
        "length": length,
    }
    for attempt in range(MAX_RETRIES):
        response = client.get(ROWS_URL, params=params)
        if response.status_code != 429 or attempt == MAX_RETRIES - 1:
            break
        wait = min(60, 5 * 2**attempt)
        print(f"Límite de peticiones de Hugging Face; reintento en {wait} s")
        time.sleep(wait)
    response.raise_for_status()
    return dict(response.json())


def _gold_mentions(row: dict[str, Any]) -> list[GoldMention]:
    text = row["paragraph_text"]
    return [
        GoldMention(
            start=a["start"],
            end=a["end"],
            text=text[a["start"] : a["end"]],
            qid=f"Q{a['qid']}" if a["qid"] else "NIL",
        )
        for a in row["paragraph_anchors"]
        if text[a["start"] : a["end"]][:1].isupper()
    ]


def download_sample(n_docs: int, seed: int) -> list[Document]:
    rng = random.Random(seed)
    docs: list[Document] = []
    with httpx.Client(timeout=30) as client:
        total = _fetch_rows(client, 0, 1)["num_rows_total"]
        while len(docs) < n_docs:
            offset = rng.randrange(0, total - BLOCK_SIZE)
            block = _fetch_rows(client, offset, BLOCK_SIZE)["rows"]
            rows = [item["row"] for item in block]
            rng.shuffle(rows)
            articles: set[str] = set()
            for row in rows:
                mentions = _gold_mentions(row)
                text = row["paragraph_text"]
                if (
                    mentions
                    and row["article_title"] not in articles
                    and MIN_CHARS <= len(text) <= MAX_CHARS
                ):
                    articles.add(row["article_title"])
                    docs.append(Document(row["article_title"], text, mentions))
                if len(articles) == DOCS_PER_BLOCK or len(docs) == n_docs:
                    break
            print(f"Descargando muestra: {len(docs)}/{n_docs}")
    return docs


def load_sample(n_docs: int, seed: int) -> list[Document]:
    """Lee la muestra de disco o la descarga (y la guarda) si no existe."""
    path = EVAL_DIR / "data" / f"wikianc_{CONFIG}_{SPLIT}_n{n_docs}_seed{seed}.jsonl"
    if not path.exists():
        docs = download_sample(n_docs, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
    with path.open(encoding="utf-8") as f:
        return [
            Document(d["title"], d["text"], [GoldMention(**m) for m in d["mentions"]])
            for d in map(json.loads, f)
        ]


# --- Evaluación --------------------------------------------------------------


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _find_overlapping(gold: GoldMention, spans: list[EntitySpan]) -> EntitySpan | None:
    return next(
        (s for s in spans if _overlaps(gold.start, gold.end, s.start_char, s.end_char)),
        None,
    )


async def evaluate_document(
    pipeline: EntityLinkingPipeline, doc: Document
) -> tuple[list[dict[str, Any]], int]:
    """Evalúa un párrafo en ambos modos.

    Returns:
        Una fila por mención anotada y el número total de entidades que
        detectó el NER (anotadas o no), para comparar lo "generoso" que es
        cada NER: Wikipedia no enlaza todas las entidades, así que las
        detecciones de más no se pueden juzgar como acierto o error.
    """
    predicted = pipeline.ner_service.extract_entities(doc.text)

    # linking: las menciones anotadas, con la etiqueta del NER si las detectó
    gold_spans = []
    for g in doc.mentions:
        match = _find_overlapping(g, predicted)
        label = match.label if match else "MISC"
        gold_spans.append(
            EntitySpan(text=g.text, start_char=g.start, end_char=g.end, label=label)
        )
    candidates_list = await asyncio.gather(
        *(pipeline.wikidata_service.get_candidates(s.text) for s in gold_spans)
    )
    linked_gold: list[LinkedEntity] = await asyncio.gather(
        *(
            pipeline.disambiguator_service.disambiguate(
                context_text=doc.text,
                entity_text=span.text,
                start_char=span.start_char,
                end_char=span.end_char,
                entity_label=span.label,
                candidates=candidates,
            )
            for span, candidates in zip(gold_spans, candidates_list)
        )
    )

    # e2e: el pipeline completo
    e2e = (await pipeline.process_text(doc.text)).entities

    rows = []
    for g, linked in zip(doc.mentions, linked_gold):
        e2e_match = next(
            (e for e in e2e if _overlaps(g.start, g.end, e.start_char, e.end_char)),
            None,
        )
        rows.append(
            {
                "doc": doc.title,
                "mention": g.text,
                "gold": g.qid,
                "ner_label": linked.label,
                "pred": linked.qid,
                "confidence": linked.confidence,
                "gold_in_candidates": g.qid in {c.qid for c in linked.candidates},
                "candidates": [f"{c.qid} {c.label}" for c in linked.candidates],
                "detected": e2e_match is not None,
                "e2e_mention": e2e_match.text if e2e_match else None,
                "e2e_pred": e2e_match.qid if e2e_match else None,
                "reasoning": linked.reasoning,
            }
        )
    return rows, len(predicted)


def _ratio(num: int, den: int) -> str:
    return f"{num / den:.3f} ({num}/{den})" if den else "n/a"


def compute_metrics(rows: list[dict[str, Any]], n_predicted: int) -> dict[str, str]:
    in_kb = [r for r in rows if r["gold"] != "NIL"]
    nil_gold = [r for r in rows if r["gold"] == "NIL"]
    nil_pred = [r for r in rows if r["pred"] == "NIL"]
    detected = [r for r in rows if r["detected"]]
    correct = [r for r in rows if r["pred"] == r["gold"]]
    return {
        # linking (menciones anotadas)
        "linking_accuracy": _ratio(len(correct), len(rows)),
        "linking_accuracy_in_kb": _ratio(
            sum(r["pred"] == r["gold"] for r in in_kb), len(in_kb)
        ),
        "candidate_recall": _ratio(
            sum(r["gold_in_candidates"] for r in in_kb), len(in_kb)
        ),
        "nil_precision": _ratio(
            sum(r["gold"] == "NIL" for r in nil_pred), len(nil_pred)
        ),
        "nil_recall": _ratio(
            sum(r["pred"] == "NIL" for r in nil_gold), len(nil_gold)
        ),
        # e2e (NER + linking)
        "ner_recall": _ratio(len(detected), len(rows)),
        "ner_detected_total": str(n_predicted),
        "e2e_recall": _ratio(sum(r["e2e_pred"] == r["gold"] for r in rows), len(rows)),
        "e2e_precision_on_annotated": _ratio(
            sum(r["e2e_pred"] == r["gold"] for r in detected), len(detected)
        ),
    }


async def run(n_docs: int, seed: int, ner: NERBackend) -> None:
    docs = load_sample(n_docs, seed)
    n_mentions = sum(len(d.mentions) for d in docs)
    n_nil = sum(m.qid == "NIL" for d in docs for m in d.mentions)
    print(f"{len(docs)} párrafos, {n_mentions} menciones anotadas ({n_nil} NIL)")
    print(f"NER: {ner}")

    pipeline = EntityLinkingPipeline(ner_backend=ner)
    rows: list[dict[str, Any]] = []
    n_predicted = 0
    # Documento a documento para no saturar la API de Wikidata
    for i, doc in enumerate(docs, 1):
        doc_rows, doc_predicted = await evaluate_document(pipeline, doc)
        rows.extend(doc_rows)
        n_predicted += doc_predicted
        print(f"\rEvaluando: {i}/{len(docs)}", end="", flush=True)
    print()

    metrics = compute_metrics(rows, n_predicted)
    for name, value in metrics.items():
        print(f"  {name:28} {value}")

    report = EVAL_DIR / "results" / f"report_{ner}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    errors = [r for r in rows if r["pred"] != r["gold"] or r["e2e_pred"] != r["gold"]]
    content = {"metrics": metrics, "errors": errors}
    report.write_text(
        json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nMétricas y errores en {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa el entity linker en WikiAnc")
    parser.add_argument("--docs", type=int, default=100, help="párrafos a evaluar")
    parser.add_argument("--seed", type=int, default=42, help="semilla de la muestra")
    parser.add_argument(
        "--ner", choices=["gliner", "spacy"], default="gliner", help="modelo NER"
    )
    args = parser.parse_args()
    asyncio.run(run(args.docs, args.seed, args.ner))


if __name__ == "__main__":
    main()
