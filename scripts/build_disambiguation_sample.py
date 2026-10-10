"""Construye una muestra para evaluar la desambiguación sobre AIDA.

Se queda con las menciones en las que el NER y el retrieval han funcionado, para
aislar el problema de la desambiguación:
- GLiNER detecta la mención con los límites exactos del gold.
- El QID correcto está entre los candidatos de Wikidata (en cualquier posición).

``gold_rank`` = 1 son casos que el primer candidato ya resuelve (control);
``gold_rank`` > 1 son los que necesitan desambiguación.

Uso:
  uv run python scripts/build_disambiguation_sample.py --docs 30
"""

import argparse
import asyncio
import json
from collections import Counter

from entity_linker.evaluation.datasets import DATA_DIR, load_aida
from entity_linker.services.ner import NERService
from entity_linker.services.wikidata import WikidataService

CANDIDATES = 20


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=30, help="documentos de AIDA test")
    args = parser.parse_args()

    docs = load_aida("test")[: args.docs]
    ner = NERService()
    wikidata = WikidataService()

    # 1. Menciones con QID que GLiNER detecta con los límites exactos
    mentions = []
    for doc in docs:
        detected = {
            (e.start_char, e.end_char): e.label for e in ner.extract_entities(doc.text)
        }
        for m in doc.mentions:
            if m.qid and (m.start, m.end) in detected:
                mentions.append((doc, m, detected[(m.start, m.end)]))

    # 2. Candidatos de Wikidata (cada texto distinto se busca una vez)
    texts = list({doc.text[m.start : m.end] for doc, m, _ in mentions})
    results = await asyncio.gather(
        *(wikidata.get_candidates(t, limit=CANDIDATES) for t in texts)
    )
    found = dict(zip(texts, results))

    # 3. Se guardan las menciones cuyo QID correcto está entre los candidatos
    path = DATA_DIR / "disambiguation_aida_test.jsonl"
    ranks = Counter()
    with path.open("w", encoding="utf-8") as f:
        for doc, m, label in mentions:
            text = doc.text[m.start : m.end]
            qids = [c.qid for c in found[text]]
            if m.qid not in qids:
                ranks["fuera de los candidatos"] += 1
                continue
            rank = qids.index(m.qid) + 1
            ranks["1" if rank == 1 else "2-10" if rank <= 10 else ">10"] += 1
            record = {
                "doc_id": doc.doc_id,
                "text": doc.text,
                "start": m.start,
                "end": m.end,
                "mention": text,
                "label": label,
                "gold": m.qid,
                "gold_rank": rank,
                "candidates": [c.model_dump() for c in found[text]],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{len(docs)} documentos, {len(mentions)} menciones con QID detectadas")
    print("Posición del QID correcto:", dict(ranks))
    print(f"Muestra guardada en {path}")


if __name__ == "__main__":
    asyncio.run(main())
