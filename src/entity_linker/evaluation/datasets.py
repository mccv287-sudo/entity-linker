"""Datasets anotados (inglés) con QIDs de Wikidata, descargados de Hugging Face.

- AIDA CoNLL-YAGO: noticias de Reuters con *todas* las entidades anotadas
  (PER, ORG, LOC, MISC) y su QID (None si no está en Wikidata). Sirve para NER
  y retrieval. https://huggingface.co/datasets/cyanic-selkie/aida-conll-yago-wikidata
- WikiAnc: párrafos de Wikipedia con sus enlaces internos y el QID al que
  apuntan. Solo sirve para retrieval: Wikipedia no enlaza todas las entidades
  y no hay etiqueta de tipo. https://huggingface.co/datasets/cyanic-selkie/wikianc
- Muestra de desambiguación: menciones de AIDA con NER correcto y QID entre los
  candidatos, generada con ``scripts/build_disambiguation_sample.py``.

Cada dataset se descarga una vez y se guarda en ``evaluation/data/``.
"""

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from entity_linker.models.schemas import WikidataCandidate

ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATA_DIR = Path(__file__).parent / "data"
DISAMBIGUATION_SAMPLE = DATA_DIR / "disambiguation_aida_test.jsonl"
PAGE_SIZE = 100  # máximo que permite la API de Hugging Face
DOCS_PER_BLOCK = 10


@dataclass(frozen=True)
class GoldMention:
    start: int
    end: int
    label: str | None  # PER, ORG, LOC, MISC; None si el dataset no lo anota
    qid: str | None  # None: NIL o sin enlace


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    mentions: list[GoldMention]


def load_aida(split: str = "test") -> list[Document]:
    path = DATA_DIR / f"aida_{split}.jsonl"
    if not path.exists():
        dataset = ("cyanic-selkie/aida-conll-yago-wikidata", "default", split)
        rows: list[dict[str, Any]] = []
        with httpx.Client(timeout=60) as client:
            while True:
                page = _fetch(client, *dataset, offset=len(rows), length=PAGE_SIZE)
                rows += [item["row"] for item in page["rows"]]
                if len(rows) >= page["num_rows_total"]:
                    break
        _write(path, [_aida_document(row) for row in rows])
    return _read(path)


def load_wikianc(
    n_docs: int = 50, split: str = "validation", seed: int = 42
) -> list[Document]:
    """Muestra aleatoria (reproducible por ``seed``) de párrafos de Wikipedia.

    Se leen bloques de filas en posiciones aleatorias y de cada uno se toman
    hasta ``DOCS_PER_BLOCK`` párrafos de artículos distintos. Solo se
    conservan los enlaces a una entidad con QID cuyo texto empieza por
    mayúscula, como aproximación a "entidad nombrada" (WikiAnc también enlaza
    conceptos comunes como "software").
    """
    path = DATA_DIR / f"wikianc_en_{split}_n{n_docs}_seed{seed}.jsonl"
    if not path.exists():
        dataset = ("cyanic-selkie/wikianc", "en", split)
        rng = random.Random(seed)
        docs: list[Document] = []
        with httpx.Client(timeout=120) as client:
            total = _fetch(client, *dataset, offset=0, length=1)["num_rows_total"]
            while len(docs) < n_docs:
                offset = rng.randrange(total - PAGE_SIZE)
                page = _fetch(client, *dataset, offset=offset, length=PAGE_SIZE)
                block = [d for d in map(_wikianc_document, page["rows"]) if d]
                by_article = {d.doc_id: d for d in block}  # uno por artículo
                docs += list(by_article.values())[:DOCS_PER_BLOCK]
        _write(path, docs[:n_docs])
    return _read(path)


def load_disambiguation_sample() -> list[dict[str, Any]]:
    """Menciones de AIDA con NER correcto y QID entre los candidatos, con sus
    candidatos de Wikidata guardados (``scripts/build_disambiguation_sample.py``).
    """
    with DISAMBIGUATION_SAMPLE.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for r in rows:
        r["candidates"] = [WikidataCandidate(**c) for c in r["candidates"]]
    return rows


def _aida_document(row: dict[str, Any]) -> Document:
    mentions = [
        GoldMention(
            e["start"], e["end"], e["tag"], f"Q{e['qid']}" if e["qid"] else None
        )
        for e in row["entities"]
    ]
    return Document(str(row["document_id"]), row["text"], mentions)


def _wikianc_document(item: dict[str, Any]) -> Document | None:
    row = item["row"]
    text = row["paragraph_text"]
    mentions = [
        GoldMention(a["start"], a["end"], None, f"Q{a['qid']}")
        for a in row["paragraph_anchors"]
        if a["qid"] and text[a["start"] : a["end"]][:1].isupper()
    ]
    if not mentions or not 200 <= len(text) <= 1500:
        return None
    return Document(row["article_title"], text, mentions)


def _fetch(
    client: httpx.Client,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
) -> dict[str, Any]:
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    }
    # Reintentos ante el límite de peticiones (429) y ante timeouts: la primera
    # lectura de una zona del dataset puede tardar más de un minuto
    for attempt in range(6):
        try:
            response = client.get(ROWS_URL, params=params)
            if response.status_code != 429:
                response.raise_for_status()
                return dict(response.json())
        except httpx.TimeoutException:
            pass
        time.sleep(5 * 2**attempt)
    raise RuntimeError(f"Hugging Face no responde para {dataset} offset={offset}")


def _write(path: Path, docs: list[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")


def _read(path: Path) -> list[Document]:
    with path.open(encoding="utf-8") as f:
        return [
            Document(d["doc_id"], d["text"], [GoldMention(**m) for m in d["mentions"]])
            for d in map(json.loads, f)
        ]
