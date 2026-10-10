"""Aprende los pesos de las señales (``WEIGHTS``) por máxima verosimilitud.

``scoring.probabilities`` es un logit condicional: la probabilidad de cada
candidato es el softmax, sobre los candidatos de su mención, de la combinación
lineal de sus señales. Este script busca los pesos que maximizan la
probabilidad del QID correcto en los documentos de train (``scipy.optimize``)
y mide la accuracy en los de test (partición por documento con crc32, la misma
del notebook), frente a los pesos actuales de ``config.py``.

Una regresión logística binaria (cada candidato por separado) no sirve: con un
candidato correcto de cada ~10 aprende a separar clases, no a ordenar los
candidatos de una mención, y en test empeoraba al primer candidato.

La señal coherence depende de qué menciones se resuelven en el paso 1, y eso
de los pesos actuales: volver a ejecutar el script tras actualizar
``config.py`` da pesos algo distintos con la misma accuracy en test.

``L2`` estabiliza los pesos (sin ella, coherence se dispara a ~18). El peso de
``llm`` no se aprende (consultar el LLM en cada mención es demasiado lento): se
fija a mano en ``config.py``.

Uso:
  uv run python scripts/tune_weights.py
"""

import asyncio
import zlib
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax

from entity_linker.config import CONFIDENT_PROB, MAX_CANDIDATES, WEIGHTS
from entity_linker.evaluation.datasets import load_disambiguation_sample
from entity_linker.evaluation.evaluate import evaluate_disambiguation
from entity_linker.services.clients.wikidata import WikidataService
from entity_linker.services.entity_types import ROOT_CLASSES
from entity_linker.services.linking.disambiguator import LLMDisambiguatorService
from entity_linker.services.linking.scoring import (
    argmax,
    coherence_score,
    context_score,
    probabilities,
    surroundings,
    type_score,
)

SIGNALS = ["prior", "context", "type", "coherence"]
L2 = 0.1


def is_train(row: dict[str, Any]) -> bool:
    return zlib.crc32(row["doc_id"].encode()) % 2 == 0


async def features(
    rows: list[dict[str, Any]], wikidata: WikidataService
) -> list[tuple[np.ndarray, int]]:
    """Por mención: señales de sus candidatos (una fila cada uno) y posición
    del correcto. Se omiten las menciones con el correcto fuera de los
    ``MAX_CANDIDATES`` primeros."""
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], []).append(r)

    mentions = []
    for doc_rows in by_doc.values():
        text = doc_rows[0]["text"]
        cands = [r["candidates"][:MAX_CANDIDATES] for r in doc_rows]
        context = [
            [
                context_score(c, surroundings(text, r["start"], r["end"]), r["mention"])
                for c in cs
            ]
            for r, cs in zip(doc_rows, cands)
        ]
        # Como el desambiguador: la coherencia se mide frente a las menciones
        # resueltas con las señales sin red (y los pesos actuales)
        resolved: dict[int, str] = {}
        for i, (cs, scores) in enumerate(zip(cands, context)):
            p = probabilities([{"context": s} for s in scores])
            if p and max(p) >= CONFIDENT_PROB:
                resolved[i] = cs[argmax(p)].qid
        classes = await wikidata.get_classes(
            [c.qid for cs in cands for c in cs], ROOT_CLASSES
        )

        for i, (r, cs) in enumerate(zip(doc_rows, cands)):
            qids = [c.qid for c in cs]
            if r["gold"] not in qids:
                continue
            others = [q for j, q in resolved.items() if j != i]
            x = [
                [
                    1 / (rank + 1),
                    context[i][rank],
                    type_score(r.get("fine_label"), c.qid, classes),
                    coherence_score(c.qid, others, classes),
                ]
                for rank, c in enumerate(cs)
            ]
            mentions.append((np.array(x), qids.index(r["gold"])))
    return mentions


def fit(mentions: list[tuple[np.ndarray, int]]) -> dict[str, float]:
    """Pesos que minimizan -log P(correcto) sumado sobre las menciones."""

    def loss(w: np.ndarray) -> float:
        nll = -sum(log_softmax(x @ w)[gold] for x, gold in mentions)
        return float(nll + L2 * w @ w)

    # Pesos ≥ 0: cada señal es evidencia a favor del candidato
    bounds = [(0, None)] * len(SIGNALS)
    weights = minimize(loss, np.ones(len(SIGNALS)), bounds=bounds).x
    return {s: round(float(w), 2) for s, w in zip(SIGNALS, weights)}


async def main() -> None:
    rows = load_disambiguation_sample()
    train = [r for r in rows if is_train(r)]
    test = [r for r in rows if not is_train(r)]
    print(f"train {len(train)} menciones, test {len(test)}")

    wikidata = WikidataService()
    learned = fit(await features(train, wikidata))

    first = sum(r["gold_rank"] == 1 for r in test) / len(test)
    print(f"{'1er candidato':12} test all {first:.3f}")
    disambiguator = LLMDisambiguatorService(wikidata, use_llm=False)
    current = dict(WEIGHTS)
    for name, weights in [("config.py", current), ("aprendidos", learned)]:
        WEIGHTS.update(weights)  # mismo dict que usa scoring
        result = await evaluate_disambiguation(test, disambiguator, use_llm=False)
        shown = {s: weights[s] for s in SIGNALS}
        print(f"{name:12} {shown} -> test {result['accuracy']}")
    WEIGHTS.update(current)
    await wikidata.aclose()


if __name__ == "__main__":
    asyncio.run(main())
