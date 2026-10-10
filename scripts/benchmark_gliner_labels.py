"""Compara conjuntos de etiquetas (prompts) de GLiNER sobre AIDA.

Uso:
  uv run python scripts/benchmark_gliner_labels.py --limit 20
"""

import argparse
import json
from pathlib import Path

from entity_linker.evaluation.datasets import load_aida
from entity_linker.evaluation.evaluate import evaluate_ner, print_metrics
from entity_linker.services.ner.gliner import NERService

LABEL_CONFIGS = {
    "A_broad": {
        "person": "PER",
        "organization": "ORG",
        "location": "LOC",
        "miscellaneous entity": "MISC",
    },
    "B_expanded": {
        "person": "PER",
        "organization": "ORG",
        "location": "LOC",
        "work of art": "MISC",
        "event": "MISC",
        "product": "MISC",
        "nationality or ethnic group": "MISC",
        "language": "MISC",
        "religion or ideology": "MISC",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    docs = load_aida(args.split)[: args.limit]
    ner = NERService(threshold=args.threshold)  # un único modelo para todo

    results = {}
    for name, labels in LABEL_CONFIGS.items():
        result = evaluate_ner(ner, docs, labels)
        del result["errors"]  # solo interesan las métricas para comparar
        results[name] = result
        print(f"\n{name} ({len(docs)} documentos)")
        print_metrics(result["metrics"])

    path = Path("reports/gliner_label_benchmark.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"split": args.split, "threshold": args.threshold, "results": results}
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados guardados en {path}")


if __name__ == "__main__":
    main()
