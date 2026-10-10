import argparse
import json

import httpx


SEARCH_URL = "https://wd-vectordb.wmcloud.org/item/query/"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

HEADERS = {
    "User-Agent": "entity-linker-semantic-test/0.1 (local research project)"
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        required=True,
        help="Mention, description or surrounding context",
    )
    parser.add_argument(
        "--instanceof",
        default=None,
        help="Optional comma-separated Wikidata class QIDs, e.g. Q5",
    )
    parser.add_argument("--lang", default="all")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.k <= 50:
        parser.error("--k debe estar entre 1 y 50")

    params = {
        "query": args.query,
        "lang": args.lang,
        "K": args.k,
        "scope": "with_sitelinks",
        "rerank": str(args.rerank).lower(),
    }

    if args.instanceof:
        params["instanceof"] = args.instanceof

    with httpx.Client(
        timeout=60.0,
        headers=HEADERS,
    ) as client:
        # 1. Recuperar candidatos por búsqueda semántica + keywords.
        response = client.get(SEARCH_URL, params=params)
        response.raise_for_status()
        payload = response.json()

        # Normalizar la respuesta sin asumir una estructura concreta.
        if isinstance(payload, list):
            results = payload
        elif isinstance(payload, dict):
            results = next(
                (
                    payload[key]
                    for key in ("results", "items", "data")
                    if isinstance(payload.get(key), list)
                ),
                None,
            )
        else:
            results = None

        if results is None:
            print("Formato de respuesta no reconocido:")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return

        qids = [
            item.get("QID")
            for item in results
            if item.get("QID")
        ]

        # 2. Recuperar etiquetas y descripciones legibles.
        details = {}

        if qids:
            detail_response = client.get(
                WIKIDATA_API,
                params={
                    "action": "wbgetentities",
                    "ids": "|".join(qids),
                    "props": "labels|descriptions",
                    "languages": "en|es",
                    "format": "json",
                },
            )
            detail_response.raise_for_status()
            details = detail_response.json().get("entities", {})

    print(f"\nQuery: {args.query}")
    print(f"Filtro P31: {args.instanceof or 'ninguno'}")
    print(f"Candidatos: {len(results)}\n")

    for rank, item in enumerate(results, start=1):
        qid = item.get("QID", "?")
        entity = details.get(qid, {})
        labels = entity.get("labels", {})
        descriptions = entity.get("descriptions", {})

        label = (
            labels.get("en", {}).get("value")
            or labels.get("es", {}).get("value")
            or qid
        )
        description = (
            descriptions.get("en", {}).get("value")
            or descriptions.get("es", {}).get("value")
            or ""
        )

        print(f"{rank}. {label} ({qid})")
        print(f"   {description}")
        print(f"   Similarity: {item.get('similarity_score')}")
        print(f"   RRF:        {item.get('rrf_score')}")
        print(f"   Source:     {item.get('source')}")

        if "reranker_score" in item:
            print(f"   Reranker:   {item['reranker_score']}")

        print(f"   https://www.wikidata.org/wiki/{qid}\n")


if __name__ == "__main__":
    main()