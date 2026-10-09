import httpx
from typing import List, Dict
from entity_linker.models.schemas import WikidataCandidate

class WikidataService:
    def __init__(self, timeout: float = 3.0):
        self.api_url = "https://www.wikidata.org/w/api.php"
        self.timeout = timeout
        self.headers = {
            "User-Agent": "EntityLinker/1.0 (technical_test@example.com) httpx/Python"
        }
        self._cache: Dict[str, List[WikidataCandidate]] = {}

    async def get_candidates(self, search_term: str, language: str = "es", limit: int = 5) -> List[WikidataCandidate]:
        cache_key = f"{search_term.lower().strip()}_{language}_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        params = {
            "action": "wbsearchentities",
            "search": search_term,
            "language": language,
            "format": "json",
            "limit": limit,
            "type": "item"
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.api_url, params=params, headers=self.headers)
                response.raise_for_status()
                data = response.json()

            candidates = []
            for item in data.get("search", []):
                candidates.append(
                    WikidataCandidate(
                        qid=item.get("id"),
                        label=item.get("label", search_term),
                        description=item.get("description"),
                        concept_uri=item.get("concepturi", f"http://www.wikidata.org/entity/{item.get('id')}")
                    )
                )
            self._cache[cache_key] = candidates
            return candidates

        except (httpx.HTTPError, Exception):
            # En caso de error de red o timeout devuelve lista vacía (resiliencia)
            return []
