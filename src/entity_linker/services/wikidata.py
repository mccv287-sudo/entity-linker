import asyncio
import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from entity_linker.models.schemas import WikidataCandidate

logger = logging.getLogger(__name__)

API_URL = "https://www.wikidata.org/w/api.php"
SPARQL_URL = "https://query.wikidata.org/sparql"
SPARQL_BATCH = 50  # elementos por consulta de clases
USER_AGENT = "EntityLinker/1.0 (technical_test@example.com) httpx/Python"
MAX_RETRIES = 5
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
CACHE_PATH = Path(".cache/wikidata.sqlite")


def normalize_mention(text: str) -> str:
    """Quita espacios, comillas y puntuación de los bordes y el posesivo.

    "Obama's" -> "Obama", ' "Reuters". ' -> "Reuters"
    """
    text = " ".join(text.split()).strip(" \"'“”‘’.,;:()")
    return re.sub(r"\s*['’]s$", "", text)


class WikidataService:
    def __init__(
        self,
        timeout: float = 5.0,
        max_concurrency: int = 4,
        cache_path: Path | None = CACHE_PATH,
    ):
        # Un cliente compartido (reutiliza conexiones) y concurrencia limitada
        # para no superar el límite de peticiones de Wikidata
        self._client = httpx.AsyncClient(
            timeout=timeout, headers={"User-Agent": USER_AGENT}
        )
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._resume_at = 0.0  # instante (monotonic) hasta el que no se pide nada
        self._cache = _open_cache(cache_path) if cache_path else None

    async def get_candidates(
        self, search_term: str, language: str = "en", limit: int = 20
    ) -> list[WikidataCandidate]:
        """Busca elementos por etiqueta o alias (búsqueda por prefijo).

        Si no hay resultados y la mención empieza por "the", reintenta sin
        el artículo ("the Beatles" -> "Beatles"). Devuelve lista vacía si
        Wikidata falla tras los reintentos.
        """
        term = normalize_mention(search_term)
        candidates = await self._search(term, language, limit)
        if not candidates and term.lower().startswith("the "):
            candidates = await self._search(term[4:], language, limit)
        return candidates

    async def _search(
        self, term: str, language: str, limit: int
    ) -> list[WikidataCandidate]:
        data = await self._get(
            {
                "action": "wbsearchentities",
                "search": term,
                "language": language,
                "uselang": language,  # etiqueta y descripción en este idioma
                "type": "item",
                "limit": limit,
                "format": "json",
            }
        )
        return [
            WikidataCandidate(
                qid=item["id"],
                label=item.get("label", term),
                description=item.get("description"),
                concept_uri=item["concepturi"],
            )
            for item in (data or {}).get("search", [])
        ]

    async def get_classes(
        self, qids: list[str]
    ) -> dict[str, tuple[set[str], set[str]]]:
        """Clases de cada elemento: (P31 directas, todos sus ancestros vía P279*).

        Las directas sirven para la coherencia entre entidades del documento;
        los ancestros, para comprobar la compatibilidad con un tipo fino
        ("Japan national football team" desciende de "sports team").
        """
        classes: dict[str, tuple[set[str], set[str]]] = {
            q: (set(), set()) for q in qids
        }
        ordered = sorted(set(qids))  # orden fijo: la misma consulta reusa la caché
        for i in range(0, len(ordered), SPARQL_BATCH):
            values = " ".join(f"wd:{q}" for q in ordered[i : i + SPARQL_BATCH])
            query = (
                f"SELECT ?item ?cls ?sup WHERE {{ VALUES ?item {{ {values} }} "
                "?item wdt:P31 ?cls . ?cls wdt:P279* ?sup }"
            )
            data = await self._get(
                {"query": query, "format": "json"}, url=SPARQL_URL, timeout=60
            )
            for row in (data or {}).get("results", {}).get("bindings", []):
                item, cls, sup = (
                    row[k]["value"].rsplit("/", 1)[-1] for k in ("item", "cls", "sup")
                )
                classes[item][0].add(cls)
                classes[item][1].add(sup)
        return classes

    async def _get(
        self,
        params: dict[str, Any],
        url: str = API_URL,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        key = url + "?" + urlencode(sorted(params.items()))
        if self._cache and (row := self._cache.execute(
            "SELECT body FROM responses WHERE key = ?", (key,)
        ).fetchone()):
            return json.loads(row[0])

        for attempt in range(MAX_RETRIES):
            try:
                async with self._semaphore:
                    # Si Wikidata ha pedido esperar (429), esperan todas
                    await asyncio.sleep(self._resume_at - time.monotonic())
                    response = await self._client.get(
                        url, params=params, timeout=timeout or self._client.timeout
                    )
                    if response.status_code == 429:
                        wait = float(response.headers.get("Retry-After", 2**attempt))
                        self._resume_at = time.monotonic() + wait
                        continue
                if response.status_code not in RETRYABLE_STATUS:
                    response.raise_for_status()
                    if self._cache:  # solo se cachean respuestas correctas
                        with self._cache:
                            self._cache.execute(
                                "INSERT OR REPLACE INTO responses VALUES (?, ?)",
                                (key, response.text),
                            )
                    return response.json()
            except httpx.TransportError:
                pass  # red o timeout: se reintenta
            except httpx.HTTPError as exc:
                logger.warning("Wikidata: %s", exc)
                return None
            await asyncio.sleep(2**attempt)
        logger.warning("Wikidata sin respuesta: %s", url)
        return None


def _open_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    # El servicio se crea al importar la app y FastAPI lo usa desde otro hilo;
    # es seguro porque todos los accesos ocurren en un único bucle asíncrono
    db = sqlite3.connect(path, check_same_thread=False)
    db.execute(
        "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, body TEXT)"
    )
    return db
