"""Cliente de Wikidata: búsqueda de candidatos y clases, con caché y reintentos."""

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

from entity_linker.config import (
    CACHE_PATH,
    SEARCH_LIMIT,
    USER_AGENT,
    WIKIDATA_API_URL,
    WIKIDATA_CONCURRENCY,
    WIKIDATA_SPARQL_URL,
    WIKIDATA_TIMEOUT,
)
from entity_linker.models.schemas import WikidataCandidate

logger = logging.getLogger(__name__)

SPARQL_BATCH = 50  # elementos por consulta de clases
MAX_RETRIES = 5
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def normalize_mention(text: str) -> str:
    """Quita espacios, comillas y puntuación de los bordes y el posesivo.

    "Obama's" -> "Obama", ' "Reuters". ' -> "Reuters"
    """
    text = " ".join(text.split()).strip(" \"'“”‘’.,;:()")
    return re.sub(r"\s*['’]s$", "", text)


class WikidataService:
    def __init__(
        self,
        timeout: float = WIKIDATA_TIMEOUT,
        max_concurrency: int = WIKIDATA_CONCURRENCY,
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

    async def aclose(self) -> None:
        """Cierra el cliente HTTP y la caché (al apagar el servicio)."""
        await self._client.aclose()
        if self._cache:
            self._cache.close()

    async def get_candidates(
        self, search_term: str, language: str = "en", limit: int = SEARCH_LIMIT
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
        self, qids: list[str], roots: set[str]
    ) -> dict[str, tuple[set[str], set[str]]]:
        """Clases de cada elemento: (P31 directas, raíces de ``roots`` de las que
        desciende vía P279*).

        Las directas sirven para la coherencia entre entidades del documento;
        las raíces, para la compatibilidad con un tipo fino ("Japan national
        football team" desciende de "sports team"). Solo se devuelven las
        raíces pedidas.

        La caché es por QID (no por consulta): cada elemento se consulta una
        sola vez aunque llegue en lotes distintos.
        """
        prefix = ",".join(sorted(roots)) + ":"  # otras raíces -> otra entrada
        classes = {q: self._cached_classes(prefix + q) for q in dict.fromkeys(qids)}
        missing = sorted(q for q, c in classes.items() if c is None)
        wanted = ", ".join(f"wd:{r}" for r in sorted(roots))
        for i in range(0, len(missing), SPARQL_BATCH):
            batch = missing[i : i + SPARQL_BATCH]
            query = (
                "SELECT DISTINCT ?item ?cls ?root WHERE { "
                f"VALUES ?item {{ {' '.join(f'wd:{q}' for q in batch)} }} "
                "?item wdt:P31 ?cls . OPTIONAL { ?cls wdt:P279* ?root . "
                f"FILTER(?root IN ({wanted})) }} }}"
            )
            data = await self._get(
                {"query": query, "format": "json"},
                url=WIKIDATA_SPARQL_URL,
                timeout=60,
                cache=False,  # se cachea cada QID por separado
            )
            if data is None:  # Wikidata no responde: sin clases, sin cachear
                continue
            found: dict[str, tuple[set[str], set[str]]] = {
                q: (set(), set()) for q in batch
            }
            for row in data.get("results", {}).get("bindings", []):
                item = row["item"]["value"].rsplit("/", 1)[-1]
                found[item][0].add(row["cls"]["value"].rsplit("/", 1)[-1])
                if "root" in row:
                    found[item][1].add(row["root"]["value"].rsplit("/", 1)[-1])
            for q, value in found.items():
                classes[q] = value
                body = {"direct": sorted(value[0]), "roots": sorted(value[1])}
                self._store(prefix + q, body)
        return {q: c or (set(), set()) for q, c in classes.items()}

    def _cached_classes(self, key: str) -> tuple[set[str], set[str]] | None:
        if not self._cache:
            return None
        row = self._cache.execute(
            "SELECT body FROM responses WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        body = json.loads(row[0])
        return set(body["direct"]), set(body["roots"])

    def _store(self, key: str, body: dict[str, Any]) -> None:
        if self._cache:
            with self._cache:
                self._cache.execute(
                    "INSERT OR REPLACE INTO responses VALUES (?, ?)",
                    (key, json.dumps(body)),
                )

    async def _get(
        self,
        params: dict[str, Any],
        url: str = WIKIDATA_API_URL,
        timeout: float | None = None,
        cache: bool = True,
    ) -> dict[str, Any] | None:
        key = url + "?" + urlencode(sorted(params.items()))
        if cache and self._cache and (row := self._cache.execute(
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
                    data = response.json()
                    if cache:  # solo se cachean respuestas correctas
                        self._store(key, data)
                    return data
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
    # Se crea en el hilo del bucle de eventos (lifespan de FastAPI)
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, body TEXT)"
    )
    return db
