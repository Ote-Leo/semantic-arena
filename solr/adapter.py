from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import httpx

from datasets.models import Chunk
from search.base import SearchEngineError
from search.models import SearchHit
from search.naming import index_name


class SolrSearchEngine:
    id = "solr"
    label = "Apache Solr"

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        configset: str = "semantic-arena",
    ) -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.configset = configset

    async def health(self) -> tuple[bool, str]:
        try:
            response = await self.client.get(
                f"{self.base_url}/admin/info/system", params={"wt": "json"}
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            return False, str(error)
        return True, "ready"

    async def index_exists(self, dataset_id: str) -> bool:
        name = index_name(dataset_id)
        response = await self._request(
            "GET",
            f"{self.base_url}/admin/cores",
            params={"action": "STATUS", "core": name, "wt": "json"},
        )
        core_status = response.json().get("status", {}).get(name, {})
        return core_status.get("name") == name

    async def delete_index(self, dataset_id: str) -> None:
        if not await self.index_exists(dataset_id):
            return
        await self._admin_action(
            "UNLOAD",
            core=index_name(dataset_id),
            deleteIndex="true",
            deleteDataDir="true",
            deleteInstanceDir="true",
        )

    async def replace_index(
        self,
        dataset_id: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
        dimensions: int,
    ) -> None:
        if len(chunks) != len(vectors):
            raise SearchEngineError(
                "Solr indexing received a different number of chunks and vectors."
            )
        name = index_name(dataset_id)
        await self.delete_index(dataset_id)
        await self._admin_action("CREATE", name=name, instanceDir=name, configSet=self.configset)
        await self._configure_schema(name, dimensions)
        documents = [
            self._index_document(chunk, vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for batch in _batches(documents, 100):
            await self._request(
                "POST",
                f"{self.base_url}/{name}/update",
                params={"commit": "false"},
                json=batch,
            )
        await self._request(
            "POST",
            f"{self.base_url}/{name}/update",
            params={"commit": "true"},
            json={"commit": {}},
        )

    async def lexical_search(self, dataset_id: str, query: str, limit: int) -> list[SearchHit]:
        name = index_name(dataset_id)
        payload = {
            "query": {
                "edismax": {
                    "query": query,
                    "qf": "title^3 text",
                    "pf": "title^8 text^2",
                    "mm": "1",
                    "uf": "-*",
                }
            },
            "limit": limit,
            "fields": _RESULT_FIELDS,
        }
        response = await self._request("POST", f"{self.base_url}/{name}/query", json=payload)
        return self._hits(response.json().get("response", {}).get("docs", []))

    async def vector_search(
        self,
        dataset_id: str,
        vector: list[float],
        limit: int,
    ) -> list[SearchHit]:
        name = index_name(dataset_id)
        vector_json = json.dumps(vector, separators=(",", ":"))
        payload = {
            "query": f"{{!knn f=vector topK={limit}}}{vector_json}",
            "limit": limit,
            "fields": _RESULT_FIELDS,
        }
        response = await self._request("POST", f"{self.base_url}/{name}/query", json=payload)
        return self._hits(response.json().get("response", {}).get("docs", []))

    async def _configure_schema(self, core: str, dimensions: int) -> None:
        vector_type = f"arena_vector_{dimensions}"
        await self._add_schema_command(
            core,
            {
                "add-field-type": {
                    "name": vector_type,
                    "class": "solr.DenseVectorField",
                    "vectorDimension": dimensions,
                    "similarityFunction": "cosine",
                    "indexed": True,
                    "stored": False,
                }
            },
        )
        fields = [
            {"name": "id", "type": "string", "stored": True, "required": True},
            {"name": "dataset_id", "type": "string", "stored": True},
            {"name": "title", "type": "text_general", "stored": True},
            {"name": "text", "type": "text_general", "stored": True},
            {"name": "source_path", "type": "string", "stored": True},
            {"name": "record_index", "type": "pint", "stored": True},
            {"name": "chunk_index", "type": "pint", "stored": True},
            {"name": "start_char", "type": "pint", "stored": True},
            {"name": "end_char", "type": "pint", "stored": True},
            {"name": "vector", "type": vector_type, "stored": False},
        ]
        for field in fields:
            await self._add_schema_command(core, {"add-field": field})

    async def _add_schema_command(self, core: str, command: dict[str, Any]) -> None:
        try:
            await self._request("POST", f"{self.base_url}/{core}/schema", json=command)
        except SearchEngineError as error:
            cause = error.__cause__
            is_duplicate = (
                isinstance(cause, httpx.HTTPStatusError)
                and "already exists" in cause.response.text.lower()
            )
            if not is_duplicate:
                raise

    async def _admin_action(self, action: str, **params: str) -> None:
        await self._request(
            "GET",
            f"{self.base_url}/admin/cores",
            params={"action": action, "wt": "json", **params},
        )

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self.client.request(method, url, **kwargs)
            response.raise_for_status()
        except httpx.HTTPError as error:
            detail = ""
            if isinstance(error, httpx.HTTPStatusError):
                detail = f" {error.response.text[:500]}"
            raise SearchEngineError(f"Solr request failed: {error}.{detail}") from error
        return response

    @staticmethod
    def _index_document(chunk: Chunk, vector: list[float]) -> dict[str, Any]:
        return {
            "id": chunk.id,
            "dataset_id": chunk.dataset_id,
            "title": chunk.title,
            "text": chunk.text,
            "source_path": chunk.source_path,
            "record_index": chunk.record_index,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "vector": vector,
        }

    @staticmethod
    def _hits(documents: list[dict[str, Any]]) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for document in documents:
            chunk = Chunk(
                id=str(document["id"]),
                dataset_id=str(document["dataset_id"]),
                source_id="",
                title=str(document.get("title", "")),
                text=str(document.get("text", "")),
                source_path=str(document.get("source_path", "")),
                record_index=int(document.get("record_index", 0)),
                chunk_index=int(document.get("chunk_index", 0)),
                start_char=int(document.get("start_char", 0)),
                end_char=int(document.get("end_char", 0)),
            )
            score = float(document["score"]) if document.get("score") is not None else None
            hits.append(SearchHit(chunk=chunk, score=score))
        return hits


_RESULT_FIELDS = [
    "id",
    "dataset_id",
    "title",
    "text",
    "source_path",
    "record_index",
    "chunk_index",
    "start_char",
    "end_char",
    "score",
]


def _batches(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
