from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

import httpx

from datasets.models import Chunk
from search.base import SearchEngineError
from search.models import SearchHit
from search.naming import index_name


class MeilisearchSearchEngine:
    id = "meilisearch"
    label = "Meilisearch"

    def __init__(self, client: httpx.AsyncClient, base_url: str, master_key: str) -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {master_key}"} if master_key else {}

    async def health(self) -> tuple[bool, str]:
        try:
            response = await self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
        except httpx.HTTPError as error:
            return False, str(error)
        return True, "ready"

    async def index_exists(self, dataset_id: str) -> bool:
        response = await self.client.get(
            f"{self.base_url}/indexes/{index_name(dataset_id)}", headers=self.headers
        )
        if response.status_code == 404:
            return False
        try:
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise SearchEngineError(f"Meilisearch index status failed: {error}.") from error
        return True

    async def delete_index(self, dataset_id: str) -> None:
        if not await self.index_exists(dataset_id):
            return
        response = await self._request("DELETE", f"/indexes/{index_name(dataset_id)}")
        await self._wait_for_task(response.json()["taskUid"])

    async def replace_index(
        self,
        dataset_id: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
        dimensions: int,
    ) -> None:
        if len(chunks) != len(vectors):
            raise SearchEngineError(
                "Meilisearch indexing received a different number of chunks and vectors."
            )
        name = index_name(dataset_id)
        await self.delete_index(dataset_id)
        response = await self._request("POST", "/indexes", json={"uid": name, "primaryKey": "id"})
        await self._wait_for_task(response.json()["taskUid"])
        response = await self._request(
            "PATCH",
            f"/indexes/{name}/settings",
            json={
                "searchableAttributes": ["title", "text"],
                "displayedAttributes": _RESULT_FIELDS,
                "embedders": {"arena": {"source": "userProvided", "dimensions": dimensions}},
            },
        )
        await self._wait_for_task(response.json()["taskUid"])
        documents = [
            self._index_document(chunk, vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for batch in _batches(documents, 100):
            response = await self._request("POST", f"/indexes/{name}/documents", json=batch)
            await self._wait_for_task(response.json()["taskUid"])

    async def lexical_search(self, dataset_id: str, query: str, limit: int) -> list[SearchHit]:
        response = await self._request(
            "POST",
            f"/indexes/{index_name(dataset_id)}/search",
            json={"q": query, "limit": limit, "showRankingScore": True},
        )
        return self._hits(response.json().get("hits", []))

    async def vector_search(
        self,
        dataset_id: str,
        vector: list[float],
        limit: int,
    ) -> list[SearchHit]:
        response = await self._request(
            "POST",
            f"/indexes/{index_name(dataset_id)}/search",
            json={
                "q": "",
                "vector": vector,
                "hybrid": {"embedder": "arena", "semanticRatio": 1.0},
                "limit": limit,
                "showRankingScore": True,
            },
        )
        return self._hits(response.json().get("hits", []))

    async def _wait_for_task(self, task_uid: int) -> None:
        for _ in range(600):
            response = await self._request("GET", f"/tasks/{task_uid}")
            task = response.json()
            if task["status"] == "succeeded":
                return
            if task["status"] in {"failed", "canceled"}:
                raise SearchEngineError(
                    f"Meilisearch task {task_uid} {task['status']}: {task.get('error')}"
                )
            await asyncio.sleep(0.1)
        raise SearchEngineError(f"Meilisearch task {task_uid} timed out.")

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self.client.request(
                method, f"{self.base_url}{path}", headers=self.headers, **kwargs
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            detail = ""
            if isinstance(error, httpx.HTTPStatusError):
                detail = f" {error.response.text[:500]}"
            raise SearchEngineError(f"Meilisearch request failed: {error}.{detail}") from error
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
            "_vectors": {"arena": vector},
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
            score = document.get("_rankingScore")
            hits.append(SearchHit(chunk=chunk, score=float(score) if score is not None else None))
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
]


def _batches(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
