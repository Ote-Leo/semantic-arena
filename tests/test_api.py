from __future__ import annotations

import time

from fastapi.testclient import TestClient

from api.app import create_app
from datasets.catalog import DatasetCatalog
from datasets.models import Chunk
from search.models import SearchHit
from settings import ArenaConfig, RuntimeSettings


class FakeEmbedder:
    dimensions = 3
    configured = True

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeEngine:
    id = "fake"
    label = "Fake Engine"

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.indexed = False
        self.staging_chunks: list[Chunk] = []

    async def health(self) -> tuple[bool, str]:
        return True, "ready"

    async def index_exists(self, dataset_id: str) -> bool:
        return self.indexed

    async def delete_index(self, dataset_id: str) -> None:
        self.indexed = False

    async def prepare_index(self, dataset_id: str, dimensions: int) -> None:
        self.staging_chunks = []

    async def index_batch(
        self,
        dataset_id: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        assert len(chunks) == len(vectors)
        assert all(len(vector) == 3 for vector in vectors)
        self.staging_chunks.extend(chunks)

    async def finalize_index(self, dataset_id: str) -> None:
        self.chunks = self.staging_chunks
        self.indexed = True

    async def abort_index(self, dataset_id: str) -> None:
        self.staging_chunks = []

    async def lexical_search(self, dataset_id: str, query: str, limit: int) -> list[SearchHit]:
        return [
            SearchHit(chunk, float(len(self.chunks) - index))
            for index, chunk in enumerate(self.chunks[:limit])
        ]

    async def vector_search(
        self,
        dataset_id: str,
        vector: list[float],
        limit: int,
    ) -> list[SearchHit]:
        return [SearchHit(chunk, 1.0) for chunk in self.chunks[:limit]]


def _client() -> TestClient:
    catalog = DatasetCatalog()
    engine = FakeEngine(catalog.chunks("demo"))
    app = create_app(
        runtime=RuntimeSettings(),
        arena_config=ArenaConfig(),
        catalog=catalog,
        embedder=FakeEmbedder(),
        engines={engine.id: engine},
    )
    return TestClient(app)


def test_api_returns_exact_evidence_and_rejects_unrelated_question() -> None:
    with _client() as client:
        supported = client.post(
            "/api/chat",
            json={
                "dataset_id": "demo",
                "query": "Which drug is used to improve blood glucose control?",
                "engine_ids": ["fake"],
            },
        )
        unsupported = client.post(
            "/api/chat",
            json={
                "dataset_id": "demo",
                "query": "Who is the president of the United States?",
                "engine_ids": ["fake"],
            },
        )

    assert supported.status_code == 200
    result = supported.json()["results"][0]
    assert result["no_data"] is False
    assert result["citations"]
    assert result["citations"][0]["quote"] in result["message"]
    assert result["citations"][0]["quote"] in result["citations"][0]["text"]
    assert unsupported.status_code == 200
    assert unsupported.json()["results"][0]["message"] == "there are no available data"


def test_api_index_job_uses_shared_chunks_and_vectors() -> None:
    with _client() as client:
        response = client.post(
            "/api/indexes",
            json={"dataset_id": "demo", "engine_ids": ["fake"]},
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        for _ in range(10):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)

    assert job["status"] == "succeeded"
    assert job["result"] == {"fake": 3}
    assert job["processed_chunks"] == 3


def test_api_lists_and_deletes_a_dataset_index() -> None:
    with _client() as client:
        create_response = client.post(
            "/api/indexes",
            json={"dataset_id": "demo", "engine_ids": ["fake"]},
        )
        job_id = create_response.json()["id"]
        for _ in range(10):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)

        before = client.get("/api/datasets").json()[0]
        delete_response = client.delete("/api/indexes/demo/fake")
        after = client.get("/api/datasets").json()[0]

    assert job["status"] == "succeeded"
    assert before["index_name"] == "arena_demo"
    assert before["engine_indexes"] == {"fake": True}
    assert delete_response.status_code == 204
    assert delete_response.content == b""
    assert after["engine_indexes"] == {"fake": False}
