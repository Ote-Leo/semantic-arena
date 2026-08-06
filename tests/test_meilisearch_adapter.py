import json

import httpx
import pytest

from datasets.models import Chunk
from meilisearch.adapter import MeilisearchSearchEngine
from search.naming import staging_index_name


@pytest.mark.asyncio
@pytest.mark.parametrize("exists", [True, False])
async def test_delete_index_waits_for_an_existing_index_deletion(exists: bool) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET" and request.url.path == "/indexes/arena_demo":
            return httpx.Response(200 if exists else 404, json={})
        if request.method == "DELETE":
            return httpx.Response(202, json={"taskUid": 7})
        if request.url.path == "/tasks/7":
            return httpx.Response(200, json={"status": "succeeded"})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = MeilisearchSearchEngine(client, "http://meili.test", "secret")

        await engine.delete_index("demo")

    delete_requests = [request for request in requests if request.method == "DELETE"]
    task_requests = [request for request in requests if request.url.path == "/tasks/7"]
    assert len(delete_requests) == int(exists)
    assert len(task_requests) == int(exists)
    assert all(request.headers["authorization"] == "Bearer secret" for request in requests)


@pytest.mark.asyncio
async def test_staged_index_is_swapped_into_place() -> None:
    requests: list[httpx.Request] = []
    staging = staging_index_name("demo")
    swapped = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal swapped
        requests.append(request)
        if request.method == "GET" and request.url.path == f"/indexes/{staging}":
            return httpx.Response(200 if swapped else 404, json={})
        if request.method == "GET" and request.url.path == "/indexes/arena_demo":
            return httpx.Response(404, json={})
        if request.method == "GET" and request.url.path == "/tasks/7":
            return httpx.Response(200, json={"status": "succeeded"})
        if request.url.path == "/swap-indexes":
            swapped = True
        return httpx.Response(202, json={"taskUid": 7})

    chunk = Chunk(
        id="chunk-1",
        dataset_id="demo",
        source_id="source-1",
        title="Demo",
        text="Source text.",
        source_path="source.md",
        record_index=0,
        chunk_index=0,
        start_char=0,
        end_char=12,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = MeilisearchSearchEngine(client, "http://meili.test", "secret")

        await engine.prepare_index("demo", dimensions=3)
        await engine.index_batch("demo", [chunk], [[0.1, 0.2, 0.3]])
        await engine.finalize_index("demo")

    document_request = next(
        request for request in requests if request.url.path.endswith("/documents")
    )
    assert document_request.url.path == f"/indexes/{staging}/documents"
    swap_request = next(request for request in requests if request.url.path == "/swap-indexes")
    assert json.loads(swap_request.read()) == [{"indexes": [staging, "arena_demo"]}]
    assert any(
        request.method == "DELETE" and request.url.path == f"/indexes/{staging}"
        for request in requests
    )
