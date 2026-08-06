import httpx
import pytest

from meilisearch.adapter import MeilisearchSearchEngine


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
