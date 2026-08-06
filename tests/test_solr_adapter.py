import json
from unittest.mock import AsyncMock

import httpx
import pytest

from datasets.models import Chunk
from search.naming import staging_index_name
from solr.adapter import SolrSearchEngine


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("core_status", "expected"),
    [({}, False), ({"name": "arena_demo"}, True)],
)
async def test_index_exists_requires_a_real_core_status(
    core_status: dict[str, str], expected: bool
) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"status": {"arena_demo": core_status}})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        engine = SolrSearchEngine(client, "http://solr.test/solr")

        exists = await engine.index_exists("demo")

    assert exists is expected


@pytest.mark.asyncio
async def test_staged_index_uses_the_configured_configset() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params.get("action") == "STATUS":
            return httpx.Response(200, json={"status": {"arena_demo": {}}})
        return httpx.Response(200, json={})

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
        engine = SolrSearchEngine(client, "http://solr.test/solr", "/configsets/arena")
        engine._configure_schema = AsyncMock()  # type: ignore[method-assign]

        await engine.prepare_index("demo", dimensions=3)
        await engine.index_batch("demo", [chunk], [[0.1, 0.2, 0.3]])
        await engine.finalize_index("demo")

    create_request = next(
        request for request in requests if request.url.params.get("action") == "CREATE"
    )
    assert create_request.url.params["configSet"] == "/configsets/arena"
    assert create_request.url.params["name"] == staging_index_name("demo")
    assert any(
        request.url.path.endswith(f"/{staging_index_name('demo')}/update")
        for request in requests
    )
    rename_request = next(
        request for request in requests if request.url.params.get("action") == "RENAME"
    )
    assert rename_request.url.params["core"] == staging_index_name("demo")
    assert rename_request.url.params["other"] == "arena_demo"


@pytest.mark.asyncio
async def test_schema_configuration_ignores_duplicate_hidden_by_error_truncation() -> None:
    duplicate_response = {
        "error": {
            "details": "x" * 600,
            "errorMessages": ["Field type 'arena_vector_3' already exists."],
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if "replace-field" in json.loads(request.content):
            return httpx.Response(200, json={})
        return httpx.Response(400, json=duplicate_response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = SolrSearchEngine(client, "http://solr.test/solr")

        await engine._configure_schema("arena_demo", dimensions=3)


@pytest.mark.asyncio
@pytest.mark.parametrize("exists", [True, False])
async def test_delete_index_unloads_an_existing_core(exists: bool) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params.get("action") == "STATUS":
            status = {"name": "arena_demo"} if exists else {}
            return httpx.Response(200, json={"status": {"arena_demo": status}})
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = SolrSearchEngine(client, "http://solr.test/solr")

        await engine.delete_index("demo")

    unload_requests = [
        request for request in requests if request.url.params.get("action") == "UNLOAD"
    ]
    assert len(unload_requests) == int(exists)
    if unload_requests:
        assert unload_requests[0].url.params["core"] == "arena_demo"
        assert unload_requests[0].url.params["deleteIndex"] == "true"
        assert unload_requests[0].url.params["deleteDataDir"] == "true"
        assert unload_requests[0].url.params["deleteInstanceDir"] == "true"
