from unittest.mock import AsyncMock

import httpx
import pytest

from datasets.models import Chunk
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
async def test_replace_index_uses_the_configured_configset() -> None:
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

        await engine.replace_index("demo", [chunk], [[0.1, 0.2, 0.3]], dimensions=3)

    create_request = next(
        request for request in requests if request.url.params.get("action") == "CREATE"
    )
    assert create_request.url.params["configSet"] == "/configsets/arena"


@pytest.mark.asyncio
async def test_schema_configuration_ignores_duplicate_hidden_by_error_truncation() -> None:
    duplicate_response = {
        "error": {
            "details": "x" * 600,
            "errorMessages": ["Field type 'arena_vector_3' already exists."],
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=duplicate_response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = SolrSearchEngine(client, "http://solr.test/solr")

        await engine._configure_schema("arena_demo", dimensions=3)
