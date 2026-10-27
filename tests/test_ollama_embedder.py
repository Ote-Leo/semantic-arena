from __future__ import annotations

import json

import httpx
import pytest

from embeddings.base import EmbeddingError
from embeddings.ollama import OllamaEmbedder


@pytest.mark.asyncio
async def test_native_embed_request_and_order() -> None:
    requests: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://ollama:11435/api/embed"
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[1, 2, 3], [4, 5, 6]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        embedder = OllamaEmbedder(client, "http://ollama:11435/api/embed", "example", 3)
        assert await embedder.embed_many([]) == []
        assert await embedder.embed_many(["first", "second"]) == [[1., 2., 3.], [4., 5., 6.]]
    assert requests == [{"model": "example", "input": ["first", "second"]}]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload,match", [
    ({"embeddings": [[1, 2]]}, "dimensions"),
    ({"embeddings": []}, "number"),
    ({"data": []}, "invalid"),
])
async def test_rejects_bad_native_embedding_responses(payload: dict, match: str) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as client:
        with pytest.raises(EmbeddingError, match=match):
            await OllamaEmbedder(client, "http://ollama/api/embed", "example", 3).embed_many(["x"])


@pytest.mark.asyncio
async def test_reports_embedding_http_failure() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(503))
    ) as client:
        with pytest.raises(EmbeddingError, match="Ollama embedding request failed"):
            await OllamaEmbedder(client, "http://ollama/api/embed", "example", 3).embed_many(["x"])
