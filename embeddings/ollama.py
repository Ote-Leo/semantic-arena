from __future__ import annotations

import httpx

from embeddings.base import EmbeddingError


class OllamaEmbedder:
    """Ollama's native /api/embed API (not its OpenAI-compatible /v1 API)."""

    def __init__(self, client: httpx.AsyncClient, url: str, model: str, dimensions: int) -> None:
        self.client = client
        self.url = url
        self.model = model
        self.dimensions = dimensions

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self.client.post(
                self.url, json={"model": self.model, "input": texts}
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise EmbeddingError(f"Ollama embedding request failed: {error}") from error
        try:
            vectors = [list(map(float, vector)) for vector in response.json()["embeddings"]]
        except (KeyError, TypeError, ValueError) as error:
            raise EmbeddingError("Ollama returned an invalid embedding response") from error
        if len(vectors) != len(texts):
            raise EmbeddingError("Ollama returned an unexpected number of vectors")
        if any(len(vector) != self.dimensions for vector in vectors):
            raise EmbeddingError(f"Ollama vector dimensions do not match {self.dimensions}")
        return vectors
