from __future__ import annotations

import httpx

from embeddings.base import EmbeddingError


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        model: str,
        dimensions: int,
        api_key: str = "",
    ) -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions
        self.api_key = api_key

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not self.configured:
            raise EmbeddingError(
                "Embedding model is not configured. Set SEMANTIC_ARENA_EMBEDDING_BASE_URL and "
                "SEMANTIC_ARENA_EMBEDDING_MODEL."
            )
        if not texts:
            return []
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = await self.client.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json={"model": self.model, "input": texts},
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise EmbeddingError(f"Embedding request failed: {error}") from error

        try:
            items = sorted(response.json()["data"], key=lambda item: item["index"])
            embeddings = [list(map(float, item["embedding"])) for item in items]
        except (KeyError, TypeError, ValueError) as error:
            raise EmbeddingError("Embedding provider returned an invalid response.") from error
        if len(embeddings) != len(texts):
            raise EmbeddingError("Embedding provider returned an unexpected number of vectors.")
        if any(len(vector) != self.dimensions for vector in embeddings):
            actual = next(len(vector) for vector in embeddings if len(vector) != self.dimensions)
            raise EmbeddingError(
                "Embedding dimensions do not match configuration: "
                f"expected {self.dimensions}, got {actual}."
            )
        return embeddings
