from __future__ import annotations

from typing import Protocol


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    dimensions: int

    async def embed_many(self, texts: list[str]) -> list[list[float]]: ...
