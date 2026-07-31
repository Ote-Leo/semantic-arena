from __future__ import annotations

import asyncio

from datasets.catalog import DatasetCatalog
from embeddings.base import Embedder
from search.base import SearchEngine


class IndexingService:
    def __init__(
        self,
        catalog: DatasetCatalog,
        embedder: Embedder,
        engines: dict[str, SearchEngine],
        embedding_batch_size: int = 64,
    ) -> None:
        self.catalog = catalog
        self.embedder = embedder
        self.engines = engines
        self.embedding_batch_size = embedding_batch_size

    async def build(self, dataset_id: str, engine_ids: list[str]) -> dict[str, int]:
        chunks = self.catalog.chunks(dataset_id)
        if not chunks:
            raise ValueError(f"Dataset '{dataset_id}' did not produce any indexable chunks.")
        vectors = await self._embed_chunks([chunk.text for chunk in chunks])
        selected_engines = [self.engines[engine_id] for engine_id in engine_ids]
        await asyncio.gather(
            *(
                engine.replace_index(dataset_id, chunks, vectors, self.embedder.dimensions)
                for engine in selected_engines
            )
        )
        return {engine.id: len(chunks) for engine in selected_engines}

    async def _embed_chunks(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.embedding_batch_size):
            batch = texts[start : start + self.embedding_batch_size]
            vectors.extend(await self.embedder.embed_many(batch))
        return vectors
