from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Iterator

from datasets.catalog import DatasetCatalog
from datasets.models import Chunk
from embeddings.base import Embedder
from search.base import SearchEngine


class IndexingService:
    def __init__(
        self,
        catalog: DatasetCatalog,
        embedder: Embedder,
        engines: dict[str, SearchEngine],
        embedding_batch_size: int = 64,
        chunk_batch_size: int = 256,
    ) -> None:
        self.catalog = catalog
        self.embedder = embedder
        self.engines = engines
        self.embedding_batch_size = embedding_batch_size
        self.chunk_batch_size = chunk_batch_size

    async def build(
        self,
        dataset_id: str,
        engine_ids: list[str],
        progress: Callable[[int], None] | None = None,
    ) -> dict[str, int]:
        selected_engines = [self.engines[engine_id] for engine_id in engine_ids]
        count = 0
        try:
            await asyncio.gather(
                *(
                    engine.prepare_index(dataset_id, self.embedder.dimensions)
                    for engine in selected_engines
                )
            )
            for chunks in _batches(self.catalog.iter_chunks(dataset_id), self.chunk_batch_size):
                vectors = await self._embed_chunks([chunk.text for chunk in chunks])
                await asyncio.gather(
                    *(
                        engine.index_batch(dataset_id, chunks, vectors)
                        for engine in selected_engines
                    )
                )
                count += len(chunks)
                if progress is not None:
                    progress(count)
            if count == 0:
                raise ValueError(f"Dataset '{dataset_id}' did not produce any indexable chunks.")
            await asyncio.gather(
                *(engine.finalize_index(dataset_id) for engine in selected_engines)
            )
        except Exception:
            await asyncio.gather(
                *(engine.abort_index(dataset_id) for engine in selected_engines),
                return_exceptions=True,
            )
            raise
        return {engine.id: count for engine in selected_engines}

    async def _embed_chunks(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.embedding_batch_size):
            batch = texts[start : start + self.embedding_batch_size]
            vectors.extend(await self.embedder.embed_many(batch))
        if len(vectors) != len(texts):
            raise ValueError("Embedding service returned a different number of vectors than texts.")
        return vectors


def _batches(items: Iterable[Chunk], size: int) -> Iterator[list[Chunk]]:
    batch: list[Chunk] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
