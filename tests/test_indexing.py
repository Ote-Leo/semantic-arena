from __future__ import annotations

from collections.abc import Iterator

import pytest

from datasets.models import Chunk
from indexing.service import IndexingService


def _chunk(index: int) -> Chunk:
    return Chunk(
        id=f"chunk-{index}",
        dataset_id="demo",
        source_id=f"source-{index}",
        title=f"Title {index}",
        text=f"Text {index}",
        source_path="source.txt",
        record_index=index,
        chunk_index=0,
        start_char=0,
        end_char=6,
    )


class StreamingCatalog:
    def __init__(self, count: int) -> None:
        self.count = count

    def iter_chunks(self, dataset_id: str) -> Iterator[Chunk]:
        for index in range(self.count):
            yield _chunk(index)


class RecordingEmbedder:
    dimensions = 3

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return [[float(text.rsplit(maxsplit=1)[-1]), 0.0, 0.0] for text in texts]


class RecordingEngine:
    label = "Recording"

    def __init__(self, engine_id: str, fail_on_batch: int | None = None) -> None:
        self.id = engine_id
        self.fail_on_batch = fail_on_batch
        self.batch_ids: list[list[str]] = []
        self.vectors: list[list[list[float]]] = []
        self.prepared = False
        self.finalized = False
        self.aborted = False

    async def prepare_index(self, dataset_id: str, dimensions: int) -> None:
        assert dimensions == 3
        self.prepared = True

    async def index_batch(
        self, dataset_id: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        self.batch_ids.append([chunk.id for chunk in chunks])
        self.vectors.append(vectors)
        if self.fail_on_batch == len(self.batch_ids):
            raise RuntimeError("index failed")

    async def finalize_index(self, dataset_id: str) -> None:
        self.finalized = True

    async def abort_index(self, dataset_id: str) -> None:
        self.aborted = True


@pytest.mark.asyncio
async def test_indexing_streams_identical_bounded_batches_to_every_engine() -> None:
    embedder = RecordingEmbedder()
    first = RecordingEngine("first")
    second = RecordingEngine("second")
    progress: list[int] = []
    service = IndexingService(
        StreamingCatalog(10),  # type: ignore[arg-type]
        embedder,
        {"first": first, "second": second},  # type: ignore[dict-item]
        embedding_batch_size=3,
        chunk_batch_size=4,
    )

    result = await service.build("demo", ["first", "second"], progress.append)

    assert result == {"first": 10, "second": 10}
    assert embedder.batch_sizes == [3, 1, 3, 1, 2]
    assert [len(batch) for batch in first.batch_ids] == [4, 4, 2]
    assert first.batch_ids == second.batch_ids
    assert first.vectors == second.vectors
    assert progress == [4, 8, 10]
    assert first.prepared and second.prepared
    assert first.finalized and second.finalized
    assert not first.aborted and not second.aborted


@pytest.mark.asyncio
async def test_indexing_aborts_every_engine_after_a_batch_failure() -> None:
    first = RecordingEngine("first")
    second = RecordingEngine("second", fail_on_batch=2)
    service = IndexingService(
        StreamingCatalog(5),  # type: ignore[arg-type]
        RecordingEmbedder(),
        {"first": first, "second": second},  # type: ignore[dict-item]
        chunk_batch_size=3,
    )

    with pytest.raises(RuntimeError, match="index failed"):
        await service.build("demo", ["first", "second"])

    assert first.aborted and second.aborted
    assert not first.finalized and not second.finalized
