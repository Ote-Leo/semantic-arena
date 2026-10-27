from __future__ import annotations

import csv
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import httpx
import pytest

from datasets.models import Chunk
from settings import ArenaConfig, RuntimeSettings

spec = importlib.util.spec_from_file_location(
    "performance_study", Path(__file__).resolve().parents[1] / "performance-study" / "__main__.py"
)
assert spec is not None and spec.loader is not None
study_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study_module)


class Catalog:
    def __init__(self, root: Path) -> None:
        self.root = root
        directory = root / "zbdoc"
        directory.mkdir()
        (directory / "manifest.yaml").write_text("id: zbdoc\n", encoding="utf-8")

    def manifest(self, dataset_id: str) -> None:
        assert dataset_id == "zbdoc"

    def iter_chunks(self, dataset_id: str):
        for index in range(5):
            yield Chunk(
                id=str(index), dataset_id=dataset_id, source_id=str(index),
                title="test", text=f"text {index}", source_path="batch.parquet",
                record_index=index, chunk_index=0, start_char=0, end_char=6,
            )


class Embedder:
    dimensions = 3

    def __init__(self) -> None:
        self.requests: list[int] = []

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.requests.append(len(texts))
        return [[float(text.split()[-1]), 0.0, 0.0] for text in texts]


class Engine:
    def __init__(self, engine_id: str, calls: list[str], fail: bool = False) -> None:
        self.id = engine_id
        self.label = engine_id
        self.calls = calls
        self.fail = fail
        self.received: list[tuple[list[str], list[list[float]]]] = []
        self.index_dataset_ids: list[str] = []
        self.aborted = False
        self.published = False

    async def health(self) -> tuple[bool, str]:
        return True, "ready"

    async def index_exists(self, dataset_id: str) -> bool:
        self.index_dataset_ids.append(dataset_id)
        return self.published

    async def prepare_index(self, dataset_id: str, dimensions: int) -> None:
        assert dimensions == 3
        self.index_dataset_ids.append(dataset_id)
        self.calls.append(self.id)

    async def index_batch(self, dataset_id: str, chunks, vectors) -> None:
        self.index_dataset_ids.append(dataset_id)
        if self.fail:
            raise RuntimeError("index failed")
        self.received.append(([chunk.id for chunk in chunks], vectors))

    async def finalize_index(self, dataset_id: str) -> None:
        self.index_dataset_ids.append(dataset_id)
        self.published = True

    async def abort_index(self, dataset_id: str) -> None:
        self.index_dataset_ids.append(dataset_id)
        self.aborted = True


@pytest.mark.asyncio
async def test_meilisearch_version_probe_authenticates() -> None:
    seen_headers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers["authorization"])
        return httpx.Response(200, json={"pkgVersion": "1.16.0"})

    runtime = RuntimeSettings(
        meilisearch_url="http://meili.test",
        meilisearch_master_key="benchmark-key",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        versions = await study_module.engine_versions(client, runtime, ["meilisearch"])
    assert versions == {"meilisearch": "1.16.0"}
    assert seen_headers == ["Bearer benchmark-key"]


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size,expected", [(1, [1, 1, 1, 1, 1]), (2, [2, 1, 2])])
async def test_study_uses_same_vectors_and_writes_metrics(
    tmp_path: Path, batch_size: int, expected: list[int]
) -> None:
    catalog = Catalog(tmp_path)
    embedder = Embedder()
    calls: list[str] = []
    solr = Engine("solr", calls)
    meili = Engine("meilisearch", calls)
    output = tmp_path / "results.csv"
    args = Namespace(
        dataset="zbdoc", output=str(output), trials=2, max_chunks=None,
        batch_embedding=False, embedding_batch_size=batch_size, replace_existing=True,
    )
    config = ArenaConfig(indexing={"chunk_batch_size": 3, "embedding_batch_size": 2})  # pyright: ignore[reportArgumentType]
    await study_module.study(
        args, catalog, embedder, [solr, meili],
        RuntimeSettings(embedding_model="example", embedding_dimensions=3), config,
    )

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert all(row["chunk_count"] == "5" and row["status"] == "ok" for row in rows)
    assert all(row["embedding_requests"] == str(len(expected)) for row in rows)
    assert all(float(row["engine_total_s"]) >= 0 for row in rows)
    assert embedder.requests == expected * 2
    assert solr.received == meili.received
    assert calls == ["solr", "meilisearch", "meilisearch", "solr"]
    assert len({row["run_id"] for row in rows}) == 1
    assert all(row["record_count"] == "5" and float(row["trial_wall_s"]) >= 0 for row in rows)
    embedding = list(csv.DictReader(
        study_module.detail_path(output, "embedding-batches").open(newline="", encoding="utf-8")
    ))
    ingestion = list(csv.DictReader(
        study_module.detail_path(output, "ingestion-batches").open(newline="", encoding="utf-8")
    ))
    chunks = list(csv.DictReader(
        study_module.detail_path(output, "chunks").open(newline="", encoding="utf-8")
    ))
    records = list(csv.DictReader(
        study_module.detail_path(output, "records").open(newline="", encoding="utf-8")
    ))
    assert len(embedding) == 2 * len(expected)
    assert [int(row["chunk_count"]) for row in embedding[:len(expected)]] == expected
    assert len(ingestion) == 8  # two submission batches per engine per trial
    assert len(chunks) == len(records) == 10
    assert all(row["embedding_batch_id"] and row["solr_batch_id"] and
               row["meilisearch_batch_id"] for row in chunks)
    assert {row["run_id"] for row in records} == {rows[0]["run_id"]}
    metadata = json.loads(study_module.detail_path(output, "run", ".json").read_text())
    assert metadata["run_id"] == rows[0]["run_id"]
    assert metadata["status"] == "completed"
    assert metadata["chunk_batch_size"] == 3


@pytest.mark.asyncio
async def test_engine_failure_is_recorded_and_does_not_block_other_engine(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path)
    failing = Engine("solr", [], fail=True)
    succeeding = Engine("meilisearch", [])
    output = tmp_path / "results.csv"
    args = Namespace(
        dataset="zbdoc", output=str(output), trials=1, max_chunks=2,
        batch_embedding=True, embedding_batch_size=None, replace_existing=False,
    )
    config = ArenaConfig(indexing={"chunk_batch_size": 3, "embedding_batch_size": 2})  # pyright: ignore[reportArgumentType]
    await study_module.study(
        args, catalog, Embedder(), [failing, succeeding],
        RuntimeSettings(embedding_model="example", embedding_dimensions=3), config,
    )
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["status"] == "failed" and "index failed" in rows[0]["error"]
    assert rows[1]["status"] == "ok" and rows[1]["chunk_count"] == "2"
    assert failing.aborted
    assert succeeding.received[0][0] == ["0", "1"]
    assert json.loads(study_module.detail_path(output, "run", ".json").read_text())["status"] == (
        "completed_with_errors"
    )
    ingestion = list(csv.DictReader(
        study_module.detail_path(output, "ingestion-batches").open(newline="", encoding="utf-8")
    ))
    assert [row["status"] for row in ingestion] == ["failed", "ok"]
    assert [row["cumulative_chunks"] for row in ingestion] == ["0", "2"]


@pytest.mark.asyncio
async def test_study_can_publish_to_separate_index_id(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path)
    engines = [Engine("solr", []), Engine("meilisearch", [])]
    output = tmp_path / "separate-index.csv"
    args = Namespace(
        dataset="zbdoc", index_dataset="presentation-demo-ollama",
        output=str(output), trials=1, max_chunks=2, batch_embedding=False,
        embedding_batch_size=1, replace_existing=False,
    )
    await study_module.study(
        args, catalog, Embedder(), engines,
        RuntimeSettings(embedding_model="example", embedding_dimensions=3), ArenaConfig(),
    )

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["dataset_id"] == "zbdoc" for row in rows)
    assert all(row["index_dataset_id"] == "presentation-demo-ollama" for row in rows)
    assert all(
        index_id == "presentation-demo-ollama"
        for engine in engines
        for index_id in engine.index_dataset_ids
    )
    metadata = json.loads(study_module.detail_path(output, "run", ".json").read_text())
    assert metadata["dataset_id"] == "zbdoc"
    assert metadata["index_dataset_id"] == "presentation-demo-ollama"


@pytest.mark.asyncio
async def test_embedding_failure_keeps_partial_results(tmp_path: Path) -> None:
    class FailingEmbedder(Embedder):
        async def embed_many(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("provider unavailable")

    output = tmp_path / "results.csv"
    args = Namespace(
        dataset="zbdoc", output=str(output), trials=1, max_chunks=2,
        batch_embedding=False, embedding_batch_size=None, replace_existing=False,
    )
    engines = [Engine("solr", []), Engine("meilisearch", [])]
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await study_module.study(
            args, Catalog(tmp_path), FailingEmbedder(), engines,
            RuntimeSettings(embedding_model="example", embedding_dimensions=3), ArenaConfig(),
        )
    with output.open(newline="") as stream:
        assert [row["status"] for row in csv.DictReader(stream)] == ["failed", "failed"]
    with study_module.detail_path(output, "embedding-batches").open(newline="") as stream:
        assert next(csv.DictReader(stream))["status"] == "failed"
    assert json.loads(study_module.detail_path(output, "run", ".json").read_text())["status"] == (
        "incomplete"
    )
    assert all(not engine.received for engine in engines)


@pytest.mark.asyncio
async def test_single_chunk_mode_and_record_shared_batches(tmp_path: Path) -> None:
    class MultiChunkCatalog(Catalog):
        def iter_chunks(self, dataset_id: str):
            for index in range(3):
                yield Chunk(
                    id=str(index), dataset_id=dataset_id, source_id="one-source",
                    title="test", text=f"text {index}", source_path="batch.parquet",
                    record_index=0, chunk_index=index, start_char=index * 6,
                    end_char=(index + 1) * 6,
                )

    output = tmp_path / "results.csv"
    args = Namespace(
        dataset="zbdoc", output=str(output), trials=1, max_chunks=None,
        batch_embedding=False, embedding_batch_size=1, chunk_batch_size=1,
        replace_existing=False, embedding_provider="ollama",
    )
    await study_module.study(
        args, MultiChunkCatalog(tmp_path), Embedder(), [Engine("solr", [])],
        RuntimeSettings(embedding_model="example", embedding_dimensions=3), ArenaConfig(),
    )
    with study_module.detail_path(output, "records").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 1
    assert records[0]["chunk_count"] == "3"
    assert records[0]["embedding_batch_ids"] == "1;2;3"
    assert records[0]["solr_batch_ids"] == "1;2;3"
    assert records[0]["meilisearch_batch_ids"] == ""
    with output.open(newline="") as stream:
        assert next(csv.DictReader(stream))["record_count"] == "1"
