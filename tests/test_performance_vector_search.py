from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import httpx
import pytest

from datasets.models import Chunk
from search.models import SearchHit
from settings import RuntimeSettings

study_spec = importlib.util.spec_from_file_location(
    "vector_search",
    Path(__file__).resolve().parents[1] / "performance-study" / "vector_search.py",
)
assert study_spec is not None and study_spec.loader is not None
study_module = importlib.util.module_from_spec(study_spec)
sys.modules[study_spec.name] = study_module
study_spec.loader.exec_module(study_module)

plot_spec = importlib.util.spec_from_file_location(
    "vector_search_plot",
    Path(__file__).resolve().parents[1] / "performance-study" / "vector_search_plot.py",
)
assert plot_spec is not None and plot_spec.loader is not None
plot_module = importlib.util.module_from_spec(plot_spec)
plot_spec.loader.exec_module(plot_module)


def chunk(chunk_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        dataset_id="demo",
        source_id=chunk_id,
        title=chunk_id,
        text=chunk_id,
        source_path="demo.md",
        record_index=0,
        chunk_index=0,
        start_char=0,
        end_char=1,
    )


class Embedder:
    dimensions = 3
    model = "test-model"

    def __init__(self) -> None:
        self.requests: list[list[str]] = []

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.requests.append(texts)
        return [[float(index), 0.0, 0.0] for index, _ in enumerate(texts)]


class Engine:
    def __init__(self, engine_id: str, order: list[str], fail: bool = False) -> None:
        self.id = engine_id
        self.order = order
        self.fail = fail
        self.vectors: list[list[float]] = []
        self.calls = 0

    async def health(self) -> tuple[bool, str]:
        return True, "ready"

    async def index_exists(self, dataset_id: str) -> bool:
        return True

    async def vector_search(self, dataset_id: str, vector: list[float], limit: int):
        self.calls += 1
        self.vectors.append(vector)
        if self.fail:
            raise RuntimeError("query failed")
        return [SearchHit(chunk(item), 1.0) for item in self.order[:limit]]


def query(query_id: str, relevant: tuple[str, ...]) -> study_module.EvaluationQuery:
    return study_module.EvaluationQuery(query_id, f"question {query_id}", relevant)


def write_query_file(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_load_queries_and_metrics(tmp_path: Path) -> None:
    path = write_query_file(tmp_path / "queries.jsonl", [
        {"query_id": "q1", "query": "question", "relevant_chunk_ids": ["b", "c"]},
    ])
    loaded = study_module.load_queries(path)
    assert loaded == [study_module.EvaluationQuery("q1", "question", ("b", "c"))]
    assert study_module.metrics(["a", "b", "c"], ["b", "c"], 2) == {
        "hits_at_k": 1,
        "recall_at_k": 0.5,
        "reciprocal_rank": 0.5,
        "ndcg_at_k": pytest.approx(
            (1 / math.log2(3)) / (1 + 1 / math.log2(3))
        ),
    }
    assert study_module.metrics(["b", "b", "c"], ["b", "c"], 3)["recall_at_k"] == 1


def test_load_unjudged_queries_requires_latency_mode(tmp_path: Path) -> None:
    path = write_query_file(tmp_path / "latency.jsonl", [
        {"query_id": "q1", "query": "question", "relevant_chunk_ids": []},
    ])
    with pytest.raises(ValueError, match="relevant_chunk_ids"):
        study_module.load_queries(path)
    assert study_module.load_queries(path, allow_unjudged=True) == [
        study_module.EvaluationQuery("q1", "question", ())
    ]


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


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"query_id": "q", "query": "x", "relevant_chunk_ids": []}], "relevant_chunk_ids"),
        (
            [
                {"query_id": "q", "query": "x", "relevant_chunk_ids": ["x"]},
                {"query_id": "q", "query": "y", "relevant_chunk_ids": ["y"]},
            ],
            "duplicate query_id",
        ),
    ],
)
def test_invalid_query_files(tmp_path: Path, rows: list[dict[str, object]], message: str) -> None:
    path = write_query_file(tmp_path / "queries.jsonl", rows)
    with pytest.raises(ValueError, match=message):
        study_module.load_queries(path)


@pytest.mark.asyncio
async def test_study_uses_shared_vectors_and_writes_query_metrics(tmp_path: Path) -> None:
    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text(
        "\n".join(
            json.dumps({
                "query_id": f"q{index}",
                "query": f"question {index}",
                "relevant_chunk_ids": ["0"],
            })
            for index in range(2)
        ) + "\n",
        encoding="utf-8",
    )
    queries = study_module.load_queries(queries_path)
    embedder = Embedder()
    solr = Engine("solr", ["0", "1"])
    meili = Engine("meilisearch", ["1", "0"])
    output = tmp_path / "vector.csv"
    args = Namespace(
        dataset="demo",
        queries=queries_path,
        output=str(output),
        top_k=1,
        trials=2,
        repetitions=2,
        warmups=1,
        query_batch_size=32,
        embedding_provider="openai",
    )
    await study_module.study(
        args,
        queries,
        embedder,
        [solr, meili],
        RuntimeSettings(embedding_model="test-model", embedding_dimensions=3),
    )

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16
    assert {row["status"] for row in rows} == {"ok"}
    assert embedder.requests == [["question 0", "question 1"]]
    assert solr.vectors == meili.vectors
    assert solr.calls == meili.calls == 12  # 2 trials * 2 queries * (1 warmup + 2 repeats)
    assert all(row["recall_at_k"] == "1.0" for row in rows if row["engine"] == "solr")
    assert all(row["recall_at_k"] == "0.0" for row in rows if row["engine"] == "meilisearch")

    with study_module.detail_path(output, "summary").open(
        newline="", encoding="utf-8"
    ) as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == 4
    assert all(row["search_count"] == "4" and row["status"] == "ok" for row in summary)
    metadata = json.loads(study_module.detail_path(output, "run", ".json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["query_count"] == 2
    assert metadata["query_embedding_s"] >= 0


@pytest.mark.asyncio
async def test_latency_only_study_omits_quality_metrics(tmp_path: Path) -> None:
    queries_path = write_query_file(tmp_path / "latency.jsonl", [
        {"query_id": "q1", "query": "question", "relevant_chunk_ids": []},
    ])
    queries = study_module.load_queries(queries_path, allow_unjudged=True)
    output = tmp_path / "latency.csv"
    args = Namespace(
        dataset="demo",
        queries=queries_path,
        output=str(output),
        top_k=1,
        trials=1,
        repetitions=2,
        warmups=0,
        query_batch_size=1,
        embedding_provider="openai",
        latency_only=True,
    )
    await study_module.study(
        args,
        queries,
        Embedder(),
        [Engine("solr", ["0"]), Engine("meilisearch", ["0"])],
        RuntimeSettings(embedding_model="test-model", embedding_dimensions=3),
    )
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["relevant_chunk_ids"] == "" for row in rows)
    assert all(row["recall_at_k"] == "" and row["ndcg_at_k"] == "" for row in rows)
    with study_module.detail_path(output, "summary").open(
        newline="", encoding="utf-8"
    ) as stream:
        summary = list(csv.DictReader(stream))
    assert all(row["mean_recall_at_k"] == "" for row in summary)
    metadata = json.loads(study_module.detail_path(output, "run", ".json").read_text())
    assert metadata["evaluation_mode"] == "latency_only"


@pytest.mark.asyncio
async def test_engine_query_failure_does_not_block_other_engine(tmp_path: Path) -> None:
    queries_path = write_query_file(tmp_path / "queries.jsonl", [
        {"query_id": "q1", "query": "question", "relevant_chunk_ids": ["0"]},
    ])
    args = Namespace(
        dataset="demo",
        queries=queries_path,
        output=str(tmp_path / "vector.csv"),
        top_k=1,
        trials=1,
        repetitions=1,
        warmups=0,
        query_batch_size=1,
        embedding_provider="openai",
    )
    await study_module.study(
        args,
        study_module.load_queries(queries_path),
        Embedder(),
        [Engine("solr", [], fail=True), Engine("meilisearch", ["0"])],
        RuntimeSettings(embedding_model="test-model", embedding_dimensions=3),
    )
    with Path(args.output).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["status"] for row in rows] == ["failed", "ok"]
    with study_module.detail_path(Path(args.output), "summary").open(
        newline="", encoding="utf-8"
    ) as stream:
        summary = list(csv.DictReader(stream))
    assert [row["status"] for row in summary] == ["failed", "ok"]
    metadata = json.loads(study_module.detail_path(
        Path(args.output), "run", ".json"
    ).read_text())
    assert metadata["status"] == "completed_with_errors"


def test_plot_writes_latency_quality_and_tradeoff_charts(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    path = tmp_path / "queries.csv"
    rows = []
    for engine, latency, recall in (("solr", "4", "1"), ("meilisearch", "6", "0.5")):
        rows.append({
            "run_id": "run",
            "trial": "1",
            "engine": engine,
            "query_id": "q1",
            "query": "question",
            "repeat": "1",
            "top_k": "2",
            "latency_ms": latency,
            "result_chunk_ids": "x",
            "relevant_chunk_ids": "x;y",
            "hits_at_k": recall,
            "recall_at_k": recall,
            "reciprocal_rank": recall,
            "ndcg_at_k": recall,
            "status": "ok",
            "error": "",
        })
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=plot_module.QUERY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    written = plot_module.plot(path, tmp_path / "plots")
    assert {item.name for item in written} == {
        "vector-search-latency.png",
        "vector-search-quality.png",
        "vector-search-tradeoff.png",
    }
    assert all(item.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for item in written)


def test_plot_latency_only_writes_no_quality_charts(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    path = tmp_path / "latency.csv"
    rows = []
    for engine, latency in (("solr", "4"), ("meilisearch", "6")):
        rows.append({
            "run_id": "run",
            "trial": "1",
            "engine": engine,
            "query_id": "q1",
            "query": "question",
            "repeat": "1",
            "top_k": "10",
            "latency_ms": latency,
            "result_chunk_ids": "x",
            "relevant_chunk_ids": "",
            "hits_at_k": "",
            "recall_at_k": "",
            "reciprocal_rank": "",
            "ndcg_at_k": "",
            "status": "ok",
            "error": "",
        })
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=plot_module.QUERY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    written = plot_module.plot(path, tmp_path / "latency-plots")
    assert {item.name for item in written} == {
        "vector-search-latency.png",
        "vector-search-latency-by-query.png",
    }


def test_plot_rejects_mixed_judged_and_unjudged_rows(tmp_path: Path) -> None:
    path = tmp_path / "mixed.csv"
    rows = []
    for query_id, recall in (("judged", "1.0"), ("unjudged", "")):
        rows.append({
            "run_id": "run",
            "trial": "1",
            "engine": "solr",
            "query_id": query_id,
            "query": "question",
            "repeat": "1",
            "top_k": "10",
            "latency_ms": "4",
            "result_chunk_ids": "x",
            "relevant_chunk_ids": "x" if recall else "",
            "hits_at_k": recall,
            "recall_at_k": recall,
            "reciprocal_rank": recall,
            "ndcg_at_k": recall,
            "status": "ok",
            "error": "",
        })
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=plot_module.QUERY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="mix judged and unjudged"):
        plot_module.plot(path, tmp_path / "plots")
