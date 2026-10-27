"""Benchmark vector-only query latency and judged retrieval quality.

Run from the repository root with:
uv run python performance-study/vector_search.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embeddings.base import Embedder  # noqa: E402
from embeddings.ollama import OllamaEmbedder  # noqa: E402
from embeddings.openai_compatible import OpenAICompatibleEmbedder  # noqa: E402
from meilisearch.adapter import MeilisearchSearchEngine  # noqa: E402
from search.base import SearchEngine  # noqa: E402
from settings import RuntimeSettings, get_runtime_settings  # noqa: E402
from solr.adapter import SolrSearchEngine  # noqa: E402

QUERY_COLUMNS = (
    "run_id",
    "trial",
    "engine",
    "query_id",
    "query",
    "repeat",
    "top_k",
    "latency_ms",
    "result_chunk_ids",
    "relevant_chunk_ids",
    "hits_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "status",
    "error",
)
SUMMARY_COLUMNS = (
    "run_id",
    "trial",
    "engine",
    "status",
    "query_count",
    "search_count",
    "top_k",
    "mean_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "mean_recall_at_k",
    "mean_reciprocal_rank",
    "mean_ndcg_at_k",
    "error_count",
)


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    id: str
    text: str
    relevant_chunk_ids: tuple[str, ...]


def detail_path(output: Path, kind: str, suffix: str = ".csv") -> Path:
    return output.with_name(f"{output.stem}-{kind}{suffix}")


def load_queries(path: Path, *, allow_unjudged: bool = False) -> list[EvaluationQuery]:
    queries: list[EvaluationQuery] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each query must be a JSON object")
            query_id, text, relevant = (
                row.get("query_id"), row.get("query"), row.get("relevant_chunk_ids")
            )
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(f"{path}:{line_number}: query_id must be a non-empty string")
            if query_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate query_id {query_id!r}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number}: query must be a non-empty string")
            if (
                not isinstance(relevant, list)
                or any(not isinstance(item, str) or not item for item in relevant)
                or len(set(relevant)) != len(relevant)
                or (not relevant and not allow_unjudged)
            ):
                raise ValueError(
                    f"{path}:{line_number}: relevant_chunk_ids must be a "
                    f"{'possibly empty' if allow_unjudged else 'non-empty'} list of "
                    "unique non-empty strings"
                )
            seen.add(query_id)
            queries.append(EvaluationQuery(query_id, text, tuple(relevant)))
    if not queries:
        raise ValueError(f"{path}: no evaluation queries")
    return queries


def metrics(result_ids: Sequence[str], relevant_ids: Sequence[str], top_k: int) -> dict[str, float]:
    relevant = set(relevant_ids)
    ranked = list(dict.fromkeys(result_ids[:top_k]))
    found = [chunk_id in relevant for chunk_id in ranked]
    hits = sum(found)
    reciprocal_rank = next((1.0 / rank for rank, hit in enumerate(found, 1) if hit), 0.0)
    dcg = sum(1.0 / math.log2(rank + 1) for rank, hit in enumerate(found, 1) if hit)
    ideal_hits = min(len(relevant), top_k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        "hits_at_k": hits,
        "recall_at_k": hits / len(relevant),
        "reciprocal_rank": reciprocal_rank,
        "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def percentile(values: Sequence[float], percentage: int) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentage / 100 * len(ordered)))
    return ordered[rank - 1]


async def embed_queries(
    embedder: Embedder, queries: list[EvaluationQuery], batch_size: int
) -> tuple[list[list[float]], float]:
    vectors: list[list[float]] = []
    started = time.perf_counter()
    for offset in range(0, len(queries), batch_size):
        batch = queries[offset : offset + batch_size]
        result = await embedder.embed_many([query.text for query in batch])
        if len(result) != len(batch):
            raise ValueError("Embedding response has the wrong number of vectors")
        for vector in result:
            if len(vector) != embedder.dimensions or any(
                not math.isfinite(value) for value in vector
            ):
                raise ValueError(
                    f"Query embedding must contain {embedder.dimensions} finite dimensions"
                )
        vectors.extend(result)
    return vectors, time.perf_counter() - started


async def study(
    args: argparse.Namespace,
    queries: list[EvaluationQuery],
    embedder: Embedder,
    engines: list[SearchEngine],
    runtime: RuntimeSettings,
    versions: dict[str, str] | None = None,
) -> None:
    has_judgments = [bool(query.relevant_chunk_ids) for query in queries]
    if any(has_judgments) and not all(has_judgments):
        raise ValueError("Evaluation queries must either all be judged or all be unjudged")
    latency_only = not any(has_judgments)
    if latency_only != getattr(args, "latency_only", False):
        raise ValueError(
            "Use --latency-only for queries without relevance judgments, and omit it "
            "when every query has relevance judgments"
        )
    for engine in engines:
        healthy, detail = await engine.health()
        if not healthy:
            raise RuntimeError(f"{engine.id} unavailable: {detail}")
        if not await engine.index_exists(args.dataset):
            raise RuntimeError(f"{engine.id} has no published index for {args.dataset}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    started_at = datetime.now(UTC).isoformat()
    query_path = Path(args.queries)
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "status": "incomplete",
        "dataset_id": args.dataset,
        "queries_sha256": hashlib.sha256(query_path.read_bytes()).hexdigest(),
        "query_count": len(queries),
        "embedding_model": getattr(embedder, "model", runtime.embedding_model),
        "embedding_dimensions": embedder.dimensions,
        "embedding_url": getattr(embedder, "url", getattr(embedder, "base_url", "")),
        "engine_order": [engine.id for engine in engines],
        "engine_versions": versions or {},
        "top_k": args.top_k,
        "trials": args.trials,
        "repetitions": args.repetitions,
        "warmups": args.warmups,
        "query_batch_size": args.query_batch_size,
        "query_embedding_s": None,
        "query_embedding_provider": args.embedding_provider,
        "evaluation_mode": "latency_only" if latency_only else "judged_quality",
    }
    run_path = detail_path(output, "run", ".json")
    run_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    vectors, embedding_s = await embed_queries(embedder, queries, args.query_batch_size)
    metadata["query_embedding_s"] = embedding_s
    metadata["status"] = "running"
    run_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    all_ok = True
    with (
        output.open("w", newline="", encoding="utf-8") as query_stream,
        detail_path(output, "summary").open("w", newline="", encoding="utf-8") as summary_stream,
    ):
        query_writer = csv.DictWriter(query_stream, fieldnames=QUERY_COLUMNS)
        summary_writer = csv.DictWriter(summary_stream, fieldnames=SUMMARY_COLUMNS)
        query_writer.writeheader()
        summary_writer.writeheader()
        for trial in range(1, args.trials + 1):
            engine_order = engines if trial % 2 else list(reversed(engines))
            for engine in engine_order:
                measurements: list[dict[str, Any]] = []
                for query, vector in zip(queries, vectors, strict=True):
                    try:
                        for _ in range(args.warmups):
                            await engine.vector_search(args.dataset, vector, args.top_k)
                        for repeat in range(1, args.repetitions + 1):
                            started = time.perf_counter()
                            hits = await engine.vector_search(args.dataset, vector, args.top_k)
                            latency_ms = (time.perf_counter() - started) * 1000
                            result_ids = [hit.chunk.id for hit in hits]
                            row: dict[str, Any] = {
                                "run_id": run_id,
                                "trial": trial,
                                "engine": engine.id,
                                "query_id": query.id,
                                "query": query.text,
                                "repeat": repeat,
                                "top_k": args.top_k,
                                "latency_ms": latency_ms,
                                "result_chunk_ids": ";".join(result_ids),
                                "relevant_chunk_ids": ";".join(query.relevant_chunk_ids),
                                **(
                                    {
                                        "hits_at_k": "",
                                        "recall_at_k": "",
                                        "reciprocal_rank": "",
                                        "ndcg_at_k": "",
                                    }
                                    if latency_only
                                    else metrics(
                                        result_ids, query.relevant_chunk_ids, args.top_k
                                    )
                                ),
                                "status": "ok",
                                "error": "",
                            }
                            measurements.append(row)
                            query_writer.writerow(row)
                            query_stream.flush()
                    except Exception as error:
                        all_ok = False
                        failed_row = {
                            "run_id": run_id,
                            "trial": trial,
                            "engine": engine.id,
                            "query_id": query.id,
                            "query": query.text,
                            "repeat": "",
                            "top_k": args.top_k,
                            "latency_ms": "",
                            "result_chunk_ids": "",
                            "relevant_chunk_ids": ";".join(query.relevant_chunk_ids),
                            "hits_at_k": "",
                            "recall_at_k": "",
                            "reciprocal_rank": "",
                            "ndcg_at_k": "",
                            "status": "failed",
                            "error": str(error),
                        }
                        query_writer.writerow(failed_row)
                        query_stream.flush()
                        break
                latencies = [float(row["latency_ms"]) for row in measurements]
                complete = len(measurements) == len(queries) * args.repetitions
                if not complete:
                    all_ok = False
                summary_writer.writerow({
                    "run_id": run_id,
                    "trial": trial,
                    "engine": engine.id,
                    "status": "ok" if complete else "failed",
                    "query_count": len(queries),
                    "search_count": len(measurements),
                    "top_k": args.top_k,
                    "mean_latency_ms": mean(latencies) if latencies else "",
                    "p50_latency_ms": percentile(latencies, 50) if latencies else "",
                    "p95_latency_ms": percentile(latencies, 95) if latencies else "",
                    "p99_latency_ms": percentile(latencies, 99) if latencies else "",
                    "mean_recall_at_k": (
                        mean(row["recall_at_k"] for row in measurements)
                        if measurements and not latency_only else ""
                    ),
                    "mean_reciprocal_rank": (
                        mean(row["reciprocal_rank"] for row in measurements)
                        if measurements and not latency_only else ""
                    ),
                    "mean_ndcg_at_k": (
                        mean(row["ndcg_at_k"] for row in measurements)
                        if measurements and not latency_only else ""
                    ),
                    "error_count": len(queries) * args.repetitions - len(measurements),
                })
                summary_stream.flush()
                print(
                    f"trial {trial} {engine.id}: "
                    f"{len(measurements)}/{len(queries) * args.repetitions} searches"
                )

    metadata["finished_at_utc"] = datetime.now(UTC).isoformat()
    metadata["status"] = "completed" if all_ok else "completed_with_errors"
    run_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="zbdoc")
    parser.add_argument("--queries", type=Path, required=True, help="Judged query JSONL file")
    parser.add_argument("--engines", nargs="+", choices=("solr", "meilisearch"),
                        default=["solr", "meilisearch"])
    parser.add_argument("--output", default="data/vector-search-study.csv")
    parser.add_argument("--embedding-provider", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--embedding-url", help="Native /api/embed URL or compatible base URL")
    parser.add_argument("--embedding-model", help="Override the configured embedding model")
    parser.add_argument("--embedding-dimensions", type=int, help="Expected vector size")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument(
        "--latency-only",
        action="store_true",
        help="Allow empty relevance judgments and omit retrieval-quality metrics",
    )
    args = parser.parse_args(argv)
    for name in ("top_k", "trials", "repetitions", "query_batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")
    if len(set(args.engines)) != len(args.engines):
        parser.error("each engine may be selected only once")
    return args


def initialize_engines(
    client: httpx.AsyncClient, runtime: RuntimeSettings
) -> dict[str, SearchEngine]:
    return {
        "solr": SolrSearchEngine(client, runtime.solr_url, runtime.solr_configset),
        "meilisearch": MeilisearchSearchEngine(
            client, runtime.meilisearch_url, runtime.meilisearch_master_key
        ),
    }


async def engine_versions(
    client: httpx.AsyncClient, runtime: RuntimeSettings, selected: list[str]
) -> dict[str, str]:
    versions: dict[str, str] = {}
    for engine in selected:
        try:
            if engine == "meilisearch":
                response = await client.get(
                    f"{runtime.meilisearch_url.rstrip('/')}/version",
                    headers=(
                        {"Authorization": f"Bearer {runtime.meilisearch_master_key}"}
                        if runtime.meilisearch_master_key else {}
                    ),
                    timeout=5.0,
                )
                response.raise_for_status()
                versions[engine] = str(response.json().get("pkgVersion", "unknown"))
            else:
                response = await client.get(
                    f"{runtime.solr_url.rstrip('/')}/admin/info/system",
                    params={"wt": "json"},
                    timeout=5.0,
                )
                response.raise_for_status()
                versions[engine] = str(
                    response.json().get("lucene", {}).get("solr-spec-version", "unknown")
                )
        except (httpx.HTTPError, ValueError, KeyError):
            versions[engine] = "unavailable"
    return versions


async def main() -> None:
    args = parse_args()
    queries = load_queries(args.queries, allow_unjudged=args.latency_only)
    runtime = get_runtime_settings()
    if args.embedding_provider == "ollama":
        url = args.embedding_url or "http://10.0.7.20:11435/api/embed"
        model = args.embedding_model or "qwen3-embedding:8b"
        dimensions = args.embedding_dimensions or 4096
        if not url.rstrip("/").endswith("/api/embed"):
            raise SystemExit("Ollama URL must point to its native /api/embed endpoint")
    else:
        url = args.embedding_url or runtime.embedding_base_url
        model = args.embedding_model or runtime.embedding_model
        dimensions = args.embedding_dimensions or runtime.embedding_dimensions
        if not url or not model:
            raise SystemExit("Configure an OpenAI-compatible embedding base URL and model")
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        embedder: Embedder = (
            OllamaEmbedder(client, url, model, dimensions)
            if args.embedding_provider == "ollama"
            else OpenAICompatibleEmbedder(
                client, url, model, dimensions, runtime.embedding_api_key
            )
        )
        engines = initialize_engines(client, runtime)
        versions = await engine_versions(client, runtime, args.engines)
        await study(
            args, queries, embedder, [engines[name] for name in args.engines], runtime, versions
        )


if __name__ == "__main__":
    asyncio.run(main())
