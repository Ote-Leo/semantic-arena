"""Measure indexing stages with identical, externally generated vectors for each engine.

Run from the repository root: uv run python performance-study/__main__.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import pickle
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

# A script inside a hyphenated directory cannot rely on the repository root being on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.catalog import DatasetCatalog  # noqa: E402
from embeddings.base import Embedder  # noqa: E402
from embeddings.ollama import OllamaEmbedder  # noqa: E402
from embeddings.openai_compatible import OpenAICompatibleEmbedder  # noqa: E402
from meilisearch.adapter import MeilisearchSearchEngine  # noqa: E402
from search.base import SearchEngine  # noqa: E402
from settings import (  # noqa: E402
    ArenaConfig,
    RuntimeSettings,
    get_arena_config,
    get_runtime_settings,
)
from solr.adapter import SolrSearchEngine  # noqa: E402

COLUMNS = (
    "run_id",
    "started_at_utc",
    "dataset_id",
    "index_dataset_id",
    "manifest_sha256",
    "model",
    "dimensions",
    "trial",
    "engine",
    "status",
    "error",
    "chunk_count",
    "record_count",
    "embedding_provider",
    "embedding_batch_size",
    "chunk_batch_size",
    "embedding_requests",
    "extraction_s",
    "embedding_s",
    "prepare_s",
    "ingestion_s",
    "finalization_s",
    "engine_total_s",
    "end_to_end_s",
    "trial_wall_s",
    "embedding_chunks_per_s",
    "engine_chunks_per_s",
)

EMBEDDING_COLUMNS = (
    "run_id", "trial", "batch_id", "chunk_count", "duration_s", "status", "error",
)
INGESTION_COLUMNS = (
    "run_id", "trial", "engine", "batch_id", "chunk_count", "cumulative_chunks",
    "duration_s", "status", "error",
)
CHUNK_COLUMNS = (
    "run_id", "trial", "chunk_id", "source_id", "source_path", "record_index",
    "chunk_index", "text_chars", "text_bytes", "embedding_batch_id",
    "embedding_duration_s", "solr_batch_id", "solr_duration_s",
    "meilisearch_batch_id", "meilisearch_duration_s",
)
RECORD_COLUMNS = (
    "run_id", "trial", "source_id", "source_path", "record_index", "chunk_count",
    "text_bytes", "embedding_batch_ids", "embedding_shared_batch_s",
    "solr_batch_ids", "solr_shared_batch_s", "meilisearch_batch_ids",
    "meilisearch_shared_batch_s",
)


def detail_path(output: Path, kind: str, suffix: str = ".csv") -> Path:
    return output.with_name(f"{output.stem}-{kind}{suffix}")


def write_records(writer: csv.DictWriter, chunk_rows: dict[str, dict[str, Any]]) -> None:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for chunk in chunk_rows.values():
        source_id = chunk["source_id"]
        key = (chunk["source_path"], chunk["record_index"])
        if key not in records:
            records[key] = {
                "run_id": chunk["run_id"], "trial": chunk["trial"],
                "source_id": source_id, "source_path": chunk["source_path"],
                "record_index": chunk["record_index"], "chunk_count": 0,
                "text_bytes": 0, "batches": {stage: {} for stage in
                                            ("embedding", "solr", "meilisearch")},
            }
        record = records[key]
        record["chunk_count"] += 1
        record["text_bytes"] += chunk["text_bytes"]
        for stage in ("embedding", "solr", "meilisearch"):
            batch_id = chunk.get(f"{stage}_batch_id")
            if batch_id is not None:
                record["batches"][stage][batch_id] = chunk[f"{stage}_duration_s"]
    for record in records.values():
        row = {key: value for key, value in record.items() if key != "batches"}
        for stage, groups in record["batches"].items():
            row[f"{stage}_batch_ids"] = ";".join(str(value) for value in sorted(groups))
            row[f"{stage}_shared_batch_s"] = sum(groups.values()) if groups else ""
        writer.writerow(row)


def batches(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def stored_batches(path: Path) -> Iterator[list[Any]]:
    # Only read temporary files written by this process; pickle is not safe for untrusted input.
    with path.open("rb") as stream:
        while True:
            try:
                yield pickle.load(stream)
            except EOFError:
                return


def seconds(start: float) -> float:
    return time.perf_counter() - start


def extract(
    catalog: DatasetCatalog,
    dataset_id: str,
    path: Path,
    size: int,
    max_chunks: int | None,
) -> tuple[int, float]:
    started = time.perf_counter()
    count = 0
    with (
        path.open("wb") as stream,
        tqdm(
            total=max_chunks,
            desc=f"Extracting {dataset_id}",
            unit="chunks",
            dynamic_ncols=True,
        ) as progress,
    ):
        for chunk_batch in batches(catalog.iter_chunks(dataset_id), size):
            if max_chunks is not None:
                chunk_batch = chunk_batch[: max_chunks - count]
            if not chunk_batch:
                break
            pickle.dump(chunk_batch, stream)
            count += len(chunk_batch)
            progress.update(len(chunk_batch))
            if max_chunks is not None and count >= max_chunks:
                break
        if progress.total is None:
            progress.total = count
            progress.refresh()
    if not count:
        raise ValueError(f"Dataset {dataset_id!r} produced no chunks")
    return count, seconds(started)


EmbeddingBatch = Callable[[int, list[Any], float, str, str], None]
IngestionBatch = Callable[[int, list[Any], float, str, str], None]


async def embed_chunks(
    embedder: Embedder, chunk_path: Path, vector_path: Path, batch_size: int,
    total_chunks: int,
    on_batch: EmbeddingBatch | None = None,
) -> tuple[int, float]:
    started = time.perf_counter()
    requests = 0
    with (
        vector_path.open("wb") as stream,
        tqdm(
            total=total_chunks,
            desc="Embedding chunks",
            unit="chunks",
            dynamic_ncols=True,
        ) as progress,
    ):
        for chunk_batch in stored_batches(chunk_path):
            vectors: list[list[float]] = []
            for text_batch in batches(chunk_batch, batch_size):
                requests += 1
                request_start = time.perf_counter()
                try:
                    result = await embedder.embed_many([chunk.text for chunk in text_batch])
                    if len(result) != len(text_batch):
                        raise ValueError("Embedding response has the wrong number of vectors")
                except Exception as error:
                    if on_batch:
                        on_batch(requests, text_batch, seconds(request_start), "failed", str(error))
                    raise
                if on_batch:
                    on_batch(requests, text_batch, seconds(request_start), "ok", "")
                vectors.extend(result)
                progress.update(len(text_batch))
            pickle.dump(vectors, stream)
    return requests, seconds(started)


async def index_engine(
    engine: SearchEngine, dataset_id: str, dimensions: int, chunk_path: Path,
    vector_path: Path, on_batch: IngestionBatch | None = None,
    timings: dict[str, float] | None = None, total_chunks: int = 0,
) -> dict[str, float]:
    timings = timings if timings is not None else {}
    try:
        started = time.perf_counter()
        try:
            await engine.prepare_index(dataset_id, dimensions)
        finally:
            timings["prepare_s"] = seconds(started)

        started = time.perf_counter()
        try:
            with tqdm(
                total=total_chunks,
                desc=f"Indexing {engine.label}",
                unit="chunks",
                dynamic_ncols=True,
            ) as progress:
                for batch_id, (chunks, vectors) in enumerate(
                    zip(stored_batches(chunk_path), stored_batches(vector_path), strict=True), 1
                ):
                    batch_start = time.perf_counter()
                    try:
                        await engine.index_batch(dataset_id, chunks, vectors)
                    except Exception as error:
                        if on_batch:
                            on_batch(batch_id, chunks, seconds(batch_start), "failed", str(error))
                        raise
                    if on_batch:
                        on_batch(batch_id, chunks, seconds(batch_start), "ok", "")
                    progress.update(len(chunks))
        finally:
            timings["ingestion_s"] = seconds(started)

        started = time.perf_counter()
        try:
            await engine.finalize_index(dataset_id)
            if not await engine.index_exists(dataset_id):
                raise RuntimeError(f"{engine.id} did not publish its index")
        finally:
            timings["finalization_s"] = seconds(started)
    except Exception:
        await engine.abort_index(dataset_id)
        raise
    return timings


def initialize_engines(client: httpx.AsyncClient,
                       runtime: RuntimeSettings) -> dict[str, SearchEngine]:
    return {
        "solr": SolrSearchEngine(client, runtime.solr_url, runtime.solr_configset),
        "meilisearch": MeilisearchSearchEngine(
            client, runtime.meilisearch_url, runtime.meilisearch_master_key
        ),
    }


async def study(
    args: argparse.Namespace, catalog: DatasetCatalog, embedder: Embedder,
    engines: list[SearchEngine], runtime: RuntimeSettings, config: ArenaConfig,
    versions: dict[str, str] | None = None,
) -> None:
    catalog.manifest(args.dataset)
    index_dataset_id = getattr(args, "index_dataset", None) or args.dataset
    manifest = catalog.root / args.dataset / "manifest.yaml"
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for engine in engines:
        healthy, detail = await engine.health()
        if not healthy:
            raise RuntimeError(f"{engine.id} unavailable: {detail}")
        if not args.replace_existing and await engine.index_exists(index_dataset_id):
            raise RuntimeError(
                f"{engine.id} already has an index for {index_dataset_id}; "
                "use --replace-existing to allow replacement"
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    started_at = datetime.now(UTC).isoformat()
    batch_size = args.embedding_batch_size or (
        config.indexing.embedding_batch_size if args.batch_embedding else 1
    )
    # Both adapters split at 100; avoid labelling several requests as a single submission.
    chunk_size = min(
        getattr(args, "chunk_batch_size", None) or config.indexing.chunk_batch_size, 100
    )
    provider = getattr(args, "embedding_provider", "openai")
    metadata = {
        "run_id": run_id, "started_at_utc": started_at, "finished_at_utc": None,
        "status": "incomplete", "dataset_id": args.dataset,
        "index_dataset_id": index_dataset_id,
        "manifest_sha256": manifest_hash, "embedding_provider": provider,
        "embedding_model": getattr(embedder, "model", runtime.embedding_model),
        "embedding_dimensions": embedder.dimensions, "embedding_batch_size": batch_size,
        "chunk_batch_size": chunk_size, "chunking": config.chunking.model_dump(),
        "engine_order": [engine.id for engine in engines], "engine_versions": versions or {},
        "trials": args.trials, "max_chunks": args.max_chunks,
        "embedding_url": getattr(embedder, "url", getattr(embedder, "base_url", "")),
    }
    run_path = detail_path(output, "run", ".json")
    run_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    failed_engines = False
    with (
        output.open("w", newline="", encoding="utf-8") as stream,
        detail_path(output, "embedding-batches").open(
            "w", newline="", encoding="utf-8"
        ) as emb_stream,
        detail_path(output, "ingestion-batches").open(
            "w", newline="", encoding="utf-8"
        ) as ing_stream,
        detail_path(output, "chunks").open("w", newline="", encoding="utf-8") as chunk_stream,
        detail_path(output, "records").open("w", newline="", encoding="utf-8") as record_stream,
    ):
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        emb_writer = csv.DictWriter(emb_stream, fieldnames=EMBEDDING_COLUMNS)
        ing_writer = csv.DictWriter(ing_stream, fieldnames=INGESTION_COLUMNS)
        chunk_writer = csv.DictWriter(chunk_stream, fieldnames=CHUNK_COLUMNS)
        record_writer = csv.DictWriter(record_stream, fieldnames=RECORD_COLUMNS)
        for csv_writer in (writer, emb_writer, ing_writer, chunk_writer, record_writer):
            csv_writer.writeheader()
        for trial in range(1, args.trials + 1):
            trial_start = time.perf_counter()
            with tempfile.TemporaryDirectory() as directory:
                chunks = Path(directory) / "chunks.pkl"
                vectors = Path(directory) / "vectors.pkl"
                count, extraction_s = extract(
                    catalog, args.dataset, chunks, chunk_size, args.max_chunks
                )
                chunk_rows: dict[str, dict[str, Any]] = {}
                for group in stored_batches(chunks):
                    for chunk in group:
                        chunk_rows[chunk.id] = {
                            "run_id": run_id, "trial": trial, "chunk_id": chunk.id,
                            "source_id": chunk.source_id, "source_path": chunk.source_path,
                            "record_index": chunk.record_index, "chunk_index": chunk.chunk_index,
                            "text_chars": len(chunk.text),
                            "text_bytes": len(chunk.text.encode("utf-8")),
                        }

                def embedding_batch(
                    batch_id: int, group: list[Any], duration: float, status: str, error: str
                ) -> None:
                    emb_writer.writerow({
                        "run_id": run_id, "trial": trial, "batch_id": batch_id,
                        "chunk_count": len(group), "duration_s": duration,
                        "status": status, "error": error,
                    })
                    emb_stream.flush()
                    for chunk in group:
                        chunk_rows[chunk.id].update(
                            embedding_batch_id=batch_id, embedding_duration_s=duration
                        )

                embedding_start = time.perf_counter()
                try:
                    requests, embedding_s = await embed_chunks(
                        embedder, chunks, vectors, batch_size, count, embedding_batch
                    )
                except Exception as error:
                    for engine in engines:
                        writer.writerow({
                            "run_id": run_id, "started_at_utc": started_at,
                            "dataset_id": args.dataset, "manifest_sha256": manifest_hash,
                            "index_dataset_id": index_dataset_id,
                            "model": getattr(embedder, "model", runtime.embedding_model),
                            "dimensions": embedder.dimensions, "trial": trial,
                            "engine": engine.id, "status": "failed", "error": str(error),
                            "chunk_count": count,
                            "record_count": len({
                                (item["source_path"], item["record_index"])
                                for item in chunk_rows.values()
                            }),
                            "embedding_provider": provider,
                            "embedding_batch_size": batch_size, "chunk_batch_size": chunk_size,
                            "extraction_s": extraction_s,
                            "embedding_s": seconds(embedding_start),
                            "trial_wall_s": seconds(trial_start),
                        })
                    for chunk_row in chunk_rows.values():
                        chunk_writer.writerow(chunk_row)  # pyright: ignore[reportArgumentType]
                    write_records(record_writer, chunk_rows)
                    stream.flush()
                    chunk_stream.flush()
                    record_stream.flush()
                    raise

                record_count = len({
                    (item["source_path"], item["record_index"])
                    for item in chunk_rows.values()
                })
                rows: list[dict[str, Any]] = []
                order = engines if trial % 2 else list(reversed(engines))
                for engine in order:
                    row: dict[str, Any] = {
                        "run_id": run_id,
                        "started_at_utc": started_at,
                        "dataset_id": args.dataset,
                        "index_dataset_id": index_dataset_id,
                        "manifest_sha256": manifest_hash,
                        "model": getattr(embedder, "model", runtime.embedding_model),
                        "dimensions": embedder.dimensions,
                        "trial": trial,
                        "engine": engine.id,
                        "status": "ok",
                        "error": "",
                        "chunk_count": count,
                        "record_count": record_count,
                        "embedding_provider": provider,
                        "embedding_batch_size": batch_size,
                        "chunk_batch_size": chunk_size,
                        "embedding_requests": requests,
                        "extraction_s": extraction_s,
                        "embedding_s": embedding_s,
                        "embedding_chunks_per_s": count / embedding_s if embedding_s else 0,
                    }
                    completed = 0

                    def ingestion_batch(
                        batch_id: int, group: list[Any], duration: float, status: str,
                        error: str,
                    ) -> None:
                        nonlocal completed
                        if status == "ok":
                            completed += len(group)
                        ing_writer.writerow({
                            "run_id": run_id, "trial": trial, "engine": engine.id,
                            "batch_id": batch_id, "chunk_count": len(group),
                            "cumulative_chunks": completed, "duration_s": duration,
                            "status": status, "error": error,
                        })
                        ing_stream.flush()
                        for chunk in group:
                            chunk_rows[chunk.id].update({
                                f"{engine.id}_batch_id": batch_id,
                                f"{engine.id}_duration_s": duration,
                            })

                    timings: dict[str, float] = {}
                    try:
                        await index_engine(
                            engine, index_dataset_id, embedder.dimensions, chunks, vectors,
                            ingestion_batch, timings, count,
                        )
                    except Exception as error:
                        failed_engines = True
                        row.update(status="failed", error=str(error), **timings)
                    else:
                        engine_s = sum(timings.values())
                        row.update(timings)
                        row.update(
                            engine_total_s=engine_s,
                            end_to_end_s=extraction_s + embedding_s + engine_s,
                            engine_chunks_per_s=count / engine_s if engine_s else 0,
                        )
                    rows.append(row)
                    print(f"trial {trial} {engine.id}: {row['status']} ({count} chunks)")
                trial_wall_s = seconds(trial_start)
                for row in rows:
                    row["trial_wall_s"] = trial_wall_s
                    writer.writerow(row)  # pyright: ignore[reportArgumentType]
                for chunk_row in chunk_rows.values():
                    chunk_writer.writerow(chunk_row)  # pyright: ignore[reportArgumentType]
                write_records(record_writer, chunk_rows)
                stream.flush()
                chunk_stream.flush()
                record_stream.flush()
    metadata["finished_at_utc"] = datetime.now(UTC).isoformat()
    metadata["status"] = "completed_with_errors" if failed_engines else "completed"
    run_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="zbdoc")
    parser.add_argument(
        "--index-dataset",
        help="Publish to a separate index ID without changing chunk IDs or source dataset",
    )
    parser.add_argument("--engines", nargs="+", choices=("solr", "meilisearch"),
                        default=["solr", "meilisearch"])
    parser.add_argument("--output", default="data/indexing-study.csv")
    parser.add_argument("--embedding-provider", choices=("ollama", "openai"), default="ollama")
    parser.add_argument(
        "--embedding-url", help="Native /api/embed URL, or OpenAI-compatible base URL"
    )
    parser.add_argument("--embedding-model", help="Override the configured model")
    parser.add_argument("--embedding-dimensions", type=int, help="Expected vector size")
    parser.add_argument("--chunk-batch-size", type=int, help="Index submission size (max 100)")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-chunks", type=int, help="Pilot run on the first N chunks")
    parser.add_argument("--batch-embedding", action="store_true",
                        help="Use config/arena.yaml's embedding_batch_size (default: 64)")
    parser.add_argument("--embedding-batch-size", type=int,
                        help="Override embedding batch size (implies batch embedding when > 1)")
    parser.add_argument("--replace-existing", action="store_true",
                        help="Allow publishing over existing indexes (required for repeat trials)")
    args = parser.parse_args()
    if args.trials < 1 or (args.max_chunks is not None and args.max_chunks < 1):
        parser.error("--trials and --max-chunks must be positive")
    if args.embedding_batch_size is not None and args.embedding_batch_size < 1:
        parser.error("--embedding-batch-size must be positive")
    if args.chunk_batch_size is not None and not 1 <= args.chunk_batch_size <= 100:
        parser.error("--chunk-batch-size must be between 1 and 100")
    if args.embedding_dimensions is not None and args.embedding_dimensions < 1:
        parser.error("--embedding-dimensions must be positive")
    if args.trials > 1 and not args.replace_existing:
        parser.error("repeat trials require --replace-existing")
    if len(set(args.engines)) != len(args.engines):
        parser.error("each engine may be selected only once")
    return args


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
                    params={"wt": "json"}, timeout=5.0
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
    runtime = get_runtime_settings()
    config = get_arena_config()
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
    catalog = DatasetCatalog(chunking=config.chunking)
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        embedder: Embedder = (
            OllamaEmbedder(client, url, model, dimensions)
            if args.embedding_provider == "ollama"
            else OpenAICompatibleEmbedder(client, url, model, dimensions, runtime.embedding_api_key)
        )
        engines = initialize_engines(client, runtime)
        versions = await engine_versions(client, runtime, args.engines)
        await study(
            args, catalog, embedder, [engines[name] for name in args.engines],
            runtime, config, versions,
        )


if __name__ == "__main__":
    asyncio.run(main())
