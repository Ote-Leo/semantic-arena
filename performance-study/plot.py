"""Plot indexing study CSVs without changing the source data.

Run from the repository root with --input data/indexing-study.csv and
--output-dir data/plots. Requires matplotlib in the active environment.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

SUMMARY = ("run_id", "trial", "engine", "status", "prepare_s", "ingestion_s",
           "finalization_s", "engine_total_s", "embedding_s", "extraction_s",
           "chunk_count", "engine_chunks_per_s")
EMBEDDING = ("run_id", "trial", "batch_id", "chunk_count", "duration_s", "status", "error")
INGESTION = ("run_id", "trial", "engine", "batch_id", "chunk_count",
             "cumulative_chunks", "duration_s", "status", "error")
CHUNKS = ("run_id", "trial", "chunk_id", "source_id", "source_path", "record_index",
          "chunk_index", "text_chars", "text_bytes", "embedding_batch_id",
          "embedding_duration_s", "solr_batch_id", "solr_duration_s",
          "meilisearch_batch_id", "meilisearch_duration_s")
RECORDS = ("run_id", "trial", "source_id", "source_path", "record_index", "chunk_count",
           "text_bytes", "embedding_batch_ids", "embedding_shared_batch_s",
           "solr_batch_ids", "solr_shared_batch_s", "meilisearch_batch_ids",
           "meilisearch_shared_batch_s")


def read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"{path}: row has more values than columns")
    return rows


def number(row: dict[str, str], column: str, path: Path, *, integer: bool = False) -> float:
    value = row[column]
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path}: invalid {column}: {value!r}") from exc
    if not math.isfinite(result) or result < 0 or (integer and not result.is_integer()):
        raise ValueError(f"{path}: invalid {column}: {value!r}")
    return result


def plot(input_path: Path, output_dir: Path) -> list[Path]:
    """Write indexing charts for successful runs; raise ValueError for unusable CSVs."""
    summary = read_csv(input_path, SUMMARY)
    good = [row for row in summary if row["status"] == "ok"]
    if not good:
        raise ValueError(f"{input_path}: no successful summary rows")
    engines = sorted({row["engine"] for row in good})
    if any(not row["run_id"] or not row["trial"] or not row["engine"] for row in good):
        raise ValueError(f"{input_path}: successful rows require run_id, trial and engine")
    for row in good:
        for field in ("prepare_s", "ingestion_s", "finalization_s", "engine_total_s",
                      "embedding_s", "extraction_s", "chunk_count", "engine_chunks_per_s"):
            number(row, field, input_path, integer=field == "chunk_count")
    successful = {(row["run_id"], row["trial"], row["engine"]) for row in good}
    trials = {(row["run_id"], row["trial"]) for row in good}

    def sidecar(suffix: str, fields: tuple[str, ...]) -> tuple[Path, list[dict[str, str]]]:
        path = input_path.with_name(input_path.stem + suffix + ".csv")
        return path, read_csv(path, fields)

    embed_path, embed_rows = sidecar("-embedding-batches", EMBEDDING)
    ingest_path, ingest_rows = sidecar("-ingestion-batches", INGESTION)
    chunks_path, chunk_rows = sidecar("-chunks", CHUNKS)
    records_path, record_rows = sidecar("-records", RECORDS)
    embedding = [row for row in embed_rows
                 if row["status"] == "ok" and (row["run_id"], row["trial"]) in trials]
    ingestion = [row for row in ingest_rows if row["status"] == "ok" and
                 (row["run_id"], row["trial"], row["engine"]) in successful]
    chunks = [row for row in chunk_rows if (row["run_id"], row["trial"]) in trials]
    records = [row for row in record_rows if (row["run_id"], row["trial"]) in trials]
    for row in embedding:
        number(row, "duration_s", embed_path)
        number(row, "chunk_count", embed_path, integer=True)
    for row in ingestion:
        for field in ("duration_s", "chunk_count", "cumulative_chunks", "batch_id"):
            number(row, field, ingest_path, integer=field != "duration_s")
    for row in chunks:
        number(row, "text_bytes", chunks_path, integer=True)
    for row in records:
        number(row, "text_bytes", records_path, integer=True)
        number(row, "chunk_count", records_path, integer=True)
    if not embedding or not ingestion:
        raise ValueError("No successful embedding and ingestion batches for successful runs")

    # Import lazily: importing this module must not select a backend or create figures.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        path = output_dir / name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    fig, ax = plt.subplots(figsize=(max(7, len(engines) * 2), 5))
    bottoms = [0.0] * len(engines)
    for field, label in (("prepare_s", "Prepare"), ("ingestion_s", "Ingest"),
                         ("finalization_s", "Finalize")):
        values = [mean(number(row, field, input_path) for row in good if row["engine"] == e)
                  for e in engines]
        ax.bar(engines, values, bottom=bottoms, label=label)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    ax.set(xlabel="Engine", ylabel="Mean stage duration per trial (s)",
           title="Engine indexing stages")
    ax.legend()
    save(fig, "engine-stages.png")

    groups = [([number(row, "duration_s", embed_path) for row in embedding], "Embedding")]
    groups.extend(([number(row, "duration_s", ingest_path) for row in ingestion
                    if row["engine"] == engine], f"{engine} ingestion") for engine in engines)
    groups = [(values, label) for values, label in groups if values]
    fig, ax = plt.subplots(figsize=(max(7, len(groups) * 2), 5))
    ax.boxplot([values for values, _ in groups], tick_labels=[label for _, label in groups])
    ax.set(xlabel="Batch type", ylabel="Batch wall-clock duration (s)",
           title="Batch latency distribution (one observation per batch)")
    save(fig, "batch-latency.png")

    fig, ax = plt.subplots(figsize=(max(7, len(engines) * 2), 5))
    ax.bar(engines, [mean(number(row, "engine_chunks_per_s", input_path)
                          for row in good if row["engine"] == engine) for engine in engines])
    ax.set(xlabel="Engine", ylabel="Mean indexing throughput (chunks/s)",
           title="Engine throughput per successful trial")
    save(fig, "throughput.png")

    series: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in ingestion:
        series[row["engine"], row["run_id"], row["trial"]].append(row)
    fig, ax = plt.subplots(figsize=(8, 5))
    for (engine, run_id, trial), rows in sorted(series.items()):
        rows.sort(key=lambda row: number(row, "batch_id", ingest_path, integer=True))
        elapsed = 0.0
        times = [0.0]
        counts = [0.0]
        for row in rows:
            elapsed += number(row, "duration_s", ingest_path)
            times.append(elapsed)
            counts.append(number(row, "cumulative_chunks", ingest_path, integer=True))
        ax.plot(times, counts, marker=".", label=f"{engine} · {run_id} trial {trial}")
    ax.set(xlabel="Cumulative ingestion batch duration (s)",
           ylabel="Cumulative chunks ingested",
           title="Ingestion progress (excludes prepare/finalize)")
    ax.legend(fontsize="small")
    save(fig, "ingestion-progress.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    if chunks:
        ax.hist([number(row, "text_bytes", chunks_path, integer=True) for row in chunks], bins=20)
    ax.set(xlabel="Chunk text size (bytes)", ylabel="Chunk count", title="Chunk size distribution")
    save(fig, "chunk-sizes.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for stage in ("embedding", *engines):
        field = f"{stage}_shared_batch_s"
        eligible = [row for row in records if row[field] and (
            stage == "embedding" or (row["run_id"], row["trial"], stage) in successful
        )]
        if eligible:
            ax.scatter(
                [number(row, "text_bytes", records_path, integer=True) for row in eligible],
                [number(row, field, records_path) for row in eligible],
                alpha=0.5, label=stage,
            )
    ax.set(xlabel="Source record chunk text (bytes)",
           ylabel="Sum of shared batch durations (s)",
           title="Record batch involvement (not exclusive record latency)")
    ax.legend(fontsize="small")
    save(fig, "record-batch-involvement.png")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/indexing-study.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/plots"))
    args = parser.parse_args()
    try:
        plot(args.input, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"plot: {exc}\n")


if __name__ == "__main__":
    main()
