"""Plot vector-search study CSVs.

Run from the repository root with --input data/vector-search-study.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

ENGINE_LABELS = {"solr": "Apache Solr", "meilisearch": "Meilisearch"}
ENGINE_COLORS = {"solr": "#60A5FA", "meilisearch": "#2DD4BF"}

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


def read_queries(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(QUERY_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"{path}: row has more values than columns")
    good = [row for row in rows if row["status"] == "ok"]
    if not good:
        raise ValueError(f"{path}: no successful query rows")
    quality_fields = ("hits_at_k", "recall_at_k", "reciprocal_rank", "ndcg_at_k")
    judged_rows = [all(row[field] for field in quality_fields) for row in good]
    if any(judged_rows) and not all(judged_rows):
        raise ValueError(f"{path}: successful rows mix judged and unjudged queries")
    for row in good:
        if not row["engine"] or not row["query_id"]:
            raise ValueError(f"{path}: successful rows require engine and query_id")
        fields = ("latency_ms", *quality_fields) if judged_rows[0] else ("latency_ms",)
        for field in fields:
            try:
                value = float(row[field])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}: invalid {field}: {row[field]!r}") from error
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{path}: invalid {field}: {row[field]!r}")
    return good


def has_quality_metrics(rows: list[dict[str, str]]) -> bool:
    return bool(rows[0]["recall_at_k"])


def plot(input_path: Path, output_dir: Path) -> list[Path]:
    rows = read_queries(input_path)
    judged = has_quality_metrics(rows)
    engines = sorted({row["engine"] for row in rows})
    latency: dict[str, list[float]] = defaultdict(list)
    quality: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        engine = row["engine"]
        latency[engine].append(float(row["latency_ms"]))
        if judged:
            for metric in ("recall_at_k", "reciprocal_rank", "ndcg_at_k"):
                quality[engine][metric].append(float(row[metric]))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        path = output_dir / name
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    fig, ax = plt.subplots(figsize=(10, 4.7))
    boxplot = ax.boxplot(
        [latency[engine] for engine in engines],
        tick_labels=[ENGINE_LABELS.get(engine, engine) for engine in engines],
        showfliers=True,
        patch_artist=True,
    )
    for engine_box, engine in zip(boxplot["boxes"], engines, strict=True):
        engine_box.set_facecolor(ENGINE_COLORS.get(engine, "#94A3B8"))
        engine_box.set_alpha(0.65)
    ax.set(
        xlabel="Engine",
        ylabel="Client-observed vector-search latency (ms)",
        title="Vector query latency distribution",
    )
    ax.grid(axis="y", alpha=0.25)
    save(fig, "vector-search-latency.png")

    if not judged:
        query_ids = sorted({row["query_id"] for row in rows})
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            grouped[row["query_id"], row["engine"]].append(float(row["latency_ms"]))
        positions = list(range(len(query_ids)))
        width = 0.8 / len(engines)
        fig, ax = plt.subplots(figsize=(max(8, len(query_ids) * 1.8), 4.7))
        for engine_index, engine in enumerate(engines):
            offsets = [
                position - 0.4 + width / 2 + engine_index * width for position in positions
            ]
            values = [mean(grouped[query_id, engine]) for query_id in query_ids]
            bars = ax.bar(
                offsets,
                values,
                width=width,
                label=ENGINE_LABELS.get(engine, engine),
                color=ENGINE_COLORS.get(engine, "#94A3B8"),
            )
            for bar, value in zip(bars, values, strict=True):
                ax.annotate(
                    f"{value:.2f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_xticks(positions, query_ids, rotation=20, ha="right")
        ax.set(
            xlabel="Query ID",
            ylabel="Mean client-observed latency (ms)",
            title="Vector query latency by query (unjudged)",
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        save(fig, "vector-search-latency-by-query.png")
        return written

    metric_names = ("recall_at_k", "reciprocal_rank", "ndcg_at_k")
    labels = ("Recall@k", "MRR", "nDCG@k")
    positions = list(range(len(metric_names)))
    width = 0.8 / len(engines)
    fig, ax = plt.subplots(figsize=(max(7, len(engines) * 2), 5))
    for engine_index, engine in enumerate(engines):
        offsets = [
            position - 0.4 + width / 2 + engine_index * width for position in positions
        ]
        values = [mean(quality[engine][metric]) for metric in metric_names]
        bars = ax.bar(
            offsets,
            values,
            width=width,
            label=ENGINE_LABELS.get(engine, engine),
            color=ENGINE_COLORS.get(engine, "#94A3B8"),
        )
        for bar, value in zip(bars, values, strict=True):
            ax.annotate(
                f"{value:.2f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 1.12)
    ax.set(xlabel="Retrieval quality metric", ylabel="Mean score",
           title="Judged vector retrieval quality")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save(fig, "vector-search-quality.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    for engine in engines:
        mean_latency = mean(latency[engine])
        mean_recall = mean(quality[engine]["recall_at_k"])
        ax.scatter(
            mean_latency,
            mean_recall,
            s=110,
            label=ENGINE_LABELS.get(engine, engine),
            color=ENGINE_COLORS.get(engine, "#94A3B8"),
        )
        ax.annotate(
            ENGINE_LABELS.get(engine, engine),
            (mean_latency, mean_recall),
            xytext=(9, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.85},
        )
    ax.set(xlabel="Mean client-observed latency (ms)", ylabel="Mean Recall@k",
           ylim=(0, 1.12), title="Latency–recall trade-off")
    ax.margins(x=0.2)
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, "vector-search-tradeoff.png")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/vector-search-study.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/vector-search-plots"))
    args = parser.parse_args()
    try:
        plot(args.input, args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(2, f"plot: {error}\n")


if __name__ == "__main__":
    main()
