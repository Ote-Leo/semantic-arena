from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "performance_plot", Path(__file__).resolve().parents[1] / "performance-study" / "plot.py"
)
assert spec is not None and spec.loader is not None
plot_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_module)


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sample(tmp_path: Path, engines: tuple[str, ...] = ("solr", "meilisearch")) -> Path:
    path = tmp_path / "indexing-study.csv"
    summary = [
        dict(run_id="run-a", trial="1", engine=engine, status="ok", prepare_s="0.2",
             ingestion_s="1.2", finalization_s="0.3", engine_total_s="1.7",
             embedding_s="0.5", extraction_s="0.1", chunk_count="3",
             engine_chunks_per_s="1.76")
        for engine in engines
    ]
    write_csv(path, plot_module.SUMMARY, summary)
    embedding = [
        dict(run_id="run-a", trial="1", batch_id=str(i), chunk_count=str(size),
             duration_s=str(duration), status="ok", error="")
        for i, (size, duration) in enumerate(((2, 0.3), (1, 0.2)))
    ]
    write_csv(path.with_name(path.stem + "-embedding-batches.csv"), plot_module.EMBEDDING,
              embedding)
    ingestion = [
        dict(run_id="run-a", trial="1", engine=engine, batch_id=str(i),
             chunk_count=str(size), cumulative_chunks=str(total), duration_s=str(duration),
             status="ok", error="")
        for engine in engines
        for i, (size, total, duration) in enumerate(((2, 2, 0.7), (1, 3, 0.5)))
    ]
    write_csv(path.with_name(path.stem + "-ingestion-batches.csv"), plot_module.INGESTION,
              ingestion)
    chunks = [
        dict(run_id="run-a", trial="1", chunk_id=str(i), source_id="source",
             source_path="source.parquet", record_index=str(i), chunk_index="0",
             text_chars="10", text_bytes=str(10 + i), embedding_batch_id=str(i // 2),
             embedding_duration_s="0.3" if i < 2 else "0.2",
             solr_batch_id=str(i // 2), solr_duration_s="0.7" if i < 2 else "0.5",
             meilisearch_batch_id=str(i // 2),
             meilisearch_duration_s="0.7" if i < 2 else "0.5")
        for i in range(3)
    ]
    write_csv(path.with_name(path.stem + "-chunks.csv"), plot_module.CHUNKS, chunks)
    records = [
        dict(run_id="run-a", trial="1", source_id="source", source_path="source.parquet",
             record_index="0", chunk_count="3", text_bytes="33", embedding_batch_ids="0;1",
             embedding_shared_batch_s="0.5", solr_batch_ids="0;1",
             solr_shared_batch_s="1.2", meilisearch_batch_ids="0;1",
             meilisearch_shared_batch_s="1.2")
    ]
    write_csv(path.with_name(path.stem + "-records.csv"), plot_module.RECORDS, records)
    return path


@pytest.mark.parametrize("engines", [("solr",), ("solr", "meilisearch")])
def test_plot_writes_pngs_for_one_or_two_engines(tmp_path: Path, engines: tuple[str, ...]) -> None:
    pytest.importorskip("matplotlib")
    path = sample(tmp_path, engines)
    output = tmp_path / "plots"
    written = plot_module.plot(path, output)
    assert {p.name for p in written} == {
        "engine-stages.png", "batch-latency.png", "throughput.png",
        "ingestion-progress.png", "chunk-sizes.png", "record-batch-involvement.png",
    }
    assert all(
        p.parent == output and p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        for p in written
    )


def test_batch_latency_counts_batches_not_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes

    path = sample(tmp_path, ("solr",))
    original = Axes.boxplot
    observed: list[list[float]] = []

    def capture(self: Axes, values, **kwargs):
        observed.extend([list(group) for group in values])
        return original(self, values, **kwargs)

    monkeypatch.setattr(Axes, "boxplot", capture)
    plot_module.plot(path, tmp_path / "plots")
    assert observed == [[0.3, 0.2], [0.7, 0.5]]


def test_failed_summary_rows_are_ignored(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    path = sample(tmp_path, ("solr",))
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=plot_module.SUMMARY)
        writer.writerow(dict(run_id="run-a", trial="1", engine="meilisearch", status="failed"))
    assert len(plot_module.plot(path, tmp_path / "plots")) == 6


@pytest.mark.parametrize("problem,match", [
    ("missing_column", "missing columns"),
    ("no_success", "no successful summary rows"),
    ("bad_duration", "invalid prepare_s"),
    ("missing_sidecar", "embedding-batches.csv"),
    ("no_batches", "No successful embedding and ingestion batches"),
])
def test_invalid_input(tmp_path: Path, problem: str, match: str) -> None:
    path = sample(tmp_path, ("solr",))
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if problem == "missing_column":
        write_csv(path, plot_module.SUMMARY[:-1], [
            {key: value for key, value in row.items() if key != "engine_chunks_per_s"}
            for row in rows
        ])
    elif problem == "no_success":
        rows[0]["status"] = "failed"
        write_csv(path, plot_module.SUMMARY, rows)
    elif problem == "bad_duration":
        rows[0]["prepare_s"] = "nan"
        write_csv(path, plot_module.SUMMARY, rows)
    elif problem == "missing_sidecar":
        path.with_name(path.stem + "-embedding-batches.csv").unlink()
    else:
        detail = path.with_name(path.stem + "-embedding-batches.csv")
        write_csv(detail, plot_module.EMBEDDING, [])
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        plot_module.plot(path, tmp_path / "plots")
    assert not (tmp_path / "plots").exists()
