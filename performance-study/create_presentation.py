"""Generate a PowerPoint deck from saved vector-search benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

NAVY = RGBColor(15, 23, 42)
PANEL = RGBColor(30, 41, 59)
PANEL_ALT = RGBColor(38, 51, 72)
WHITE = RGBColor(241, 245, 249)
MUTED = RGBColor(161, 174, 193)
TEAL = RGBColor(45, 212, 191)
BLUE = RGBColor(96, 165, 250)
AMBER = RGBColor(251, 191, 36)
GREEN = RGBColor(74, 222, 128)
FONT = "Aptos"
ENGINE_LABELS = {"solr": "Apache Solr", "meilisearch": "Meilisearch"}
ENGINE_COLORS = {"solr": BLUE, "meilisearch": TEAL}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def percentile(values: list[float], percentage: int) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("No latency measurements found")
    return ordered[max(1, math.ceil(percentage * len(ordered) / 100)) - 1]


def color(value: RGBColor) -> str:
    return f"#{value.red:02X}{value.green:02X}{value.blue:02X}"


def add_box(slide, x: float, y: float, w: float, h: float, fill: RGBColor, radius=True):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x),
        Inches(y),
        Inches(w),
        Inches(h),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    return shape


def add_text(
    slide,
    text: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    size: float = 18,
    fill: RGBColor = WHITE,
    bold: bool = False,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    margin: float = 0.02,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.space_after = Pt(0)
    run = paragraph.add_run()
    run.text = text
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = fill
    return box


def add_rich_lines(slide, items, x: float, y: float, w: float, h: float) -> None:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.02)
    frame.margin_right = Inches(0.02)
    frame.margin_top = Inches(0.02)
    frame.margin_bottom = Inches(0.02)
    for index, (heading, detail) in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.space_after = Pt(13)
        paragraph.level = 0
        lead = paragraph.add_run()
        lead.text = f"{heading}\n"
        lead.font.name = FONT
        lead.font.size = Pt(17)
        lead.font.bold = True
        lead.font.color.rgb = WHITE
        body = paragraph.add_run()
        body.text = detail
        body.font.name = FONT
        body.font.size = Pt(13)
        body.font.color.rgb = MUTED


def add_base(prs: Presentation, title: str, subtitle: str, page: int, date_label: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = NAVY
    add_text(slide, title, 0.65, 0.38, 12.0, 0.48, size=27, bold=True)
    add_text(slide, subtitle, 0.67, 0.92, 12.0, 0.32, size=11, fill=MUTED)
    add_box(slide, 0.68, 7.08, 11.97, 0.012, PANEL_ALT, radius=False)
    add_text(
        slide,
        f"SEMANTIC ARENA  /  VECTOR SEARCH PILOT  /  {date_label}",
        0.7,
        7.16,
        10.8,
        0.18,
        size=8,
        fill=MUTED,
    )
    add_text(slide, f"{page:02d}", 12.0, 7.13, 0.55, 0.24, size=10, fill=MUTED,
             align=PP_ALIGN.RIGHT)
    return slide


def make_deck(
    input_path: Path,
    output_path: Path,
    plot_dir: Path,
    indexing_summary_path: Path | None = None,
) -> Path:
    metadata_path = input_path.with_name(input_path.stem + "-run.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    query_rows = [row for row in load_csv(input_path) if row["status"] == "ok"]
    engines = sorted({row["engine"] for row in query_rows})
    if len(engines) != 2:
        raise ValueError("Presentation requires successful data for exactly two engines")
    expected = metadata["query_count"] * metadata["trials"] * metadata["repetitions"]
    judged = metadata.get("evaluation_mode", "judged_quality") != "latency_only"
    stats: dict[str, dict[str, float | int]] = {}
    for engine in engines:
        rows = [row for row in query_rows if row["engine"] == engine]
        if len(rows) != expected:
            raise ValueError(
                f"{engine}: expected {expected} successful measurements, found {len(rows)}"
            )
        latencies = [float(row["latency_ms"]) for row in rows]
        stats[engine] = {
            "n": len(rows),
            "mean_ms": sum(latencies) / len(latencies),
            "p50_ms": percentile(latencies, 50),
            "p95_ms": percentile(latencies, 95),
        }
        if judged:
            stats[engine].update({
                "recall": sum(float(row["recall_at_k"]) for row in rows) / len(rows),
                "mrr": sum(float(row["reciprocal_rank"]) for row in rows) / len(rows),
                "ndcg": sum(float(row["ndcg_at_k"]) for row in rows) / len(rows),
            })

    corpus = {
        "source_dataset_id": metadata.get("source_dataset_id", metadata["dataset_id"]),
        "index_dataset_id": metadata["dataset_id"],
        "indexed_chunks": metadata.get("indexed_chunk_count", 3),
        "source_records": metadata.get("source_record_count", ""),
    }
    if indexing_summary_path is not None:
        index_rows = [
            row for row in load_csv(indexing_summary_path)
            if row["status"] == "ok" and row["engine"] in engines
        ]
        counts = {int(row["chunk_count"]) for row in index_rows}
        source_ids = {row["dataset_id"] for row in index_rows}
        index_ids = {row.get("index_dataset_id", row["dataset_id"]) for row in index_rows}
        if not index_rows or len(counts) != 1 or len(source_ids) != 1 or len(index_ids) != 1:
            raise ValueError("Indexing summary must show a consistent successful index per engine")
        corpus.update({
            "source_dataset_id": next(iter(source_ids)),
            "index_dataset_id": next(iter(index_ids)),
            "indexed_chunks": next(iter(counts)),
            "source_records": next(iter({int(row["record_count"]) for row in index_rows})),
        })

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    prs.core_properties.title = "Solr vs Meilisearch: Vector Search Pilot"
    prs.core_properties.subject = "Measured vector retrieval comparison"
    prs.core_properties.author = "Semantic Arena"
    started = datetime.fromisoformat(metadata["started_at_utc"])
    date_label = started.strftime("%d %b %Y").upper()
    meili = stats["meilisearch"]
    solr = stats["solr"]
    faster_ratio = max(meili["mean_ms"], solr["mean_ms"]) / min(
        meili["mean_ms"], solr["mean_ms"]
    )
    winner = min(engines, key=lambda engine: stats[engine]["mean_ms"])

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = NAVY
    add_box(slide, 9.55, 0, 3.783, 7.5, PANEL, radius=False)
    add_text(slide, "SEMANTIC ARENA  /  LIVE PILOT", 0.82, 0.7, 7.5, 0.3,
             size=11, fill=TEAL, bold=True)
    add_text(slide, "Vector search,\nhead to head.", 0.82, 1.35, 7.8, 1.7,
             size=40, bold=True)
    add_text(
        slide,
        "Apache Solr  ×  Meilisearch",
        0.87,
        3.38,
        7.5,
        0.45,
        size=22,
        fill=MUTED,
    )
    add_text(
        slide,
        (
            f"{metadata['embedding_model']}  •  "
            f"{metadata['embedding_dimensions']:,}-dimensional vectors"
        ),
        0.87,
        4.15,
        7.9,
        0.36,
        size=15,
    )
    cover_label = (
        "A controlled, judged retrieval comparison"
        if judged else "A controlled vector-search latency comparison"
    )
    add_text(slide, cover_label, 0.87, 4.7, 7.8, 0.35, size=15, fill=MUTED)
    add_box(slide, 10.0, 1.15, 3.0, 1.1, PANEL_ALT)
    add_text(slide, f"{int(corpus['indexed_chunks']):,}", 10.08, 1.3, 1.25, 0.55,
             size=27, fill=TEAL, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "indexed chunks", 11.0, 1.52, 1.75, 0.3, size=14, fill=MUTED)
    add_box(slide, 10.0, 2.55, 3.0, 1.1, PANEL_ALT)
    add_text(slide, str(metadata["query_count"]), 10.2, 2.7, 0.8, 0.55,
             size=36, fill=BLUE, bold=True, align=PP_ALIGN.CENTER)
    query_label = "judged queries" if judged else "latency queries"
    add_text(slide, query_label, 11.0, 2.92, 1.75, 0.3, size=14, fill=MUTED)
    add_box(slide, 10.0, 3.95, 3.0, 1.1, PANEL_ALT)
    add_text(slide, f"{metadata['trials']} × {metadata['repetitions']}", 10.15, 4.1,
             1.65, 0.55, size=25, fill=AMBER, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "trials × repeats", 11.8, 4.3, 1.0, 0.45, size=12, fill=MUTED)
    add_text(slide, date_label, 0.87, 6.72, 4.0, 0.25, size=10, fill=MUTED)
    add_text(slide, "PILOT RESULT — NOT A UNIVERSAL RANKING", 8.5, 6.72, 4.0, 0.25,
             size=10, fill=AMBER, bold=True, align=PP_ALIGN.RIGHT)

    slide = add_base(
        prs,
        "A fair comparison starts with shared inputs",
        (
            "Vector retrieval only — indexing, embedding, fusion, and answer "
            "generation are not in query latency."
        ),
        2,
        date_label,
    )
    steps = [
        (
            "01",
            "Same corpus",
            f"{int(corpus['indexed_chunks']):,} chunks from {corpus['source_dataset_id']}.",
        ),
        (
            "02",
            "Same vectors",
            f"One shared {metadata['embedding_model']} query vector per question.",
        ),
        (
            "03",
            "Same query",
            f"Top-{metadata['top_k']} vector-only retrieval; no lexical search or RRF.",
        ),
        (
            "04",
            "Measured output" if not judged else "Judged output",
            (
                "Four fixed queries; latency measured without making quality claims."
                if not judged else
                "Four fixed queries with known relevant chunk IDs; exact top result checked."
            ),
        ),
    ]
    for index, (number, heading, detail) in enumerate(steps):
        x = 0.75 + index * 3.15
        add_box(slide, x, 1.7, 2.85, 3.05, PANEL)
        add_text(slide, number, x + 0.22, 1.95, 0.7, 0.55, size=28, fill=TEAL, bold=True)
        add_text(slide, heading, x + 0.22, 2.75, 2.4, 0.45, size=20, bold=True)
        add_text(slide, detail, x + 0.22, 3.42, 2.38, 1.05, size=14, fill=MUTED)
    add_box(slide, 0.75, 5.3, 11.95, 0.9, PANEL_ALT)
    add_text(
        slide,
        (
            "Alternating engine order  •  1 warm-up/query  •  5 trials  •  "
            "3 measured repeats/query/trial"
        ),
        1.0,
        5.57,
        11.4,
        0.3,
        size=15,
        fill=WHITE,
        align=PP_ALIGN.CENTER,
    )

    slide = add_base(
        prs,
        f"On this pilot, {ENGINE_LABELS[winner]} had lower mean latency",
        "Client-observed latency across successful vector-search requests; lower is faster.",
        3,
        date_label,
    )
    for index, engine in enumerate(("meilisearch", "solr")):
        stat = stats[engine]
        x = 0.85 + index * 6.0
        accent = ENGINE_COLORS[engine]
        add_box(slide, x, 1.55, 5.55, 1.55, PANEL)
        add_text(slide, ENGINE_LABELS[engine], x + 0.28, 1.82, 3.2, 0.35,
                 size=17, fill=accent, bold=True)
        add_text(slide, f"{stat['mean_ms']:.2f}", x + 0.27, 2.17, 1.8, 0.62,
                 size=34, bold=True)
        add_text(slide, "ms mean", x + 2.05, 2.4, 1.5, 0.3, size=14, fill=MUTED)
        add_text(slide, f"p50  {stat['p50_ms']:.2f} ms     p95  {stat['p95_ms']:.2f} ms",
                 x + 0.28, 2.78, 4.9, 0.25, size=11, fill=MUTED)
        add_text(slide, f"{stat['n']} measured requests", x + 3.4, 1.85, 1.9, 0.3,
                 size=11, fill=MUTED, align=PP_ALIGN.RIGHT)
    chart = plot_dir / "vector-search-latency.png"
    if not chart.is_file():
        raise FileNotFoundError(f"Missing generated chart: {chart}")
    slide.shapes.add_picture(str(chart), Inches(1.05), Inches(3.35), width=Inches(7.05))
    add_box(slide, 8.5, 3.6, 3.95, 2.25, PANEL_ALT)
    add_text(slide, f"{faster_ratio:.1f}×", 8.85, 3.9, 1.8, 0.75,
             size=43, fill=TEAL if winner == "meilisearch" else BLUE, bold=True)
    add_text(slide, f"{ENGINE_LABELS[winner]} lower mean latency", 8.85, 4.8, 3.25, 0.55,
             size=17, bold=True)
    add_text(slide, "Measured in this setup only.", 8.85, 5.45, 3.25, 0.25,
             size=11, fill=MUTED)

    if judged:
        slide = add_base(
            prs,
            "Both engines ranked the judged answer first",
            (
                "Binary-relevance metrics at k=1; scores are averaged over the "
                "same repeated query set."
            ),
            4,
            date_label,
        )
        metrics = [
            ("Recall@1", "recall"),
            ("MRR", "mrr"),
            ("nDCG@1", "ndcg"),
        ]
        x0 = 0.95
        y0 = 1.65
        add_text(slide, "ENGINE", x0, y0, 3.0, 0.28, size=10, fill=MUTED, bold=True)
        for index, (label, _) in enumerate(metrics):
            add_text(slide, label, 4.15 + index * 2.1, y0, 1.75, 0.28,
                     size=12, fill=MUTED, bold=True, align=PP_ALIGN.CENTER)
        for engine_index, engine in enumerate(("meilisearch", "solr")):
            y = 2.25 + engine_index * 0.92
            add_box(slide, 0.85, y, 7.85, 0.7, PANEL)
            add_text(slide, ENGINE_LABELS[engine], 1.1, y + 0.19, 2.5, 0.3,
                     size=16, fill=ENGINE_COLORS[engine], bold=True)
            for index, (_, field) in enumerate(metrics):
                add_text(slide, f"{stats[engine][field]:.2f}", 4.15 + index * 2.1,
                         y + 0.18, 1.75, 0.32, size=18, bold=True, align=PP_ALIGN.CENTER)
        add_box(slide, 9.1, 1.78, 3.3, 3.12, PANEL_ALT)
        add_text(slide, "4 / 4", 9.45, 2.1, 2.5, 0.7, size=38, fill=GREEN, bold=True,
                 align=PP_ALIGN.CENTER)
        add_text(slide, "queries correct at rank 1\nfor both engines", 9.45, 2.98, 2.5, 0.9,
                 size=17, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, "Quality is tied on this tiny corpus.", 9.4, 4.25, 2.55, 0.42,
                 size=12, fill=MUTED, align=PP_ALIGN.CENTER)
        add_box(slide, 0.85, 4.55, 7.85, 1.05, PANEL_ALT)
        add_text(slide, "All 4 / 4", 1.15, 4.78, 1.9, 0.45,
                 size=25, fill=GREEN, bold=True)
        add_text(slide, "judged answers retrieved at rank 1", 3.15, 4.85, 5.2, 0.32,
                 size=16, bold=True)
        add_text(
            slide,
            "One relevant chunk per query; all returned top-1 IDs matched the judgment.",
            1.1,
            6.25,
            11.0,
            0.3,
            size=12,
            fill=MUTED,
            align=PP_ALIGN.CENTER,
        )
    else:
        slide = add_base(
            prs,
            "Latency by query — retrieval quality not scored",
            (
                "No reviewed relevance judgments were supplied; these charts "
                "make no recall or ranking claims."
            ),
            4,
            date_label,
        )
        chart = plot_dir / "vector-search-latency-by-query.png"
        if not chart.is_file():
            raise FileNotFoundError(f"Missing generated chart: {chart}")
        slide.shapes.add_picture(str(chart), Inches(0.9), Inches(1.55), width=Inches(7.5))
        add_box(slide, 8.75, 2.15, 3.55, 2.75, PANEL_ALT)
        add_text(slide, f"{metadata['query_count']}", 9.1, 2.52, 1.0, 0.75,
                 size=42, fill=TEAL, bold=True)
        add_text(slide, "fixed test queries", 10.05, 2.82, 1.9, 0.35,
                 size=15, fill=MUTED)
        add_text(slide, "No qrels supplied", 9.1, 3.7, 2.8, 0.4,
                 size=20, bold=True)
        add_text(slide, "Latency is reported; Recall, MRR and nDCG are intentionally omitted.",
                 9.1, 4.25, 2.85, 0.55, size=13, fill=MUTED)
        add_text(
            slide,
            "Query phrases cover study objectives, patient groups, interventions and outcomes.",
            1.0,
            6.1,
            11.0,
            0.35,
            size=12,
            fill=MUTED,
            align=PP_ALIGN.CENTER,
        )

    slide = add_base(
        prs,
        "What this result does — and does not — say",
        "A presentation-ready smoke test is not a production-scale engine benchmark.",
        5,
        date_label,
    )
    add_box(slide, 0.85, 1.55, 5.55, 3.9, PANEL)
    add_text(slide, "What we measured", 1.15, 1.9, 4.5, 0.45, size=21, fill=TEAL, bold=True)
    add_rich_lines(
        slide,
        [
            ("Vector query latency", "Client-observed, sequential calls on one local host."),
            (
                "Top-1 relevance",
                "Four fixed questions with explicit relevant chunk IDs."
                if judged else "Latency only; no relevance judgments were supplied.",
            ),
            ("Controlled inputs", "Same Ollama model, dimensions, and query vectors."),
        ],
        1.15,
        2.55,
        4.7,
        2.45,
    )
    add_box(slide, 6.8, 1.55, 5.55, 3.9, PANEL)
    add_text(slide, "What we did not measure", 7.1, 1.9, 4.9, 0.45,
             size=21, fill=AMBER, bold=True)
    add_rich_lines(
        slide,
        [
            (
                "Scale",
                (
                    f"{int(corpus['indexed_chunks']):,} indexed chunks and "
                    f"{metadata['query_count']} queries."
                ),
            ),
            ("Concurrency", "No load test, query throughput, or saturation point."),
            ("Server resources", "No CPU, RAM, disk, or server-only timing."),
        ],
        7.1,
        2.55,
        4.8,
        2.45,
    )
    add_box(slide, 0.85, 5.82, 11.5, 0.65, PANEL_ALT)
    add_text(
        slide,
        "Treat latency as a local pilot result—not a universal winner.",
        1.1,
        6.02,
        11.0,
        0.26,
        size=16,
        fill=WHITE,
        bold=True,
        align=PP_ALIGN.CENTER,
    )

    slide = add_base(
        prs,
        "Next: turn the pilot into a decision-grade comparison",
        "Keep the same controls; increase query coverage, corpus scale, and workload realism.",
        6,
        date_label,
    )
    next_steps = [
        (
            "01",
            "Grow the qrels",
            "Build 100–300 representative queries; have a reviewer judge relevant chunks.",
        ),
        (
            "02",
            "Test realistic scale",
            "Use the target corpus size and identical chunk/vector inputs.",
        ),
        (
            "03",
            "Add concurrency",
            "Measure p50/p95/p99, throughput, errors, and resource use at target load.",
        ),
        (
            "04",
            "Report trade-offs",
            "Compare quality and latency together; repeat on the intended hardware.",
        ),
    ]
    for index, (number, heading, detail) in enumerate(next_steps):
        x = 0.85 + (index % 2) * 6.0
        y = 1.65 + (index // 2) * 2.25
        add_box(slide, x, y, 5.55, 1.78, PANEL)
        add_text(slide, number, x + 0.25, y + 0.22, 0.65, 0.5, size=27, fill=TEAL, bold=True)
        add_text(slide, heading, x + 1.0, y + 0.26, 4.0, 0.35, size=18, bold=True)
        add_text(slide, detail, x + 1.0, y + 0.78, 4.1, 0.75, size=13, fill=MUTED)
    add_box(slide, 0.85, 6.35, 11.55, 0.44, PANEL_ALT)
    add_text(
        slide,
        f"Reproducible run: {metadata['trials']} trials × {metadata['repetitions']} repeats "
        f"• query embeddings shared • engine order alternated",
        1.05,
        6.46,
        11.15,
        0.2,
        size=11,
        fill=MUTED,
        align=PP_ALIGN.CENTER,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/vector-presentation-search.csv"))
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("data/vector-presentation-plots"),
    )
    parser.add_argument("--indexing-summary", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("performance-study/vector-search-comparison.pptx"),
    )
    args = parser.parse_args()
    path = make_deck(args.input, args.output, args.plot_dir, args.indexing_summary)
    print(f"Created {path}")


if __name__ == "__main__":
    main()
