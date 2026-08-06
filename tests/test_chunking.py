from itertools import pairwise

from datasets.catalog import DatasetCatalog
from datasets.chunking import chunk_document
from datasets.models import SourceDocument


def test_demo_dataset_produces_stable_chunks() -> None:
    catalog = DatasetCatalog()

    first = catalog.chunks("demo")
    catalog.clear_cache("demo")
    second = catalog.chunks("demo")

    assert first
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert all(chunk.dataset_id == "demo" for chunk in first)
    assert all(chunk.text for chunk in first)


def test_catalog_exposes_supported_demo_sources() -> None:
    catalog = DatasetCatalog()

    chunks = catalog.chunks("demo")

    assert {chunk.source_path for chunk in chunks} == {
        "documents/metformin.md",
        "documents/trial.jsonl",
    }


def test_short_paragraphs_do_not_cause_one_character_chunk_progress() -> None:
    text = "\n\n".join(f"Paragraph {index} has useful content." for index in range(50))
    document = SourceDocument(
        id="paragraphs",
        title="Paragraphs",
        text=text,
        source_path="paragraphs.txt",
        record_index=0,
    )

    chunks = chunk_document(
        document,
        dataset_id="test",
        max_characters=200,
        overlap_characters=30,
    )

    assert len(chunks) < 15
    assert all(len(chunk.text) <= 200 for chunk in chunks)
    assert all(
        later.start_char - earlier.start_char >= 140
        for earlier, later in pairwise(chunks)
    )
