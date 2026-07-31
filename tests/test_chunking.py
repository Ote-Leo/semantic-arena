from datasets.catalog import DatasetCatalog


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
