from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datasets.catalog import DatasetCatalog
from settings import ChunkingConfig


def _write_dataset(root: Path, records: list[dict[str, object]]) -> None:
    dataset = root / "clinical"
    dataset.mkdir()
    (dataset / "manifest.yaml").write_text(
        """id: clinical
name: Clinical trials
sources:
  - glob: records.parquet
    format: parquet
    id_field: id
    json_field: source_data
    duplicate_id_policy: skip-identical
    title_fields:
      - drksId
      - trialDescriptions[].title
    text_fields:
      - trialStatus
      - trialDescriptions
      - recruitment.status
""",
        encoding="utf-8",
    )
    pq.write_table(pa.Table.from_pylist(records), dataset / "records.parquet")


def _payload(title: str = "Über trial") -> str:
    return json.dumps(
        {
            "drksId": "DRKS00000001",
            "trialStatus": "RECRUITING",
            "trialDescriptions": [
                {
                    "idLocale": {"locale": "de"},
                    "title": title,
                    "summary": "Decoded summary",
                }
            ],
            "recruitment": {"status": "RECRUITING"},
            "trialContacts": [{"email": "private@example.test"}],
        }
    )


def test_catalog_extracts_selected_json_fields_and_skips_identical_ids(tmp_path: Path) -> None:
    payload = _payload()
    _write_dataset(
        tmp_path,
        [
            {"id": "DRKS00000001", "source_data": payload},
            {"id": "DRKS00000001", "source_data": payload},
        ],
    )
    catalog = DatasetCatalog(
        root=tmp_path,
        chunking=ChunkingConfig(max_characters=10_000, overlap_characters=100),
    )

    chunks = catalog.chunks("clinical")

    assert len(chunks) == 1
    assert chunks[0].source_id == "DRKS00000001"
    assert chunks[0].title == "DRKS00000001 | Über trial"
    assert "Trial Status: RECRUITING" in chunks[0].text
    assert "Summary: Decoded summary" in chunks[0].text
    assert "private@example.test" not in chunks[0].text
    assert "\\u00dc" not in chunks[0].text


def test_catalog_rejects_conflicting_duplicate_ids(tmp_path: Path) -> None:
    _write_dataset(
        tmp_path,
        [
            {"id": "DRKS00000001", "source_data": _payload()},
            {"id": "DRKS00000001", "source_data": _payload("Different title")},
        ],
    )
    catalog = DatasetCatalog(root=tmp_path)

    with pytest.raises(ValueError, match="Conflicting duplicate source ID"):
        catalog.chunks("clinical")


def test_zbdoc_manifest_is_discoverable_without_loading_the_corpus() -> None:
    catalog = DatasetCatalog()

    manifest = catalog.manifest("zbdoc")

    assert manifest.name == "DRKS Clinical Trials"
    assert manifest.sources[0].json_field == "source_data"
    assert manifest.sources[0].duplicate_id_policy == "skip-identical"
    assert "trialDescriptions" in manifest.sources[0].text_fields
