from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from datasets.chunking import chunk_document
from datasets.models import Chunk, DatasetManifest, SourceDocument, SourceSpec, manifest_path
from settings import ROOT_DIR, ChunkingConfig


class DatasetNotFoundError(ValueError):
    pass


class DatasetCatalog:
    def __init__(
        self,
        root: Path | None = None,
        chunking: ChunkingConfig | None = None,
    ) -> None:
        self.root = root or ROOT_DIR / "sample-data"
        self.chunking = chunking or ChunkingConfig()
        self._chunk_cache: dict[str, list[Chunk]] = {}

    def manifests(self) -> list[DatasetManifest]:
        manifests: list[DatasetManifest] = []
        if not self.root.exists():
            return manifests
        for path in sorted(self.root.glob("*/manifest.yaml")):
            manifests.append(self._read_manifest(path))
        return manifests

    def manifest(self, dataset_id: str) -> DatasetManifest:
        path = manifest_path(self.root / dataset_id)
        if not path.is_file():
            raise DatasetNotFoundError(f"Unknown dataset: {dataset_id}")
        manifest = self._read_manifest(path)
        if manifest.id != dataset_id:
            raise DatasetNotFoundError(
                f"Dataset manifest ID does not match directory: {dataset_id}"
            )
        return manifest

    def chunks(self, dataset_id: str) -> list[Chunk]:
        if dataset_id not in self._chunk_cache:
            manifest = self.manifest(dataset_id)
            chunks: list[Chunk] = []
            for document in self._documents(manifest):
                chunks.extend(
                    chunk_document(
                        document,
                        dataset_id,
                        self.chunking.max_characters,
                        self.chunking.overlap_characters,
                    )
                )
            self._chunk_cache[dataset_id] = chunks
        return self._chunk_cache[dataset_id]

    def chunk(self, dataset_id: str, chunk_id: str) -> Chunk:
        for chunk in self.chunks(dataset_id):
            if chunk.id == chunk_id:
                return chunk
        raise DatasetNotFoundError(f"Unknown chunk: {chunk_id}")

    def clear_cache(self, dataset_id: str | None = None) -> None:
        if dataset_id is None:
            self._chunk_cache.clear()
        else:
            self._chunk_cache.pop(dataset_id, None)

    def _read_manifest(self, path: Path) -> DatasetManifest:
        with path.open(encoding="utf-8") as manifest_file:
            payload = yaml.safe_load(manifest_file) or {}
        return DatasetManifest.model_validate(payload)

    def _documents(self, manifest: DatasetManifest) -> Iterator[SourceDocument]:
        dataset_dir = self.root / manifest.id
        record_index = 0
        for source in manifest.sources:
            for path in sorted(dataset_dir.glob(source.glob)):
                for item in self._read_source(path, source):
                    text = self._field(item, source.text_field, "text")
                    if not text.strip():
                        continue
                    title = self._field(item, source.title_field, path.stem)
                    source_id = self._field(item, source.id_field, "") or self._stable_source_id(
                        path, record_index, text
                    )
                    yield SourceDocument(
                        id=source_id,
                        title=title,
                        text=text,
                        source_path=path.relative_to(dataset_dir).as_posix(),
                        record_index=record_index,
                    )
                    record_index += 1

    def _read_source(self, path: Path, source: SourceSpec) -> Iterator[dict[str, Any]]:
        source_format = source.format.lower()
        if source_format in {"text", "txt", "markdown", "md"}:
            yield {"text": path.read_text(encoding="utf-8"), "title": path.stem}
            return
        if source_format == "json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload if isinstance(payload, list) else [payload]
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError(f"JSON records must be objects: {path}")
                yield record
            return
        if source_format in {"jsonl", "ndjson"}:
            with path.open(encoding="utf-8") as source_file:
                for line_number, line in enumerate(source_file, start=1):
                    if line.strip():
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError(f"JSONL record {line_number} is not an object: {path}")
                        yield record
            return
        if source_format == "parquet":
            for batch in pq.ParquetFile(path).iter_batches():
                for record in batch.to_pylist():
                    yield record
            return
        raise ValueError(f"Unsupported dataset format '{source.format}' in {path}")

    @staticmethod
    def _field(record: dict[str, Any], field: str | None, default: str) -> str:
        value = record.get(field or default, default)
        if value is None:
            return default
        return str(value)

    @staticmethod
    def _stable_source_id(path: Path, record_index: int, text: str) -> str:
        source = f"{path.as_posix()}\x1f{record_index}\x1f{text}"
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]
