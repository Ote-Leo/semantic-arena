from __future__ import annotations

import hashlib
import json
import re
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
            self._chunk_cache[dataset_id] = list(self.iter_chunks(dataset_id))
        return self._chunk_cache[dataset_id]

    def iter_chunks(self, dataset_id: str) -> Iterator[Chunk]:
        manifest = self.manifest(dataset_id)
        for document in self._documents(manifest):
            yield from chunk_document(
                document,
                dataset_id,
                self.chunking.max_characters,
                self.chunking.overlap_characters,
            )

    def chunk(self, dataset_id: str, chunk_id: str) -> Chunk:
        chunks = self._chunk_cache.get(dataset_id)
        for chunk in chunks if chunks is not None else self.iter_chunks(dataset_id):
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
        seen_source_ids: dict[str, str] = {}
        for source in manifest.sources:
            for path in sorted(dataset_dir.glob(source.glob)):
                for item in self._read_source(path, source):
                    content = self._content_record(item, source, path, record_index)
                    text = self._document_text(content, source)
                    if not text.strip():
                        continue
                    title = self._document_title(content, source, path.stem)
                    source_id = self._field(item, source.id_field, "") or self._stable_source_id(
                        path, record_index, text
                    )
                    if source.id_field and source_id in seen_source_ids:
                        fingerprint = self._document_fingerprint(title, text)
                        if source.duplicate_id_policy == "skip-identical":
                            if seen_source_ids[source_id] == fingerprint:
                                continue
                            raise ValueError(
                                f"Conflicting duplicate source ID '{source_id}' in {path}"
                            )
                        if source.duplicate_id_policy == "error":
                            raise ValueError(f"Duplicate source ID '{source_id}' in {path}")
                    elif source.id_field:
                        seen_source_ids[source_id] = self._document_fingerprint(title, text)
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
    def _content_record(
        record: dict[str, Any], source: SourceSpec, path: Path, record_index: int
    ) -> dict[str, Any]:
        if source.json_field is None:
            return record
        raw_content = record.get(source.json_field)
        if not isinstance(raw_content, str):
            raise ValueError(
                f"JSON field '{source.json_field}' must be a string in {path} record {record_index}"
            )
        try:
            content = json.loads(raw_content)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in field '{source.json_field}' in {path} record {record_index}"
            ) from error
        if not isinstance(content, dict):
            raise ValueError(
                f"JSON field '{source.json_field}' must contain an object in {path} "
                f"record {record_index}"
            )
        return content

    def _document_text(self, record: dict[str, Any], source: SourceSpec) -> str:
        if not source.text_fields:
            return self._field(record, source.text_field, "text")
        sections = [self._render_path(record, path) for path in source.text_fields]
        return "\n\n".join(section for section in sections if section)

    def _document_title(self, record: dict[str, Any], source: SourceSpec, default: str) -> str:
        if not source.title_fields:
            return self._field(record, source.title_field, default)
        values: list[str] = []
        for path in source.title_fields:
            for value in self._resolve_path(record, path):
                values.extend(self._scalar_values(value))
        unique_values = list(dict.fromkeys(value for value in values if value.strip()))
        return " | ".join(unique_values) or default

    def _render_path(self, record: dict[str, Any], path: str) -> str:
        values = self._resolve_path(record, path)
        if not values:
            return ""
        label = self._label(path)
        if all(not isinstance(value, (dict, list)) for value in values):
            rendered = [text for value in values if (text := self._scalar_text(value))]
            return f"{label}: {' | '.join(rendered)}" if rendered else ""
        rendered = [text for value in values if (text := self._render_value(value))]
        body = "\n\n".join(rendered)
        return f"{label}\n{body}" if body else ""

    @staticmethod
    def _resolve_path(record: dict[str, Any], path: str) -> list[Any]:
        values: list[Any] = [record]
        for raw_component in path.split("."):
            flatten = raw_component.endswith("[]")
            component = raw_component[:-2] if flatten else raw_component
            resolved: list[Any] = []
            for value in values:
                if not isinstance(value, dict) or component not in value:
                    continue
                child = value[component]
                if flatten:
                    if isinstance(child, list):
                        resolved.extend(child)
                else:
                    resolved.append(child)
            values = resolved
        return values

    def _render_value(self, value: Any) -> str:
        if value is None or value == "" or value == [] or value == {}:
            return ""
        if isinstance(value, dict):
            lines: list[str] = []
            for key, child in value.items():
                rendered = self._render_value(child)
                if not rendered:
                    continue
                if isinstance(child, (dict, list)):
                    lines.append(f"{self._label(key)}\n{rendered}")
                else:
                    lines.append(f"{self._label(key)}: {rendered}")
            return "\n".join(lines)
        if isinstance(value, list):
            return "\n\n".join(rendered for item in value if (rendered := self._render_value(item)))
        return self._scalar_text(value)

    def _scalar_values(self, value: Any) -> list[str]:
        if isinstance(value, dict):
            return [item for child in value.values() for item in self._scalar_values(child)]
        if isinstance(value, list):
            return [item for child in value for item in self._scalar_values(child)]
        rendered = self._scalar_text(value)
        return [rendered] if rendered else []

    @staticmethod
    def _scalar_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)

    @staticmethod
    def _label(path: str) -> str:
        leaf = path.rsplit(".", maxsplit=1)[-1].removesuffix("[]")
        words = re.sub(r"(?<!^)(?=[A-Z])", " ", leaf).replace("_", " ")
        return words.title()

    @staticmethod
    def _document_fingerprint(title: str, text: str) -> str:
        return hashlib.sha256(f"{title}\x1f{text}".encode()).hexdigest()

    @staticmethod
    def _stable_source_id(path: Path, record_index: int, text: str) -> str:
        source = f"{path.as_posix()}\x1f{record_index}\x1f{text}"
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]
