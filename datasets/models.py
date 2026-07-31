from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field


class SourceSpec(BaseModel):
    glob: str
    format: str
    text_field: str | None = None
    title_field: str | None = None
    id_field: str | None = None


class DatasetManifest(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str
    description: str = ""
    sources: list[SourceSpec]


@dataclass(frozen=True, slots=True)
class SourceDocument:
    id: str
    title: str
    text: str
    source_path: str
    record_index: int


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    dataset_id: str
    source_id: str
    title: str
    text: str
    source_path: str
    record_index: int
    chunk_index: int
    start_char: int
    end_char: int

    @property
    def citation_label(self) -> str:
        suffix = f"record {self.record_index}, chars {self.start_char}-{self.end_char}"
        return f"{self.title} ({self.source_path}, {suffix})"


def manifest_path(dataset_dir: Path) -> Path:
    return dataset_dir / "manifest.yaml"
