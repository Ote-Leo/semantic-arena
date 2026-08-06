from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SourceSpec(BaseModel):
    glob: str
    format: str
    text_field: str | None = None
    text_fields: list[str] = Field(default_factory=list)
    title_field: str | None = None
    title_fields: list[str] = Field(default_factory=list)
    id_field: str | None = None
    json_field: str | None = None
    duplicate_id_policy: Literal["allow", "error", "skip-identical"] = "allow"

    @model_validator(mode="after")
    def validate_extraction(self) -> SourceSpec:
        if self.text_field and self.text_fields:
            raise ValueError("Use either text_field or text_fields, not both.")
        if self.title_field and self.title_fields:
            raise ValueError("Use either title_field or title_fields, not both.")
        if self.duplicate_id_policy != "allow" and not self.id_field:
            raise ValueError("duplicate_id_policy requires id_field.")
        return self


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
