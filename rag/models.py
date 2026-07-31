from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from datasets.models import Chunk

NO_DATA_MESSAGE = "there are no available data"


class QuoteSelection(BaseModel):
    chunk_id: str
    quote: str = Field(min_length=1)


class SelectionPayload(BaseModel):
    selections: list[QuoteSelection] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Citation:
    chunk: Chunk
    quote: str


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    message: str
    citations: list[Citation]
    no_data: bool
