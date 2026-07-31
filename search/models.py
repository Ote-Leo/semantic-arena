from __future__ import annotations

from dataclasses import dataclass

from datasets.models import Chunk


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk: Chunk
    score: float | None


@dataclass(frozen=True, slots=True)
class Candidate:
    chunk: Chunk
    fused_score: float
    lexical_rank: int | None
    vector_rank: int | None
    lexical_score: float | None
    vector_score: float | None
