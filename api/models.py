from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class EngineStatusResponse(BaseModel):
    id: str
    label: str
    available: bool
    detail: str


class HealthResponse(BaseModel):
    engines: list[EngineStatusResponse]
    embedding_configured: bool
    selector_mode: str


class DatasetResponse(BaseModel):
    id: str
    name: str
    description: str
    index_name: str
    engine_indexes: dict[str, bool]


class IndexRequest(BaseModel):
    dataset_id: str
    engine_ids: list[str] = Field(min_length=1)


class JobResponse(BaseModel):
    id: str
    dataset_id: str
    engine_ids: list[str]
    status: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    result: dict[str, int] | None
    error: str | None


class ChatRequest(BaseModel):
    dataset_id: str
    query: str = Field(min_length=1, max_length=8_000)
    engine_ids: list[str] = Field(min_length=1)

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Query must contain text.")
        return normalized


class CitationResponse(BaseModel):
    chunk_id: str
    quote: str
    title: str
    source_path: str
    record_index: int
    chunk_index: int
    start_char: int
    end_char: int


class CandidateResponse(BaseModel):
    chunk_id: str
    title: str
    source_path: str
    excerpt: str
    fused_score: float
    lexical_rank: int | None
    vector_rank: int | None
    lexical_score: float | None
    vector_score: float | None


class TimingsResponse(BaseModel):
    lexical_ms: float
    vector_ms: float
    fusion_ms: float
    selection_ms: float
    total_ms: float


class EngineAnswerResponse(BaseModel):
    engine_id: str
    engine_label: str
    status: str
    message: str | None = None
    no_data: bool | None = None
    citations: list[CitationResponse] = Field(default_factory=list)
    candidates: list[CandidateResponse] = Field(default_factory=list)
    timings: TimingsResponse | None = None
    error: str | None = None


class ChatResponse(BaseModel):
    dataset_id: str
    query: str
    embedding_ms: float
    results: list[EngineAnswerResponse]


class ChunkCitationResponse(BaseModel):
    chunk_id: str
    dataset_id: str
    title: str
    text: str
    source_path: str
    record_index: int
    chunk_index: int
    start_char: int
    end_char: int
