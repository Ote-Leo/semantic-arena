from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_prefix="SEMANTIC_ARENA_",
        extra="ignore",
    )

    solr_url: str = "http://localhost:8983/solr"
    solr_configset: str = "semantic-arena"
    meilisearch_url: str = "http://localhost:7700"
    meilisearch_master_key: str = "change-this-development-key"
    embedding_base_url: str = ""
    embedding_model: str = ""
    embedding_dimensions: int = 768
    embedding_api_key: str = ""
    selector_base_url: str = ""
    selector_model: str = ""
    selector_api_key: str = ""


class ChunkingConfig(BaseModel):
    max_characters: int = Field(default=1_200, ge=100)
    overlap_characters: int = Field(default=160, ge=0)


class RetrievalConfig(BaseModel):
    lexical_limit: int = Field(default=20, ge=1)
    vector_limit: int = Field(default=20, ge=1)
    final_limit: int = Field(default=8, ge=1)
    lexical_weight: float = Field(default=1.0, ge=0)
    vector_weight: float = Field(default=1.0, ge=0)
    rrf_k: int = Field(default=60, ge=1)


class AnsweringConfig(BaseModel):
    maximum_quotes: int = Field(default=3, ge=1)


class ArenaConfig(BaseModel):
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    answering: AnsweringConfig = Field(default_factory=AnsweringConfig)


@lru_cache
def get_runtime_settings() -> RuntimeSettings:
    return RuntimeSettings()


@lru_cache
def get_arena_config() -> ArenaConfig:
    config_path = ROOT_DIR / "config" / "arena.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file) or {}
    return ArenaConfig.model_validate(payload)
