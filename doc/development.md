# Development Guide

## Repository Layout

The repository uses a flat, top-level layout rather than a `src/semantic_arena` package:

```text
semantic-arena/
|-- api/                 # HTTP routes and request/response models
|-- datasets/            # Dataset manifests, loaders, and chunking
|-- doc/                 # Project documentation
|-- embeddings/          # Configurable embedding providers
|-- indexing/            # Engine-independent indexing orchestration
|-- jobs/                # Index build status and background jobs
|-- rag/                 # Evidence selection and validation
|-- search/              # Shared search interfaces and RRF fusion
|-- solr/                # Apache Solr adapter and configuration
|-- meilisearch/         # Meilisearch adapter and configuration
|-- frontend/            # Chat and search comparison interface
|-- sample-data/         # Source datasets to index
|   `-- demo/
|-- scripts/             # Local development and data utilities
|-- config/              # Non-secret application configuration
|-- tests/
|-- main.py              # Application entry point
|-- compose.yaml         # Local Solr and Meilisearch services
|-- pyproject.toml
|-- .env.example
`-- README.md
```

## Checks

Run the test and lint suites before committing changes:

```bash
uv run pytest -q
uv run ruff check .
```

## Development Status

The first working vertical slice is implemented:

- FastAPI API and browser-based comparison interface
- Docker Compose configuration for Apache Solr and Meilisearch
- Local text, Markdown, JSON, JSONL, and Parquet ingestion
- Shared deterministic chunking and OpenAI-compatible embeddings
- Solr and Meilisearch lexical/vector adapters with shared weighted RRF
- Indexing jobs, health reporting, source citation lookup, and engine-isolated errors
- Deterministic extractive fallback and optional OpenAI-compatible evidence selector
- Unit and API tests for stable chunks, RRF, strict quote validation, indexing, and no-data fallback

The remaining work is production hardening: persistent job/index metadata, versioned physical
indexes, evaluation-file tooling, Docker-backed integration tests, and broader source-format
fixtures.
