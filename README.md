# Semantic Arena

Semantic Arena is a controlled comparison environment for retrieval-augmented generation (RAG)
systems. It compares Apache Solr and Meilisearch using identical source documents, chunks,
embeddings, queries, and evidence-selection rules.

The system answers questions only from documents explicitly indexed in a selected local dataset.
It is not a general-purpose chatbot. Answers must be exact quotes from retrieved chunks and include
source citations. When no retrieved passage supports an answer, the response is exactly:

```text
there are no available data
```

## Quick Start

Requirements:

- Python 3.14 and `uv`
- Docker with permission to access its daemon
- An OpenAI-compatible embedding endpoint, local or hosted

Set up the local configuration and dependencies:

```bash
cp .env.example .env
uv sync --group dev
docker compose up -d
```

Edit `.env` to point `SEMANTIC_ARENA_EMBEDDING_BASE_URL` and
`SEMANTIC_ARENA_EMBEDDING_MODEL` at a compatible embedding service. The configured embedding
dimension must exactly match the selected model's vector size.

Start the web application:

```bash
uv run uvicorn api.app:app --reload
```

Open <http://127.0.0.1:8000>, select the `demo` dataset and one or both available engines, then
select **Index selected**. Indexing generates document vectors once and sends identical chunks and
vectors to every selected engine.

Run the checks with:

```bash
uv run pytest -q
uv run ruff check .
```

If Docker reports a permission error for `/var/run/docker.sock`, grant the current user access to
the Docker daemon before starting the Compose services.

## Documentation

- [Architecture and grounding](doc/architecture.md)
- [Datasets and Parquet tools](doc/datasets.md)
- [API reference](doc/api.md)
- [Development guide and project status](doc/development.md)

## Current Capabilities

- FastAPI API and browser-based comparison interface
- Docker Compose services for Apache Solr and Meilisearch
- Local text, Markdown, JSON, JSONL, and Parquet ingestion
- Shared deterministic chunking and OpenAI-compatible embeddings
- Solr and Meilisearch lexical/vector adapters with shared weighted RRF
- Indexing jobs, health reporting, citation lookup, and engine-isolated errors
- Deterministic extractive fallback and optional OpenAI-compatible evidence selector
