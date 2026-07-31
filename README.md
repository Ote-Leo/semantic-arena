# Semantic Arena

Semantic Arena is a controlled comparison environment for retrieval-augmented generation (RAG) systems. Its first search-engine adapters are:

- Apache Solr
- Meilisearch

The system answers questions only from documents explicitly indexed in a selected local dataset. It is not a general-purpose chatbot.

## Grounding Policy

Semantic Arena has a strict no-improvisation policy:

- It does not browse the internet or provide models with browsing tools.
- It does not use model knowledge as an answer source.
- An answer is displayed only when it is an exact quote from a retrieved, indexed chunk.
- Every displayed quote has a citation to its source document and location.
- If no retrieved passage directly supports an answer, the system responds exactly:

  ```text
  there are no available data
  ```

For example, a dataset of medical documents must not answer a question about the President of the United States unless an indexed document contains that information, regardless of whether it is correct or generally known.

This policy prevents unsupported answers. It does not eliminate false abstentions: an answer may exist in a dataset but remain undiscovered if retrieval misses the relevant passage.

## Comparison Design

The arena compares retrieval engines, not different document-processing or embedding pipelines. Both engines receive:

- The same source documents.
- The same deterministic chunks and chunk IDs.
- The same document embeddings.
- The same query embedding.
- The same retrieval settings where comparable.
- The same evidence-selection policy.

Each selected engine performs full-text and vector retrieval. The backend then applies a shared weighted reciprocal-rank fusion (RRF) algorithm to its lexical and semantic result lists. This makes ranking comparison controlled and repeatable instead of comparing unrelated native hybrid-ranking behavior.

Apache Solr will use eDisMax for full-text retrieval and `DenseVectorField` for vector retrieval. Meilisearch will use its regular full-text API and a `userProvided` embedder, allowing the backend to send the exact same precomputed vectors used by Solr.

## Answer Pipeline

```text
Dataset files
  -> dataset loader and deterministic chunker
  -> shared embedding model
  -> Solr and Meilisearch indexes

User query
  -> one shared query embedding
  -> lexical + vector retrieval in every selected engine
  -> shared weighted RRF per engine
  -> evidence selection from retrieved chunks
  -> exact-quote and citation validation
  -> cited extracts or "there are no available data"
```

The evidence-selection model can select passages but cannot author the rendered response. It returns candidate chunk IDs and quotes. The backend rejects a selection unless:

1. The chunk ID belongs to the selected dataset and search result.
2. The quote is an exact substring of that stored chunk.
3. The corresponding citation metadata is available.

Only validated stored text is rendered in the frontend.

## Repository Layout

The repository uses a flat, top-level layout rather than a `src/semantic_arena` package:

```text
semantic-arena/
├── api/                 # HTTP routes and request/response models
├── datasets/            # Dataset manifests, loaders, and chunking
├── embeddings/          # Configurable embedding providers
├── indexing/            # Engine-independent indexing orchestration
├── jobs/                # Index build status and background jobs
├── rag/                 # Evidence selection and validation
├── search/              # Shared search interfaces and RRF fusion
├── solr/                # Apache Solr adapter and configuration
├── meilisearch/         # Meilisearch adapter and configuration
├── frontend/            # Chat and search comparison interface
├── sample-data/         # Source datasets to index
│   └── demo/
├── config/              # Non-secret application configuration
├── tests/
├── main.py              # Application entry point
├── compose.yaml         # Local Solr and Meilisearch services
├── pyproject.toml
├── .env.example
└── README.md
```

Generated indexes and engine data are stored in Docker volumes and are not committed to the repository.

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
select **Index selected**. Indexing generates document vectors once and sends the identical chunks
and vectors to every selected engine.

Run the test suite with:

```bash
uv run pytest -q
```

If Docker reports a permission error for `/var/run/docker.sock`, grant the current user access to
the Docker daemon before starting the Compose services.

## Datasets

Datasets are local-only and live under `sample-data/<dataset-id>/`. A dataset has a manifest describing its files and fields. For example:

```yaml
id: wikitext-103
name: Wikitext 103
sources:
  - glob: documents/*.parquet
    format: parquet
    text_field: text
    title_field: title
```

The initial ingestion formats are:

- Plain text (`.txt`)
- Markdown (`.md`)
- JSON arrays
- JSON Lines (`.jsonl`)
- Parquet

Chunking will be deterministic and paragraph-aware, retaining source path, record number, chunk number, and character offsets. These fields provide stable citations and ensure both engines index identical records.

## Model Configuration

Embedding and evidence-selection models are independently configurable roles.

- OpenAI-compatible endpoints support local services such as Ollama, LM Studio, and vLLM.
- Compatible hosted APIs are supported through the same configuration mechanism.
- Model endpoint, model ID, API key, dimensions, and timeouts are environment-backed settings.
- Secrets are not committed to the repository.
- Changing an embedding model, embedding dimensions, or chunking settings marks indexes stale and requires reindexing.
- Changing the evidence-selection model does not require reindexing.

Cloud profiles can send documents, retrieved excerpts, and queries to a third party. Local profiles are preferred when data must remain on the machine.

## Frontend

The frontend will provide:

- A multi-select control for Solr and Meilisearch.
- A dataset selector with index readiness and indexing status.
- Explicit index and reindex actions with progress reporting.
- A chat-style query input.
- Side-by-side engine output on desktop and stacked cards on mobile.
- Exact quoted answers with expandable citations.
- Retrieval diagnostics showing lexical, vector, and fused candidates.
- Per-engine timings for retrieval, fusion, evidence selection, and total response time.
- Isolated error displays so a failed engine does not hide another engine's result.

Changing datasets resets chat context to prevent cross-dataset evidence contamination.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Report backend, model, Solr, and Meilisearch availability. |
| `GET /api/engines` | List available engines and their status. |
| `GET /api/datasets` | List dataset manifests and per-engine index status. |
| `POST /api/indexes` | Start indexing a selected dataset into selected engines. |
| `GET /api/jobs/{job_id}` | Return indexing progress, outcome, or error. |
| `POST /api/chat` | Run a query against selected engines concurrently. |
| `GET /api/citations/{dataset_id}/{chunk_id}` | Return validated source metadata for a citation. |

The current MVP rebuilds the deterministic dataset index for each selected engine. Build failures are
preserved in the job record and reported to the frontend. Versioned, atomic index replacement is a
planned production-hardening improvement.

## Metrics and Evaluation

Responses will report:

- Shared query-embedding time.
- Lexical retrieval time.
- Vector retrieval time.
- Fusion time.
- Evidence-selection time.
- Total latency.
- Candidate and citation counts.
- Raw engine ranks and scores, labelled as engine-specific values.

Dataset-local evaluation files will support measurement of Recall@k, MRR, nDCG, answerability accuracy, citation validity, and latency percentiles.

## Development Status

The first working vertical slice is implemented:

- FastAPI API and browser-based comparison interface.
- Docker Compose configuration for Apache Solr and Meilisearch.
- Local text, Markdown, JSON, JSONL, and Parquet ingestion.
- Shared deterministic chunking and OpenAI-compatible embeddings.
- Solr and Meilisearch lexical/vector adapters with shared weighted RRF.
- Indexing jobs, health reporting, source citation lookup, and engine-isolated errors.
- Deterministic extractive fallback and optional OpenAI-compatible evidence selector.
- Unit and API tests for stable chunks, RRF, strict quote validation, indexing, and no-data fallback.

The remaining work is production hardening: persistent job/index metadata, versioned physical indexes,
evaluation-file tooling, Docker-backed integration tests, and broader source-format fixtures.

## Verification Requirements

The completed project must verify:

- Both engines receive identical chunk IDs and vectors.
- Unsupported questions return `there are no available data`.
- Fabricated or altered model quotes are rejected before display.
- Every rendered answer includes a valid citation to local indexed material.
- A failure in one selected engine does not affect output from another.
- The frontend works on desktop and mobile layouts.
