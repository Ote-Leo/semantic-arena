# Architecture and Grounding

## Grounding Policy

Semantic Arena has a strict no-improvisation policy:

- It does not browse the internet or provide models with browsing tools.
- It does not use model knowledge as an answer source.
- An answer is displayed only when it is an exact quote from a retrieved, indexed chunk.
- Every displayed quote has a citation to its source document and location.
- If no retrieved passage directly supports an answer, the system responds exactly
  `there are no available data`.

For example, a dataset of medical documents must not answer a question about the President of the
United States unless an indexed document contains that information, regardless of whether it is
correct or generally known.

This policy prevents unsupported answers. It does not eliminate false abstentions: an answer may
exist in a dataset but remain undiscovered if retrieval misses the relevant passage.

## Comparison Design

The arena compares retrieval engines, not different document-processing or embedding pipelines.
Both engines receive:

- The same source documents
- The same deterministic chunks and chunk IDs
- The same document embeddings
- The same query embedding
- The same retrieval settings where comparable
- The same evidence-selection policy

Each selected engine performs full-text and vector retrieval. The backend then applies a shared
weighted reciprocal-rank fusion (RRF) algorithm to its lexical and semantic result lists. This
makes ranking comparison controlled and repeatable instead of comparing unrelated native
hybrid-ranking behavior.

Apache Solr uses eDisMax for full-text retrieval and `DenseVectorField` for vector retrieval.
Meilisearch uses its regular full-text API and a `userProvided` embedder, allowing the backend to
send the exact same precomputed vectors used by Solr.

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

The evidence-selection model can select passages but cannot author the rendered response. It
returns candidate chunk IDs and quotes. The backend rejects a selection unless:

1. The chunk ID belongs to the selected dataset and search result.
2. The quote is an exact substring of that stored chunk.
3. The corresponding citation metadata is available.

Only validated stored text is rendered in the frontend.

## Model Configuration

Embedding and evidence-selection models are independently configurable roles.

- OpenAI-compatible endpoints support local services such as Ollama, LM Studio, and vLLM.
- Compatible hosted APIs are supported through the same configuration mechanism.
- Model endpoint, model ID, API key, dimensions, and timeouts are environment-backed settings.
- Secrets are not committed to the repository.
- Changing an embedding model, embedding dimensions, or chunking settings marks indexes stale and
  requires reindexing.
- Changing the evidence-selection model does not require reindexing.

Cloud profiles can send documents, retrieved excerpts, and queries to a third party. Local
profiles are preferred when data must remain on the machine.

## Frontend

The frontend provides:

- A multi-select control for Solr and Meilisearch
- A dataset selector with index readiness and indexing status
- Explicit index and reindex actions with progress reporting
- A chat-style query input
- Side-by-side engine output on desktop and stacked cards on mobile
- Exact quoted answers with expandable citations
- Retrieval diagnostics showing lexical, vector, and fused candidates
- Per-engine timings for retrieval, fusion, evidence selection, and total response time
- Isolated error displays so a failed engine does not hide another engine's result

Changing datasets resets chat context to prevent cross-dataset evidence contamination.

## Metrics and Evaluation

Responses report:

- Shared query-embedding time
- Lexical retrieval time
- Vector retrieval time
- Fusion time
- Evidence-selection time
- Total latency
- Candidate and citation counts
- Raw engine ranks and scores, labelled as engine-specific values

Dataset-local evaluation files will support measurement of Recall@k, MRR, nDCG, answerability
accuracy, citation validity, and latency percentiles.

## Verification Requirements

The project must verify:

- Both engines receive identical chunk IDs and vectors.
- Unsupported questions return `there are no available data`.
- Fabricated or altered model quotes are rejected before display.
- Every rendered answer includes a valid citation to local indexed material.
- A failure in one selected engine does not affect output from another.
- The frontend works on desktop and mobile layouts.
