# Indexing and vector-search performance studies

Compare Solr and Meilisearch indexing the **same manifest-selected chunks and vectors**.
The study uses `DatasetCatalog` and the existing engine adapters; it does not change
production indexing. The current `sample-data/zbdoc/manifest.yaml` selects only
`batch_0_1000.parquet` (not every Parquet file). One source record can produce several
chunks. Results are saved under ignored `data/` by default; each invocation replaces
files with the same `--output` stem, so use different names to retain runs.

## Setup

From the repository root, with Python 3.14 and `uv`:

```sh
uv sync --group dev --group study
# Configure .env for the engines (copy .env.example if needed). Ensure the
# Meilisearch master key in .env matches the one used by Docker Compose.
docker compose up -d --wait
```

The default benchmark embedder calls Ollama's **native** endpoint
`http://10.0.7.20:11435/api/embed` with model `qwen3-embedding:8b` and expects
4096-dimensional vectors. This sends manifest-extracted chunk text to that host;
make sure you trust it and that it is accessible from the machine running the
study. The script does not launch or pull the embedding model. Override its URL,
model or expected dimensions with `--embedding-url`, `--embedding-model`, and
`--embedding-dimensions`. The request count and vector size are checked before
submitting batches to either engine. To use the original OpenAI-compatible
`/embeddings` API instead, pass `--embedding-provider openai` and configure
`SEMANTIC_ARENA_EMBEDDING_BASE_URL`, `SEMANTIC_ARENA_EMBEDDING_MODEL`, and
`SEMANTIC_ARENA_EMBEDDING_DIMENSIONS` (or use the corresponding CLI overrides).

```sh
# Pilot. WARNING: this publishes a partial index for zbdoc on both engines.
uv run python performance-study/__main__.py --max-chunks 100

# Full manifest scope, reuse each trial's vectors for both engines.
uv run python performance-study/__main__.py --batch-embedding --trials 3 --replace-existing

# One embedding request and one engine submission per chunk, for measured
# per-chunk client-observed latency (slower and not representative of throughput).
uv run python performance-study/__main__.py --embedding-batch-size 1 --chunk-batch-size 1 --replace-existing

# Read only the saved CSVs; no engine or embedding service needed for plotting.
uv run --group study python performance-study/plot.py --input data/indexing-study.csv --output-dir data/plots
```

The benchmark checks engine health and refuses to replace published indexes
without `--replace-existing`. Repeated trials require it. Use a dedicated dataset
or isolated engine instances if existing search results must be preserved. Engine
order alternates across trials. Each trial extracts and embeds the manifest scope
once and sends the **same** chunks and vectors to both engines, sequentially.
Extraction, embedding, and per-engine indexing display live `tqdm` progress; CSV
sidecars are flushed as embedding and indexing batches finish.
The script does not start or stop Docker containers and does not delete existing
published indexes unless replacement is explicitly requested. It uses temporary
pickle spools that are removed after a trial; do not load spools from untrusted
sources. The adapters' staging, finalization, and abort behavior is preserved.

## Saved results

For `--output data/indexing-study.csv`, the script also writes:

- `data/indexing-study-run.json`: run ID, timestamps, manifest hash, chunking,
  endpoint/model settings, selected engines, and discovered engine versions.
  No API key is saved.
- `data/indexing-study-embedding-batches.csv`: one row per embedding request,
  including trial, batch ID, chunk count, wall-clock duration, and status.
- `data/indexing-study-ingestion-batches.csv`: one row per engine submission,
  including trial, batch ID, cumulative successful chunk count, duration, and
  status. Meilisearch waits for task completion; Solr commit is measured later.
- `data/indexing-study-chunks.csv`: per-chunk source/record IDs and positions,
  text character/byte sizes, and IDs/durations of its embedding and engine batches.
  It contains **no chunk text or vectors**.
- `data/indexing-study-records.csv`: source record ID, number and bytes of
  chunks, the distinct batch IDs that contained its chunks, and the sum of those
  **shared batch durations**. A batch can contain multiple records; these are
  involvement times, not exclusive per-record CPU or elapsed latency. Do not
  add these across records to calculate totals.

The summary CSV has one row per engine/trial. `extraction_s` covers manifest
loading, chunking and spooling; `embedding_s` includes provider requests and
vector spooling; `prepare_s` covers staging index/schema setup; `ingestion_s`
includes batch submissions; `finalization_s` includes commit/swap/publication
check. `engine_total_s` is their sum, while `end_to_end_s` is a hypothetical
**single-engine** extraction + embedding + engine total. `trial_wall_s` is the
actual wall time of the dual-engine trial, including benchmark instrumentation.
Throughput is measured in chunks/second, not source records/second. Rows with
`status=failed` are excluded from performance plots but retain error information.

Both adapters split submissions above 100 chunks. This script caps its effective
`chunk_batch_size` at 100 so each measured ingestion batch maps to one adapter
call. The default embedding batch size is one unless `--batch-embedding` is used;
that flag selects the configured batch size (currently 64). With batching, a
chunk row's `*_duration_s` is the **same shared duration for every chunk in its
batch**. It must not be interpreted as an independently measured per-chunk time.
To measure per-chunk embedding and ingestion latency, use both batch sizes set
to one; this changes the workload, so compare throughput runs separately.

`plot.py` produces PNGs for engine-stage means, batch-latency distributions,
throughput, ingestion progress, chunk sizes, and source-record shared-batch
involvement versus text size. Batch latency plots have one observation per batch,
not per chunk, and omit failed runs. Index existence is
checked after publication, but the benchmark does not independently verify
indexed document counts. These wall-clock metrics include client serialization,
transport, and temporary spool I/O; they do not measure engine CPU, RAM, disk,
or server-only processing time. Record hardware, competing load, and endpoint
versions when interpreting the charts.

## Vector-search comparison

`vector_search.py` runs vector-only retrieval against already-published indexes.
It does not create, replace, or delete indexes. Both engines receive the same
query vectors generated once per run. Use the same embedding model, dimensions,
dataset, and chunking configuration that were used to build the indexes; the
study checks that indexes exist but cannot verify their build configuration.

The evaluation file is JSON Lines with one judged query per line. Each
`relevant_chunk_ids` list contains the chunk IDs judged relevant to that query:

```json
{"query_id":"example-1","query":"What does the source say?","relevant_chunk_ids":["chunk-id"]}
```

The small `demo-queries.jsonl` file supplies four example judgments for the
bundled `demo` dataset. It is suitable for validating the workflow and
presentation output, not for drawing general conclusions about either engine.
For a meaningful comparison on another dataset, curate a representative,
fixed query set and manually judge its relevant chunk IDs.

```sh
# Ensure Docker services are running and configure the embedding endpoint.
uv sync --group dev --group study
docker compose up -d --wait

# Build dedicated Ollama-backed indexes from the demo source. This does not replace
# any existing `demo` indexes; use a unique index ID for a fresh comparison.
uv run python performance-study/__main__.py \
  --dataset demo \
  --index-dataset vector-presentation-ollama \
  --embedding-provider ollama \
  --output data/vector-presentation-indexing.csv

# Search-only benchmark; no index mutation. Repeated trials alternate engine order.
uv run python performance-study/vector_search.py \
  --dataset vector-presentation-ollama \
  --queries performance-study/demo-queries.jsonl \
  --embedding-provider ollama \
  --trials 5 --repetitions 3 --warmups 1

# Create presentation charts from saved results without calling any services.
uv run --group study python performance-study/vector_search_plot.py \
  --input data/vector-search-study.csv \
  --output-dir data/vector-search-plots

# Generate an editable PowerPoint deck from the saved measurements and charts.
uv run --with python-pptx python performance-study/create_presentation.py \
  --input data/vector-search-study.csv \
  --plot-dir data/vector-search-plots \
  --output performance-study/vector-search-comparison.pptx
```

Configure `.env` with the OpenAI-compatible embedding settings before using
`--embedding-provider openai`, or use `--embedding-provider ollama` and the
corresponding CLI overrides. The indexing study's `--index-dataset` separates
the published index ID from the source dataset ID; this is useful for safely
testing a new embedding model without replacing indexes built with another
model or dimension. Keep the same index ID for the subsequent vector-search
study. The query text is sent to the selected embedding endpoint.
Each query is embedded once, outside measured engine-search latency; the same
vector is reused for every engine and trial. The default query batch size is 32
and can be changed with `--query-batch-size`. Queries run sequentially;
`--warmups` are excluded from measurements, while `--repetitions` are recorded
individually. Engine order alternates by trial.

### Latency-only study on the local zbdoc manifest

The current `zbdoc` manifest selects only `batch_0_1000.parquet`, not the
complete set of local batches. With the current chunking settings, that scope
produces 11,648 chunks. The supplied `zbdoc-latency-queries.jsonl` contains
four fixed queries with empty relevance lists: use `--latency-only` to run them
without making Recall/MRR/nDCG claims.

```sh
# This publishes a separate 4,096-dimensional index; it does not overwrite `zbdoc`.
uv run python performance-study/__main__.py \
  --dataset zbdoc \
  --index-dataset zbdoc-presentation-ollama \
  --embedding-provider ollama \
  --batch-embedding \
  --output data/zbdoc-presentation-indexing.csv

uv run python performance-study/vector_search.py \
  --dataset zbdoc-presentation-ollama \
  --queries performance-study/zbdoc-latency-queries.jsonl \
  --embedding-provider ollama \
  --latency-only \
  --top-k 10 --trials 5 --repetitions 3 --warmups 1 \
  --output data/zbdoc-presentation-search.csv

uv run --group study python performance-study/vector_search_plot.py \
  --input data/zbdoc-presentation-search.csv \
  --output-dir data/zbdoc-presentation-plots

uv run --with python-pptx python performance-study/create_presentation.py \
  --input data/zbdoc-presentation-search.csv \
  --plot-dir data/zbdoc-presentation-plots \
  --indexing-summary data/zbdoc-presentation-indexing.csv \
  --output performance-study/zbdoc-vector-search-comparison.pptx
```

In latency-only mode, the query CSV leaves relevance metrics blank and plotting
produces latency charts only. If every query has relevance judgments, omit
`--latency-only`; quality and latency plots will then be produced.

For `--output data/vector-search-study.csv`, the study writes:

- `data/vector-search-study-run.json`: query-file hash, model/dimensions, engine
  versions, trial settings, shared query-embedding time, and run status.
- `data/vector-search-study.csv`: one row per query, engine, trial, and measured
  repetition, including ranked chunk IDs, latency, and judged quality metrics.
- `data/vector-search-study-summary.csv`: per-engine/per-trial mean, p50, p95,
  and p99 latency plus mean Recall@k, MRR, and nDCG@k.
- `data/vector-search-plots/`: latency distribution, judged quality, and
  latency-versus-recall charts.

Recall@k is the fraction of a query's judged relevant chunks retrieved in the
first k results. MRR measures the rank of the first relevant chunk, and nDCG@k
uses binary relevance from the supplied judgments. All quality results depend
on the completeness and correctness of those judgments. A missing judgment can
make good retrieval look poor; unjudged results are not automatically treated
as relevant.

Latency is client-observed and includes request serialization and transport;
it is not server-only processing time. Queries run sequentially, so these
charts do not characterize concurrent throughput, resource consumption, or
indexing speed. Engine-native similarity scores are not compared because their
scales and meanings are not necessarily equivalent. Record machine load,
hardware, index settings, embedding model, and engine versions for the
presentation, and report quality alongside latency rather than claiming a
universal winner.

## Follow-up TODOs

- [ ] Build a representative set of 100–300 queries and have reviewers judge
  relevant chunks; report Recall@k, MRR, and nDCG alongside latency.
- [ ] Repeat the comparison at the full intended corpus size, keeping chunks
  and vectors identical between engines.
- [ ] Run concurrent-load tests at realistic levels and record p50/p95/p99
  latency, throughput, and errors.
- [ ] Monitor server CPU, memory, and disk during indexing and search.
- [ ] Repeat indexing under controlled conditions to assess run-to-run
  variation; the current indexing comparison is based on one run.
