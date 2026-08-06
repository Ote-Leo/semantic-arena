# Datasets and Parquet Tools

## Dataset Layout

Datasets are local-only and live under `sample-data/<dataset-id>/`. Each dataset has a manifest
describing its files and fields. For example:

```yaml
id: wikitext-103
name: Wikitext 103
sources:
  - glob: documents/*.parquet
    format: parquet
    text_field: text
    title_field: title
```

Supported ingestion formats are:

- Plain text (`.txt`)
- Markdown (`.md`)
- JSON arrays
- JSON Lines (`.jsonl`)
- Parquet

Chunking is deterministic and paragraph-aware. Each chunk retains its source path, record number,
chunk number, and character offsets. These fields provide stable citations and ensure both engines
index identical records.

### JSON Stored in a Column

When a Parquet or JSON field contains a serialized JSON object, manifests can select nested content
without indexing the serialized representation:

```yaml
sources:
  - glob: batch_*.parquet
    format: parquet
    id_field: id
    json_field: source_data
    duplicate_id_policy: skip-identical
    title_fields:
      - drksId
      - trialDescriptions[].title
    text_fields:
      - trialStatus
      - trialDescriptions
      - studiedHealthConditions
      - recruitment
```

Paths use dots for nested objects and `[]` to project every item in an array. Selected objects and
arrays are rendered as labeled human-readable sections. `duplicate_id_policy: skip-identical`
removes repeated IDs only when their extracted title and text match; conflicting records fail the
index build.

The `zbdoc` manifest indexes identifiers, bilingual descriptions, conditions, study design,
outcomes, recruitment and eligibility, result summaries, publications, and non-contact ethics
decisions. Direct contacts, email addresses, phone numbers, street addresses, and raw URLs are
deliberately excluded.

The current local corpus contains 21,728 unique trials and produces 251,996 chunks with the default
chunking configuration. A complete build requires about 3,938 embedding requests at the configured
batch size of 64, plus enough Docker disk space for the text and configured-dimension vectors in
both engines. Confirm the embedding endpoint, model cost, and available disk space before starting
it.

Large datasets are extracted, embedded, and submitted to each selected engine in bounded batches.
The UI reports the number of processed chunks while staging indexes are built. A completed staging
index replaces the prior index; failed builds are removed without publishing partial data.

## Parquet Tools

Local Parquet files can be viewed and inspected with the PyArrow-based utility in `scripts/`:

```bash
uv run python scripts/parquet_tools.py show --head 20 data.parquet
uv run python scripts/parquet_tools.py show --columns id,title --format github data.parquet
uv run python scripts/parquet_tools.py show --format html --expand-json --head 10 data.parquet > table.html
uv run python scripts/parquet_tools.py csv --columns id,title "data/*.parquet"
uv run python scripts/parquet_tools.py record --row 0 data.parquet
uv run python scripts/parquet_tools.py record --id DRKS00039129 --format html \
  "sample-data/zbdoc/*.parquet" > record.html
uv run python scripts/parquet_tools.py inspect data.parquet
uv run python scripts/parquet_tools.py inspect --format text data.parquet
```

`show` and `csv` accept multiple files or quoted local glob expressions. Their `--head` limit
applies across all matching files. For HTML output, `--expand-json` detects JSON strings and renders
their objects and arrays as collapsible nested tables. `record` selects one record by its zero-based
row or by an ID and expands JSON strings automatically; it outputs formatted JSON by default or a
self-contained HTML document with `--format html`. `inspect` outputs formatted JSON by default,
including the Arrow data schema, Parquet physical columns, row groups, compression, and column
statistics. Use `--format text` for the compact human-readable view.

From WSL2, a generated document can be opened in a system-wide Windows Brave installation with:

```bash
"/mnt/c/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe" \
  "$(wslpath -w "$PWD/record.html")"
```

Run any command with `--help` for its complete options.

Generated indexes and engine data are stored in Docker volumes and are not committed to the
repository.
