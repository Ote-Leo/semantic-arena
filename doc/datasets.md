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

## Parquet Tools

Local Parquet files can be viewed and inspected with the PyArrow-based utility in `scripts/`:

```bash
uv run python scripts/parquet_tools.py show --head 20 data.parquet
uv run python scripts/parquet_tools.py show --columns id,title --format github data.parquet
uv run python scripts/parquet_tools.py show --format html data.parquet > table.html
uv run python scripts/parquet_tools.py csv --columns id,title "data/*.parquet"
uv run python scripts/parquet_tools.py inspect data.parquet
```

`show` and `csv` accept multiple files or quoted local glob expressions. Their `--head` limit
applies across all matching files. Run any command with `--help` for its complete options.

Generated indexes and engine data are stored in Docker volumes and are not committed to the
repository.
