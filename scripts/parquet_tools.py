from __future__ import annotations

import argparse
import csv
import glob
import html
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TextIO

import pyarrow as pa
import pyarrow.parquet as pq


class CommandError(Exception):
    pass


def _columns(value: str) -> list[str]:
    columns = [column.strip() for column in value.split(",") if column.strip()]
    if not columns:
        raise argparse.ArgumentTypeError("at least one column is required")
    return columns


def _head(value: str) -> int:
    try:
        limit = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if limit < -1:
        raise argparse.ArgumentTypeError("must be -1 or greater")
    return limit


def _flatten_columns(groups: list[list[str]] | None) -> list[str]:
    return [column for group in groups or [] for column in group]


def _expand_files(expressions: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for expression in expressions:
        matches = sorted(glob.glob(expression)) if glob.has_magic(expression) else [expression]
        if not matches:
            raise CommandError(f"no files match: {expression}")
        for match in matches:
            path = Path(match)
            if not path.is_file():
                raise CommandError(f"file not found: {path}")
            paths.append(path)
    return paths


def _selected_columns(parquet_file: pq.ParquetFile, requested: Sequence[str]) -> list[str]:
    available = parquet_file.schema_arrow.names
    if not requested:
        return available
    missing = [column for column in requested if column not in available]
    if missing:
        raise CommandError(f"unknown columns: {', '.join(missing)}")
    return list(requested)


def _projected_schema(schema: pa.Schema, columns: Sequence[str]) -> pa.Schema:
    return pa.schema([schema.field(column) for column in columns])


def _iter_batches(
    paths: Sequence[Path], requested: Sequence[str], head: int
) -> tuple[list[str], Iterator[pa.RecordBatch]]:
    first_file = pq.ParquetFile(paths[0])
    columns = _selected_columns(first_file, requested)
    expected_schema = _projected_schema(first_file.schema_arrow, columns)

    def batches() -> Iterator[pa.RecordBatch]:
        remaining = head
        for path in paths:
            parquet_file = pq.ParquetFile(path)
            current_columns = _selected_columns(parquet_file, requested)
            current_schema = _projected_schema(parquet_file.schema_arrow, current_columns)
            if not current_schema.equals(expected_schema, check_metadata=False):
                raise CommandError(f"selected schema does not match the first file: {path}")
            if remaining == 0:
                break
            batch_size = min(65_536, remaining) if remaining > 0 else 65_536
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
                if remaining >= 0 and batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)
                yield batch
                if remaining >= 0:
                    remaining -= batch.num_rows
                    if remaining == 0:
                        break

    return columns, batches()


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def _render_psql(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row, strict=True)]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def render_row(row: Sequence[str]) -> str:
        cells = (value.ljust(width) for value, width in zip(row, widths, strict=True))
        return "| " + " | ".join(cells) + " |"

    separator = "|" + "+".join("-" * (width + 2) for width in widths) + "|"
    lines = [border, render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    lines.append(border)
    return "\n".join(lines)


def _render_github(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row, strict=True)]

    def render_row(row: Sequence[str]) -> str:
        cells = (value.ljust(width) for value, width in zip(row, widths, strict=True))
        return "| " + " | ".join(cells) + " |"

    lines = [render_row(headers), render_row(["-" * max(3, width) for width in widths])]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def _render_html(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["<table>", "  <thead>", "    <tr>"]
    lines.extend(f"      <th>{html.escape(header)}</th>" for header in headers)
    lines.extend(["    </tr>", "  </thead>", "  <tbody>"])
    for row in rows:
        lines.append("    <tr>")
        lines.extend(f"      <td>{html.escape(value)}</td>" for value in row)
        lines.append("    </tr>")
    lines.extend(["  </tbody>", "</table>"])
    return "\n".join(lines)


def show(paths: Sequence[Path], columns: Sequence[str], head: int, table_format: str) -> None:
    headers, batches = _iter_batches(paths, columns, head)
    rows = [
        [_display_value(record[column]) for column in headers]
        for batch in batches
        for record in batch.to_pylist()
    ]
    renderers = {"psql": _render_psql, "github": _render_github, "html": _render_html}
    renderer = renderers[table_format]
    print(renderer(headers, rows))


def write_csv(
    paths: Sequence[Path], columns: Sequence[str], head: int, output: TextIO | None = None
) -> None:
    output = output or sys.stdout
    headers, batches = _iter_batches(paths, columns, head)
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(headers)
    for batch in batches:
        for record in batch.to_pylist():
            writer.writerow([record[column] for column in headers])


def _compression_summary(metadata: pq.FileMetaData, column_index: int) -> str:
    if metadata.num_row_groups == 0:
        return "N/A"
    chunks = [
        metadata.row_group(index).column(column_index)
        for index in range(metadata.num_row_groups)
    ]
    codecs = ", ".join(dict.fromkeys(chunk.compression for chunk in chunks))
    uncompressed = sum(chunk.total_uncompressed_size for chunk in chunks)
    compressed = sum(chunk.total_compressed_size for chunk in chunks)
    if not uncompressed:
        return f"{codecs} (space_saved: N/A)"
    space_saved = (1 - compressed / uncompressed) * 100
    return f"{codecs} (space_saved: {space_saved:.0f}%)"


def inspect_file(path: Path, output: TextIO | None = None) -> None:
    output = output or sys.stdout
    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.metadata
    schema = parquet_file.schema
    print("############ file meta data ############", file=output)
    print(f"created_by: {metadata.created_by}", file=output)
    print(f"num_columns: {metadata.num_columns}", file=output)
    print(f"num_rows: {metadata.num_rows}", file=output)
    print(f"num_row_groups: {metadata.num_row_groups}", file=output)
    print(f"format_version: {metadata.format_version}", file=output)
    print(f"serialized_size: {metadata.serialized_size}", file=output)

    print("\n############ Columns ############", file=output)
    for index in range(len(schema)):
        print(schema.column(index).path, file=output)

    for index in range(len(schema)):
        column = schema.column(index)
        print(f"\n############ Column({column.path}) ############", file=output)
        print(f"name: {column.name}", file=output)
        print(f"path: {column.path}", file=output)
        print(f"max_definition_level: {column.max_definition_level}", file=output)
        print(f"max_repetition_level: {column.max_repetition_level}", file=output)
        print(f"physical_type: {column.physical_type}", file=output)
        print(f"logical_type: {column.logical_type}", file=output)
        print(f"converted_type (legacy): {column.converted_type}", file=output)
        print(f"compression: {_compression_summary(metadata, index)}", file=output)

    print("\n############ Row Groups ############", file=output)
    for index in range(metadata.num_row_groups):
        row_group = metadata.row_group(index)
        print(f"row_group {index}:", file=output)
        print(f"  num_rows: {row_group.num_rows}", file=output)
        print(f"  total_byte_size: {row_group.total_byte_size}", file=output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parquet-tools", description="Inspect local Parquet files"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_parser = subparsers.add_parser("show", help="Show data as a human-readable table")
    show_parser.add_argument("file", metavar="FILE", nargs="+")
    show_parser.add_argument(
        "--format", "-f", choices=("psql", "github", "html"), default="psql"
    )
    show_parser.add_argument("--columns", "-c", action="append", type=_columns)
    show_parser.add_argument("--head", "-n", type=_head, default=-1)

    csv_parser = subparsers.add_parser("csv", help="Write data as CSV")
    csv_parser.add_argument("file", metavar="FILE", nargs="+")
    csv_parser.add_argument("--columns", "-c", action="append", type=_columns)
    csv_parser.add_argument("--head", "-n", type=_head, default=-1)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect file metadata and schema")
    inspect_parser.add_argument("file", metavar="FILE")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            inspect_file(_expand_files([args.file])[0])
        else:
            paths = _expand_files(args.file)
            columns = _flatten_columns(args.columns)
            if args.command == "show":
                show(paths, columns, args.head, args.format)
            else:
                write_csv(paths, columns, args.head)
    except (CommandError, OSError, pa.ArrowException) as error:
        print(f"parquet-tools: error: {error}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
