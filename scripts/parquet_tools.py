from __future__ import annotations

import argparse
import csv
import glob
import html
import json
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


def _row_number(value: str) -> int:
    try:
        row = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if row < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return row


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


_HTML_STYLE = """
:root { color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
body { margin: 0; padding: 1.5rem; background: #f4f6f8; color: #18212b; }
h1 { margin: 0 0 1rem; font-size: 1.25rem; }
.table-scroll { overflow: auto; max-height: calc(100vh - 5rem); border: 1px solid #c7d0d9; }
table { border-collapse: collapse; background: #fff; }
table.data { width: max-content; min-width: 100%; }
th, td { padding: .55rem .7rem; border: 1px solid #d8dee4; text-align: left; vertical-align: top; }
table.data > thead > tr > th { position: sticky; top: 0; z-index: 1; background: #e9eef3; }
table.data > tbody > tr:nth-child(even) > td { background: #f8fafb; }
td { max-width: 42rem; overflow-wrap: anywhere; }
table.nested { width: 100%; margin-top: .4rem; font-size: .9rem; }
table.nested th { white-space: nowrap; background: #eef3f7; }
details { min-width: 14rem; }
summary { cursor: pointer; color: #275d85; font-weight: 600; }
.null, .empty { color: #6b7280; font-style: italic; }
.scalar { white-space: pre-wrap; }
@media (prefers-color-scheme: dark) {
  body { background: #11161c; color: #e6edf3; }
  table { background: #161b22; }
  .table-scroll { border-color: #3d444d; }
  th, td { border-color: #3d444d; }
  table.data > thead > tr > th, table.nested th { background: #222a33; }
  table.data > tbody > tr:nth-child(even) > td { background: #1b222b; }
  summary { color: #79b8e8; }
}
""".strip()


def _decoded_json(value: Any) -> Any:
    if not isinstance(value, str) or not value.lstrip().startswith(("{", "[")):
        return value
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return value
    return decoded if isinstance(decoded, (dict, list)) else value


def _html_scalar(value: Any) -> str:
    if value is None:
        return '<span class="null">null</span>'
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, bytes):
        return html.escape(value.hex())
    return f'<span class="scalar">{html.escape(str(value))}</span>'


def _html_collection(value: dict[str, Any] | list[Any], depth: int) -> str:
    if not value:
        label = "{}" if isinstance(value, dict) else "[]"
        return f'<span class="empty">{label}</span>'

    open_attribute = " open" if depth == 0 else ""
    if isinstance(value, dict):
        rows = "".join(
            "<tr>"
            f"<th>{html.escape(str(key))}</th>"
            f"<td>{_html_value(item, expand_json=True, depth=depth + 1)}</td>"
            "</tr>"
            for key, item in value.items()
        )
        return (
            f'<details{open_attribute}><summary>object ({len(value)} fields)</summary>'
            f'<table class="nested"><tbody>{rows}</tbody></table></details>'
        )

    if all(isinstance(item, dict) for item in value):
        keys = list(dict.fromkeys(key for item in value for key in item))
        header = "<th>#</th>" + "".join(f"<th>{html.escape(str(key))}</th>" for key in keys)
        rows = "".join(
            "<tr>"
            f"<th>{index}</th>"
            + "".join(
                f"<td>{_html_value(item.get(key), expand_json=True, depth=depth + 1)}</td>"
                for key in keys
            )
            + "</tr>"
            for index, item in enumerate(value)
        )
        table = (
            '<table class="nested">'
            f"<thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"
        )
    else:
        rows = "".join(
            "<tr>"
            f"<th>{index}</th>"
            f"<td>{_html_value(item, expand_json=True, depth=depth + 1)}</td>"
            "</tr>"
            for index, item in enumerate(value)
        )
        table = f'<table class="nested"><tbody>{rows}</tbody></table>'
    return (
        f'<details{open_attribute}><summary>array ({len(value)} items)</summary>'
        f"{table}</details>"
    )


def _html_value(value: Any, *, expand_json: bool, depth: int = 0) -> str:
    if expand_json:
        value = _decoded_json(value)
    if isinstance(value, (dict, list)):
        return _html_collection(value, depth)
    return _html_scalar(value)


def _html_document(body: str, title: str) -> str:
    escaped_title = html.escape(title)
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{escaped_title}</title>",
            f"<style>{_HTML_STYLE}</style>",
            "</head>",
            "<body>",
            f"<h1>{escaped_title}</h1>",
            body,
            "</body>",
            "</html>",
        ]
    )


def _render_html(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    expand_json: bool,
    title: str = "Parquet data",
) -> str:
    header = "".join(f"<th>{html.escape(column)}</th>" for column in headers)
    body_rows = "".join(
        "<tr>"
        + "".join(f"<td>{_html_value(value, expand_json=expand_json)}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    table = (
        '<div class="table-scroll"><table class="data">'
        f"<thead><tr>{header}</tr></thead><tbody>{body_rows}</tbody>"
        "</table></div>"
    )
    return _html_document(table, title)


def show(
    paths: Sequence[Path],
    columns: Sequence[str],
    head: int,
    table_format: str,
    expand_json: bool = False,
) -> None:
    headers, batches = _iter_batches(paths, columns, head)
    records = [
        record
        for batch in batches
        for record in batch.to_pylist()
    ]
    if table_format == "html":
        rows = [[record[column] for column in headers] for record in records]
        print(_render_html(headers, rows, expand_json=expand_json))
        return
    if expand_json:
        raise CommandError("--expand-json requires --format html")
    rows = [[_display_value(record[column]) for column in headers] for record in records]
    renderer = _render_psql if table_format == "psql" else _render_github
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


def _record_at(parquet_file: pq.ParquetFile, row: int) -> dict[str, Any]:
    remaining = row
    for batch in parquet_file.iter_batches(batch_size=1_024):
        if remaining < batch.num_rows:
            return batch.slice(remaining, 1).to_pylist()[0]
        remaining -= batch.num_rows
    raise CommandError(f"row index out of range: {row}")


def _find_record_by_row(paths: Sequence[Path], row: int) -> dict[str, Any]:
    remaining = row
    total_rows = 0
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        file_rows = parquet_file.metadata.num_rows
        total_rows += file_rows
        if remaining < file_rows:
            return _record_at(parquet_file, remaining)
        remaining -= file_rows
    raise CommandError(f"row index {row} is out of range for {total_rows} rows")


def _find_record_by_id(
    paths: Sequence[Path], identifier: str, id_column: str
) -> dict[str, Any]:
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        _selected_columns(parquet_file, [id_column])
        row_offset = 0
        for batch in parquet_file.iter_batches(batch_size=1_024, columns=[id_column]):
            for index, value in enumerate(batch.column(0).to_pylist()):
                if str(value) == identifier:
                    return _record_at(parquet_file, row_offset + index)
            row_offset += batch.num_rows
    raise CommandError(f"record not found where {id_column}={identifier!r}")


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, bytes):
        return value.hex()
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _expanded_record(record: dict[str, Any]) -> dict[str, Any]:
    return {column: _json_compatible(_decoded_json(value)) for column, value in record.items()}


def _render_record_html(record: dict[str, Any], title: str) -> str:
    rows = "".join(
        "<tr>"
        f"<th>{html.escape(column)}</th>"
        f"<td>{_html_value(value, expand_json=True)}</td>"
        "</tr>"
        for column, value in record.items()
    )
    table = (
        '<div class="table-scroll"><table class="data record">'
        f"<tbody>{rows}</tbody></table></div>"
    )
    return _html_document(table, title)


def show_record(
    paths: Sequence[Path],
    *,
    row: int | None,
    identifier: str | None,
    id_column: str,
    output_format: str,
    output: TextIO | None = None,
) -> None:
    output = output or sys.stdout
    if identifier is None:
        selected_row = 0 if row is None else row
        record = _find_record_by_row(paths, selected_row)
        title = f"Parquet record {selected_row}"
    else:
        record = _find_record_by_id(paths, identifier, id_column)
        title = f"Parquet record {identifier}"
    expanded = _expanded_record(record)
    if output_format == "json":
        json.dump(expanded, output, indent=2, ensure_ascii=False)
        print(file=output)
    else:
        print(_render_record_html(expanded, title), file=output)


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


def _decoded_metadata(metadata: dict[bytes, bytes] | None) -> dict[str, str]:
    if not metadata:
        return {}
    return {
        key.decode("utf-8", errors="backslashreplace"): value.decode(
            "utf-8", errors="backslashreplace"
        )
        for key, value in metadata.items()
    }


def _field_description(field: pa.Field) -> dict[str, Any]:
    description: dict[str, Any] = {
        "name": field.name,
        "type": str(field.type),
        "nullable": field.nullable,
    }
    metadata = _decoded_metadata(field.metadata)
    if metadata:
        description["metadata"] = metadata
    child_count = getattr(field.type, "num_fields", 0)
    if child_count:
        description["children"] = [
            _field_description(field.type.field(index)) for index in range(child_count)
        ]
    return description


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.hex()
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _statistics_description(statistics: pq.Statistics | None) -> dict[str, Any] | None:
    if statistics is None:
        return None
    description: dict[str, Any] = {
        "null_count": statistics.null_count,
        "distinct_count": statistics.distinct_count,
        "num_values": statistics.num_values,
    }
    if statistics.has_min_max:
        description["min"] = _json_value(statistics.min)
        description["max"] = _json_value(statistics.max)
    return description


def _inspect_description(parquet_file: pq.ParquetFile) -> dict[str, Any]:
    metadata = parquet_file.metadata
    arrow_schema = parquet_file.schema_arrow
    parquet_schema = parquet_file.schema
    description: dict[str, Any] = {
        "file_metadata": {
            "created_by": metadata.created_by,
            "num_columns": metadata.num_columns,
            "num_rows": metadata.num_rows,
            "num_row_groups": metadata.num_row_groups,
            "format_version": metadata.format_version,
            "serialized_size": metadata.serialized_size,
            "key_value_metadata": _decoded_metadata(metadata.metadata),
        },
        "schema": {
            "fields": [_field_description(field) for field in arrow_schema],
            "metadata": _decoded_metadata(arrow_schema.metadata),
        },
        "parquet_columns": [],
        "row_groups": [],
    }
    for index in range(len(parquet_schema)):
        column = parquet_schema.column(index)
        description["parquet_columns"].append(
            {
                "name": column.name,
                "path": column.path,
                "max_definition_level": column.max_definition_level,
                "max_repetition_level": column.max_repetition_level,
                "physical_type": column.physical_type,
                "logical_type": str(column.logical_type),
                "converted_type": column.converted_type,
                "compression": _compression_summary(metadata, index),
            }
        )
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        row_group_description = {
            "index": row_group_index,
            "num_rows": row_group.num_rows,
            "total_byte_size": row_group.total_byte_size,
            "columns": [],
        }
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            row_group_description["columns"].append(
                {
                    "path": column.path_in_schema,
                    "physical_type": column.physical_type,
                    "num_values": column.num_values,
                    "compression": column.compression,
                    "encodings": list(column.encodings),
                    "total_compressed_size": column.total_compressed_size,
                    "total_uncompressed_size": column.total_uncompressed_size,
                    "statistics": _statistics_description(column.statistics),
                }
            )
        description["row_groups"].append(row_group_description)
    return description


def _inspect_text(parquet_file: pq.ParquetFile, output: TextIO) -> None:
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


def inspect_file(path: Path, output_format: str = "json", output: TextIO | None = None) -> None:
    output = output or sys.stdout
    parquet_file = pq.ParquetFile(path)
    if output_format == "json":
        json.dump(_inspect_description(parquet_file), output, indent=2, ensure_ascii=False)
        print(file=output)
    else:
        _inspect_text(parquet_file, output)


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
    show_parser.add_argument(
        "--expand-json",
        action="store_true",
        help="Render JSON strings as nested tables in HTML output",
    )

    csv_parser = subparsers.add_parser("csv", help="Write data as CSV")
    csv_parser.add_argument("file", metavar="FILE", nargs="+")
    csv_parser.add_argument("--columns", "-c", action="append", type=_columns)
    csv_parser.add_argument("--head", "-n", type=_head, default=-1)

    record_parser = subparsers.add_parser("record", help="Show one record with JSON expanded")
    record_parser.add_argument("file", metavar="FILE", nargs="+")
    selector = record_parser.add_mutually_exclusive_group()
    selector.add_argument("--row", type=_row_number, help="Zero-based row across all input files")
    selector.add_argument("--id", help="Value to find in the ID column")
    record_parser.add_argument("--id-column", default="id", help="ID column name (default: id)")
    record_parser.add_argument(
        "--format", "-f", choices=("json", "html"), default="json"
    )

    inspect_parser = subparsers.add_parser("inspect", help="Inspect file metadata and schema")
    inspect_parser.add_argument("file", metavar="FILE")
    inspect_parser.add_argument(
        "--format",
        "-f",
        choices=("json", "text"),
        default="json",
        help="Output format (default: json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            inspect_file(_expand_files([args.file])[0], args.format)
        elif args.command == "record":
            show_record(
                _expand_files(args.file),
                row=args.row,
                identifier=args.id,
                id_column=args.id_column,
                output_format=args.format,
            )
        else:
            paths = _expand_files(args.file)
            columns = _flatten_columns(args.columns)
            if args.command == "show":
                show(paths, columns, args.head, args.format, args.expand_json)
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
