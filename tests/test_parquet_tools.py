from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.parquet_tools import main


@pytest.fixture
def parquet_file(tmp_path: Path) -> Path:
    path = tmp_path / "records.parquet"
    table = pa.table(
        {
            "id": [1, 2, 3],
            "name": ["alpha", "comma,value", None],
            "active": [True, False, True],
        }
    )
    pq.write_table(table, path, row_group_size=2, compression="snappy")
    return path


@pytest.fixture
def json_parquet_file(tmp_path: Path) -> Path:
    path = tmp_path / "json-records.parquet"
    source_data = [
        json.dumps(
            {
                "trialStatus": "UPDATED",
                "contacts": [{"name": "Alice & Bob", "role": "owner"}],
            }
        ),
        json.dumps({"trialStatus": "REGISTERED", "contacts": []}),
    ]
    pq.write_table(
        pa.table({"id": ["DRKS00000001", "DRKS00000002"], "source_data": source_data}),
        path,
    )
    return path


def test_show_filters_columns_and_rows(
    parquet_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(
        ["show", "--format", "github", "--columns", "name,active", "-n", "2", str(parquet_file)]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "| name" in captured.out
    assert "active" in captured.out
    assert "alpha" in captured.out
    assert "comma,value" in captured.out
    assert "id" not in captured.out
    assert captured.err == ""


def test_show_outputs_escaped_html(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "html.parquet"
    pq.write_table(pa.table({"name<unsafe>": ["A & <script>alert('x')</script>"]}), path)

    result = main(["show", "--format", "html", str(path)])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out.startswith("<!doctype html>\n<html")
    assert '<table class="data">' in captured.out
    assert "<th>name&lt;unsafe&gt;</th>" in captured.out
    assert "A &amp; &lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in captured.out
    assert "<script>" not in captured.out
    assert captured.out.rstrip().endswith("</html>")
    assert captured.err == ""


def test_show_expands_json_as_nested_html_tables(
    json_parquet_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(
        [
            "show",
            "--format",
            "html",
            "--expand-json",
            "--head",
            "1",
            str(json_parquet_file),
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert '<summary>object (2 fields)</summary>' in captured.out
    assert "<th>trialStatus</th>" in captured.out
    assert '<summary>array (1 items)</summary>' in captured.out
    assert "<th>name</th>" in captured.out
    assert "Alice &amp; Bob" in captured.out
    assert "&quot;trialStatus&quot;" not in captured.out
    assert captured.err == ""


def test_expand_json_requires_html(
    json_parquet_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(["show", "--expand-json", str(json_parquet_file)])

    captured = capsys.readouterr()
    assert result == 1
    assert "--expand-json requires --format html" in captured.err


def test_record_outputs_parsed_json(
    json_parquet_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(["record", "--row", "1", str(json_parquet_file)])

    captured = capsys.readouterr()
    record = json.loads(captured.out)
    assert result == 0
    assert record["id"] == "DRKS00000002"
    assert record["source_data"] == {"trialStatus": "REGISTERED", "contacts": []}
    assert captured.err == ""


def test_record_finds_id_and_outputs_nested_html(
    json_parquet_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(
        [
            "record",
            "--id",
            "DRKS00000001",
            "--format",
            "html",
            str(json_parquet_file),
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "<title>Parquet record DRKS00000001</title>" in captured.out
    assert "<th>source_data</th>" in captured.out
    assert "<th>trialStatus</th>" in captured.out
    assert "Alice &amp; Bob" in captured.out
    assert captured.err == ""


def test_record_reports_missing_id(
    json_parquet_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(["record", "--id", "missing", str(json_parquet_file)])

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert "record not found where id='missing'" in captured.err


def test_csv_writes_valid_escaped_output(
    parquet_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(["csv", "-c", "id,name", str(parquet_file)])

    captured = capsys.readouterr()
    assert result == 0
    assert list(csv.reader(io.StringIO(captured.out))) == [
        ["id", "name"],
        ["1", "alpha"],
        ["2", "comma,value"],
        ["3", ""],
    ]
    assert captured.err == ""


def test_head_applies_across_globbed_files(
    parquet_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    second_path = tmp_path / "second.parquet"
    pq.write_table(pq.read_table(parquet_file), second_path)

    result = main(["csv", "--head", "4", str(tmp_path / "*.parquet")])

    captured = capsys.readouterr()
    rows = list(csv.reader(io.StringIO(captured.out)))
    assert result == 0
    assert len(rows) == 5
    assert rows[0] == ["id", "name", "active"]


def test_inspect_reports_schema_and_row_groups(
    parquet_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(["inspect", str(parquet_file)])

    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert result == 0
    assert captured.out.startswith('{\n  "file_metadata"')
    assert document["file_metadata"]["num_columns"] == 3
    assert document["file_metadata"]["num_rows"] == 3
    assert document["file_metadata"]["num_row_groups"] == 2
    assert document["schema"]["fields"] == [
        {"name": "id", "type": "int64", "nullable": True},
        {"name": "name", "type": "string", "nullable": True},
        {"name": "active", "type": "bool", "nullable": True},
    ]
    assert document["parquet_columns"][1]["physical_type"] == "BYTE_ARRAY"
    assert document["parquet_columns"][1]["compression"].startswith("SNAPPY")
    assert document["row_groups"][1]["index"] == 1
    assert captured.err == ""


def test_inspect_supports_text_format(
    parquet_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(["inspect", "--format", "text", str(parquet_file)])

    captured = capsys.readouterr()
    assert result == 0
    assert "num_columns: 3" in captured.out
    assert "############ Column(name) ############" in captured.out
    assert "row_group 1:" in captured.out
    assert captured.err == ""


def test_inspect_json_describes_nested_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "nested.parquet"
    nested_type = pa.struct([pa.field("tags", pa.list_(pa.string()), nullable=False)])
    pq.write_table(pa.table({"record": pa.array([{"tags": ["one"]}], type=nested_type)}), path)

    result = main(["inspect", str(path)])

    captured = capsys.readouterr()
    field = json.loads(captured.out)["schema"]["fields"][0]
    assert result == 0
    assert field["name"] == "record"
    assert field["children"][0]["name"] == "tags"
    assert field["children"][0]["nullable"] is False
    assert field["children"][0]["children"][0]["name"] == "element"


def test_unknown_column_returns_error(
    parquet_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(["show", "--columns", "missing", str(parquet_file)])

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert "unknown columns: missing" in captured.err


def test_missing_file_returns_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = main(["inspect", str(tmp_path / "missing.parquet")])

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert "file not found" in captured.err
