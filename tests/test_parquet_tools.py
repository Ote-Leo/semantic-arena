from __future__ import annotations

import csv
import io
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
    assert captured.out.startswith("<table>\n  <thead>")
    assert "<th>name&lt;unsafe&gt;</th>" in captured.out
    assert "<td>A &amp; &lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;</td>" in captured.out
    assert "<script>" not in captured.out
    assert captured.out.rstrip().endswith("</table>")
    assert captured.err == ""


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
    assert result == 0
    assert "num_columns: 3" in captured.out
    assert "num_rows: 3" in captured.out
    assert "num_row_groups: 2" in captured.out
    assert "############ Column(name) ############" in captured.out
    assert "physical_type: BYTE_ARRAY" in captured.out
    assert "compression: SNAPPY" in captured.out
    assert "row_group 1:" in captured.out
    assert captured.err == ""


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
