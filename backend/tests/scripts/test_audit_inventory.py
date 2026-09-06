import csv
from pathlib import Path

from scripts import audit_inventory
from scripts.audit_inventory import build_inventory, find_duplicate_relative_paths, write_inventory_csv


def _make_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_build_inventory_matches_created_files(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tmp_dataset"
    _make_file(dataset_root / "class_a" / "signal1.csv", b"a,b\n1,2\n")
    _make_file(dataset_root / "class_a" / "signal2.csv", b"a,b\n3,4\n")
    _make_file(dataset_root / "class_b" / "signal3.txt", b"not a csv")

    rows = build_inventory(dataset_root)

    assert len(rows) == 3
    assert {row["relative_path"] for row in rows} == {
        "class_a/signal1.csv",
        "class_a/signal2.csv",
        "class_b/signal3.txt",
    }

    signal1 = next(row for row in rows if row["relative_path"] == "class_a/signal1.csv")
    assert signal1["filename"] == "signal1.csv"
    assert signal1["extension"] == ".csv"
    assert signal1["size_bytes"] == len(b"a,b\n1,2\n")

    signal3 = next(row for row in rows if row["relative_path"] == "class_b/signal3.txt")
    assert signal3["extension"] == ".txt"


def test_build_inventory_is_deterministically_ordered(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tmp_dataset"
    _make_file(dataset_root / "class_b" / "z.csv", b"z")
    _make_file(dataset_root / "class_a" / "a.csv", b"a")

    rows = build_inventory(dataset_root)

    assert [row["relative_path"] for row in rows] == ["class_a/a.csv", "class_b/z.csv"]
    # Re-running against the same filesystem layout must produce the same order.
    assert [row["relative_path"] for row in build_inventory(dataset_root)] == [
        "class_a/a.csv",
        "class_b/z.csv",
    ]


def test_build_inventory_raises_on_missing_root(tmp_path: Path) -> None:
    missing_root = tmp_path / "does_not_exist"

    try:
        build_inventory(missing_root)
        raised = False
    except FileNotFoundError:
        raised = True

    assert raised


def test_build_inventory_returns_empty_list_for_empty_root(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty_dataset"
    empty_root.mkdir()

    assert build_inventory(empty_root) == []


def test_find_duplicate_relative_paths_detects_duplicates() -> None:
    rows = [
        {"relative_path": "a.csv"},
        {"relative_path": "b.csv"},
        {"relative_path": "a.csv"},
    ]

    assert find_duplicate_relative_paths(rows) == ["a.csv"]


def test_find_duplicate_relative_paths_empty_when_unique() -> None:
    rows = [{"relative_path": "a.csv"}, {"relative_path": "b.csv"}]

    assert find_duplicate_relative_paths(rows) == []


def test_write_inventory_csv_writes_expected_columns(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tmp_dataset"
    _make_file(dataset_root / "class_a" / "signal1.csv", b"abc")

    rows = build_inventory(dataset_root)
    output_csv = tmp_path / "out" / "file_inventory.csv"
    write_inventory_csv(rows, output_csv)

    with output_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["relative_path", "filename", "extension", "size_bytes"]
        written_rows = list(reader)

    assert len(written_rows) == 1
    assert written_rows[0]["relative_path"] == "class_a/signal1.csv"
    assert written_rows[0]["size_bytes"] == "3"


def test_main_fails_with_nonzero_exit_when_dataset_root_missing(tmp_path: Path, monkeypatch) -> None:
    missing_root = tmp_path / "does_not_exist"
    output_csv = tmp_path / "out" / "file_inventory.csv"
    monkeypatch.setattr(audit_inventory, "DATASET_ROOT", missing_root)
    monkeypatch.setattr(audit_inventory, "OUTPUT_CSV", output_csv)

    exit_code = audit_inventory.main()

    assert exit_code != 0
    assert not output_csv.exists()


def test_main_fails_with_nonzero_exit_when_dataset_root_empty(tmp_path: Path, monkeypatch) -> None:
    empty_root = tmp_path / "empty_dataset"
    empty_root.mkdir()
    output_csv = tmp_path / "out" / "file_inventory.csv"
    monkeypatch.setattr(audit_inventory, "DATASET_ROOT", empty_root)
    monkeypatch.setattr(audit_inventory, "OUTPUT_CSV", output_csv)

    exit_code = audit_inventory.main()

    assert exit_code != 0
    assert not output_csv.exists()


def test_main_writes_csv_and_returns_zero_for_populated_root(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "tmp_dataset"
    _make_file(dataset_root / "class_a" / "signal1.csv", b"abc")
    _make_file(dataset_root / "class_b" / "signal2.csv", b"defgh")
    output_csv = tmp_path / "out" / "file_inventory.csv"
    monkeypatch.setattr(audit_inventory, "DATASET_ROOT", dataset_root)
    monkeypatch.setattr(audit_inventory, "OUTPUT_CSV", output_csv)

    exit_code = audit_inventory.main()

    assert exit_code == 0
    assert output_csv.exists()

    with output_csv.open(newline="", encoding="utf-8") as f:
        written_rows = list(csv.DictReader(f))
    assert len(written_rows) == 2
