"""TASK 1.5.1 — recursively inventories the MAFAULDA dataset under data/raw/mafaulda/.

Filesystem inventory only: no CSV parsing, no channel/class interpretation, no
train/val/test split, no windowing. Those belong to TASK 1.5.2 and later. This
script never modifies data/raw/mafaulda/ — it only reads file metadata.

Usage (from the repository root):
    uv run python backend/scripts/audit_inventory.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "data" / "raw" / "mafaulda"
OUTPUT_CSV = REPO_ROOT / "docs" / "dataset_audit" / "file_inventory.csv"

FIELDNAMES = ["relative_path", "filename", "extension", "size_bytes"]


def build_inventory(dataset_root: Path) -> list[dict[str, str | int]]:
    """Recursively lists every file under `dataset_root`, sorted by relative path.

    Raises FileNotFoundError if `dataset_root` does not exist. Returns an empty
    list if it exists but contains no files — the caller decides how to treat
    that (see `main`), since "missing" and "empty" are different failure modes.
    """
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    rows: list[dict[str, str | int]] = []
    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        rows.append(
            {
                "relative_path": path.relative_to(dataset_root).as_posix(),
                "filename": path.name,
                "extension": path.suffix,
                "size_bytes": path.stat().st_size,
            }
        )

    rows.sort(key=lambda row: row["relative_path"])
    return rows


def find_duplicate_relative_paths(rows: list[dict[str, str | int]]) -> list[str]:
    """Returns any `relative_path` values that appear more than once, sorted."""
    counts = Counter(row["relative_path"] for row in rows)
    return sorted(path for path, count in counts.items() if count > 1)


def write_inventory_csv(rows: list[dict[str, str | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    try:
        rows = build_inventory(DATASET_ROOT)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print(f"ERROR: No MAFAULDA dataset files found in {DATASET_ROOT}", file=sys.stderr)
        return 1

    duplicates = find_duplicate_relative_paths(rows)
    if duplicates:
        print(f"ERROR: duplicate relative_path entries found: {duplicates}", file=sys.stderr)
        return 1

    write_inventory_csv(rows, OUTPUT_CSV)
    print(f"Wrote {len(rows)} file(s) to {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
