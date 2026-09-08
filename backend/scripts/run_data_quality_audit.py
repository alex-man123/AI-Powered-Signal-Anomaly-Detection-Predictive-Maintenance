"""TASK 1.5.8 — runs the data-quality audit over every file in file_inventory.csv.

Read-only with respect to data/raw/mafaulda/: processes one file at a time via
app.datasets.data_quality.analyze_file_quality, discarding each DataFrame before moving
to the next. Prints a summary plus every problematic file to stdout (the source used to
write docs/dataset_audit/data_quality_report.md by hand) and, if --json is given,
writes the full per-file results there for inspection (not a committed deliverable).

Usage (from the repository root):
    uv run python backend/scripts/run_data_quality_audit.py [--json PATH]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.datasets.data_quality import STATUS_OK, analyze_file_quality  # noqa: E402

DATASET_ROOT = REPO_ROOT / "data" / "raw" / "mafaulda"
INVENTORY_CSV = REPO_ROOT / "docs" / "dataset_audit" / "file_inventory.csv"


def main() -> int:
    json_out = None
    if "--json" in sys.argv:
        json_out = Path(sys.argv[sys.argv.index("--json") + 1])

    with INVENTORY_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    reports = []
    for i, row in enumerate(rows):
        relative_path = row["relative_path"]
        report = analyze_file_quality(DATASET_ROOT / relative_path, relative_path)
        reports.append(report)
        if (i + 1) % 100 == 0:
            print(f"...{i + 1}/{len(rows)} files analyzed", file=sys.stderr)

    status_counts: dict[str, int] = {}
    for r in reports:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    missing_value_files = [r for r in reports if r.status == STATUS_OK and r.has_missing_values]
    constant_files = [r for r in reports if r.status == STATUS_OK and r.constant_channels]
    zero_files = [r for r in reports if r.status == STATUS_OK and r.zero_channels]
    error_files = [r for r in reports if r.status != STATUS_OK]

    print(f"Total files in inventory: {len(rows)}")
    print(f"Total files analyzed: {len(reports)}")
    print(f"Status counts: {status_counts}")
    print(f"Files with missing values: {len(missing_value_files)}")
    print(f"Files with >=1 constant (non-zero) channel: {len(constant_files)}")
    print(f"Files with >=1 zero-signal channel: {len(zero_files)}")
    print(f"Files with read/structure/non-numeric errors: {len(error_files)}")

    print()
    print("=== problematic files (missing values) ===")
    for r in missing_value_files:
        print(f"{r.relative_path}: missing={r.missing_values} ({r.missing_percentage:.4f}%)")

    print()
    print("=== problematic files (constant non-zero channel) ===")
    for r in constant_files:
        cols = [(c.column, c.min) for c in r.constant_channels]
        print(f"{r.relative_path}: constant columns={cols}")

    print()
    print("=== problematic files (zero-signal channel) ===")
    for r in zero_files:
        cols = [c.column for c in r.zero_channels]
        print(f"{r.relative_path}: zero columns={cols}")

    print()
    print("=== read/structure/non-numeric errors ===")
    for r in error_files:
        print(f"{r.relative_path}: status={r.status} error={r.error_message}")

    # column-count consistency (bonus check, TASK 1.5.3-adjacent, "other observed issues")
    column_counts = {r.columns for r in reports if r.status == STATUS_OK}
    print()
    print(f"=== distinct column counts observed across all analyzed files: {column_counts} ===")

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        serializable = [
            {
                "relative_path": r.relative_path,
                "status": r.status,
                "rows": r.rows,
                "columns": r.columns,
                "missing_values": r.missing_values,
                "missing_percentage": r.missing_percentage,
                "error_message": r.error_message,
                "channels": [
                    {"column": c.column, "min": c.min, "max": c.max, "std": c.std, "classification": c.classification}
                    for c in r.channels
                ],
            }
            for r in reports
        ]
        json_out.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        print(f"\nWrote full per-file results to {json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
