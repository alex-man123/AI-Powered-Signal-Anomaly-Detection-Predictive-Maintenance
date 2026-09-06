"""TASK 1.5.3 — empirical inspection of MAFAULDA signal structure (columns, dtypes,
row counts) for a small, deterministic sample of real files.

Read-only: only reads file content for inspection, never writes to
data/raw/mafaulda/. No resampling, filtering, feature extraction, or splitting here —
this is audit-only, feeding docs/dataset_audit/signal_structure.md.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPO_ROOT / "data" / "raw" / "mafaulda"
INVENTORY_CSV = REPO_ROOT / "docs" / "dataset_audit" / "file_inventory.csv"


@dataclass(frozen=True)
class FileStructure:
    relative_path: str
    state: str
    rows: int
    columns: int
    column_dtypes: tuple[str, ...]
    has_missing_values: bool


def select_audit_files() -> list[tuple[str, str]]:
    """Deterministically selects one representative file per distinct state.

    Uses the first occurrence of each state in file_inventory.csv (TASK 1.5.1's own
    deterministic, relative_path-sorted ordering) — not a manual/arbitrary pick.
    Returns a list of (relative_path, state) pairs, sorted by state name.
    """
    with INVENTORY_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    selected: dict[str, str] = {}
    for row in rows:
        state = row["relative_path"].split("/")[0]
        if state not in selected:
            selected[state] = row["relative_path"]

    return [(path, state) for state, path in sorted(selected.items())]


def inspect_file(relative_path: str, state: str) -> FileStructure:
    """Reads one real MAFAULDA file with pandas and reports its observed structure.

    The files have no header row (verified during this audit), so columns are read
    positionally (header=None) — they are never renamed or assigned assumed sensor names.
    """
    df = pd.read_csv(DATASET_ROOT / relative_path, header=None)
    return FileStructure(
        relative_path=relative_path,
        state=state,
        rows=df.shape[0],
        columns=df.shape[1],
        column_dtypes=tuple(str(dtype) for dtype in df.dtypes),
        has_missing_values=bool(df.isna().any().any()),
    )
