"""TASK 1.5.4 — class distribution and rotation-frequency provenance audit.

Reads only `docs/dataset_audit/file_inventory.csv` (TASK 1.5.1) and reuses
`parse_recording_state` (TASK 1.5.2) — never re-scans or reads content from
data/raw/mafaulda/ directly, so this stays fast regardless of dataset size.

Rotation frequency provenance is intentionally strict: a value extracted from a
filename is always labeled `rotation_source="filename"`, never `"tachometer"` —
this project has no locally-processed tachometer channel, only a number embedded in
a filename. See docs/dataset_audit/class_distribution.md for the full write-up of why
that value is interpreted as Hz (not RPM) and is never presented as a direct physical
measurement.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from app.datasets.mafaulda_parser import UNKNOWN, parse_recording_state

REPO_ROOT = Path(__file__).resolve().parents[3]
INVENTORY_CSV = REPO_ROOT / "docs" / "dataset_audit" / "file_inventory.csv"

ROTATION_SOURCES = frozenset({"tachometer", "metadata", "filename", "unavailable"})

# AC2's exact threshold rule: a class below this fraction of the mean file count per
# class is flagged as severely underrepresented.
SEVERE_UNDERREPRESENTATION_THRESHOLD = 0.10


@dataclass(frozen=True)
class RotationInfo:
    relative_path: str
    state: str
    rotation_frequency_hz: float | None
    rotation_source: str
    evidence: str


@dataclass(frozen=True)
class ImbalanceAnalysis:
    mean_files_per_class: float
    percentage_of_mean: dict[str, float]
    severely_underrepresented: list[str]


def load_inventory_rows(inventory_csv: Path = INVENTORY_CSV) -> list[dict[str, str]]:
    with inventory_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_class_distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    """Counts files per state (including UNKNOWN, if any — never dropped silently)."""
    counts: dict[str, int] = {}
    for row in rows:
        state = parse_recording_state(row["relative_path"]).state
        counts[state] = counts.get(state, 0) + 1
    return counts


def analyze_imbalance(class_counts: dict[str, int]) -> ImbalanceAnalysis:
    """Applies the exact AC2 rule: class_count < 0.10 * mean_files_per_class ->
    severely underrepresented. Excludes UNKNOWN from the "class" population used to
    compute the mean — UNKNOWN is a data-quality bucket, not a state to balance against.
    """
    real_classes = {state: count for state, count in class_counts.items() if state != UNKNOWN}

    if not real_classes:
        return ImbalanceAnalysis(mean_files_per_class=0.0, percentage_of_mean={}, severely_underrepresented=[])

    mean_files_per_class = sum(real_classes.values()) / len(real_classes)
    percentage_of_mean = {
        state: (count / mean_files_per_class) * 100 if mean_files_per_class else 0.0
        for state, count in real_classes.items()
    }
    severely_underrepresented = sorted(
        state
        for state, count in real_classes.items()
        if count < SEVERE_UNDERREPRESENTATION_THRESHOLD * mean_files_per_class
    )

    return ImbalanceAnalysis(
        mean_files_per_class=mean_files_per_class,
        percentage_of_mean=percentage_of_mean,
        severely_underrepresented=severely_underrepresented,
    )


def extract_rotation_info(relative_path: str, state: str) -> RotationInfo:
    """Extracts a rotation-frequency value from the filename, with explicit provenance.

    This dataset has no header, no metadata file, and no locally-processed tachometer
    channel (see docs/dataset_audit/signal_structure.md) — so the only possible sources
    here are "filename" or "unavailable", never "tachometer"/"metadata" (those branches
    exist in the schema for completeness/future datasets, but are never reachable for
    MAFAULDA as currently inventoried).
    """
    filename = relative_path.rsplit("/", maxsplit=1)[-1]
    stem = filename[: -len(".csv")] if filename.endswith(".csv") else filename

    try:
        value = float(stem)
    except ValueError:
        return RotationInfo(
            relative_path=relative_path,
            state=state,
            rotation_frequency_hz=None,
            rotation_source="unavailable",
            evidence=f"filename '{filename}' does not parse as a decimal number",
        )

    return RotationInfo(
        relative_path=relative_path,
        state=state,
        rotation_frequency_hz=value,
        rotation_source="filename",
        evidence=f"filename '{filename}'",
    )


def build_rotation_audit(rows: list[dict[str, str]]) -> list[RotationInfo]:
    infos = []
    for row in rows:
        state = parse_recording_state(row["relative_path"]).state
        infos.append(extract_rotation_info(row["relative_path"], state))
    return infos
