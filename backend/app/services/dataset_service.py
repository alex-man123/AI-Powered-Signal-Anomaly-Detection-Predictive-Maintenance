"""TASK 10.2 -- derives real dataset metadata for the API layer from the
project's ALREADY-ESTABLISHED sources of truth:

- `data/processed/split_manifest.json` (TASK 1.5.7/2.2's own sole source of truth
  for which real files exist and how many there are) -- for `signal_count`.
- `app.datasets.mafaulda_parser.parse_recording_state` (TASK 1.5.x -- pure string
  parsing of a relative path, no file I/O) -- for the real, distinct label set.
- `app.datasets.validators`'s audit-confirmed constants (TASK 2.3):
  `SAMPLING_RATE_HZ`, `NUM_CHANNELS`, `MINIMUM_SIGNAL_LENGTH` -- for
  `sampling_rate`/`channels`/`samples_per_signal`.

Deliberately does NOT query `app.models.signal`'s `Dataset`/`Signal` SQLAlchemy
tables: TASK 2.2's `loader.py` module docstring states explicitly that it never
persists to those tables ("No DB persistence -- this module returns validated
Pydantic objects; writing them to the signals/datasets tables is a future
service's responsibility") -- confirmed directly against the project's real
database (`sqlite3.OperationalError: no such table: datasets`/`signals`), so
there is no populated table this service could honestly read metadata from.
Building a NEW ingestion pipeline to populate them would be scope creep well
beyond "implement two GET endpoints".

Deliberately does NOT call `app.datasets.loader.load_recording`/`load_split`/
`load_all_splits`: those functions read and structurally validate the ACTUAL
CSV content of every file they touch (`analyze_file_quality`) -- calling them
for all 880 real files just to answer a metadata request would mean reading
essentially the entire ~4 GB dataset on every `GET /api/datasets/{id}`, which
this task explicitly forbids ("Nu încărca inutil întregul dataset de ~4 GB doar
pentru endpoint").

Exactly ONE dataset currently exists in this project -- MAFAULDA (no second
dataset/ingestion pipeline exists yet; CWRU is unimplemented future Phase 13
work per blueprint.md section 33). `MAFAULDA_DATASET_ID = 1` mirrors
`app.datasets.loader.load_recording`'s own `dataset_id: int = 1` default (the id
this data would belong to were it ever persisted) -- not an invented numbering
scheme.

"samples" in backlog TASK 10.2's AC1 ("samples, sampling rate, channels") is
ambiguous between two equally real, audit-confirmed facts: how many recordings
this dataset has (880, from the split manifest) and how many raw time-domain
samples each recording contains (250,000 -- `MINIMUM_SIGNAL_LENGTH`, audit-
confirmed zero-variance across all 880 real files). Rather than silently picking
one reading, both are exposed: `signal_count` (recordings) and
`samples_per_signal` (samples per recording).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.datasets.mafaulda_parser import UNKNOWN, parse_recording_state
from app.datasets.validators import MINIMUM_SIGNAL_LENGTH, NUM_CHANNELS, SAMPLING_RATE_HZ
from app.models.signal import SignalLabel

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "split_manifest.json"

MAFAULDA_DATASET_ID = 1
MAFAULDA_DATASET_NAME = "mafaulda"


class DatasetServiceError(ValueError):
    """Raised when real dataset metadata cannot be honestly produced (e.g. a
    missing/corrupt split manifest, or a manifest file path that does not map
    to a known label) -- never silently replaced with a placeholder value."""


class DatasetNotFoundError(DatasetServiceError):
    """Raised for a dataset id this project does not have. The route layer
    (`app.api.routes.datasets`) converts this into HTTP 404, never a 500."""


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise DatasetServiceError(f"split manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as f:
        return json.load(f)


def _signal_count(splits: dict[str, list[str]]) -> int:
    return sum(len(relative_paths) for relative_paths in splits.values())


def _distinct_labels(splits: dict[str, list[str]]) -> list[SignalLabel]:
    labels: set[SignalLabel] = set()
    for relative_paths in splits.values():
        for relative_path in relative_paths:
            mapping = parse_recording_state(relative_path)
            if mapping.state == UNKNOWN:
                raise DatasetServiceError(
                    f"{relative_path!r} (from the split manifest) does not map to a "
                    "known SignalLabel -- refusing to report dataset metadata derived "
                    "from an unmapped file"
                )
            labels.add(SignalLabel(mapping.state))
    return sorted(labels, key=lambda label: label.value)


def list_datasets(*, manifest_path: Path = DEFAULT_MANIFEST_PATH) -> list[dict[str, Any]]:
    """Summary metadata for every dataset this project currently has --
    exactly one, MAFAULDA (see module docstring)."""
    manifest = _load_manifest(manifest_path)
    splits = manifest.get("splits", {})

    return [
        {
            "id": MAFAULDA_DATASET_ID,
            "name": MAFAULDA_DATASET_NAME,
            "signal_count": _signal_count(splits),
        }
    ]


def get_dataset_detail(dataset_id: int, *, manifest_path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """Real detail metadata for one dataset.

    Raises:
        DatasetNotFoundError: if `dataset_id` does not identify a real dataset
            this project has (currently, anything other than `MAFAULDA_DATASET_ID`).
    """
    if dataset_id != MAFAULDA_DATASET_ID:
        raise DatasetNotFoundError(f"dataset {dataset_id} does not exist")

    manifest = _load_manifest(manifest_path)
    splits = manifest.get("splits", {})

    return {
        "id": MAFAULDA_DATASET_ID,
        "name": MAFAULDA_DATASET_NAME,
        "signal_count": _signal_count(splits),
        "sampling_rate": SAMPLING_RATE_HZ,
        "channels": list(range(NUM_CHANNELS)),
        "samples_per_signal": MINIMUM_SIGNAL_LENGTH,
        "labels": _distinct_labels(splits),
    }
