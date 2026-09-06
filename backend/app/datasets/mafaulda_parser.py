"""TASK 1.5.2 — maps a MAFAULDA `relative_path` (as produced by TASK 1.5.1's
`docs/dataset_audit/file_inventory.csv`) to the recording's state and, where present,
its operating condition.

Naming convention actually observed under data/raw/mafaulda/ during this audit (see
docs/dataset_audit/recording_mapping.md for the full write-up — this is NOT copied from
MAFAULDA documentation, it is derived from the real folder listing):

    <state>/<filename>.csv              (state == "normal" — no condition sub-folder)
    <state>/<condition>/<filename>.csv   (every other state — one condition sub-folder)

The filename itself (e.g. "12.288.csv") is a plain decimal number. It encodes no state
information as far as this task determines — its meaning (likely a rotation-speed-related
value) is out of scope here; see later TASK 1.5.x. Nothing about CSV *content* is read.

This module expects POSIX-style paths (forward slashes), exactly what
`build_inventory()`/`file_inventory.csv` from TASK 1.5.1 already produce.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

UNKNOWN = "UNKNOWN"

# Exactly the four top-level folder names found under data/raw/mafaulda/ during the
# TASK 1.5.1 audit (docs/dataset_audit/file_inventory.csv) — NOT taken from MAFAULDA
# documentation. Anything else maps to UNKNOWN, never guessed.
KNOWN_STATES = frozenset(
    {
        "normal",
        "imbalance",
        "horizontal-misalignment",
        "vertical-misalignment",
    }
)


@dataclass(frozen=True)
class RecordingMapping:
    relative_path: str
    state: str
    condition: str | None


def parse_recording_state(relative_path: str) -> RecordingMapping:
    """Derives (state, condition) from a `relative_path` exactly as it appears in
    docs/dataset_audit/file_inventory.csv.

    Returns state=UNKNOWN (condition=None) for anything that does not match the
    exact two-level shape above with a recognized top-level state — never guessed,
    never silently dropped.
    """
    parts = PurePosixPath(relative_path).parts

    if len(parts) == 2:
        state_candidate, condition = parts[0], None
    elif len(parts) == 3:
        state_candidate, condition = parts[0], parts[1]
    else:
        return RecordingMapping(relative_path=relative_path, state=UNKNOWN, condition=None)

    if state_candidate not in KNOWN_STATES:
        return RecordingMapping(relative_path=relative_path, state=UNKNOWN, condition=None)

    return RecordingMapping(relative_path=relative_path, state=state_candidate, condition=condition)
