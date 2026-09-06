"""TASK 1.5.5 — replication/condition-group analysis.

Group definition (per TASK 1.5.5's own spec): (state, approximate rotation frequency).
This intentionally does NOT include the severity/condition sub-folder (e.g. "0.5mm",
"10g") — see docs/dataset_audit/replication_analysis.md for why that matters: a group
here can span multiple different severities that happen to share a target speed, which
is a different thing from "the same physical setup recorded more than once". A finer,
severity-aware grouping (`build_condition_groups_with_severity`) is provided
specifically to check that distinction against the real data, not to replace the
task-defined grouping.

Tolerance: exact match, zero tolerance. Justified empirically (see
docs/dataset_audit/replication_analysis.md): the minimum gap between any two distinct
`rotation_frequency_hz` values in the real dataset is 0.2048 Hz, with zero pairs closer
than 0.01 Hz — values are already cleanly discrete, so no fuzzy/tolerance-based
bucketing is needed or justified for this dataset as currently inventoried.

`UNKNOWN_ROTATION_KEY` groups recordings whose rotation_source == "unavailable"
separately, per state — never dropped, never guessed. It happens to be unreachable for
this dataset (TASK 1.5.4 found rotation_source="filename" for all 880 files), but the
code path is real and tested.

Reads only docs/dataset_audit/file_inventory.csv (via class_distribution_audit) — never
data/raw/mafaulda/ content, never modifies anything.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from app.datasets.class_distribution_audit import RotationInfo, extract_rotation_info
from app.datasets.mafaulda_parser import parse_recording_state

UNKNOWN_ROTATION_KEY = "UNKNOWN_ROTATION"

GroupKey = tuple[str, "float | str"]


@dataclass(frozen=True)
class ConditionGroup:
    state: str
    rotation_frequency_hz: float | None  # None for the UNKNOWN_ROTATION bucket
    relative_paths: tuple[str, ...]
    rotation_sources: tuple[str, ...]  # distinct sources observed within this group

    @property
    def file_count(self) -> int:
        return len(self.relative_paths)

    @property
    def is_replicated(self) -> bool:
        return self.file_count >= 2


def group_key_for(info: RotationInfo) -> GroupKey:
    if info.rotation_source == "unavailable":
        return (info.state, UNKNOWN_ROTATION_KEY)
    return (info.state, info.rotation_frequency_hz)


def build_condition_groups(rotation_infos: list[RotationInfo]) -> list[ConditionGroup]:
    """Groups by (state, exact rotation_frequency_hz) — the literal TASK 1.5.5
    definition, ignoring severity/condition. See module docstring."""
    buckets: dict[GroupKey, list[RotationInfo]] = defaultdict(list)
    for info in rotation_infos:
        buckets[group_key_for(info)].append(info)

    groups = []
    for (state, freq_or_unknown), infos in buckets.items():
        is_unknown = freq_or_unknown == UNKNOWN_ROTATION_KEY
        groups.append(
            ConditionGroup(
                state=state,
                rotation_frequency_hz=None if is_unknown else freq_or_unknown,
                relative_paths=tuple(sorted(info.relative_path for info in infos)),
                rotation_sources=tuple(sorted({info.rotation_source for info in infos})),
            )
        )

    return sorted(
        groups,
        key=lambda g: (g.state, g.rotation_frequency_hz if g.rotation_frequency_hz is not None else float("inf")),
    )


def summarize(groups: list[ConditionGroup]) -> dict[str, int]:
    return {
        "total_groups": len(groups),
        "replicated_groups": sum(1 for g in groups if g.is_replicated),
        "singleton_groups": sum(1 for g in groups if not g.is_replicated),
        "total_recordings": sum(g.file_count for g in groups),
    }


def build_condition_groups_with_severity(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str | None, "float | str"], list[str]]:
    """Finer grouping — (state, condition/severity, rotation_frequency_hz) — used only
    to check whether the coarser `build_condition_groups` "replication" signal
    reflects repeated identical physical setups, or just a shared target speed across
    different severities. See docs/dataset_audit/replication_analysis.md."""
    buckets: dict[tuple, list[str]] = defaultdict(list)
    for row in rows:
        mapping = parse_recording_state(row["relative_path"])
        info = extract_rotation_info(row["relative_path"], mapping.state)
        freq_key = UNKNOWN_ROTATION_KEY if info.rotation_source == "unavailable" else info.rotation_frequency_hz
        buckets[(mapping.state, mapping.condition, freq_key)].append(row["relative_path"])
    return buckets


def find_same_size_duplicates(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    """Cheap, sufficient check for potential byte-identical files: groups
    relative_paths by size_bytes (a necessary precondition for byte-identical content).
    Never reads/hashes file content — see docs/dataset_audit/replication_analysis.md
    for why that's unnecessary for this dataset (every file already has a unique
    size_bytes, which alone rules out any byte-for-byte duplicate)."""
    by_size: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_size[row["size_bytes"]].append(row["relative_path"])
    return {size: paths for size, paths in by_size.items() if len(paths) > 1}
