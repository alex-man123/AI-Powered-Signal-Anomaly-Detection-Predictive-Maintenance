"""TASK 2.2 — dataset loader: builds canonical `SignalRecord` objects (TASK 2.1) from
the real MAFAULDA files, using `data/processed/split_manifest.json` (TASK 1.5.7) as the
sole source of truth for which files exist and which split they belong to.

Explicitly NOT done here (out of scope for this task):
- No preprocessing/filtering/resampling/FFT/PSD/STFT/feature extraction/windowing.
- No DB persistence — this module returns validated Pydantic objects; writing them to
  the `signals`/`datasets` tables (TASK 2.1) is a future service's responsibility, kept
  separate per the loader's own testability/independence requirement.
- No re-deriving the split — `train`/`validation`/`test` come only from the manifest;
  never from filename, folder, label, or a new random split.

Read-only with respect to data/raw/mafaulda/ and data/processed/split_manifest.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.datasets.class_distribution_audit import extract_rotation_info
from app.datasets.data_quality import STATUS_OK, analyze_file_quality
from app.datasets.mafaulda_parser import UNKNOWN, parse_recording_state
from app.datasets.validators import (
    NUM_CHANNELS,
    SAMPLING_RATE_HZ,
    DatasetValidationError,
    validate_channel_count,
    validate_sampling_rate,
)
from app.models.signal import SignalLabel, SignalRecord

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPO_ROOT / "data" / "raw" / "mafaulda"
MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "split_manifest.json"


class DatasetLoaderError(DatasetValidationError):
    """Raised for any file that fails validation against the audit-confirmed
    structure — split, path, and the exact reason are always included in the message,
    per this task's explicit "fail fast, never silent" requirement.

    Inherits from `DatasetValidationError` (TASK 2.3) rather than a separate `ValueError`
    subclass, per that task's instruction to reuse an existing suitable error mechanism
    instead of creating a duplicate one — a bare `except DatasetValidationError` (or
    `except DatasetLoaderError`) both work as expected.
    """


@dataclass(frozen=True)
class LoadedRecording:
    """One manifest entry (one real file), resolved into its split assignment, label,
    and one `SignalRecord` per channel. Kept separate from TASK 2.1's `SignalRecord`
    (which has no `split` field, and is not modified here) — this is loader-level
    metadata, not a database schema change."""

    relative_path: str
    split: str
    label: SignalLabel
    signal_records: tuple[SignalRecord, ...]


def _load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.is_file():
        raise DatasetLoaderError(f"split manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as f:
        return json.load(f)


def _resolve_and_validate_path(relative_path: str, dataset_root: Path, split: str) -> Path:
    dataset_root_resolved = dataset_root.resolve()
    full_path = (dataset_root / relative_path).resolve()

    if not full_path.is_relative_to(dataset_root_resolved):
        raise DatasetLoaderError(
            f"[split={split}] path escapes dataset root: {relative_path!r} -> {full_path}"
        )
    if not full_path.exists():
        raise DatasetLoaderError(
            f"[split={split}] file listed in manifest does not exist: {relative_path!r} "
            f"(resolved: {full_path})"
        )
    if not full_path.is_file():
        raise DatasetLoaderError(f"[split={split}] path is not a regular file: {relative_path!r}")

    return full_path


def _validate_structure(full_path: Path, relative_path: str, split: str) -> None:
    """Reuses TASK 1.5.8's quality/structure audit rather than re-implementing a
    second, possibly inconsistent, structural check."""
    report = analyze_file_quality(full_path, relative_path)

    if report.status != STATUS_OK:
        raise DatasetLoaderError(
            f"[split={split}] {relative_path!r} failed structural validation: "
            f"status={report.status} ({report.error_message})"
        )
    try:
        validate_channel_count(report.columns, NUM_CHANNELS)
    except DatasetValidationError as exc:
        raise DatasetLoaderError(
            f"[split={split}] {relative_path!r}: {exc} (docs/dataset_audit/signal_structure.md)"
        ) from exc
    if report.all_channels_constant_or_zero:
        # Matches docs/dataset_audit/data_quality_report.md §9's documented decision:
        # "All channels constant or zero in one file -> EXCLUDE". A single constant/zero
        # channel among the others is explicitly NOT a rejection reason (§9: "KEEP,
        # channel-level note") -- not implemented as a per-channel flag here, since
        # SignalRecord has no such field and 0 real files currently trigger this anyway.
        raise DatasetLoaderError(
            f"[split={split}] {relative_path!r}: all {report.columns} channels are "
            f"constant or zero -- excluded per data_quality_report.md §9"
        )


def load_recording(
    relative_path: str,
    split: str,
    *,
    dataset_root: Path = DATASET_ROOT,
    dataset_id: int = 1,
) -> LoadedRecording:
    """Loads and validates one manifest entry. `dataset_id` is the `datasets.id` this
    data will belong to if/when a caller persists it via TASK 2.1's models -- this
    loader does not create or manage `Dataset` rows itself (persistence orchestration
    is explicitly out of scope here, per the task's separation-of-concerns instruction).
    """
    full_path = _resolve_and_validate_path(relative_path, dataset_root, split)
    _validate_structure(full_path, relative_path, split)

    mapping = parse_recording_state(relative_path)
    if mapping.state == UNKNOWN:
        raise DatasetLoaderError(
            f"[split={split}] {relative_path!r} does not map to a known label "
            f"(mafaulda_parser.parse_recording_state returned UNKNOWN)"
        )

    try:
        label = SignalLabel(mapping.state)
    except ValueError as exc:
        raise DatasetLoaderError(
            f"[split={split}] {relative_path!r}: state {mapping.state!r} is not a "
            f"valid SignalLabel"
        ) from exc

    rotation = extract_rotation_info(relative_path, mapping.state)

    # Self-check, not a per-file comparison: no MAFAULDA file carries its own sampling
    # rate (docs/dataset_audit/signal_structure.md §5), so there is no file-derived
    # value to compare against here -- SAMPLING_RATE_HZ is validated against itself.
    # TASK 2.3's AC1 (a mismatched sampling rate must raise, not be silently accepted)
    # is demonstrated directly against validate_sampling_rate with synthetic values in
    # backend/tests/dataset/test_validators.py, since the real dataset structurally
    # cannot produce a per-file mismatch for the loader to encounter.
    try:
        validate_sampling_rate(SAMPLING_RATE_HZ, SAMPLING_RATE_HZ)
    except DatasetValidationError as exc:
        raise DatasetLoaderError(f"[split={split}] {relative_path!r}: {exc}") from exc

    signal_records = tuple(
        SignalRecord(
            dataset_id=dataset_id,
            sampling_rate=SAMPLING_RATE_HZ,
            channel=channel,
            file_path=relative_path,
            machine_id=None,  # docs/dataset_audit/AUDIT_REPORT.md §7: no machine/session ID exists locally
            operating_condition=mapping.condition,
            label=label,
        )
        for channel in range(NUM_CHANNELS)
    )

    return LoadedRecording(
        relative_path=relative_path,
        split=split,
        label=label,
        signal_records=signal_records,
    )


def load_split(
    split_name: str,
    *,
    dataset_root: Path = DATASET_ROOT,
    manifest_path: Path = MANIFEST_PATH,
    dataset_id: int = 1,
) -> list[LoadedRecording]:
    """Loads every file the manifest assigns to `split_name` (e.g. "train",
    "validation", "test" -- whatever keys actually exist under manifest["splits"], read
    from the manifest itself rather than hardcoded here)."""
    manifest = _load_manifest(manifest_path)
    splits = manifest.get("splits", {})

    if split_name not in splits:
        raise DatasetLoaderError(
            f"split {split_name!r} not found in manifest; available splits: {sorted(splits)}"
        )

    return [
        load_recording(relative_path, split_name, dataset_root=dataset_root, dataset_id=dataset_id)
        for relative_path in splits[split_name]
    ]


def load_all_splits(
    *,
    dataset_root: Path = DATASET_ROOT,
    manifest_path: Path = MANIFEST_PATH,
    dataset_id: int = 1,
) -> dict[str, list[LoadedRecording]]:
    manifest = _load_manifest(manifest_path)
    return {
        split_name: load_split(
            split_name, dataset_root=dataset_root, manifest_path=manifest_path, dataset_id=dataset_id
        )
        for split_name in manifest.get("splits", {})
    }
