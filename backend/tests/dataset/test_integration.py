"""TASK 2.5 — end-to-end dataset integration test.

Exercises the REAL pipeline (real files -> real loader -> real validators -> real
windowing) on a small, deterministic real subset selected from the real
data/processed/split_manifest.json. No component's logic is reimplemented here: this
file only calls backend/app/datasets/loader.py, backend/app/datasets/validators.py,
backend/app/datasets/data_quality.py, and backend/app/signal_processing/windowing.py.

The one piece of genuinely new glue code is reading a recording's actual sample values
from its source CSV (via pandas) -- necessary because, by design (TASK 2.1/2.2), the
loader's SignalRecord only carries a file_path reference, never the raw values, so
*something* has to bridge "here is a validated recording" to "here is an array
windowing.py can segment". That bridging does not exist anywhere else in the codebase
yet and is not a restatement of any existing component's logic.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

from app.datasets.data_quality import analyze_file_quality
from app.datasets.loader import DATASET_ROOT as REAL_DATASET_ROOT
from app.datasets.loader import LoadedRecording, load_all_splits
from app.datasets.validators import SAMPLING_RATE_HZ, validate_signal
from app.signal_processing.windowing import Window, create_windows

# Real files, taken directly from data/processed/split_manifest.json (same
# already-vetted subset used by TASK 2.2/2.3's tests) -- not invented, not globbed.
REAL_SUBSET = {
    "train": [
        "horizontal-misalignment/0.5mm/12.288.csv",
        "horizontal-misalignment/0.5mm/13.5168.csv",
        "horizontal-misalignment/0.5mm/14.5408.csv",
    ],
    "validation": [
        "horizontal-misalignment/0.5mm/20.48.csv",
        "horizontal-misalignment/0.5mm/21.504.csv",
        "horizontal-misalignment/0.5mm/26.8288.csv",
    ],
    "test": [
        "horizontal-misalignment/0.5mm/18.8416.csv",
        "horizontal-misalignment/0.5mm/25.8048.csv",
        "horizontal-misalignment/0.5mm/33.1776.csv",
    ],
}

# Arbitrary, reasonable, explicitly-documented choice for THIS integration test only --
# not a project-wide windowing configuration decision (blueprint.md defines no concrete
# window_size/overlap; that choice belongs to a later DSP-configuration task).
WINDOW_SIZE = 1024
OVERLAP = 0.5
CHANNEL = 0  # arbitrary single channel, for demonstrating the pipeline once per file


def _write_manifest(tmp_path: Path, splits: dict[str, list[str]]) -> Path:
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps({"splits": splits}), encoding="utf-8")
    return manifest_path


def _read_channel_values(relative_path: str, channel: int) -> list[float]:
    """Glue code: reads one channel's raw values from the real source file. Not a
    restatement of data_quality.py (which computes stats, never returns raw values)
    or loader.py (which never reads file content into memory at all)."""
    df = pd.read_csv(REAL_DATASET_ROOT / relative_path, header=None)
    return df[channel].tolist()


@pytest.fixture(scope="module")
def pipeline_result(tmp_path_factory: pytest.TempPathFactory):
    """Runs the real pipeline once (file -> loader -> validation -> windowing) over
    the real subset, so the three test functions below can each inspect a different
    aspect without repeating the (relatively slow, real-file-reading) work."""
    tmp_path = tmp_path_factory.mktemp("integration_manifest")
    manifest_path = _write_manifest(tmp_path, REAL_SUBSET)

    loaded = load_all_splits(dataset_root=REAL_DATASET_ROOT, manifest_path=manifest_path)

    windows_by_split: dict[str, list[Window]] = defaultdict(list)
    recording_ids_by_split: dict[str, set[str]] = defaultdict(set)
    quality_reports: dict[str, object] = {}

    for split_name, recordings in loaded.items():
        for recording in recordings:
            recording_ids_by_split[split_name].add(recording.relative_path)

            # --- validation stage (real validators, real per-file shape) ---
            report = analyze_file_quality(
                REAL_DATASET_ROOT / recording.relative_path, recording.relative_path
            )
            quality_reports[recording.relative_path] = report
            signal_record = recording.signal_records[CHANNEL]
            validate_signal(
                sampling_rate=signal_record.sampling_rate,
                shape=(report.rows, report.columns),
            )  # raises if invalid -- never caught/suppressed

            # --- windowing stage (real windowing, on real channel values) ---
            values = _read_channel_values(recording.relative_path, CHANNEL)
            windows = create_windows(
                values,
                recording_id=recording.relative_path,
                split=recording.split,
                window_size=WINDOW_SIZE,
                overlap=OVERLAP,
            )
            windows_by_split[split_name].extend(windows)

    return {
        "loaded": loaded,
        "windows_by_split": dict(windows_by_split),
        "recording_ids_by_split": dict(recording_ids_by_split),
        "quality_reports": quality_reports,
    }


def test_real_dataset_pipeline_end_to_end(pipeline_result) -> None:
    """AC1: file -> loader -> validation -> windowing runs on the real subset without
    errors, and every stage's output is checked explicitly (not just "no exception")."""
    loaded: dict[str, list[LoadedRecording]] = pipeline_result["loaded"]

    assert set(loaded.keys()) == {"train", "validation", "test"}
    for split_name, relative_paths in REAL_SUBSET.items():
        recordings = loaded[split_name]
        assert len(recordings) == 3

        for recording in recordings:
            # --- loader stage ---
            assert recording.relative_path in relative_paths
            assert recording.split == split_name
            assert recording.label.value == "horizontal-misalignment"
            assert len(recording.signal_records) == 8
            for sr in recording.signal_records:
                assert sr.file_path == recording.relative_path
                assert sr.operating_condition == "0.5mm"

            # --- validation stage already ran in the fixture; re-confirm explicitly ---
            report = pipeline_result["quality_reports"][recording.relative_path]
            assert report.status == "OK"
            assert report.rows == 250_000
            assert report.columns == 8
            validate_signal(
                sampling_rate=recording.signal_records[CHANNEL].sampling_rate,
                shape=(report.rows, report.columns),
            )  # must not raise

    # --- windowing stage ---
    windows_by_split = pipeline_result["windows_by_split"]
    for split_name in REAL_SUBSET:
        assert len(windows_by_split[split_name]) > 0
        for window in windows_by_split[split_name]:
            assert window.length == WINDOW_SIZE
            assert 0 <= window.start_sample < window.end_sample <= 250_000


def test_window_counts_are_positive_and_reported(pipeline_result) -> None:
    """AC2: total windows per split are computed (never hardcoded) and reported."""
    windows_by_split = pipeline_result["windows_by_split"]
    recording_ids_by_split = pipeline_result["recording_ids_by_split"]

    print("\nDataset integration summary")
    print("---------------------------")
    print(f"{'Split':<12}{'Recordings':>12}{'Windows':>10}{'Windows/Recording':>20}")

    total_windows = 0
    total_recordings = 0
    for split_name in ("train", "validation", "test"):
        n_recordings = len(recording_ids_by_split[split_name])
        n_windows = len(windows_by_split[split_name])
        windows_per_recording = n_windows / n_recordings if n_recordings else 0
        print(f"{split_name:<12}{n_recordings:>12}{n_windows:>10}{windows_per_recording:>20.1f}")

        total_windows += n_windows
        total_recordings += n_recordings

        assert n_recordings > 0
        assert n_windows > 0, f"split {split_name!r} produced zero windows"

    print(f"{'total':<12}{total_recordings:>12}{total_windows:>10}")

    # Reasonableness, evaluated against the real recordings actually used (not an
    # arbitrary global threshold): every real MAFAULDA file has exactly 250,000
    # samples (confirmed dataset-wide in TASK 1.5.8's full audit), and this subset
    # uses 3 recordings per split -- so with a fixed window_size/overlap, each split
    # is expected to produce the SAME number of windows here. That is not a fragile
    # coincidence; it is a direct, explainable consequence of the audited dataset's
    # uniform recording length, not of this test's specific file selection.
    window_counts = {split: len(windows_by_split[split]) for split in ("train", "validation", "test")}
    assert len(set(window_counts.values())) == 1, (
        f"expected equal window counts across splits given 3 equal-length recordings "
        f"per split, got {window_counts}"
    )


def test_windows_preserve_recording_and_split_provenance(pipeline_result) -> None:
    """Split isolation end-to-end: every window's split matches its source
    recording's split, and no recording_id/window crosses a split boundary."""
    recording_ids_by_split = pipeline_result["recording_ids_by_split"]
    windows_by_split = pipeline_result["windows_by_split"]

    train_ids = recording_ids_by_split["train"]
    val_ids = recording_ids_by_split["validation"]
    test_ids = recording_ids_by_split["test"]

    assert train_ids & val_ids == set()
    assert train_ids & test_ids == set()
    assert val_ids & test_ids == set()

    for split_name, expected_ids in (
        ("train", train_ids),
        ("validation", val_ids),
        ("test", test_ids),
    ):
        for window in windows_by_split[split_name]:
            assert window.split == split_name
            assert window.recording_id in expected_ids
            # never leaked into a different split's recording set
            other_splits_ids = (train_ids | val_ids | test_ids) - expected_ids
            assert window.recording_id not in other_splits_ids


def test_window_count_matches_independent_formula_for_one_real_recording(pipeline_result) -> None:
    """Mathematical verification (independent of windowing.py's own
    compute_window_count -- this formula is written fresh here, not imported)."""
    quality_reports = pipeline_result["quality_reports"]
    windows_by_split = pipeline_result["windows_by_split"]

    relative_path = REAL_SUBSET["train"][0]
    report = quality_reports[relative_path]
    n = report.rows  # 250,000, confirmed by the real file itself in this test run
    w = WINDOW_SIZE
    overlap = OVERLAP
    step = w * (1 - overlap)

    assert n >= w
    expected = int((n - w) // step) + 1

    actual = sum(1 for window in windows_by_split["train"] if window.recording_id == relative_path)

    assert actual == expected
