from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.datasets.loader import DATASET_ROOT as REAL_DATASET_ROOT
from app.datasets.loader import load_all_splits
from app.datasets.validators import SAMPLING_RATE_HZ
from app.features.extractor import FeatureExtractionError, extract_feature_matrix, extract_features
from app.features.registry import FEATURE_REGISTRY, FREQUENCY_FEATURE_REGISTRY
from app.features.time_domain import FeatureError
from app.signal_processing.windowing import Window, create_windows

FS = SAMPLING_RATE_HZ
NPERSEG = 256
NOVERLAP = 128
TOTAL_FEATURE_COUNT = len(FEATURE_REGISTRY) + len(FREQUENCY_FEATURE_REGISTRY)


def _synthetic_signal(n: int = 4096, fs: float = FS) -> np.ndarray:
    """Deterministic (no randomness) synthetic window: a sine plus a fixed harmonic
    -- long enough for the default NPERSEG/NOVERLAP Welch configuration used
    throughout this file."""
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 400 * t)


# --- AC1: determinism ---


def test_ac1_extract_features_is_deterministic_for_the_same_window() -> None:
    signal = _synthetic_signal()

    result_1 = extract_features(signal, FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    result_2 = extract_features(signal, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    assert result_1.keys() == result_2.keys()
    for name in result_1:
        assert result_1[name] == pytest.approx(result_2[name]), f"feature {name!r} differed between runs"


def test_ac1_extract_feature_matrix_is_deterministic_for_the_same_windows() -> None:
    windows = create_windows(
        list(_synthetic_signal()), recording_id="synthetic", split="train", window_size=1024, overlap=0.5
    )

    matrix_1 = extract_feature_matrix(windows, FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    matrix_2 = extract_feature_matrix(windows, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    pd.testing.assert_frame_equal(matrix_1, matrix_2)


# --- AC3: column count / names / order derived from the registries, never hard-coded ---


def test_ac3_single_window_feature_count_matches_registry_sizes() -> None:
    signal = _synthetic_signal()

    result = extract_features(signal, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    assert len(result) == len(FEATURE_REGISTRY) + len(FREQUENCY_FEATURE_REGISTRY)


def test_ac3_single_window_feature_names_and_order_match_registries() -> None:
    signal = _synthetic_signal()

    result = extract_features(signal, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    expected_order = list(FEATURE_REGISTRY.keys()) + list(FREQUENCY_FEATURE_REGISTRY.keys())
    assert list(result.keys()) == expected_order


def test_ac3_matrix_column_count_matches_registry_sizes() -> None:
    windows = create_windows(
        list(_synthetic_signal()), recording_id="synthetic", split="train", window_size=1024, overlap=0.5
    )

    matrix = extract_feature_matrix(windows, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    assert matrix.shape[1] == len(FEATURE_REGISTRY) + len(FREQUENCY_FEATURE_REGISTRY)


def test_ac3_matrix_column_names_and_order_match_registries() -> None:
    windows = create_windows(
        list(_synthetic_signal()), recording_id="synthetic", split="train", window_size=1024, overlap=0.5
    )

    matrix = extract_feature_matrix(windows, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    expected_columns = list(FEATURE_REGISTRY.keys()) + list(FREQUENCY_FEATURE_REGISTRY.keys())
    assert list(matrix.columns) == expected_columns


def test_ac3_matrix_row_count_matches_window_count() -> None:
    windows = create_windows(
        list(_synthetic_signal()), recording_id="synthetic", split="train", window_size=1024, overlap=0.5
    )

    matrix = extract_feature_matrix(windows, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    assert matrix.shape[0] == len(windows) > 0


def test_matrix_of_zero_windows_has_correct_columns_and_zero_rows() -> None:
    matrix = extract_feature_matrix([], FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    assert matrix.shape == (0, TOTAL_FEATURE_COUNT)
    assert list(matrix.columns) == list(FEATURE_REGISTRY.keys()) + list(FREQUENCY_FEATURE_REGISTRY.keys())


# --- registry-driven: adding a feature to either registry appears automatically ---


def test_extractor_is_registry_driven_for_time_domain_registry() -> None:
    """Mutates the SAME dict object `extract_features` iterates (imported by
    reference, not copied) -- demonstrating the extractor has no hard-coded
    knowledge of individual feature names. Cleans up afterward so no other test is
    affected."""

    def dummy_feature(signal: np.ndarray) -> float:
        return 12345.0

    FEATURE_REGISTRY["__test_dummy_time__"] = dummy_feature
    try:
        result = extract_features(_synthetic_signal(), FS, nperseg=NPERSEG, noverlap=NOVERLAP)
        assert result["__test_dummy_time__"] == 12345.0
        assert len(result) == len(FEATURE_REGISTRY) + len(FREQUENCY_FEATURE_REGISTRY)
    finally:
        del FEATURE_REGISTRY["__test_dummy_time__"]


def test_extractor_is_registry_driven_for_frequency_domain_registry() -> None:
    def dummy_frequency_feature(frequencies: np.ndarray, power: np.ndarray) -> float:
        return 6789.0

    FREQUENCY_FEATURE_REGISTRY["__test_dummy_freq__"] = dummy_frequency_feature
    try:
        result = extract_features(_synthetic_signal(), FS, nperseg=NPERSEG, noverlap=NOVERLAP)
        assert result["__test_dummy_freq__"] == 6789.0
        assert len(result) == len(FEATURE_REGISTRY) + len(FREQUENCY_FEATURE_REGISTRY)
    finally:
        del FREQUENCY_FEATURE_REGISTRY["__test_dummy_freq__"]


# --- error handling: a failing feature must propagate, never be silently defaulted ---


def test_extract_features_propagates_underlying_error_for_degenerate_signal() -> None:
    """A constant signal makes `skewness` (the 7th entry in FEATURE_REGISTRY's
    insertion order, the first to divide-by-zero) raise `FeatureError` -- the
    extractor must wrap and propagate it, not swallow it or substitute a value."""
    constant_signal = np.full(2048, 5.0)

    with pytest.raises(FeatureExtractionError, match="skewness") as exc_info:
        extract_features(constant_signal, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    assert isinstance(exc_info.value.__cause__, FeatureError)


def test_extract_features_propagates_underlying_error_from_injected_failing_feature() -> None:
    def broken_feature(signal: np.ndarray) -> float:
        raise RuntimeError("synthetic failure for this test only")

    FEATURE_REGISTRY["__test_broken__"] = broken_feature
    try:
        with pytest.raises(FeatureExtractionError, match="__test_broken__") as exc_info:
            extract_features(_synthetic_signal(), FS, nperseg=NPERSEG, noverlap=NOVERLAP)
        assert isinstance(exc_info.value.__cause__, RuntimeError)
    finally:
        del FEATURE_REGISTRY["__test_broken__"]


# --- edge cases ---


def test_extract_features_on_pure_sine_produces_finite_values() -> None:
    signal = _synthetic_signal()

    result = extract_features(signal, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    assert all(np.isfinite(value) for value in result.values())


def test_extract_features_does_not_modify_input_signal() -> None:
    signal = list(_synthetic_signal())
    snapshot = list(signal)

    extract_features(signal, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    assert signal == snapshot


def test_extract_feature_matrix_does_not_reorder_windows() -> None:
    """Row i must correspond exactly to windows[i] -- verified here by giving each
    window a distinct, identifiable mean (via a per-window constant additive
    offset) and checking the matrix's 'mean' column matches, in order."""
    base = _synthetic_signal(n=1024)
    offsets = [0.0, 100.0, -50.0]
    windows = [
        Window(recording_id="w", split="train", start_sample=0, end_sample=1024, values=tuple(base + offset))
        for offset in offsets
    ]

    matrix = extract_feature_matrix(windows, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    for i, offset in enumerate(offsets):
        assert matrix.iloc[i]["mean"] == pytest.approx(np.mean(base) + offset)


# --- real-data validation (AC2 + Definition of Done) ---
#
# Same already-vetted real subset used by TASK 2.5's own integration test (data/raw/
# mafaulda/horizontal-misalignment/0.5mm/*.csv, referenced via the real split_manifest
# .json) -- not a new split, not a dataset scan.
REAL_SUBSET = {
    "train": [
        "horizontal-misalignment/0.5mm/12.288.csv",
        "horizontal-misalignment/0.5mm/13.5168.csv",
    ],
    "validation": [
        "horizontal-misalignment/0.5mm/20.48.csv",
    ],
    "test": [
        "horizontal-misalignment/0.5mm/18.8416.csv",
    ],
}
REAL_WINDOW_SIZE = 1024
REAL_OVERLAP = 0.5
REAL_CHANNEL = 0


def _write_manifest(tmp_path: Path, splits: dict[str, list[str]]) -> Path:
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps({"splits": splits}), encoding="utf-8")
    return manifest_path


@pytest.fixture(scope="module")
def real_windows_by_split(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[Window]]:
    """Loads a small real MAFAULDA subset (2 train / 1 validation / 1 test
    recordings) via the real loader + real windowing, exactly as TASK 2.5's own
    integration test does -- no new split, no full-dataset scan."""
    tmp_path = tmp_path_factory.mktemp("extractor_real_data")
    manifest_path = _write_manifest(tmp_path, REAL_SUBSET)

    loaded = load_all_splits(dataset_root=REAL_DATASET_ROOT, manifest_path=manifest_path)

    windows_by_split: dict[str, list[Window]] = {}
    for split_name, recordings in loaded.items():
        windows_by_split[split_name] = []
        for recording in recordings:
            df = pd.read_csv(REAL_DATASET_ROOT / recording.relative_path, header=None)
            values = df[REAL_CHANNEL].tolist()
            windows = create_windows(
                values,
                recording_id=recording.relative_path,
                split=recording.split,
                window_size=REAL_WINDOW_SIZE,
                overlap=REAL_OVERLAP,
            )
            windows_by_split[split_name].extend(windows)

    return windows_by_split


def test_ac2_real_train_feature_matrix_contains_no_nan_or_inf(real_windows_by_split) -> None:
    train_windows = real_windows_by_split["train"]
    assert len(train_windows) > 0, "test setup error: expected at least one real train window"

    matrix = extract_feature_matrix(train_windows, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    assert matrix.shape == (len(train_windows), TOTAL_FEATURE_COUNT)
    assert np.isfinite(matrix.to_numpy()).all(), matrix[~np.isfinite(matrix.to_numpy()).all(axis=1)]


def test_real_validation_feature_matrix_contains_no_nan_or_inf(real_windows_by_split) -> None:
    validation_windows = real_windows_by_split["validation"]
    assert len(validation_windows) > 0

    matrix = extract_feature_matrix(validation_windows, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    assert matrix.shape == (len(validation_windows), TOTAL_FEATURE_COUNT)
    assert np.isfinite(matrix.to_numpy()).all()


def test_real_test_split_feature_matrix_contains_no_nan_or_inf(real_windows_by_split) -> None:
    test_windows = real_windows_by_split["test"]
    assert len(test_windows) > 0

    matrix = extract_feature_matrix(test_windows, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    assert matrix.shape == (len(test_windows), TOTAL_FEATURE_COUNT)
    assert np.isfinite(matrix.to_numpy()).all()


def test_real_splits_remain_separate_after_extraction(real_windows_by_split) -> None:
    """No leakage: each split's windows come from only that split's recordings, and
    extraction never merges/concatenates across splits."""
    train_ids = {w.recording_id for w in real_windows_by_split["train"]}
    validation_ids = {w.recording_id for w in real_windows_by_split["validation"]}
    test_ids = {w.recording_id for w in real_windows_by_split["test"]}

    assert train_ids & validation_ids == set()
    assert train_ids & test_ids == set()
    assert validation_ids & test_ids == set()

    for split_name, ids in (("train", train_ids), ("validation", validation_ids), ("test", test_ids)):
        for window in real_windows_by_split[split_name]:
            assert window.split == split_name
            assert window.recording_id in ids
