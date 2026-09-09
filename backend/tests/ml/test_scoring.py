from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve

from app.datasets.loader import DATASET_ROOT as REAL_DATASET_ROOT
from app.datasets.loader import load_all_splits
from app.datasets.validators import SAMPLING_RATE_HZ
from app.features.extractor import extract_feature_matrix
from app.ml.isolation_forest import score as raw_isolation_forest_score
from app.ml.isolation_forest import train_isolation_forest
from app.ml.scoring import (
    ScoringError,
    calibrate,
    calibrate_threshold,
    classify,
    compute_normalized_scores,
    fit_score_normalizer,
    normalize_scores,
    to_anomaly_score,
)
from app.signal_processing.windowing import Window, create_windows


# --- score orientation ---


def test_to_anomaly_score_flips_sign() -> None:
    raw = np.array([0.1, -0.2, 0.0])
    np.testing.assert_array_equal(to_anomaly_score(raw), np.array([-0.1, 0.2, 0.0]))


def test_isolation_forest_anomaly_is_more_negative_raw_and_higher_anomaly_score() -> None:
    """End-to-end confirmation (real sklearn IsolationForest, not assumed): a clear
    outlier's raw decision_function is more negative than a normal point's, and
    after to_anomaly_score() it is correspondingly higher (more anomalous)."""
    rng = np.random.default_rng(0)
    normal_train = pd.DataFrame(rng.normal(0, 1, size=(200, 2)), columns=["a", "b"])
    labels = ["normal"] * 200
    model, scaler = train_isolation_forest(normal_train, labels, random_state=42)

    normal_point = pd.DataFrame([[0.0, 0.0]], columns=["a", "b"])
    outlier_point = pd.DataFrame([[50.0, 50.0]], columns=["a", "b"])

    raw_normal = raw_isolation_forest_score(model, scaler, normal_point)
    raw_outlier = raw_isolation_forest_score(model, scaler, outlier_point)
    assert raw_outlier[0] < raw_normal[0]

    anomaly_normal = to_anomaly_score(raw_normal)
    anomaly_outlier = to_anomaly_score(raw_outlier)
    assert anomaly_outlier[0] > anomaly_normal[0]


# --- Test 1 — AC1: validation-only calibration ---


def test_ac1_calibrate_threshold_signature_has_no_test_data_parameter() -> None:
    for name in inspect.signature(calibrate_threshold).parameters:
        assert "test" not in name.lower(), f"calibrate_threshold must not accept a {name!r} parameter"
    for name in inspect.signature(calibrate).parameters:
        assert "test" not in name.lower(), f"calibrate must not accept a {name!r} parameter"


def test_ac1_threshold_depends_only_on_validation_input() -> None:
    validation_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    threshold_1 = calibrate_threshold(validation_scores, threshold_method="percentile", percentile_value=90)

    # These hypothetical "test" distributions are never passed to calibrate_threshold
    # anywhere -- there is no parameter to pass them through. Recomputing with the
    # identical validation input must give the identical threshold regardless of
    # what a caller's test data looks like elsewhere in their program.
    _hypothetical_test_b1 = np.array([0.99] * 20)
    _hypothetical_test_b2 = np.array([0.01] * 20)

    threshold_2 = calibrate_threshold(validation_scores, threshold_method="percentile", percentile_value=90)

    assert threshold_1 == threshold_2


# --- Test 2 — AC2: normalization range, extreme synthetic inputs ---


def test_ac2_normalized_scores_always_in_unit_interval_for_extreme_inputs() -> None:
    params = fit_score_normalizer(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))

    extreme_inputs = np.array([-1_000_000.0, -1.0, 0.0, 1.0, 1_000_000.0])
    normalized = normalize_scores(extreme_inputs, params)

    assert np.all(normalized >= 0.0)
    assert np.all(normalized <= 1.0)


# --- Test 3 — validation min/max mapping ---


def test_validation_min_and_max_map_to_0_and_1() -> None:
    params = fit_score_normalizer(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))

    assert params.validation_min == pytest.approx(0.1)
    assert params.validation_max == pytest.approx(0.5)

    normalized = normalize_scores(np.array([0.1, 0.5]), params)
    np.testing.assert_allclose(normalized, [0.0, 1.0])


# --- Test 4 — values outside validation range are clipped ---


def test_values_outside_validation_range_are_clipped_to_0_and_1() -> None:
    params = fit_score_normalizer(np.array([0.0, 1.0, 2.0]))  # min=0, max=2

    normalized = normalize_scores(np.array([-5.0, 5.0]), params)

    np.testing.assert_allclose(normalized, [0.0, 1.0])


# --- Test 5 — constant validation distribution ---


def test_constant_validation_distribution_does_not_divide_by_zero() -> None:
    params = fit_score_normalizer(np.array([3.0, 3.0, 3.0]))

    assert params.validation_min == params.validation_max == pytest.approx(3.0)

    # Exact match to the single known reference point -> 0.0 (not anomalous
    # relative to what's known); anything else -> 1.0 (maximally out-of-distribution,
    # per the module's documented convention for a degenerate reference).
    normalized = normalize_scores(np.array([3.0, 3.0, 5.0, -1.0]), params)
    np.testing.assert_allclose(normalized, [0.0, 0.0, 1.0, 1.0])


# --- Test 6 — percentile method ---


def test_percentile_method_matches_expected_value_without_labels() -> None:
    validation_normalized = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    threshold = calibrate_threshold(
        validation_normalized, threshold_method="percentile", percentile_value=90
    )

    expected = np.percentile(validation_normalized, 90)
    assert threshold == pytest.approx(expected)


def test_percentile_method_with_labels_uses_normal_only_subset() -> None:
    """blueprint.md line 258: percentile of scores on NORMAL data specifically --
    including fault scores (much higher here) would inflate the threshold."""
    validation_normalized = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.9, 0.95, 0.99])
    labels = ["normal"] * 5 + ["horizontal-misalignment"] * 3

    threshold_normal_only = calibrate_threshold(
        validation_normalized, threshold_method="percentile", percentile_value=95, validation_labels=labels
    )
    threshold_all = calibrate_threshold(
        validation_normalized, threshold_method="percentile", percentile_value=95, validation_labels=None
    )

    expected_normal_only = np.percentile([0.1, 0.15, 0.2, 0.25, 0.3], 95)
    assert threshold_normal_only == pytest.approx(expected_normal_only)
    assert threshold_normal_only < threshold_all


# --- Test 7 — invalid percentile ---


def test_invalid_percentile_value_raises() -> None:
    validation_normalized = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ScoringError, match="percentile_value"):
        calibrate_threshold(validation_normalized, threshold_method="percentile", percentile_value=-1)
    with pytest.raises(ScoringError, match="percentile_value"):
        calibrate_threshold(validation_normalized, threshold_method="percentile", percentile_value=101)


# --- Test 8 — F1 optimal ---


def test_f1_optimal_threshold_matches_independently_computed_value() -> None:
    validation_normalized = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.6, 0.8, 0.9])
    labels = ["normal"] * 5 + ["horizontal-misalignment"] * 3

    threshold = calibrate_threshold(
        validation_normalized, threshold_method="validation_f1_optimal", validation_labels=labels
    )

    # Independently recomputed with sklearn primitives directly in the test, not by
    # calling this module's own implementation.
    y = [0] * 5 + [1] * 3
    precision, recall, thresholds = precision_recall_curve(y, validation_normalized)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1])
    expected = thresholds[np.argmax(f1)]

    assert threshold == pytest.approx(expected)


# --- Test 9 — F1 without fault labels ---


def test_f1_optimal_without_fault_examples_raises() -> None:
    validation_normalized = np.array([0.1, 0.2, 0.3])
    labels = ["normal"] * 3

    with pytest.raises(ScoringError, match="fault"):
        calibrate_threshold(
            validation_normalized, threshold_method="validation_f1_optimal", validation_labels=labels
        )


def test_f1_optimal_without_any_labels_raises() -> None:
    validation_normalized = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ScoringError, match="validation_labels"):
        calibrate_threshold(
            validation_normalized, threshold_method="validation_f1_optimal", validation_labels=None
        )


# --- Test 10 — AC3: both methods produce different, verified results ---


def test_ac3_percentile_and_f1_optimal_produce_different_thresholds() -> None:
    validation_normalized = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.6, 0.8, 0.9])
    labels = ["normal"] * 7 + ["horizontal-misalignment"] * 3

    percentile_threshold = calibrate_threshold(
        validation_normalized, threshold_method="percentile", percentile_value=95, validation_labels=labels
    )
    f1_threshold = calibrate_threshold(
        validation_normalized, threshold_method="validation_f1_optimal", validation_labels=labels
    )

    print(f"\npercentile_threshold={percentile_threshold} f1_threshold={f1_threshold}")

    assert percentile_threshold == pytest.approx(0.385)
    assert f1_threshold == pytest.approx(0.6)
    assert percentile_threshold != pytest.approx(f1_threshold)


# --- Test 11 — unknown threshold method ---


def test_unknown_threshold_method_raises() -> None:
    validation_normalized = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ScoringError, match="Unknown threshold_method"):
        calibrate_threshold(validation_normalized, threshold_method="something_invalid")


# --- Test 12 — full score application ---


def test_full_pipeline_new_scores_are_normalized_and_classified_sensibly() -> None:
    raw_validation_scores = np.array([0.05, 0.02, -0.01, 0.1, -0.3])
    anomaly_raw_validation = to_anomaly_score(raw_validation_scores)

    calibration = calibrate(anomaly_raw_validation, threshold_method="percentile", percentile_value=80)

    new_raw_scores = np.array([0.09, -0.35])  # one normal-like, one clearly anomalous
    new_anomaly_raw = to_anomaly_score(new_raw_scores)
    normalized_new = normalize_scores(new_anomaly_raw, calibration.normalization)
    decisions = classify(normalized_new, calibration)

    assert decisions.dtype == bool
    assert decisions.shape == (2,)
    assert decisions[0] == np.bool_(False)
    assert decisions[1] == np.bool_(True)


# --- Test 13 — deterministic calibration ---


def test_calibration_is_deterministic() -> None:
    validation_raw = np.array([0.1, 0.05, -0.02, 0.2, -0.1, 0.15])
    labels = ["normal"] * 4 + ["horizontal-misalignment"] * 2

    calibration_1 = calibrate(validation_raw, threshold_method="validation_f1_optimal", validation_labels=labels)
    calibration_2 = calibrate(validation_raw, threshold_method="validation_f1_optimal", validation_labels=labels)

    assert calibration_1 == calibration_2


# --- Test 14 — score/label length mismatch ---


def test_scores_labels_length_mismatch_raises() -> None:
    validation_normalized = np.array([0.1, 0.2, 0.3])
    labels = ["normal", "normal"]  # too few

    with pytest.raises(ScoringError, match="align"):
        calibrate_threshold(validation_normalized, threshold_method="percentile", validation_labels=labels)


# --- additional validation: empty / NaN / Inf ---


def test_fit_score_normalizer_raises_on_empty() -> None:
    with pytest.raises(ScoringError, match="empty"):
        fit_score_normalizer(np.array([]))


def test_fit_score_normalizer_raises_on_nan() -> None:
    with pytest.raises(ScoringError, match="NaN"):
        fit_score_normalizer(np.array([0.1, np.nan, 0.3]))


def test_fit_score_normalizer_raises_on_inf() -> None:
    with pytest.raises(ScoringError, match="NaN"):
        fit_score_normalizer(np.array([0.1, np.inf, 0.3]))


def test_normalize_scores_raises_on_nan_input() -> None:
    params = fit_score_normalizer(np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ScoringError, match="NaN"):
        normalize_scores(np.array([0.1, np.nan]), params)


def test_calibrate_threshold_raises_on_empty_validation_scores() -> None:
    with pytest.raises(ScoringError, match="empty"):
        calibrate_threshold(np.array([]), threshold_method="percentile")


# --- input immutability ---


def test_normalize_scores_does_not_modify_input() -> None:
    params = fit_score_normalizer(np.array([0.0, 1.0]))
    raw = np.array([-5.0, 0.5, 5.0])
    snapshot = raw.copy()

    normalize_scores(raw, params)

    np.testing.assert_array_equal(raw, snapshot)


# --- real-data validation (Definition of Done) ---
#
# Real MAFAULDA subset from data/processed/split_manifest.json: 2 normal train
# recordings (TASK 6.2's own choice), 1 normal + 1 non-normal VALIDATION recording
# (so both percentile and validation_f1_optimal are meaningfully exercisable on real
# data) -- not a new split, not a full dataset scan, test split untouched entirely.
REAL_TRAIN_NORMAL = ["normal/12.288.csv", "normal/16.1792.csv"]
REAL_VALIDATION_NORMAL = ["normal/14.336.csv"]
REAL_VALIDATION_NON_NORMAL = ["horizontal-misalignment/0.5mm/20.48.csv"]
REAL_WINDOW_SIZE = 1024
REAL_OVERLAP = 0.5
REAL_CHANNEL = 0
REAL_NPERSEG = 256
REAL_NOVERLAP = 128


def _write_manifest(tmp_path: Path, splits: dict[str, list[str]]) -> Path:
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps({"splits": splits}), encoding="utf-8")
    return manifest_path


@pytest.fixture(scope="module")
def real_calibration_inputs(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[IsolationForest, object, pd.DataFrame, list[str]]:
    """Real loader + real windowing + real TASK 5.3 feature extraction + real TASK
    6.2 training -- no new split, no full-dataset scan. Returns the trained model,
    its scaler, the VALIDATION feature matrix, and VALIDATION labels only (test
    split is never even loaded)."""
    tmp_path = tmp_path_factory.mktemp("scoring_real_data")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "train": REAL_TRAIN_NORMAL,
            "validation": REAL_VALIDATION_NORMAL + REAL_VALIDATION_NON_NORMAL,
            "test": [],
        },
    )

    loaded = load_all_splits(dataset_root=REAL_DATASET_ROOT, manifest_path=manifest_path)

    def _windows_and_labels(split_name: str) -> tuple[list[Window], list[str]]:
        windows: list[Window] = []
        labels: list[str] = []
        for recording in loaded[split_name]:
            df = pd.read_csv(REAL_DATASET_ROOT / recording.relative_path, header=None)
            values = df[REAL_CHANNEL].tolist()
            recording_windows = create_windows(
                values,
                recording_id=recording.relative_path,
                split=recording.split,
                window_size=REAL_WINDOW_SIZE,
                overlap=REAL_OVERLAP,
            )
            windows.extend(recording_windows)
            labels.extend([recording.label.value] * len(recording_windows))
        return windows, labels

    train_windows, train_labels = _windows_and_labels("train")
    validation_windows, validation_labels = _windows_and_labels("validation")

    train_matrix = extract_feature_matrix(
        train_windows, SAMPLING_RATE_HZ, nperseg=REAL_NPERSEG, noverlap=REAL_NOVERLAP
    )
    validation_matrix = extract_feature_matrix(
        validation_windows, SAMPLING_RATE_HZ, nperseg=REAL_NPERSEG, noverlap=REAL_NOVERLAP
    )

    model, scaler = train_isolation_forest(train_matrix, train_labels, random_state=42)

    return model, scaler, validation_matrix, validation_labels


def test_real_data_percentile_calibration(
    real_calibration_inputs: tuple[IsolationForest, object, pd.DataFrame, list[str]],
) -> None:
    model, scaler, validation_matrix, validation_labels = real_calibration_inputs
    assert "normal" in validation_labels and "horizontal-misalignment" in validation_labels, (
        "test setup error: expected both classes in the real validation subset"
    )

    raw_validation_scores = raw_isolation_forest_score(model, scaler, validation_matrix)
    anomaly_raw_validation = to_anomaly_score(raw_validation_scores)

    calibration = calibrate(
        anomaly_raw_validation,
        threshold_method="percentile",
        percentile_value=95,
        validation_labels=validation_labels,
    )

    assert 0.0 <= calibration.threshold <= 1.0
    assert np.isfinite(calibration.threshold)


def test_real_data_f1_optimal_calibration_and_classification(
    real_calibration_inputs: tuple[IsolationForest, object, pd.DataFrame, list[str]],
) -> None:
    model, scaler, validation_matrix, validation_labels = real_calibration_inputs

    raw_validation_scores = raw_isolation_forest_score(model, scaler, validation_matrix)
    anomaly_raw_validation = to_anomaly_score(raw_validation_scores)

    calibration = calibrate(
        anomaly_raw_validation, threshold_method="validation_f1_optimal", validation_labels=validation_labels
    )

    normalized_validation = normalize_scores(anomaly_raw_validation, calibration.normalization)
    decisions = classify(normalized_validation, calibration)

    assert decisions.shape == (len(validation_matrix),)
    assert decisions.dtype == bool
    assert np.isfinite(normalized_validation).all()
    assert np.all(normalized_validation >= 0.0) and np.all(normalized_validation <= 1.0)
