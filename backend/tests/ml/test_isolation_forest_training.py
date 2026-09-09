from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from app.datasets.loader import DATASET_ROOT as REAL_DATASET_ROOT
from app.datasets.loader import load_all_splits
from app.datasets.validators import SAMPLING_RATE_HZ
from app.features.extractor import extract_feature_matrix
from app.ml.isolation_forest import (
    DEFAULT_MODEL_FILENAME,
    IsolationForestTrainingError,
    load_model,
    save_model,
    score,
    select_normal_samples,
    train_isolation_forest,
)
from app.ml.scaling import ScalingError, load_scaler, save_scaler
from app.signal_processing.windowing import Window, create_windows


def _synthetic_dataset() -> tuple[pd.DataFrame, list[str]]:
    """20 'normal' rows clustered near 0, 3 'horizontal-misalignment' rows clustered
    far away (~+50), 2 'imbalance' rows clustered far away in the other direction
    (~-50) -- deterministic (fixed seed), designed so that an implementation which
    accidentally trains on ALL rows (instead of only 'normal') is easy to detect,
    both by row count and by the resulting scaler's mean being dragged away from 0."""
    rng = np.random.default_rng(0)
    normal = pd.DataFrame(rng.normal(0.0, 1.0, size=(20, 3)), columns=["mean", "rms", "kurtosis"])
    anomaly_a = pd.DataFrame(rng.normal(50.0, 1.0, size=(3, 3)), columns=["mean", "rms", "kurtosis"])
    anomaly_b = pd.DataFrame(rng.normal(-50.0, 1.0, size=(2, 3)), columns=["mean", "rms", "kurtosis"])

    features = pd.concat([normal, anomaly_a, anomaly_b], ignore_index=True)
    labels = ["normal"] * 20 + ["horizontal-misalignment"] * 3 + ["imbalance"] * 2

    return features, labels


# --- Test 1 (CRITICAL) — AC1 zero label leakage ---


def test_ac1_select_normal_samples_keeps_only_normal_rows() -> None:
    features, labels = _synthetic_dataset()

    normal_features = select_normal_samples(features, labels)

    assert len(normal_features) == 20
    # Every selected row must come from the 'normal' block (index 0-19) -- none of
    # the anomaly rows (index 20-24) leaked through.
    assert list(normal_features.index) == list(range(20))


def test_ac1_isolation_forest_fit_is_called_with_only_normal_samples(monkeypatch) -> None:
    """Detects label leakage at the ACTUAL sklearn call site, not just in a helper
    function -- would fail if `train_isolation_forest` were changed to fit on all
    train rows (e.g. `model.fit(all_features)`), even if `select_normal_samples`
    itself remained correct."""
    features, labels = _synthetic_dataset()

    captured: dict[str, np.ndarray] = {}
    original_fit = IsolationForest.fit

    def spy_fit(self, X, y=None, sample_weight=None):
        captured["X"] = np.asarray(X).copy()
        return original_fit(self, X, y=y, sample_weight=sample_weight)

    monkeypatch.setattr(IsolationForest, "fit", spy_fit)

    train_isolation_forest(features, labels, random_state=42)

    assert "X" in captured, "IsolationForest.fit was never called"
    assert captured["X"].shape[0] == 20, (
        f"expected exactly the 20 normal samples to reach .fit(), got {captured['X'].shape[0]}"
    )
    # Exactly the DSP feature columns, never a label-derived extra column.
    assert captured["X"].shape[1] == features.shape[1] == 3


def test_ac1_features_labels_length_mismatch_raises() -> None:
    features, labels = _synthetic_dataset()

    with pytest.raises(IsolationForestTrainingError, match="rows"):
        select_normal_samples(features, labels[:-1])


# --- Test 2 — number of samples used for training ---


def test_number_of_training_samples_equals_normal_count_not_total_train_count() -> None:
    features, labels = _synthetic_dataset()

    normal_features = select_normal_samples(features, labels)

    total_train_count = len(features)
    normal_count = sum(1 for label in labels if label == "normal")

    assert normal_count == 20
    assert total_train_count == 25
    assert len(normal_features) == normal_count
    assert len(normal_features) != total_train_count


# --- Test 3 (AC2) — decision_function ---


def test_ac2_decision_function_single_window_produces_finite_score() -> None:
    features, labels = _synthetic_dataset()
    model, scaler = train_isolation_forest(features, labels, random_state=42)

    new_window = pd.DataFrame([[0.1, 0.2, 0.3]], columns=["mean", "rms", "kurtosis"])
    scores = score(model, scaler, new_window)

    assert scores.shape == (1,)
    assert np.isfinite(scores).all()


def test_ac2_decision_function_multiple_windows_produces_correct_shape() -> None:
    features, labels = _synthetic_dataset()
    model, scaler = train_isolation_forest(features, labels, random_state=42)

    new_windows = pd.DataFrame(
        np.random.default_rng(1).normal(0, 1, size=(7, 3)), columns=["mean", "rms", "kurtosis"]
    )
    scores = score(model, scaler, new_windows)

    assert scores.shape == (7,)
    assert np.isfinite(scores).all()


# --- Test 4/5 (AC3) — reproducibility ---


def test_ac3_repeated_training_with_same_seed_produces_identical_scores() -> None:
    features, labels = _synthetic_dataset()
    new_windows = pd.DataFrame(
        np.random.default_rng(2).normal(0, 1, size=(5, 3)), columns=["mean", "rms", "kurtosis"]
    )

    model_1, scaler_1 = train_isolation_forest(features, labels, random_state=42)
    model_2, scaler_2 = train_isolation_forest(features, labels, random_state=42)

    scores_1 = score(model_1, scaler_1, new_windows)
    scores_2 = score(model_2, scaler_2, new_windows)

    np.testing.assert_allclose(scores_1, scores_2)


def test_different_seeds_can_produce_different_scores() -> None:
    """Sanity check that the seed is actually wired to something meaningful (not a
    no-op) -- not a strict requirement, but confirms the reproducibility test above
    isn't vacuously true because the model ignores random_state entirely."""
    features, labels = _synthetic_dataset()
    new_windows = pd.DataFrame(
        np.random.default_rng(2).normal(0, 1, size=(5, 3)), columns=["mean", "rms", "kurtosis"]
    )

    model_a, scaler_a = train_isolation_forest(features, labels, random_state=1)
    model_b, scaler_b = train_isolation_forest(features, labels, random_state=2)

    scores_a = score(model_a, scaler_a, new_windows)
    scores_b = score(model_b, scaler_b, new_windows)

    assert not np.allclose(scores_a, scores_b)


# --- Test 6 — no normal samples ---


def test_training_with_zero_normal_samples_raises_explicitly() -> None:
    features = pd.DataFrame(
        np.random.default_rng(3).normal(0, 1, size=(10, 3)), columns=["mean", "rms", "kurtosis"]
    )
    labels = ["horizontal-misalignment"] * 5 + ["imbalance"] * 5

    with pytest.raises(IsolationForestTrainingError, match="normal"):
        train_isolation_forest(features, labels, random_state=42)


# --- Test 7 — invalid feature shape at scoring time ---


def test_score_with_mismatched_feature_count_raises_explicitly() -> None:
    features, labels = _synthetic_dataset()
    model, scaler = train_isolation_forest(features, labels, random_state=42)

    wrong_shape = pd.DataFrame(
        np.random.default_rng(4).normal(0, 1, size=(2, 4)), columns=["mean", "rms", "kurtosis", "extra"]
    )

    # Reused, not reimplemented: TASK 5.4's own ScalingError, propagated unchanged.
    with pytest.raises(ScalingError, match="columns"):
        score(model, scaler, wrong_shape)


# --- Test 8 — model (+ scaler) serialization ---


def test_model_and_scaler_serialization_roundtrip_produces_identical_scores(tmp_path: Path) -> None:
    features, labels = _synthetic_dataset()
    model, scaler = train_isolation_forest(features, labels, random_state=42)

    model_path = save_model(model, tmp_path / DEFAULT_MODEL_FILENAME)
    scaler_path = save_scaler(scaler, tmp_path / "scaler_v1.pkl")

    assert model_path.exists()
    assert model_path.name == "isolation_forest_v1.pkl"
    assert scaler_path.exists()

    reloaded_model = load_model(model_path)
    reloaded_scaler = load_scaler(scaler_path)

    new_windows = pd.DataFrame(
        np.random.default_rng(5).normal(0, 1, size=(6, 3)), columns=["mean", "rms", "kurtosis"]
    )

    original_scores = score(model, scaler, new_windows)
    reloaded_scores = score(reloaded_model, reloaded_scaler, new_windows)

    np.testing.assert_allclose(original_scores, reloaded_scores)


# --- input immutability ---


def test_train_isolation_forest_does_not_modify_input_features_or_labels() -> None:
    features, labels = _synthetic_dataset()
    features_snapshot = features.copy()
    labels_snapshot = list(labels)

    train_isolation_forest(features, labels, random_state=42)

    pd.testing.assert_frame_equal(features, features_snapshot)
    assert labels == labels_snapshot


# --- real-data validation (Definition of Done) ---
#
# 2 real 'normal' train recordings + 1 real non-normal train recording (to prove the
# filter genuinely excludes it on real data, not just synthetic data), 1 real
# 'normal' validation recording for scoring. Picked directly from data/processed/
# split_manifest.json (33 real normal train files exist there) -- not a new split,
# not a full dataset scan.
REAL_TRAIN_NORMAL = ["normal/12.288.csv", "normal/16.1792.csv"]
REAL_TRAIN_NON_NORMAL = ["horizontal-misalignment/0.5mm/12.288.csv"]
REAL_VALIDATION_NORMAL = ["normal/17.2032.csv"]
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
def real_train_features_and_labels(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Loads a small real MAFAULDA subset (2 normal + 1 non-normal train recordings,
    1 normal validation recording) via the real loader + real windowing + real TASK
    5.3 feature extraction -- no new split, no full-dataset scan."""
    tmp_path = tmp_path_factory.mktemp("isolation_forest_real_data")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "train": REAL_TRAIN_NORMAL + REAL_TRAIN_NON_NORMAL,
            "validation": REAL_VALIDATION_NORMAL,
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
    validation_windows, _validation_labels = _windows_and_labels("validation")

    train_matrix = extract_feature_matrix(
        train_windows, SAMPLING_RATE_HZ, nperseg=REAL_NPERSEG, noverlap=REAL_NOVERLAP
    )
    validation_matrix = extract_feature_matrix(
        validation_windows, SAMPLING_RATE_HZ, nperseg=REAL_NPERSEG, noverlap=REAL_NOVERLAP
    )

    return train_matrix, train_labels, validation_matrix


def test_real_data_training_uses_only_normal_windows(
    real_train_features_and_labels: tuple[pd.DataFrame, list[str], pd.DataFrame], monkeypatch
) -> None:
    train_matrix, train_labels, _validation_matrix = real_train_features_and_labels

    assert "horizontal-misalignment" in train_labels, (
        "test setup error: expected at least one real non-normal recording's windows in train"
    )
    normal_count = sum(1 for label in train_labels if label == "normal")
    non_normal_count = len(train_labels) - normal_count
    assert normal_count > 0
    assert non_normal_count > 0

    captured: dict[str, np.ndarray] = {}
    original_fit = IsolationForest.fit

    def spy_fit(self, X, y=None, sample_weight=None):
        captured["X"] = np.asarray(X).copy()
        return original_fit(self, X, y=y, sample_weight=sample_weight)

    monkeypatch.setattr(IsolationForest, "fit", spy_fit)

    train_isolation_forest(train_matrix, train_labels, random_state=42)

    assert captured["X"].shape[0] == normal_count


def test_real_data_training_and_scoring_completes(
    real_train_features_and_labels: tuple[pd.DataFrame, list[str], pd.DataFrame],
) -> None:
    train_matrix, train_labels, validation_matrix = real_train_features_and_labels

    model, scaler = train_isolation_forest(train_matrix, train_labels, random_state=42)

    validation_scores = score(model, scaler, validation_matrix)

    assert validation_scores.shape == (len(validation_matrix),)
    assert np.isfinite(validation_scores).all()
