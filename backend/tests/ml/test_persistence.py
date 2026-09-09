from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.datasets.loader import DATASET_ROOT as REAL_DATASET_ROOT
from app.datasets.loader import load_all_splits
from app.datasets.validators import SAMPLING_RATE_HZ
from app.features.extractor import extract_feature_matrix
from app.ml.inference import (
    ModelPersistenceError,
    default_model_path,
    default_scaler_path,
    load_model,
    load_scaler,
    predict,
    save_model,
    save_scaler,
)
from app.ml.isolation_forest import train_isolation_forest
from app.ml.scaling import ScalingError
from app.signal_processing.windowing import Window, create_windows


def _trained_model_and_scaler(seed: int = 42):
    rng = np.random.default_rng(0)
    train_features = pd.DataFrame(
        rng.normal(0, 1, size=(200, 3)), columns=["mean", "rms", "kurtosis"]
    )
    labels = ["normal"] * 200
    return train_isolation_forest(train_features, labels, random_state=seed)


# --- AC1 (CRITICAL) — save/load consistency ---


def test_ac1_single_sample_predictions_identical_after_save_and_load(tmp_path: Path) -> None:
    model, scaler = _trained_model_and_scaler()
    new_sample = pd.DataFrame([[0.1, -0.2, 0.05]], columns=["mean", "rms", "kurtosis"])

    predictions_before = predict(model, scaler, new_sample)

    model_path = save_model(model, tmp_path / "isolation_forest_v1.pkl")
    scaler_path = save_scaler(scaler, tmp_path / "scaler_v1.pkl")

    reloaded_model = load_model(model_path)
    reloaded_scaler = load_scaler(scaler_path)

    predictions_after = predict(reloaded_model, reloaded_scaler, new_sample)

    # joblib serializes a fitted estimator's exact internal state (no
    # recomputation happens on reload) -- verified independently before writing
    # this test (see TASK 6.4's own report), so bit-exact equality is the
    # documented tolerance here, not an approximate one.
    assert predictions_before.shape == predictions_after.shape
    np.testing.assert_array_equal(predictions_before, predictions_after)


def test_ac1_multiple_samples_predictions_identical_after_save_and_load(tmp_path: Path) -> None:
    model, scaler = _trained_model_and_scaler()
    rng = np.random.default_rng(99)
    new_samples = pd.DataFrame(
        rng.normal(0, 1, size=(15, 3)), columns=["mean", "rms", "kurtosis"]
    )

    predictions_before = predict(model, scaler, new_samples)

    model_path = save_model(model, tmp_path / "isolation_forest_v1.pkl")
    scaler_path = save_scaler(scaler, tmp_path / "scaler_v1.pkl")
    reloaded_model = load_model(model_path)
    reloaded_scaler = load_scaler(scaler_path)

    predictions_after = predict(reloaded_model, reloaded_scaler, new_samples)

    assert predictions_before.shape == (15,) == predictions_after.shape
    np.testing.assert_array_equal(predictions_before, predictions_after)


def test_ac1_reloaded_scaler_parameters_are_bit_identical(tmp_path: Path) -> None:
    _model, scaler = _trained_model_and_scaler()

    scaler_path = save_scaler(scaler, tmp_path / "scaler_v1.pkl")
    reloaded_scaler = load_scaler(scaler_path)

    np.testing.assert_array_equal(reloaded_scaler.mean_, scaler.mean_)
    np.testing.assert_array_equal(reloaded_scaler.scale_, scaler.scale_)


def test_no_retraining_happens_on_load_model_parameters_unchanged(tmp_path: Path) -> None:
    """Loaded model's own hyperparameters/fitted attributes match the original
    exactly -- if `load_model` accidentally created-and-fit a NEW model instead of
    deserializing, `n_estimators`/`estimators_` would very likely differ."""
    model, _scaler = _trained_model_and_scaler()

    model_path = save_model(model, tmp_path / "isolation_forest_v1.pkl")
    reloaded_model = load_model(model_path)

    assert reloaded_model.n_estimators == model.n_estimators
    assert reloaded_model.random_state == model.random_state
    assert len(reloaded_model.estimators_) == len(model.estimators_)


# --- artifact file checks ---


def test_save_model_creates_non_empty_file(tmp_path: Path) -> None:
    model, _scaler = _trained_model_and_scaler()

    path = save_model(model, tmp_path / "isolation_forest_v1.pkl")

    assert path.exists()
    assert path.is_file()
    assert path.stat().st_size > 0


def test_save_scaler_creates_non_empty_file(tmp_path: Path) -> None:
    _model, scaler = _trained_model_and_scaler()

    path = save_scaler(scaler, tmp_path / "scaler_v1.pkl")

    assert path.exists()
    assert path.is_file()
    assert path.stat().st_size > 0


# --- missing artifact ---


def test_load_model_raises_explicitly_for_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(ModelPersistenceError, match="not found"):
        load_model(tmp_path / "does_not_exist.pkl")


def test_load_scaler_raises_explicitly_for_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(ModelPersistenceError, match="not found"):
        load_scaler(tmp_path / "does_not_exist.pkl")


# --- corrupted artifact ---


def test_load_model_raises_explicitly_for_corrupted_artifact(tmp_path: Path) -> None:
    corrupted_path = tmp_path / "corrupted.pkl"
    corrupted_path.write_bytes(b"this is not a valid pickle/joblib file at all \x00\x01\x02")

    with pytest.raises(ModelPersistenceError, match="Failed to deserialize"):
        load_model(corrupted_path)


def test_load_scaler_raises_explicitly_for_corrupted_artifact(tmp_path: Path) -> None:
    corrupted_path = tmp_path / "corrupted.pkl"
    corrupted_path.write_bytes(b"also not a valid pickle/joblib file \xff\xfe\xfd")

    with pytest.raises(ModelPersistenceError, match="Failed to deserialize"):
        load_scaler(corrupted_path)


def test_load_model_raises_explicitly_for_wrong_artifact_type(tmp_path: Path) -> None:
    """A validly-pickled object that is simply not an IsolationForest (e.g. a
    scaler saved at the model's path by mistake) must be rejected explicitly, not
    silently accepted as if it were usable for inference."""
    _model, scaler = _trained_model_and_scaler()
    wrong_path = save_scaler(scaler, tmp_path / "wrong_type.pkl")

    with pytest.raises(ModelPersistenceError, match="IsolationForest"):
        load_model(wrong_path)


def test_load_scaler_raises_explicitly_for_wrong_artifact_type(tmp_path: Path) -> None:
    model, _scaler = _trained_model_and_scaler()
    wrong_path = save_model(model, tmp_path / "wrong_type.pkl")

    with pytest.raises(ModelPersistenceError, match="StandardScaler"):
        load_scaler(wrong_path)


# --- inference shape validation ---


def test_predict_with_wrong_number_of_features_raises_explicitly(tmp_path: Path) -> None:
    model, scaler = _trained_model_and_scaler()
    model_path = save_model(model, tmp_path / "isolation_forest_v1.pkl")
    scaler_path = save_scaler(scaler, tmp_path / "scaler_v1.pkl")
    reloaded_model = load_model(model_path)
    reloaded_scaler = load_scaler(scaler_path)

    wrong_shape = pd.DataFrame(
        np.random.default_rng(1).normal(0, 1, size=(2, 4)),
        columns=["mean", "rms", "kurtosis", "extra"],
    )

    # Reused, not reimplemented: TASK 5.4's own ScalingError, propagated unchanged.
    with pytest.raises(ScalingError, match="columns"):
        predict(reloaded_model, reloaded_scaler, wrong_shape)


# --- default path resolution ---


def test_default_model_and_scaler_paths_use_repo_models_directory() -> None:
    model_path = default_model_path()
    scaler_path = default_scaler_path()

    assert model_path.name == "isolation_forest_v1.pkl"
    assert scaler_path.name == "scaler_v1.pkl"
    assert model_path.parent == scaler_path.parent
    assert model_path.parent.name == "models"


# --- real-data validation (Definition of Done) ---
#
# Real MAFAULDA subset (2 normal train recordings, matching TASK 6.2's own choice)
# from data/processed/split_manifest.json -- not a new split, not a full dataset
# scan. Saves the REAL artifacts to this project's actual default models/ directory
# (already .gitignore'd for *.pkl, already containing a .gitkeep placeholder),
# demonstrating the literal deliverable this task asks for
# ("models/isolation_forest_v1.pkl"), in addition to the tmp_path-isolated tests
# above (which avoid any stale-artifact interference between test runs).
REAL_TRAIN_NORMAL = ["normal/12.288.csv", "normal/16.1792.csv"]
REAL_WINDOW_SIZE = 1024
REAL_OVERLAP = 0.5
REAL_CHANNEL = 0
REAL_NPERSEG = 256
REAL_NOVERLAP = 128


def _write_manifest(tmp_path: Path, splits: dict[str, list[str]]) -> Path:
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps({"splits": splits}), encoding="utf-8")
    return manifest_path


def test_real_data_end_to_end_save_load_and_infer(tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_path = tmp_path_factory.mktemp("inference_real_data")
    manifest_path = _write_manifest(tmp_path, {"train": REAL_TRAIN_NORMAL, "validation": [], "test": []})

    loaded = load_all_splits(dataset_root=REAL_DATASET_ROOT, manifest_path=manifest_path)

    windows: list[Window] = []
    labels: list[str] = []
    for recording in loaded["train"]:
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

    train_matrix = extract_feature_matrix(windows, SAMPLING_RATE_HZ, nperseg=REAL_NPERSEG, noverlap=REAL_NOVERLAP)

    model, scaler = train_isolation_forest(train_matrix, labels, random_state=42)

    predictions_before = predict(model, scaler, train_matrix)

    # Real default artifact paths -- this task's literal expected deliverable.
    model_path = save_model(model)
    scaler_path = save_scaler(scaler)

    assert model_path == default_model_path()
    assert model_path.exists() and model_path.stat().st_size > 0
    assert scaler_path.exists() and scaler_path.stat().st_size > 0

    reloaded_model = load_model()
    reloaded_scaler = load_scaler()

    predictions_after = predict(reloaded_model, reloaded_scaler, train_matrix)

    np.testing.assert_array_equal(predictions_before, predictions_after)
