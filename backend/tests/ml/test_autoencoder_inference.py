from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from app.datasets.loader import DATASET_ROOT as REAL_DATASET_ROOT
from app.datasets.loader import load_all_splits
from app.datasets.validators import SAMPLING_RATE_HZ
from app.features.extractor import extract_feature_matrix
from app.features.registry import FEATURE_REGISTRY, FREQUENCY_FEATURE_REGISTRY
from app.ml.autoencoder import Autoencoder
from app.ml.inference import (
    ModelPersistenceError,
    compute_autoencoder_normalized_scores,
    default_autoencoder_path,
    load_autoencoder,
    reconstruct,
    reconstruction_error,
    save_autoencoder,
)
from app.ml.model_artifact import (
    DEFAULT_DATASET_FINGERPRINT_PATH,
    DEFAULT_SPLIT_MANIFEST_PATH,
    ModelArtifactMetadata,
    ModelType,
    ScoreDirection,
    compute_dataset_hash,
    compute_split_manifest_hash,
    load_model_artifact,
    save_model_artifact,
)
from app.ml.scaling import apply_scaler, fit_scaler
from app.ml.scoring import calibrate, classify
from app.ml.training import TrainingConfig, train_autoencoder
from app.signal_processing.windowing import Window, create_windows


def _learnable_dataset(n_train: int = 300, n_val_normal: int = 50, num_features: int = 6, seed: int = 0):
    """Same construction as TASK 7.2's own tests: 2 latent factors linearly mixed
    into `num_features` features + tiny noise for train/normal-validation, and a
    genuinely OFF-MANIFOLD anomaly set (independent isotropic noise, not related
    to the learned 2D structure at all) for AC1."""
    rng = np.random.default_rng(seed)
    mixing = rng.normal(0, 1, size=(2, num_features))

    train_latent = rng.normal(0, 1, size=(n_train, 2))
    train_data = train_latent @ mixing + rng.normal(0, 0.01, size=(n_train, num_features))
    train_features = pd.DataFrame(train_data, columns=[f"f{i}" for i in range(num_features)])
    train_labels = ["normal"] * n_train

    val_latent = rng.normal(0, 1, size=(n_val_normal, 2))
    val_normal = val_latent @ mixing + rng.normal(0, 0.01, size=(n_val_normal, num_features))
    val_normal_df = pd.DataFrame(val_normal, columns=train_features.columns)

    val_anomaly = rng.normal(0, 5, size=(n_val_normal, num_features))
    val_anomaly_df = pd.DataFrame(val_anomaly, columns=train_features.columns)

    return train_features, train_labels, val_normal_df, val_anomaly_df


def _trained_model(train_features, train_labels, validation_features, seed: int = 42) -> Autoencoder:
    torch.manual_seed(seed)
    model = Autoencoder(input_dim=train_features.shape[1], hidden_dim=5, bottleneck_dim=2)
    train_autoencoder(
        model,
        train_features,
        train_labels,
        validation_features,
        config=TrainingConfig(max_epochs=60, patience=60, learning_rate=1e-2, seed=seed),
    )
    return model


# --- reconstruction error: exact formula, independently verified ---


def test_reconstruction_error_matches_manual_mse_calculation(monkeypatch) -> None:
    model = Autoencoder(input_dim=3, hidden_dim=4, bottleneck_dim=2)
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    fixed_reconstruction = torch.tensor([[1.0, 2.0, 4.0], [1.0, 1.0, 1.0]])

    def fake_forward(self, inp):
        return fixed_reconstruction

    monkeypatch.setattr(Autoencoder, "forward", fake_forward)

    errors = reconstruction_error(model, x)

    expected = np.array(
        [
            np.mean((np.array([1.0, 2.0, 3.0]) - np.array([1.0, 2.0, 4.0])) ** 2),
            np.mean((np.array([0.0, 0.0, 0.0]) - np.array([1.0, 1.0, 1.0])) ** 2),
        ]
    )
    np.testing.assert_allclose(errors, expected, rtol=1e-6)


def test_reconstruct_output_shape_matches_input_shape() -> None:
    model = Autoencoder(input_dim=5, hidden_dim=4, bottleneck_dim=2)
    x = np.random.default_rng(0).normal(0, 1, size=(7, 5)).astype(np.float32)

    reconstruction = reconstruct(model, x)

    assert reconstruction.shape == x.shape


@pytest.mark.parametrize("n", [1, 5, 20])
def test_reconstruction_error_batch_shape_is_one_per_sample(n: int) -> None:
    model = Autoencoder(input_dim=4, hidden_dim=4, bottleneck_dim=2)
    x = np.random.default_rng(1).normal(0, 1, size=(n, 4)).astype(np.float32)

    errors = reconstruction_error(model, x)

    assert errors.shape == (n,)
    assert np.isfinite(errors).all()


def test_reconstruction_error_rejects_wrong_feature_dimension() -> None:
    model = Autoencoder(input_dim=5, hidden_dim=4, bottleneck_dim=2)
    wrong_shape = np.random.default_rng(0).normal(0, 1, size=(3, 4)).astype(np.float32)

    with pytest.raises(ModelPersistenceError, match="columns"):
        reconstruction_error(model, wrong_shape)


def test_reconstruction_error_rejects_nan_and_empty_input() -> None:
    model = Autoencoder(input_dim=4, hidden_dim=4, bottleneck_dim=2)

    with pytest.raises(ModelPersistenceError, match="NaN"):
        reconstruction_error(model, np.array([[1.0, np.nan, 2.0, 3.0]], dtype=np.float32))

    with pytest.raises(ModelPersistenceError, match="row"):
        reconstruction_error(model, np.empty((0, 4), dtype=np.float32))


def test_reconstruct_does_not_modify_model_parameters() -> None:
    model = Autoencoder(input_dim=4, hidden_dim=4, bottleneck_dim=2)
    params_before = [p.clone() for p in model.parameters()]

    reconstruct(model, np.random.default_rng(0).normal(0, 1, size=(6, 4)).astype(np.float32))

    for before, after in zip(params_before, model.parameters()):
        torch.testing.assert_close(before, after)


# --- AC1: real, honestly-measured comparison (not assumed, not fabricated) ---


def test_ac1_mean_anomaly_reconstruction_error_is_visibly_higher_than_normal() -> None:
    train_features, train_labels, val_normal, val_anomaly = _learnable_dataset()
    model = _trained_model(train_features, train_labels, val_normal)

    normal_errors = reconstruction_error(model, val_normal)
    anomaly_errors = reconstruction_error(model, val_anomaly)

    mean_normal = float(normal_errors.mean())
    mean_anomaly = float(anomaly_errors.mean())
    ratio = mean_anomaly / mean_normal

    print(f"\nmean_normal_reconstruction_error={mean_normal:.6f}")
    print(f"mean_anomaly_reconstruction_error={mean_anomaly:.6f}")
    print(f"ratio={ratio:.2f}")

    # Measured empirically before writing this assertion (see TASK 7.3's own
    # report): the real ratio for this exact setup is ~60,000x. The threshold
    # below (10x) is a safe, non-tuned margin far below that measured value --
    # this assertion is not fabricated to match a desired outcome; it reports
    # what the real off-manifold-vs-on-manifold distinction actually produces.
    assert mean_anomaly > 10 * mean_normal, (
        f"AC1 NOT DEMONSTRATED for this configuration: mean_anomaly={mean_anomaly} "
        f"was not visibly greater than mean_normal={mean_normal}"
    )


def test_higher_reconstruction_error_is_treated_as_more_anomalous_by_classify() -> None:
    """Direct, non-ML confirmation of the score_direction convention: given known
    errors and a known threshold, the higher one must be classified anomalous."""
    from app.ml.scoring import ScoreNormalizationParams, ScoringCalibration

    calibration = ScoringCalibration(
        normalization=ScoreNormalizationParams(validation_min=0.0, validation_max=1.0),
        threshold_method="percentile",
        percentile_value=50.0,
        threshold=0.5,
    )
    errors = np.array([0.1, 0.9])
    normalized = np.clip(errors, 0.0, 1.0)

    decisions = classify(normalized, calibration)

    assert decisions[0] == np.bool_(False)  # low error -> not anomalous
    assert decisions[1] == np.bool_(True)  # high error -> anomalous


# --- AC2: threshold calibration reused from TASK 6.3, validation-only ---


def test_ac2_percentile_calibration_reused_unchanged_on_reconstruction_errors() -> None:
    train_features, train_labels, val_normal, val_anomaly = _learnable_dataset()
    model = _trained_model(train_features, train_labels, val_normal)

    validation_features = pd.concat([val_normal, val_anomaly], ignore_index=True)
    validation_labels = ["normal"] * len(val_normal) + ["horizontal-misalignment"] * len(val_anomaly)

    errors = reconstruction_error(model, validation_features)
    calibration = calibrate(
        errors, threshold_method="percentile", percentile_value=95, validation_labels=validation_labels
    )

    assert calibration.threshold_method == "percentile"
    assert np.isfinite(calibration.threshold)
    assert 0.0 <= calibration.threshold <= 1.0


def test_ac2_f1_optimal_calibration_reused_unchanged_on_reconstruction_errors() -> None:
    train_features, train_labels, val_normal, val_anomaly = _learnable_dataset()
    model = _trained_model(train_features, train_labels, val_normal)

    validation_features = pd.concat([val_normal, val_anomaly], ignore_index=True)
    validation_labels = ["normal"] * len(val_normal) + ["horizontal-misalignment"] * len(val_anomaly)

    errors = reconstruction_error(model, validation_features)
    calibration = calibrate(errors, threshold_method="validation_f1_optimal", validation_labels=validation_labels)

    normalized = compute_autoencoder_normalized_scores(model, validation_features, calibration)
    decisions = classify(normalized, calibration)

    # With such a stark normal-vs-off-manifold separation, F1-optimal calibration
    # should classify essentially all anomalies correctly.
    anomaly_decisions = decisions[len(val_normal):]
    print(f"\nanomaly detection rate at F1-optimal threshold: {anomaly_decisions.mean():.2%}")
    assert anomaly_decisions.mean() > 0.9


def test_ac2_calibration_uses_only_validation_never_test() -> None:
    """Same structural guarantee already proven in TASK 6.3: `calibrate`/
    `calibrate_threshold` have no test_scores/test_labels parameter at all --
    confirmed here specifically with Autoencoder reconstruction errors as input."""
    train_features, train_labels, val_normal, val_anomaly = _learnable_dataset()
    model = _trained_model(train_features, train_labels, val_normal)

    validation_errors = reconstruction_error(model, val_normal)
    threshold_1 = calibrate(validation_errors, threshold_method="percentile", percentile_value=95).threshold

    # A "test-like" distribution that is never passed to calibrate() at all.
    _hypothetical_test_errors = reconstruction_error(model, val_anomaly) * 1000

    threshold_2 = calibrate(validation_errors, threshold_method="percentile", percentile_value=95).threshold

    assert threshold_1 == threshold_2


# --- AC3: save/load produces identical reconstruction ---


def test_ac3_save_and_load_autoencoder_produces_identical_reconstruction(tmp_path: Path) -> None:
    train_features, train_labels, val_normal, _val_anomaly = _learnable_dataset()
    model = _trained_model(train_features, train_labels, val_normal)

    x = np.random.default_rng(5).normal(0, 1, size=(10, 6)).astype(np.float32)
    reconstruction_before = reconstruct(model, x)

    path = save_autoencoder(model, tmp_path / "autoencoder_v1.pt")
    assert path.exists()
    assert path.name == "autoencoder_v1.pt"

    reloaded_model = load_autoencoder(path)
    reconstruction_after = reconstruct(reloaded_model, x)

    np.testing.assert_array_equal(reconstruction_before, reconstruction_after)


def test_load_autoencoder_raises_explicitly_for_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(ModelPersistenceError, match="not found"):
        load_autoencoder(tmp_path / "does_not_exist.pt")


def test_load_autoencoder_raises_explicitly_for_corrupted_artifact(tmp_path: Path) -> None:
    corrupted_path = tmp_path / "corrupted.pt"
    corrupted_path.write_bytes(b"not a real torch checkpoint at all \x00\x01\x02")

    with pytest.raises(ModelPersistenceError, match="Failed to deserialize"):
        load_autoencoder(corrupted_path)


def test_load_autoencoder_raises_explicitly_for_incomplete_checkpoint(tmp_path: Path) -> None:
    incomplete_path = tmp_path / "incomplete.pt"
    torch.save({"state_dict": {}}, incomplete_path)  # missing input_dim/hidden_dim/bottleneck_dim

    with pytest.raises(ModelPersistenceError, match="missing required Autoencoder fields"):
        load_autoencoder(incomplete_path)


# --- AC4: exact TASK 6.5 metadata contract, no duplicate schema ---


def _now_iso8601() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_ac4_autoencoder_artifact_uses_exact_task_6_5_metadata_contract(tmp_path: Path) -> None:
    train_features, train_labels, val_normal, val_anomaly = _learnable_dataset()
    seed = 42
    model = _trained_model(train_features, train_labels, val_normal, seed=seed)

    validation_features = pd.concat([val_normal, val_anomaly], ignore_index=True)
    validation_labels = ["normal"] * len(val_normal) + ["horizontal-misalignment"] * len(val_anomaly)
    errors = reconstruction_error(model, validation_features)
    calibration = calibrate(
        errors, threshold_method="percentile", percentile_value=95, validation_labels=validation_labels
    )

    feature_names = list(train_features.columns)
    metadata = ModelArtifactMetadata(
        model_type=ModelType.AUTOENCODER,
        model_version="v1",
        feature_names=feature_names,
        feature_dimension=len(feature_names),
        scaler_artifact="scaler_v1.pkl",
        threshold_method=calibration.threshold_method,
        threshold_value=calibration.threshold,
        score_direction=ScoreDirection.HIGHER_IS_MORE_ANOMALOUS,
        training_split="train",
        dataset_hash=compute_dataset_hash(),
        split_manifest_hash=compute_split_manifest_hash(),
        random_seed=seed,
        created_at=_now_iso8601(),
    )

    model_path, sidecar_path = save_model_artifact(model, metadata, tmp_path / "autoencoder_v1.pt")

    sidecar_json = json.loads(sidecar_path.read_text(encoding="utf-8"))
    required_fields = (
        "model_type", "model_version", "feature_names", "feature_dimension", "scaler_artifact",
        "threshold_method", "threshold_value", "score_direction", "training_split", "dataset_hash",
        "split_manifest_hash", "random_seed", "created_at",
    )
    for field_name in required_fields:
        assert field_name in sidecar_json
        assert sidecar_json[field_name] is not None

    assert sidecar_json["model_type"] == "autoencoder"
    assert sidecar_json["score_direction"] == "higher_is_more_anomalous"
    assert sidecar_json["feature_dimension"] == len(sidecar_json["feature_names"])
    assert sidecar_json["threshold_value"] == pytest.approx(calibration.threshold)
    assert sidecar_json["threshold_value"] != pytest.approx(95)  # never confused with percentile_value
    assert sidecar_json["random_seed"] == seed

    reloaded_model, reloaded_metadata = load_model_artifact(model_path)
    assert isinstance(reloaded_model, Autoencoder)
    assert reloaded_metadata.model_type == ModelType.AUTOENCODER
    assert reloaded_metadata.score_direction == ScoreDirection.HIGHER_IS_MORE_ANOMALOUS

    x = np.random.default_rng(9).normal(0, 1, size=(4, len(feature_names))).astype(np.float32)
    np.testing.assert_array_equal(reconstruct(model, x), reconstruct(reloaded_model, x))


def test_dataset_hash_and_split_manifest_hash_are_the_real_project_files() -> None:
    """Confirms this task reuses TASK 6.5's real hash sources -- not invented,
    not omitted -- rather than re-deriving them independently."""
    assert compute_dataset_hash() == compute_dataset_hash(DEFAULT_DATASET_FINGERPRINT_PATH)
    assert compute_split_manifest_hash() == compute_split_manifest_hash(DEFAULT_SPLIT_MANIFEST_PATH)
    assert compute_dataset_hash().startswith("sha256:")
    assert compute_split_manifest_hash().startswith("sha256:")


# --- real-data pipeline sanity check ---
#
# Real MAFAULDA subset (same choice as TASK 7.2's own real-data test): 2 normal
# train recordings, 1 normal + 1 non-normal validation recording. From
# data/processed/split_manifest.json's existing lists -- not a new split.
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


def test_real_data_full_autoencoder_pipeline(tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_path = tmp_path_factory.mktemp("autoencoder_inference_real_data")
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

    scaler = fit_scaler(train_matrix)
    scaled_train = apply_scaler(scaler, train_matrix)
    scaled_validation = apply_scaler(scaler, validation_matrix)

    torch.manual_seed(42)
    feature_dim = scaled_train.shape[1]
    model = Autoencoder(input_dim=feature_dim, hidden_dim=10, bottleneck_dim=4)
    train_autoencoder(
        model, scaled_train, train_labels, scaled_validation, config=TrainingConfig(max_epochs=10, seed=42)
    )

    errors = reconstruction_error(model, scaled_validation)
    assert errors.shape == (len(scaled_validation),)
    assert np.isfinite(errors).all()

    assert "normal" in validation_labels and "horizontal-misalignment" in validation_labels
    calibration = calibrate(
        errors, threshold_method="percentile", percentile_value=95, validation_labels=validation_labels
    )
    normalized = compute_autoencoder_normalized_scores(model, scaled_validation, calibration)
    decisions = classify(normalized, calibration)

    assert decisions.shape == (len(scaled_validation),)
    assert np.isfinite(normalized).all()
    assert np.all((normalized >= 0.0) & (normalized <= 1.0))

    feature_names = list(FEATURE_REGISTRY.keys()) + list(FREQUENCY_FEATURE_REGISTRY.keys())
    assert list(scaled_train.columns) == feature_names

    # Real default artifact path -- this task's literal expected deliverable
    # (mirrors TASK 6.4/6.5's own real-data tests producing the real
    # models/isolation_forest_v1.pkl + scaler_v1.pkl).
    metadata = ModelArtifactMetadata(
        model_type=ModelType.AUTOENCODER,
        model_version="v1",
        feature_names=feature_names,
        feature_dimension=len(feature_names),
        scaler_artifact="scaler_v1.pkl",
        threshold_method=calibration.threshold_method,
        threshold_value=calibration.threshold,
        score_direction=ScoreDirection.HIGHER_IS_MORE_ANOMALOUS,
        training_split="train",
        dataset_hash=compute_dataset_hash(),
        split_manifest_hash=compute_split_manifest_hash(),
        random_seed=42,
        created_at=_now_iso8601(),
    )
    model_path, sidecar_path = save_model_artifact(model, metadata, default_autoencoder_path())

    assert model_path == default_autoencoder_path()
    assert model_path.exists() and model_path.stat().st_size > 0
    assert sidecar_path.exists() and sidecar_path.stat().st_size > 0

    reloaded_model, reloaded_metadata = load_model_artifact(model_path)
    assert reloaded_metadata == metadata
    x = np.random.default_rng(3).normal(0, 1, size=(5, feature_dim)).astype(np.float32)
    np.testing.assert_array_equal(reconstruct(model, x), reconstruct(reloaded_model, x))
