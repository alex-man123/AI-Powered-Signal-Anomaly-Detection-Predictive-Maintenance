from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from app.ml.inference import predict
from app.ml.isolation_forest import train_isolation_forest
from app.ml.model_artifact import (
    ModelArtifactError,
    ModelArtifactMetadata,
    ModelType,
    ScoreDirection,
    compute_sha256,
    load_model_artifact,
    save_model_artifact,
)

REQUIRED_FIELDS = (
    "model_type",
    "model_version",
    "feature_names",
    "feature_dimension",
    "scaler_artifact",
    "threshold_method",
    "threshold_value",
    "score_direction",
    "training_split",
    "dataset_hash",
    "split_manifest_hash",
    "random_seed",
    "created_at",
)


def _now_iso8601() -> str:
    return datetime.now(timezone.utc).isoformat()


def _valid_metadata_kwargs(**overrides) -> dict:
    defaults = dict(
        model_type=ModelType.ISOLATION_FOREST,
        model_version="v1",
        feature_names=["rms", "kurtosis", "crest_factor"],
        feature_dimension=3,
        scaler_artifact="scaler_v1.pkl",
        threshold_method="percentile",
        threshold_value=0.73,
        score_direction=ScoreDirection.HIGHER_IS_MORE_ANOMALOUS,
        training_split="train",
        dataset_hash="sha256:" + "a" * 64,
        split_manifest_hash="sha256:" + "b" * 64,
        random_seed=42,
        created_at=_now_iso8601(),
    )
    defaults.update(overrides)
    return defaults


def _trained_model_and_scaler(seed: int = 42):
    rng = np.random.default_rng(0)
    train_features = pd.DataFrame(
        rng.normal(0, 1, size=(200, 3)), columns=["mean", "rms", "kurtosis"]
    )
    labels = ["normal"] * 200
    return train_isolation_forest(train_features, labels, random_state=seed)


# --- Test 1 — full metadata schema (AC1) ---


def test_ac1_all_required_fields_present_and_non_null_after_save_and_load(tmp_path: Path) -> None:
    model, _scaler = _trained_model_and_scaler()
    metadata = ModelArtifactMetadata(**_valid_metadata_kwargs())

    model_path, sidecar_path = save_model_artifact(model, metadata, tmp_path / "isolation_forest_v1.pkl")

    sidecar_json = json.loads(sidecar_path.read_text(encoding="utf-8"))
    for field_name in REQUIRED_FIELDS:
        assert field_name in sidecar_json, f"missing field {field_name!r}"
        assert sidecar_json[field_name] is not None, f"field {field_name!r} is null"

    _reloaded_model, reloaded_metadata = load_model_artifact(model_path)
    for field_name in REQUIRED_FIELDS:
        assert getattr(reloaded_metadata, field_name) is not None


# --- Test 2 — feature dimension consistency ---


def test_feature_dimension_must_equal_len_feature_names() -> None:
    # Valid: consistent.
    ModelArtifactMetadata(**_valid_metadata_kwargs(feature_names=["a", "b"], feature_dimension=2))

    # Invalid: mismatched -- must raise explicitly, never auto-corrected.
    with pytest.raises(ValidationError, match="feature_dimension"):
        ModelArtifactMetadata(**_valid_metadata_kwargs(feature_names=["a", "b"], feature_dimension=3))


# --- Test 3 — SHA-256 correctness (independent verification) ---


def test_compute_sha256_matches_independently_computed_digest(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_bytes(b"some real file content for hashing, not a fixture-only string")

    result = compute_sha256(file_path)

    # Computed directly with hashlib here in the test, not by calling this
    # module's own compute_sha256 a second time.
    expected_digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    assert result == f"sha256:{expected_digest}"


# --- Test 4 (AC2, CRITICAL) — single-character mutation changes the hash ---


def test_ac2_single_character_mutation_changes_hash(tmp_path: Path) -> None:
    manifest_a = tmp_path / "manifest_a.json"
    manifest_a.write_text('{"splits": {"train": ["a.csv", "b.csv"]}}', encoding="utf-8")
    hash_a = compute_sha256(manifest_a)

    manifest_b = tmp_path / "manifest_b.json"
    # Single character changed: "a.csv" -> "a.csw".
    manifest_b.write_text('{"splits": {"train": ["a.csw", "b.csv"]}}', encoding="utf-8")
    hash_b = compute_sha256(manifest_b)

    assert hash_a != hash_b


def test_ac2_mutating_a_file_in_place_changes_its_own_hash(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"splits": {"train": ["12.288.csv"]}}', encoding="utf-8")
    hash_before = compute_sha256(manifest)

    manifest.write_text('{"splits": {"train": ["12.289.csv"]}}', encoding="utf-8")
    hash_after = compute_sha256(manifest)

    assert hash_before != hash_after


# --- Test 5 — hash format ---


def test_hash_format_must_be_sha256_prefix_and_64_hex_chars() -> None:
    with pytest.raises(ValidationError, match="dataset_hash|format"):
        ModelArtifactMetadata(**_valid_metadata_kwargs(dataset_hash="not-a-real-hash"))

    with pytest.raises(ValidationError, match="split_manifest_hash|format"):
        ModelArtifactMetadata(**_valid_metadata_kwargs(split_manifest_hash="sha256:tooshort"))

    with pytest.raises(ValidationError):
        # Uppercase hex is rejected -- format requires lowercase.
        ModelArtifactMetadata(**_valid_metadata_kwargs(dataset_hash="sha256:" + "A" * 64))


def test_compute_sha256_output_matches_required_format(tmp_path: Path) -> None:
    file_path = tmp_path / "f.txt"
    file_path.write_text("hello", encoding="utf-8")

    result = compute_sha256(file_path)

    assert result.startswith("sha256:")
    digest_part = result.removeprefix("sha256:")
    assert len(digest_part) == 64
    assert all(c in "0123456789abcdef" for c in digest_part)


# --- Test 6 — sidecar naming ---


def test_sidecar_has_same_basename_as_model_artifact(tmp_path: Path) -> None:
    model, _scaler = _trained_model_and_scaler()
    metadata = ModelArtifactMetadata(**_valid_metadata_kwargs())

    model_path, sidecar_path = save_model_artifact(
        model, metadata, tmp_path / "isolation_forest_v1.pkl"
    )

    assert model_path.name == "isolation_forest_v1.pkl"
    assert sidecar_path.name == "isolation_forest_v1.json"
    assert sidecar_path.exists()


# --- Test 7 — save + load consistency ---


def test_save_and_load_round_trip_preserves_metadata(tmp_path: Path) -> None:
    model, _scaler = _trained_model_and_scaler()
    metadata = ModelArtifactMetadata(**_valid_metadata_kwargs())

    model_path, _sidecar_path = save_model_artifact(model, metadata, tmp_path / "isolation_forest_v1.pkl")
    _reloaded_model, reloaded_metadata = load_model_artifact(model_path)

    assert reloaded_metadata == metadata


# --- Test 8 — missing sidecar ---


def test_load_model_artifact_raises_explicitly_when_sidecar_missing(tmp_path: Path) -> None:
    model, _scaler = _trained_model_and_scaler()
    metadata = ModelArtifactMetadata(**_valid_metadata_kwargs())
    model_path, sidecar_path = save_model_artifact(model, metadata, tmp_path / "isolation_forest_v1.pkl")

    sidecar_path.unlink()  # model exists, sidecar deliberately removed

    with pytest.raises(ModelArtifactError, match="sidecar"):
        load_model_artifact(model_path)


def test_load_model_artifact_raises_explicitly_when_model_missing(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactError, match="not found"):
        load_model_artifact(tmp_path / "does_not_exist.pkl")


# --- Test 9 — invalid sidecar ---


def test_load_model_artifact_raises_explicitly_for_incomplete_sidecar(tmp_path: Path) -> None:
    model, _scaler = _trained_model_and_scaler()
    metadata = ModelArtifactMetadata(**_valid_metadata_kwargs())
    model_path, sidecar_path = save_model_artifact(model, metadata, tmp_path / "isolation_forest_v1.pkl")

    # Overwrite with an incomplete sidecar (missing required fields) -- must not
    # be silently accepted or filled in with defaults.
    sidecar_path.write_text(json.dumps({"model_type": "isolation_forest"}), encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="Invalid or incomplete"):
        load_model_artifact(model_path)


def test_load_model_artifact_raises_explicitly_for_malformed_json_sidecar(tmp_path: Path) -> None:
    model, _scaler = _trained_model_and_scaler()
    metadata = ModelArtifactMetadata(**_valid_metadata_kwargs())
    model_path, sidecar_path = save_model_artifact(model, metadata, tmp_path / "isolation_forest_v1.pkl")

    sidecar_path.write_text("{ this is not valid json at all", encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="Invalid or incomplete"):
        load_model_artifact(model_path)


# --- Test 10 (AC3, CRITICAL) — metadata reflects real, differing configuration ---


def test_ac3_two_artifacts_with_different_configurations_have_different_metadata(tmp_path: Path) -> None:
    model, _scaler = _trained_model_and_scaler()

    metadata_a = ModelArtifactMetadata(
        **_valid_metadata_kwargs(threshold_method="percentile", threshold_value=0.73)
    )
    metadata_b = ModelArtifactMetadata(
        **_valid_metadata_kwargs(threshold_method="validation_f1_optimal", threshold_value=0.61)
    )

    path_a, _sidecar_a = save_model_artifact(model, metadata_a, tmp_path / "artifact_a.pkl")
    path_b, _sidecar_b = save_model_artifact(model, metadata_b, tmp_path / "artifact_b.pkl")

    _model_a, reloaded_a = load_model_artifact(path_a)
    _model_b, reloaded_b = load_model_artifact(path_b)

    assert reloaded_a.threshold_method != reloaded_b.threshold_method
    assert reloaded_a.threshold_value != reloaded_b.threshold_value
    assert reloaded_a.threshold_method == "percentile"
    assert reloaded_a.threshold_value == pytest.approx(0.73)
    assert reloaded_b.threshold_method == "validation_f1_optimal"
    assert reloaded_b.threshold_value == pytest.approx(0.61)


def test_unknown_threshold_method_is_rejected() -> None:
    with pytest.raises(ValidationError, match="threshold_method"):
        ModelArtifactMetadata(**_valid_metadata_kwargs(threshold_method="made_up_method"))


# --- Test 11 — score direction ---


def test_score_direction_supports_higher_is_more_anomalous() -> None:
    metadata = ModelArtifactMetadata(
        **_valid_metadata_kwargs(score_direction=ScoreDirection.HIGHER_IS_MORE_ANOMALOUS)
    )
    assert metadata.score_direction == ScoreDirection.HIGHER_IS_MORE_ANOMALOUS


def test_score_direction_supports_lower_is_more_anomalous() -> None:
    metadata = ModelArtifactMetadata(
        **_valid_metadata_kwargs(score_direction=ScoreDirection.LOWER_IS_MORE_ANOMALOUS)
    )
    assert metadata.score_direction == ScoreDirection.LOWER_IS_MORE_ANOMALOUS


def test_invalid_score_direction_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelArtifactMetadata(**_valid_metadata_kwargs(score_direction="sideways"))


# --- Test 12 — Autoencoder schema compatibility (NOT training/persistence) ---


def test_autoencoder_model_type_is_accepted_by_the_schema() -> None:
    metadata = ModelArtifactMetadata(
        **_valid_metadata_kwargs(
            model_type=ModelType.AUTOENCODER,
            model_version="v1",
            scaler_artifact="scaler_v1.pkl",
        )
    )
    assert metadata.model_type == ModelType.AUTOENCODER


def test_save_model_artifact_rejects_autoencoder_model_type_explicitly(tmp_path: Path) -> None:
    """Schema support != serialization support: the metadata schema accepts
    'autoencoder', but no Autoencoder serializer exists yet (TASK 7.3) -- this
    must fail loudly, not silently pretend to save a torch model via joblib."""
    fake_model = object()
    metadata = ModelArtifactMetadata(**_valid_metadata_kwargs(model_type=ModelType.AUTOENCODER))

    with pytest.raises(ModelArtifactError, match="autoencoder"):
        save_model_artifact(fake_model, metadata, tmp_path / "autoencoder_v1.pt")


def test_load_model_artifact_rejects_autoencoder_sidecar_explicitly(tmp_path: Path) -> None:
    model_path = tmp_path / "autoencoder_v1.pt"
    model_path.write_bytes(b"not a real torch checkpoint, just a placeholder for this test")
    sidecar_path = tmp_path / "autoencoder_v1.json"
    metadata = ModelArtifactMetadata(**_valid_metadata_kwargs(model_type=ModelType.AUTOENCODER))
    sidecar_path.write_text(metadata.model_dump_json(), encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="autoencoder"):
        load_model_artifact(model_path)


# --- Test 13 — random seed ---


def test_random_seed_is_required_and_must_be_integer() -> None:
    metadata = ModelArtifactMetadata(**_valid_metadata_kwargs(random_seed=123))
    assert metadata.random_seed == 123

    with pytest.raises(ValidationError):
        ModelArtifactMetadata(**_valid_metadata_kwargs(random_seed="not-an-int"))


# --- Test 14 — created_at ISO8601 ---


def test_created_at_must_be_valid_iso8601() -> None:
    valid = _now_iso8601()
    metadata = ModelArtifactMetadata(**_valid_metadata_kwargs(created_at=valid))
    # Must round-trip through datetime.fromisoformat without error.
    datetime.fromisoformat(metadata.created_at)

    with pytest.raises(ValidationError, match="created_at"):
        ModelArtifactMetadata(**_valid_metadata_kwargs(created_at="not a timestamp at all"))


def test_training_split_only_accepts_train() -> None:
    ModelArtifactMetadata(**_valid_metadata_kwargs(training_split="train"))

    with pytest.raises(ValidationError):
        ModelArtifactMetadata(**_valid_metadata_kwargs(training_split="validation"))
    with pytest.raises(ValidationError):
        ModelArtifactMetadata(**_valid_metadata_kwargs(training_split="test"))


# --- Test 15 — model inference compatibility (no retraining, TASK 6.4 preserved) ---


def test_model_inference_identical_after_artifact_save_and_load(tmp_path: Path) -> None:
    model, scaler = _trained_model_and_scaler()
    metadata = ModelArtifactMetadata(**_valid_metadata_kwargs())

    new_samples = pd.DataFrame(
        np.random.default_rng(7).normal(0, 1, size=(10, 3)), columns=["mean", "rms", "kurtosis"]
    )
    predictions_before = predict(model, scaler, new_samples)

    model_path, _sidecar_path = save_model_artifact(model, metadata, tmp_path / "isolation_forest_v1.pkl")
    reloaded_model, reloaded_metadata = load_model_artifact(model_path)

    predictions_after = predict(reloaded_model, scaler, new_samples)

    np.testing.assert_array_equal(predictions_before, predictions_after)
    assert reloaded_model.n_estimators == model.n_estimators
    assert len(reloaded_model.estimators_) == len(model.estimators_)
    assert reloaded_metadata.random_seed == 42
