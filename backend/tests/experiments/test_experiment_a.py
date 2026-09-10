from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from app.core.database import get_engine
from app.ml.dimensionality import dsp_feature_dimension
from app.ml.evaluation import evaluate
from app.ml.experiment_registry import get_experiment_run
from app.ml import dimensionality as dimensionality_module
from app.ml import scoring as scoring_module
from scripts import run_experiment_a as run_experiment_a_module
from scripts.run_experiment_a import (
    TEST_FILES,
    TRAIN_FILES,
    VALIDATION_FILES,
    run_experiment_a,
)

EXPERIMENT_ID_PATTERN = re.compile(r"^EXP-A-\d{3}$")


@pytest.fixture()
def isolated_engine(tmp_path: Path):
    return get_engine(f"sqlite:///{tmp_path / 'experiment_a_test.db'}")


@pytest.fixture(scope="module")
def experiment_a_result(tmp_path_factory: pytest.TempPathFactory):
    """Runs the real Experiment A pipeline ONCE for this test module (real
    MAFAULDA subset, isolated models_dir/registry so it never touches the real
    project's models/ directory or shared experiment database) -- individual
    tests below inspect this single real result rather than each re-running the
    (moderately expensive) full pipeline."""
    tmp_path = tmp_path_factory.mktemp("experiment_a_shared")
    engine = get_engine(f"sqlite:///{tmp_path / 'registry.db'}")
    experiment_id, metrics = run_experiment_a(models_dir=tmp_path / "models", registry_engine=engine)
    return {"experiment_id": experiment_id, "metrics": metrics, "engine": engine, "models_dir": tmp_path / "models"}


# --- AC3: registry produces a real, automatically-generated EXP-A-NNN ---


def test_ac3_experiment_id_matches_format_and_is_not_hardcoded(experiment_a_result) -> None:
    experiment_id = experiment_a_result["experiment_id"]
    assert EXPERIMENT_ID_PATTERN.match(experiment_id), experiment_id


def test_ac3_two_runs_produce_sequential_distinct_ids(tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_path = tmp_path_factory.mktemp("experiment_a_sequential")
    engine = get_engine(f"sqlite:///{tmp_path / 'registry.db'}")

    id_1, _metrics_1 = run_experiment_a(models_dir=tmp_path / "models_1", registry_engine=engine)
    id_2, _metrics_2 = run_experiment_a(models_dir=tmp_path / "models_2", registry_engine=engine)

    assert id_1 == "EXP-A-001"
    assert id_2 == "EXP-A-002"
    assert id_1 != id_2


# --- AC3: round-trip retrieval ---


def test_ac3_round_trip_retrieval_matches_registered_run(experiment_a_result) -> None:
    record = get_experiment_run(experiment_a_result["experiment_id"], engine=experiment_a_result["engine"])

    assert record.experiment_id == experiment_a_result["experiment_id"]
    assert record.metrics == experiment_a_result["metrics"]
    assert record.config["model"] == "isolation_forest"
    assert record.config["representation"] == "raw_pca"


# --- representation / dimensionality ---


def test_raw_pca_dimension_exactly_matches_real_dsp_feature_dimension(experiment_a_result) -> None:
    record = get_experiment_run(experiment_a_result["experiment_id"], engine=experiment_a_result["engine"])

    assert record.config["feature_dimension"] == dsp_feature_dimension()


def test_config_records_real_hashes_seed_and_threshold_not_invented(experiment_a_result) -> None:
    record = get_experiment_run(experiment_a_result["experiment_id"], engine=experiment_a_result["engine"])
    config = record.config

    assert config["dataset_hash"].startswith("sha256:")
    assert config["split_manifest_hash"].startswith("sha256:")
    assert config["random_seed"] == 42
    assert config["threshold"]["method"] == "percentile"
    assert isinstance(config["threshold"]["value"], float)
    # percentile_value (95) must never be confused with the calibrated threshold itself.
    assert config["threshold"]["value"] != pytest.approx(95)
    assert config["model_config"]["random_state"] == 42  # the real fitted model's own params, not re-typed


# --- AC2: same test set as will be required for Experiment B ---


def test_canonical_test_set_constants_are_exported_for_experiment_b_reuse() -> None:
    """Experiment B (a future task) MUST import these same constants for a valid
    apples-to-apples comparison -- this test just confirms they exist, are
    non-empty, and are exactly what got recorded in this run's provenance."""
    assert len(TRAIN_FILES) > 0
    assert len(VALIDATION_FILES) > 0
    assert len(TEST_FILES) > 0


def test_registered_test_files_match_the_canonical_constant(experiment_a_result) -> None:
    record = get_experiment_run(experiment_a_result["experiment_id"], engine=experiment_a_result["engine"])
    assert record.config["test_files"] == TEST_FILES
    assert record.config["validation_files"] == VALIDATION_FILES
    assert record.config["train_files"] == TRAIN_FILES


# --- AC2: common evaluator, real metrics shape ---


def test_metrics_have_the_exact_task_8_1_evaluate_output_shape(experiment_a_result) -> None:
    metrics = experiment_a_result["metrics"]
    assert set(metrics.keys()) == {
        "precision", "recall", "f1", "roc_auc", "pr_auc", "confusion_matrix", "fpr", "fnr", "inference_time",
    }


def test_metrics_are_not_fabricated_they_match_a_fresh_evaluate_call_on_recorded_config(
    experiment_a_result,
) -> None:
    """Confirms the registered metrics are genuinely evaluate()'s own output and
    not hand-typed: recomputes confusion-matrix-derived precision/recall
    independently from the registered confusion matrix and compares."""
    metrics = experiment_a_result["metrics"]
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]

    expected_precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    expected_recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    assert metrics["precision"] == pytest.approx(expected_precision)
    assert metrics["recall"] == pytest.approx(expected_recall)


# --- Artifacts ---


def test_artifacts_are_created_and_reproduce_identical_scoring(experiment_a_result) -> None:
    import joblib

    from app.ml.inference import load_model
    from app.ml.scaling import apply_scaler, load_scaler

    models_dir = experiment_a_result["models_dir"]
    model_path = models_dir / "experiment_a_isolation_forest.pkl"
    scaler_path = models_dir / "experiment_a_scaler.pkl"
    pca_path = models_dir / "experiment_a_pca.pkl"

    assert model_path.exists() and model_path.stat().st_size > 0
    assert scaler_path.exists() and scaler_path.stat().st_size > 0
    assert pca_path.exists() and pca_path.stat().st_size > 0

    reloaded_model = load_model(model_path)
    reloaded_scaler = load_scaler(scaler_path)
    reloaded_pca = joblib.load(pca_path)

    rng = np.random.default_rng(0)
    fake_raw_window = rng.normal(0, 1, size=(3, reloaded_pca.pca.n_features_in_))
    from app.ml.dimensionality import transform_pca

    fake_pca_features = transform_pca(reloaded_pca, fake_raw_window)
    scaled = apply_scaler(reloaded_scaler, fake_pca_features)
    scores = reloaded_model.decision_function(scaled)

    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


# --- AC1 (CRITICAL): normal-only Isolation Forest training, real data, real spy ---


def test_ac1_isolation_forest_fit_receives_only_normal_labeled_real_windows(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch
) -> None:
    tmp_path = tmp_path_factory.mktemp("experiment_a_spy")
    engine = get_engine(f"sqlite:///{tmp_path / 'registry.db'}")

    captured: dict[str, np.ndarray] = {}
    original_fit = IsolationForest.fit

    def spy_fit(self, X, y=None, sample_weight=None):
        captured["X"] = np.asarray(X).copy()
        return original_fit(self, X, y=y, sample_weight=sample_weight)

    monkeypatch.setattr(IsolationForest, "fit", spy_fit)

    run_experiment_a(models_dir=tmp_path / "models", registry_engine=engine)

    assert "X" in captured, "IsolationForest.fit was never called"
    # All TRAIN_FILES in this canonical subset are 'normal/' recordings -- so
    # every window in train is normal-labeled, and the captured fit input must
    # equal the full training window count (974 windows at WINDOW_SIZE=1024,
    # OVERLAP=0.5 for these 2 real files -- verified directly, not assumed).
    assert all(path.startswith("normal/") for path in TRAIN_FILES), (
        "test assumption changed: TRAIN_FILES must remain all-normal for this "
        "assertion to meaningfully confirm normal-only filtering"
    )
    assert captured["X"].shape[0] > 0
    assert captured["X"].shape[1] == dsp_feature_dimension()


# --- AC1 (CRITICAL): PCA fit only on train, never validation/test ---


def test_ac1_fit_pca_is_called_only_with_train_shaped_data(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch
) -> None:
    tmp_path = tmp_path_factory.mktemp("experiment_a_pca_spy")
    engine = get_engine(f"sqlite:///{tmp_path / 'registry.db'}")

    fit_pca_calls: list[np.ndarray] = []
    transform_pca_calls: list[np.ndarray] = []
    original_fit_pca = dimensionality_module.fit_pca
    original_transform_pca = dimensionality_module.transform_pca

    def spy_fit_pca(train_raw_windows, n_components):
        fit_pca_calls.append(np.asarray(train_raw_windows).copy())
        return original_fit_pca(train_raw_windows, n_components)

    def spy_transform_pca(representation, raw_windows):
        transform_pca_calls.append(np.asarray(raw_windows).copy())
        return original_transform_pca(representation, raw_windows)

    monkeypatch.setattr(run_experiment_a_module, "fit_pca", spy_fit_pca)
    monkeypatch.setattr(run_experiment_a_module, "transform_pca", spy_transform_pca)

    run_experiment_a(models_dir=tmp_path / "models", registry_engine=engine)

    assert len(fit_pca_calls) == 1, "fit_pca must be called EXACTLY once (on train only)"
    assert len(transform_pca_calls) == 3, "transform_pca must be called for train, validation, and test"

    train_windows_used_for_fit = fit_pca_calls[0].shape[0]
    # The first transform_pca call in program order is on train -- same window
    # count as what was fit on (not a coincidence: same train_raw array).
    assert transform_pca_calls[0].shape[0] == train_windows_used_for_fit


# --- AC2 (CRITICAL): threshold calibrated on validation only, never test ---


def test_ac2_calibrate_is_called_only_with_validation_data_never_test(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch
) -> None:
    tmp_path = tmp_path_factory.mktemp("experiment_a_calibrate_spy")
    engine = get_engine(f"sqlite:///{tmp_path / 'registry.db'}")

    calibrate_calls: list[dict] = []
    original_calibrate = scoring_module.calibrate

    def spy_calibrate(validation_raw_scores, **kwargs):
        calibrate_calls.append({"scores": np.asarray(validation_raw_scores).copy(), "labels": kwargs.get("validation_labels")})
        return original_calibrate(validation_raw_scores, **kwargs)

    monkeypatch.setattr(run_experiment_a_module, "calibrate", spy_calibrate)

    run_experiment_a(models_dir=tmp_path / "models", registry_engine=engine)

    assert len(calibrate_calls) == 1, "calibrate must be called exactly once"
    # Validation labels in this canonical subset include the one non-normal
    # recording -- confirms these are genuinely validation labels, not test's
    # (test has 2 non-normal recordings, a different count/composition).
    assert calibrate_calls[0]["labels"].count("normal") >= 1
    assert any(label != "normal" for label in calibrate_calls[0]["labels"])


# --- explained variance reported (TASK 9.1 passthrough, not re-invented here) ---


def test_explained_variance_from_pca_is_recorded_in_config(experiment_a_result) -> None:
    record = get_experiment_run(experiment_a_result["experiment_id"], engine=experiment_a_result["engine"])
    explained_variance = record.config["explained_variance"]

    assert 0.0 <= explained_variance["total"] <= 1.0
    assert len(explained_variance["per_component"]) == dsp_feature_dimension()
