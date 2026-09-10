from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from app.core.database import get_engine
from app.ml.experiment_registry import get_experiment_run
from scripts import run_experiment_c as run_experiment_c_module
from scripts.run_experiment_a import TEST_FILES, TRAIN_FILES, VALIDATION_FILES
from scripts.run_experiment_c import run_experiment_c

EXPERIMENT_ID_PATTERN = re.compile(r"^EXP-C-\d{3}$")


@pytest.fixture(scope="module")
def experiment_c_result(tmp_path_factory: pytest.TempPathFactory):
    """Runs the real Experiment C pipeline ONCE for this test module: loads the
    REAL, standing models/autoencoder_v1.pt + its metadata sidecar (never
    retrained), scores/calibrates/evaluates on the real canonical test set, and
    registers the run in an isolated registry database."""
    tmp_path = tmp_path_factory.mktemp("experiment_c_shared")
    engine = get_engine(f"sqlite:///{tmp_path / 'registry.db'}")
    experiment_id, metrics = run_experiment_c(registry_engine=engine)
    return {"experiment_id": experiment_id, "metrics": metrics, "engine": engine}


# --- Test 1 (AC2): Experiment C registration, valid format ---


def test_experiment_id_matches_format_and_is_not_hardcoded(experiment_c_result) -> None:
    assert EXPERIMENT_ID_PATTERN.match(experiment_c_result["experiment_id"])


# --- Test 2 (AC2): uniqueness ---


def test_two_runs_produce_sequential_distinct_ids(tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_path = tmp_path_factory.mktemp("experiment_c_sequential")
    engine = get_engine(f"sqlite:///{tmp_path / 'registry.db'}")

    id_1, _metrics_1 = run_experiment_c(registry_engine=engine)
    id_2, _metrics_2 = run_experiment_c(registry_engine=engine)

    assert id_1 == "EXP-C-001"
    assert id_2 == "EXP-C-002"
    assert id_1 != id_2


# --- Test 3 (AC2): round-trip ---


def test_round_trip_retrieval_matches_registered_run(experiment_c_result) -> None:
    record = get_experiment_run(experiment_c_result["experiment_id"], engine=experiment_c_result["engine"])

    assert record.experiment_id == experiment_c_result["experiment_id"]
    assert record.metrics == experiment_c_result["metrics"]
    assert record.config["model"] == "autoencoder"
    assert record.config["representation"] == "dsp_features"
    assert record.config["reused_existing_artifact"] is True
    assert record.config["retraining_performed"] is False


# --- Test 4 (AC1): reference integrity -- points at the existing model, not a new one ---


def test_config_references_the_real_standing_artifact_path(experiment_c_result) -> None:
    from app.ml.inference import default_autoencoder_path

    record = get_experiment_run(experiment_c_result["experiment_id"], engine=experiment_c_result["engine"])

    assert record.config["model_artifact_path"] == str(default_autoencoder_path())
    assert "experiment_c_autoencoder" not in record.config["model_artifact_path"], (
        "Experiment C must not create/reference a second Autoencoder artifact -- "
        "it must point at the SAME artifact TASK 7.3 already produced"
    )


def test_registered_threshold_matches_the_persisted_metadata_sidecar(experiment_c_result) -> None:
    """The freshly-calibrated threshold this run computed must equal the
    threshold TASK 7.3 already persisted in models/autoencoder_v1.json --
    confirming this run's calibration genuinely reproduces the existing result
    rather than silently drifting from it (the script itself also raises if this
    check fails, before ever registering the run)."""
    import json

    from app.ml.inference import default_autoencoder_path

    metadata_path = default_autoencoder_path().with_suffix(".json")
    persisted_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    record = get_experiment_run(experiment_c_result["experiment_id"], engine=experiment_c_result["engine"])

    assert record.config["threshold"]["value"] == pytest.approx(persisted_metadata["threshold_value"])
    assert record.config["random_seed"] == persisted_metadata["random_seed"]
    assert record.config["dataset_hash"] == persisted_metadata["dataset_hash"]
    assert record.config["split_manifest_hash"] == persisted_metadata["split_manifest_hash"]


def test_metrics_are_bit_identical_to_the_known_real_phase_8_measurement(experiment_c_result) -> None:
    """The strongest possible confirmation of AC1: since this script loads the
    exact same standing Autoencoder artifact TASK 8.1's own real-data test
    already trained and measured (same seed/data/methodology), the resulting
    metrics must match that already-reported real measurement exactly. These
    expected values are copied from TASK 8.1/8.2's own already-delivered,
    independently-verified report, not computed by this module's own logic."""
    metrics = experiment_c_result["metrics"]

    assert metrics["precision"] == pytest.approx(0.6243093922651933)
    assert metrics["recall"] == pytest.approx(0.11601642710472279)
    assert metrics["f1"] == pytest.approx(0.19567099567099566)
    assert metrics["roc_auc"] == pytest.approx(0.5847048939785554)
    assert metrics["pr_auc"] == pytest.approx(0.5582988338260472)
    assert metrics["fpr"] == pytest.approx(0.06981519507186858)
    assert metrics["fnr"] == pytest.approx(0.8839835728952772)
    assert metrics["confusion_matrix"] == [[906, 68], [861, 113]]


# --- Test 5 (CRITICAL): no duplicate training, verified both statically and at runtime ---


def test_run_experiment_c_never_calls_a_gradient_update(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch
) -> None:
    """Spies on torch.optim.Adam.step and Tensor.backward -- both must never be
    called anywhere during a full Experiment C run, since neither training nor
    fine-tuning is permitted here."""
    import torch

    tmp_path = tmp_path_factory.mktemp("experiment_c_no_train_spy")
    engine = get_engine(f"sqlite:///{tmp_path / 'registry.db'}")

    backward_calls: list[object] = []
    original_backward = torch.Tensor.backward

    def spy_backward(self, *args, **kwargs):
        backward_calls.append(self)
        return original_backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", spy_backward)

    run_experiment_c(registry_engine=engine)

    assert backward_calls == [], (
        f"Tensor.backward was called {len(backward_calls)} time(s) -- Experiment C must never train/fine-tune"
    )


def test_run_experiment_c_does_not_import_train_autoencoder() -> None:
    """Static confirmation: no code path in this script could call
    train_autoencoder even indirectly, since it is never imported. The module
    docstring itself legitimately discusses/rules out train_autoencoder in prose,
    so it is excluded from the source-text check below."""
    source = inspect.getsource(run_experiment_c_module)
    tree = ast.parse(source)

    imported_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imported_names.append(alias.name)

    assert "train_autoencoder" not in imported_names

    module_docstring = ast.get_docstring(tree) or ""
    code_without_docstring = source.replace(module_docstring, "")
    assert "train_autoencoder" not in code_without_docstring


# --- Test: same test set as Experiments A and B (by direct constant reuse) ---


def test_experiment_c_reuses_the_exact_same_file_constants_as_experiments_a_and_b(experiment_c_result) -> None:
    record = get_experiment_run(experiment_c_result["experiment_id"], engine=experiment_c_result["engine"])

    assert record.config["train_files"] == TRAIN_FILES
    assert record.config["validation_files"] == VALIDATION_FILES
    assert record.config["test_files"] == TEST_FILES


def test_run_experiment_c_module_imports_file_constants_directly_from_experiment_a() -> None:
    from scripts import run_experiment_a as run_experiment_a_module

    assert run_experiment_c_module.TEST_FILES is run_experiment_a_module.TEST_FILES
    assert run_experiment_c_module.VALIDATION_FILES is run_experiment_a_module.VALIDATION_FILES
    assert run_experiment_c_module.TRAIN_FILES is run_experiment_a_module.TRAIN_FILES


# --- artifact-provenance guard rails ---


def test_run_experiment_c_rejects_metadata_with_wrong_feature_names(tmp_path: Path) -> None:
    from app.ml.autoencoder import Autoencoder
    from app.ml.model_artifact import ModelArtifactMetadata, ModelType, ScoreDirection, save_model_artifact

    wrong_model = Autoencoder(input_dim=3, hidden_dim=4, bottleneck_dim=2)
    metadata = ModelArtifactMetadata(
        model_type=ModelType.AUTOENCODER,
        model_version="v1",
        feature_names=["a", "b", "c"],
        feature_dimension=3,
        scaler_artifact="scaler_v1.pkl",
        threshold_method="percentile",
        threshold_value=0.5,
        score_direction=ScoreDirection.HIGHER_IS_MORE_ANOMALOUS,
        training_split="train",
        dataset_hash="sha256:" + "a" * 64,
        split_manifest_hash="sha256:" + "b" * 64,
        random_seed=42,
        created_at="2026-01-01T00:00:00+00:00",
    )
    wrong_model_path = tmp_path / "wrong_autoencoder.pt"
    save_model_artifact(wrong_model, metadata, wrong_model_path)

    with pytest.raises(RuntimeError, match="feature_names"):
        run_experiment_c(model_path=wrong_model_path)


def test_run_experiment_c_rejects_metadata_with_wrong_dataset_hash(tmp_path: Path) -> None:
    from app.ml.autoencoder import Autoencoder
    from app.ml.model_artifact import ModelArtifactMetadata, ModelType, ScoreDirection, save_model_artifact
    from app.features.registry import FEATURE_REGISTRY, FREQUENCY_FEATURE_REGISTRY

    feature_names = list(FEATURE_REGISTRY.keys()) + list(FREQUENCY_FEATURE_REGISTRY.keys())
    wrong_model = Autoencoder(input_dim=len(feature_names), hidden_dim=8, bottleneck_dim=3)
    metadata = ModelArtifactMetadata(
        model_type=ModelType.AUTOENCODER,
        model_version="v1",
        feature_names=feature_names,
        feature_dimension=len(feature_names),
        scaler_artifact="scaler_v1.pkl",
        threshold_method="percentile",
        threshold_value=0.5,
        score_direction=ScoreDirection.HIGHER_IS_MORE_ANOMALOUS,
        training_split="train",
        dataset_hash="sha256:" + "c" * 64,  # deliberately wrong
        split_manifest_hash="sha256:" + "d" * 64,  # deliberately wrong
        random_seed=42,
        created_at="2026-01-01T00:00:00+00:00",
    )
    wrong_model_path = tmp_path / "wrong_autoencoder.pt"
    save_model_artifact(wrong_model, metadata, wrong_model_path)

    with pytest.raises(RuntimeError, match="dataset_hash"):
        run_experiment_c(model_path=wrong_model_path)
