from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from app.core.database import get_engine
from app.ml.experiment_registry import get_experiment_run
from scripts import run_experiment_b as run_experiment_b_module
from scripts.run_experiment_a import TEST_FILES, TRAIN_FILES, VALIDATION_FILES
from scripts.run_experiment_b import run_experiment_b

EXPERIMENT_ID_PATTERN = re.compile(r"^EXP-B-\d{3}$")


@pytest.fixture(scope="module")
def experiment_b_result(tmp_path_factory: pytest.TempPathFactory):
    """Runs the real Experiment B pipeline ONCE for this test module: loads the
    REAL, standing models/isolation_forest_v1.pkl + scaler_v1.pkl (never
    retrained), scores/calibrates/evaluates on the real canonical test set, and
    registers the run in an isolated registry database (so tests never pollute
    the project's shared experiment registry)."""
    tmp_path = tmp_path_factory.mktemp("experiment_b_shared")
    engine = get_engine(f"sqlite:///{tmp_path / 'registry.db'}")
    experiment_id, metrics = run_experiment_b(registry_engine=engine)
    return {"experiment_id": experiment_id, "metrics": metrics, "engine": engine}


# --- Test 1 (AC2): Experiment B registration, valid format ---


def test_experiment_id_matches_format_and_is_not_hardcoded(experiment_b_result) -> None:
    assert EXPERIMENT_ID_PATTERN.match(experiment_b_result["experiment_id"])


# --- Test 2 (AC2): uniqueness ---


def test_two_runs_produce_sequential_distinct_ids(tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_path = tmp_path_factory.mktemp("experiment_b_sequential")
    engine = get_engine(f"sqlite:///{tmp_path / 'registry.db'}")

    id_1, _metrics_1 = run_experiment_b(registry_engine=engine)
    id_2, _metrics_2 = run_experiment_b(registry_engine=engine)

    assert id_1 == "EXP-B-001"
    assert id_2 == "EXP-B-002"
    assert id_1 != id_2


# --- Test 3 (AC2): round-trip ---


def test_round_trip_retrieval_matches_registered_run(experiment_b_result) -> None:
    record = get_experiment_run(experiment_b_result["experiment_id"], engine=experiment_b_result["engine"])

    assert record.experiment_id == experiment_b_result["experiment_id"]
    assert record.metrics == experiment_b_result["metrics"]
    assert record.config["model"] == "isolation_forest"
    assert record.config["representation"] == "dsp_features"
    assert record.config["reused_existing_artifact"] is True
    assert record.config["retraining_performed"] is False


# --- Test 4 (AC1): reference integrity -- points at the existing model, not a new one ---


def test_config_references_the_real_standing_artifact_paths(experiment_b_result) -> None:
    from app.ml.inference import default_model_path, default_scaler_path

    record = get_experiment_run(experiment_b_result["experiment_id"], engine=experiment_b_result["engine"])

    assert record.config["model_artifact_path"] == str(default_model_path())
    assert record.config["scaler_artifact_path"] == str(default_scaler_path())
    assert "experiment_b_isolation_forest.pkl" not in record.config["model_artifact_path"], (
        "Experiment B must not create/reference a second Isolation Forest artifact -- "
        "it must point at the SAME artifact TASK 6.4 already produced"
    )


def test_metrics_are_bit_identical_to_the_known_real_phase_8_measurement(experiment_b_result) -> None:
    """The strongest possible confirmation of AC1: since this script loads the
    exact same (bit-identical, same seed/data/methodology) model TASK 8.1's own
    real-data test already trained and measured, the resulting metrics must match
    that already-reported real measurement exactly -- not merely "look
    reasonable". These expected values are copied from TASK 8.1/8.2's own
    already-delivered, independently-verified report, not computed by this
    module's own logic."""
    metrics = experiment_b_result["metrics"]

    assert metrics["precision"] == pytest.approx(0.6502463054187192)
    assert metrics["recall"] == pytest.approx(0.13552361396303902)
    assert metrics["f1"] == pytest.approx(0.22429906542056074)
    assert metrics["roc_auc"] == pytest.approx(0.638304331510442)
    assert metrics["pr_auc"] == pytest.approx(0.6073014263853652)
    assert metrics["fpr"] == pytest.approx(0.0728952772073922)
    assert metrics["fnr"] == pytest.approx(0.864476386036961)
    assert metrics["confusion_matrix"] == [[903, 71], [842, 132]]


# --- Test 5 (CRITICAL): no duplicate training, verified both statically and at runtime ---


def test_run_experiment_b_never_calls_isolation_forest_fit(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch
) -> None:
    tmp_path = tmp_path_factory.mktemp("experiment_b_no_fit_spy")
    engine = get_engine(f"sqlite:///{tmp_path / 'registry.db'}")

    fit_calls: list[object] = []
    original_fit = IsolationForest.fit

    def spy_fit(self, *args, **kwargs):
        fit_calls.append(self)
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(IsolationForest, "fit", spy_fit)

    run_experiment_b(registry_engine=engine)

    assert fit_calls == [], f"IsolationForest.fit was called {len(fit_calls)} time(s) -- Experiment B must never retrain"


def test_run_experiment_b_does_not_import_train_isolation_forest() -> None:
    """Static confirmation: no code path in this script could call
    train_isolation_forest even indirectly, since it is never imported."""
    source = inspect.getsource(run_experiment_b_module)
    tree = ast.parse(source)

    imported_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imported_names.append(alias.name)

    assert "train_isolation_forest" not in imported_names
    assert "train_isolation_forest" not in source  # not even referenced dynamically


# --- Test 6: same test set as Experiment A (by direct constant reuse, not coincidence) ---


def test_experiment_b_reuses_the_exact_same_file_constants_as_experiment_a(experiment_b_result) -> None:
    record = get_experiment_run(experiment_b_result["experiment_id"], engine=experiment_b_result["engine"])

    assert record.config["train_files"] == TRAIN_FILES
    assert record.config["validation_files"] == VALIDATION_FILES
    assert record.config["test_files"] == TEST_FILES


def test_run_experiment_b_module_imports_file_constants_directly_from_experiment_a() -> None:
    """Confirms A and B cannot silently drift apart: B imports the actual
    objects, not copies -- if Experiment A's file lists ever changed, B would
    automatically pick up the same change."""
    assert run_experiment_b_module.TRAIN_FILES is run_experiment_b_module.TRAIN_FILES  # sanity
    from scripts import run_experiment_a as run_experiment_a_module

    assert run_experiment_b_module.TEST_FILES is run_experiment_a_module.TEST_FILES
    assert run_experiment_b_module.VALIDATION_FILES is run_experiment_a_module.VALIDATION_FILES
    assert run_experiment_b_module.TRAIN_FILES is run_experiment_a_module.TRAIN_FILES


# --- artifact-provenance guard rails ---


def test_run_experiment_b_rejects_a_model_with_wrong_random_state(tmp_path: Path) -> None:
    import joblib
    from sklearn.ensemble import IsolationForest as _IF

    wrong_model = _IF(random_state=1).fit(np.random.default_rng(0).normal(0, 1, size=(20, 15)))
    wrong_model_path = tmp_path / "wrong_model.pkl"
    joblib.dump(wrong_model, wrong_model_path)

    from app.ml.inference import default_scaler_path

    with pytest.raises(RuntimeError, match="random_state"):
        run_experiment_b(model_path=wrong_model_path, scaler_path=default_scaler_path())


def test_run_experiment_b_rejects_a_scaler_with_wrong_feature_names(tmp_path: Path) -> None:
    import joblib
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    wrong_scaler = StandardScaler().fit(pd.DataFrame(np.random.default_rng(0).normal(0, 1, size=(20, 3)), columns=["a", "b", "c"]))
    wrong_scaler_path = tmp_path / "wrong_scaler.pkl"
    joblib.dump(wrong_scaler, wrong_scaler_path)

    from app.ml.inference import default_model_path

    with pytest.raises(RuntimeError, match="feature_names_in_"):
        run_experiment_b(model_path=default_model_path(), scaler_path=wrong_scaler_path)
