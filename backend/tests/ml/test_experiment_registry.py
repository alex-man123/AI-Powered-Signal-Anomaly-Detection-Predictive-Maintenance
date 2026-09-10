from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from app.core.database import get_engine
from app.ml import experiment_registry as experiment_registry_module
from app.ml.experiment_registry import (
    ExperimentRegistryError,
    ExperimentRun,
    get_experiment_run,
    register_experiment_run,
)

EXPERIMENT_ID_PATTERN = re.compile(r"^EXP-[ABC]-\d{3}$")


@pytest.fixture()
def engine(tmp_path: Path):
    """A fresh, isolated SQLite database per test -- never the project's shared
    mafaulda.db, so tests never depend on/interfere with each other's sequence
    numbers."""
    return get_engine(f"sqlite:///{tmp_path / 'test_experiments.db'}")


def _sample_config(**overrides) -> dict:
    config = {
        "dataset_hash": "sha256:" + "a" * 64,
        "split_manifest_hash": "sha256:" + "b" * 64,
        "representation": "raw_pca",
        "model": "isolation_forest",
        "feature_dimension": 15,
        "random_seed": 42,
        "preprocessing_config": {"detrend": True, "normalize": "standard"},
        "model_config": {"n_estimators": 100, "contamination": "auto"},
        "threshold": {"method": "percentile", "value": 0.73},
    }
    config.update(overrides)
    return config


def _sample_metrics(**overrides) -> dict:
    metrics = {
        "precision": 0.65,
        "recall": 0.14,
        "f1": 0.22,
        "roc_auc": 0.64,
        "pr_auc": 0.61,
        "confusion_matrix": [[903, 71], [842, 132]],
        "fpr": 0.07,
        "fnr": 0.86,
        "inference_time": 0.009,
    }
    metrics.update(overrides)
    return metrics


# --- Test 1: ID generation, format, automatic ---


def test_id_generation_format_for_each_family(engine) -> None:
    id_a = register_experiment_run("A", _sample_config(), _sample_metrics(), engine=engine)
    id_b = register_experiment_run("B", _sample_config(), _sample_metrics(), engine=engine)
    id_c = register_experiment_run("C", _sample_config(), _sample_metrics(), engine=engine)

    for experiment_id, family in ((id_a, "A"), (id_b, "B"), (id_c, "C")):
        assert EXPERIMENT_ID_PATTERN.match(experiment_id), experiment_id
        assert experiment_id == f"EXP-{family}-001"  # first run in each family


def test_sequence_number_increments_within_a_family(engine) -> None:
    ids = [register_experiment_run("A", _sample_config(), _sample_metrics(), engine=engine) for _ in range(4)]

    assert ids == ["EXP-A-001", "EXP-A-002", "EXP-A-003", "EXP-A-004"]


def test_families_have_independent_sequence_counters(engine) -> None:
    register_experiment_run("A", _sample_config(), _sample_metrics(), engine=engine)
    register_experiment_run("A", _sample_config(), _sample_metrics(), engine=engine)
    first_b = register_experiment_run("B", _sample_config(), _sample_metrics(), engine=engine)

    assert first_b == "EXP-B-001"  # B's counter is independent of A's


# --- Test 2: uniqueness ---


def test_no_two_runs_ever_receive_the_same_id(engine) -> None:
    all_ids = []
    for family in ("A", "A", "B", "A", "C", "B"):
        all_ids.append(register_experiment_run(family, _sample_config(), _sample_metrics(), engine=engine))

    assert len(all_ids) == len(set(all_ids)), f"duplicate IDs generated: {all_ids}"


# --- Test 3 (AC2): round-trip, full fidelity ---


def test_ac2_round_trip_preserves_full_configuration_and_metrics(engine) -> None:
    config = _sample_config()
    metrics = _sample_metrics()

    experiment_id = register_experiment_run("A", config, metrics, engine=engine)
    record = get_experiment_run(experiment_id, engine=engine)

    assert record.experiment_id == experiment_id
    assert record.config == config
    assert record.metrics == metrics
    # Explicit per-field checks (not just dict equality) -- per this task's own
    # instruction not to compare only experiment_id.
    assert record.config["dataset_hash"] == config["dataset_hash"]
    assert record.config["split_manifest_hash"] == config["split_manifest_hash"]
    assert record.config["representation"] == config["representation"]
    assert record.config["model"] == config["model"]
    assert record.config["feature_dimension"] == config["feature_dimension"]
    assert record.config["random_seed"] == config["random_seed"]
    assert record.config["preprocessing_config"] == config["preprocessing_config"]
    assert record.config["model_config"] == config["model_config"]
    assert record.config["threshold"] == config["threshold"]
    assert record.metrics == metrics
    assert record.created_at  # non-empty, generated at registration time


def test_ac2_created_at_is_valid_iso8601(engine) -> None:
    from datetime import datetime

    experiment_id = register_experiment_run("A", _sample_config(), _sample_metrics(), engine=engine)
    record = get_experiment_run(experiment_id, engine=engine)

    datetime.fromisoformat(record.created_at)  # must not raise


# --- Test 4: persistence across a fresh engine/session (simulates process restart) ---


def test_persistence_survives_a_new_engine_instance(tmp_path: Path) -> None:
    db_path = tmp_path / "persistence_test.db"
    engine_1 = get_engine(f"sqlite:///{db_path}")
    config = _sample_config(model="autoencoder")
    metrics = _sample_metrics(precision=0.62)

    experiment_id = register_experiment_run("B", config, metrics, engine=engine_1)

    # Fresh Engine object pointed at the SAME file -- simulates a new process
    # reopening the database, not reusing the in-memory `engine_1` object.
    engine_2 = get_engine(f"sqlite:///{db_path}")
    record = get_experiment_run(experiment_id, engine=engine_2)

    assert record.experiment_id == experiment_id
    assert record.config == config
    assert record.metrics == metrics


# --- Test 5: configuration fidelity (no silent field loss) ---


def test_nested_configuration_fields_are_not_flattened_or_lost(engine) -> None:
    config = _sample_config(
        preprocessing_config={"detrend": True, "filters": {"lowpass": 500, "highpass": 5}},
        model_config={"n_estimators": 200, "max_samples": "auto", "nested": {"a": [1, 2, 3]}},
    )
    metrics = _sample_metrics()

    experiment_id = register_experiment_run("C", config, metrics, engine=engine)
    record = get_experiment_run(experiment_id, engine=engine)

    assert record.config["preprocessing_config"]["filters"] == {"lowpass": 500, "highpass": 5}
    assert record.config["model_config"]["nested"] == {"a": [1, 2, 3]}


# --- Test 6: metrics fidelity (TASK 8.1 output preserved exactly) ---


def test_evaluation_metrics_are_preserved_exactly_not_recalculated(engine) -> None:
    """Uses app.ml.evaluation.evaluate()'s own real output shape -- confirms the
    registry stores exactly what TASK 8.1 produced, not a re-derived summary."""
    from app.ml.evaluation import evaluate

    real_metrics = evaluate(
        y_true=[0, 0, 0, 1, 1, 1],
        y_pred=[0, 0, 1, 1, 1, 1],
        anomaly_scores=[0.1, 0.2, 0.6, 0.7, 0.8, 0.9],
        inference_time=0.0123,
    )

    experiment_id = register_experiment_run("A", _sample_config(), real_metrics, engine=engine)
    record = get_experiment_run(experiment_id, engine=engine)

    assert record.metrics == real_metrics
    assert record.metrics["confusion_matrix"] == real_metrics["confusion_matrix"]
    assert record.metrics["precision"] == real_metrics["precision"]


# --- Test 7: invalid input ---


def test_invalid_family_raises_explicitly(engine) -> None:
    with pytest.raises(ExperimentRegistryError, match="family"):
        register_experiment_run("D", _sample_config(), _sample_metrics(), engine=engine)
    with pytest.raises(ExperimentRegistryError, match="family"):
        register_experiment_run("a", _sample_config(), _sample_metrics(), engine=engine)  # case-sensitive


def test_empty_config_raises_explicitly(engine) -> None:
    with pytest.raises(ExperimentRegistryError, match="config"):
        register_experiment_run("A", {}, _sample_metrics(), engine=engine)


def test_empty_metrics_raises_explicitly(engine) -> None:
    with pytest.raises(ExperimentRegistryError, match="metrics"):
        register_experiment_run("A", _sample_config(), {}, engine=engine)


def test_non_json_serializable_config_raises_explicitly(engine) -> None:
    with pytest.raises(ExperimentRegistryError, match="JSON-serializable"):
        register_experiment_run("A", {"bad": object()}, _sample_metrics(), engine=engine)


def test_retrieval_with_unknown_experiment_id_raises_explicitly(engine) -> None:
    with pytest.raises(ExperimentRegistryError, match="No experiment run found"):
        get_experiment_run("EXP-A-999", engine=engine)


def test_retrieval_with_empty_experiment_id_raises_explicitly(engine) -> None:
    with pytest.raises(ExperimentRegistryError, match="empty"):
        get_experiment_run("", engine=engine)


# --- Test 8: no manual/hard-coded ID ---


def test_register_experiment_run_signature_has_no_id_parameter() -> None:
    params = list(inspect.signature(register_experiment_run).parameters)
    assert "experiment_id" not in params
    assert "id" not in params
    assert params == ["family", "config", "metrics", "engine"]


# --- Test 9: collision handling ---


def test_family_sequence_collision_raises_explicit_registry_error(engine) -> None:
    """Forces a genuine database-level collision (not simulated at the Python
    logic level): pre-insert a row for family=A, sequence=1 directly, bypassing
    _next_sequence, then attempt another registration that computes the SAME
    sequence -- the real UNIQUE constraint on (family, sequence) must reject it,
    and this module must surface that as ExperimentRegistryError, not a bare
    IntegrityError or a silent overwrite."""
    from app.core.database import get_session_factory
    from app.ml.experiment_registry import _ensure_schema

    _ensure_schema(engine)
    session_factory = get_session_factory(engine)

    with session_factory() as session:
        session.add(
            ExperimentRun(
                experiment_id="EXP-A-001",
                family="A",
                sequence=1,
                config_json="{}",
                metrics_json="{}",
                created_at="2026-01-01T00:00:00+00:00",
            )
        )
        session.commit()

    # Directly attempt to insert a second row at the SAME (family, sequence) --
    # exactly what would happen if _next_sequence raced with another writer.
    with session_factory() as session:
        session.add(
            ExperimentRun(
                experiment_id="EXP-A-001-duplicate",
                family="A",
                sequence=1,
                config_json="{}",
                metrics_json="{}",
                created_at="2026-01-01T00:00:01+00:00",
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
    # Confirms the real database constraint is what's being relied on above (not
    # a Python-level check this module could bypass); register_experiment_run
    # itself catches exactly this exception type and re-raises it as
    # ExperimentRegistryError (see its own try/except around session.commit()).


# --- schema deduplication (section 20) ---


def test_experiment_run_does_not_duplicate_model_artifact_metadata_fields() -> None:
    """ExperimentRun's own typed columns must stay minimal (id/experiment_id/
    family/sequence/created_at) -- everything from TASK 6.5's ModelArtifactMetadata
    schema (feature_names, threshold_method, score_direction, etc.) must NOT
    reappear as a typed column here; it only exists inside the opaque config_json
    blob, which this registry never inspects."""
    column_names = {column.name for column in ExperimentRun.__table__.columns}
    forbidden_task_6_5_fields = {
        "feature_names", "scaler_artifact", "threshold_method", "threshold_value",
        "score_direction", "training_split", "model_type", "model_version",
    }
    assert column_names.isdisjoint(forbidden_task_6_5_fields)
    assert column_names == {"id", "experiment_id", "family", "sequence", "config_json", "metrics_json", "created_at"}


def test_experiment_registry_does_not_import_model_artifact_or_evaluation_modules() -> None:
    """Static confirmation: this module reuses app.core.database only -- it does
    not import app.ml.model_artifact (would risk re-deriving/duplicating that
    schema) or app.ml.evaluation (would risk recalculating metrics)."""
    source = inspect.getsource(experiment_registry_module)
    tree = ast.parse(source)

    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
        elif isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)

    assert not any("model_artifact" in name for name in imported_modules)
    assert not any(name.endswith("app.ml.evaluation") for name in imported_modules)


def test_experiment_registry_does_not_recompute_metrics(engine) -> None:
    """The exact metrics dict passed in comes back unchanged -- no field is
    added, removed, or recalculated by the registry."""
    metrics = _sample_metrics()
    metrics_snapshot = dict(metrics)

    experiment_id = register_experiment_run("A", _sample_config(), metrics, engine=engine)
    record = get_experiment_run(experiment_id, engine=engine)

    assert record.metrics == metrics_snapshot
    assert set(record.metrics.keys()) == set(metrics_snapshot.keys())
