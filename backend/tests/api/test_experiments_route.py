"""TASK 10.6 -- tests for `GET /api/experiments`, `GET /api/experiments/{id}`.

Uses the real `app.main.app` through `TestClient` against TASK 9.5's REAL,
already-generated artifacts (`docs/results/central_experiment_results.json`,
`docs/results/experiment_{a,b,c}_reproducibility.json`) -- never a mock, and
cross-checked directly against those same real files read independently here
(never copied literal values that could silently drift). A separate section
proves the endpoints genuinely read from disk (not a hard-coded Python
literal) by pointing the service at a temporary, deliberately-modified copy of
the artifacts -- the REAL project files under `docs/results/` are never
touched by these tests.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import experiment_service

client = TestClient(app)

REAL_AGGREGATED_JSON = json.loads(experiment_service.AGGREGATED_JSON_PATH.read_text(encoding="utf-8"))
REAL_REPRODUCIBILITY_ARTIFACTS = {
    name: json.loads(path.read_text(encoding="utf-8"))
    for name, path in experiment_service.REPRODUCIBILITY_PATHS.items()
}


# ---------------------------------------------------------------------------
# GET /api/experiments
# ---------------------------------------------------------------------------


def test_list_experiments_returns_200() -> None:
    response = client.get("/api/experiments")

    assert response.status_code == 200


def test_list_experiments_includes_a_b_c() -> None:
    response = client.get("/api/experiments")
    experiments = [row["experiment"] for row in response.json()]

    assert set(experiments) == {"A", "B", "C"}


def test_list_experiments_each_row_has_an_experiment_id() -> None:
    response = client.get("/api/experiments")

    for row in response.json():
        assert row["experiment_id"]


def test_list_experiments_metrics_match_the_real_artifact_exactly() -> None:
    """Cross-checked against the same real `central_experiment_results.json`
    read independently here -- proves the route is not returning a
    hard-coded/stale copy."""
    response = client.get("/api/experiments")

    assert response.json() == pytest.approx(REAL_AGGREGATED_JSON["matrix"], nan_ok=True)


def test_list_experiments_route_calls_the_real_service_not_a_hardcoded_list(monkeypatch) -> None:
    calls: list[str] = []
    real_fn = experiment_service.list_experiments

    def spy():
        calls.append("list_experiments")
        return real_fn()

    monkeypatch.setattr(experiment_service, "list_experiments", spy)

    response = client.get("/api/experiments")

    assert response.status_code == 200
    assert calls == ["list_experiments"]


# ---------------------------------------------------------------------------
# GET /api/experiments/{id}
# ---------------------------------------------------------------------------


def test_get_experiment_detail_returns_200_for_a_real_existing_id() -> None:
    real_id = REAL_REPRODUCIBILITY_ARTIFACTS["B"]["experiment_id"]

    response = client.get(f"/api/experiments/{real_id}")

    assert response.status_code == 200


def test_get_experiment_detail_matches_the_real_artifact_exactly() -> None:
    real_artifact = REAL_REPRODUCIBILITY_ARTIFACTS["C"]
    real_id = real_artifact["experiment_id"]

    response = client.get(f"/api/experiments/{real_id}")
    body = response.json()

    for key, value in real_artifact.items():
        if isinstance(value, (int, float)):
            assert body[key] == pytest.approx(value, nan_ok=True)
        else:
            assert body[key] == value


def test_get_experiment_detail_includes_configuration_and_metrics() -> None:
    real_id = REAL_REPRODUCIBILITY_ARTIFACTS["B"]["experiment_id"]

    response = client.get(f"/api/experiments/{real_id}")
    body = response.json()

    assert "preprocessing_config" in body
    assert "model_config" in body
    assert "feature_set" in body
    assert "threshold_method" in body
    assert "threshold_value" in body
    assert "metrics" in body
    assert set(body["metrics"].keys()) == {
        "precision", "recall", "f1", "roc_auc", "pr_auc", "confusion_matrix", "fpr", "fnr", "inference_time",
    }


def test_get_experiment_detail_returns_404_for_a_nonexistent_id() -> None:
    response = client.get("/api/experiments/EXP-Z-999")

    assert response.status_code == 404
    assert response.status_code != 500
    assert "EXP-Z-999" in response.json()["detail"]


def test_get_experiment_detail_returns_404_for_the_superseded_exp_a_001_id() -> None:
    """Disclosed discrepancy (see experiment_service.py docstring): TASK 9.5
    re-registered Experiment A as EXP-A-002 after fixing a real PCA
    determinism bug -- EXP-A-001 is real but superseded, and must correctly
    404, not be silently redirected/aliased."""
    response = client.get("/api/experiments/EXP-A-001")

    assert response.status_code == 404


def test_get_experiment_detail_includes_dataset_and_split_hashes() -> None:
    real_id = REAL_REPRODUCIBILITY_ARTIFACTS["A"]["experiment_id"]

    response = client.get(f"/api/experiments/{real_id}")
    body = response.json()

    assert body["dataset_hash"] == REAL_REPRODUCIBILITY_ARTIFACTS["A"]["dataset_hash"]
    assert body["split_manifest_hash"] == REAL_REPRODUCIBILITY_ARTIFACTS["A"]["split_manifest_hash"]


def test_get_experiment_detail_returned_id_matches_the_requested_id() -> None:
    for artifact in REAL_REPRODUCIBILITY_ARTIFACTS.values():
        real_id = artifact["experiment_id"]

        response = client.get(f"/api/experiments/{real_id}")

        assert response.json()["experiment_id"] == real_id


def test_get_experiment_detail_derives_feature_dimension_correctly_for_all_three() -> None:
    """Experiment A's feature_set is a raw_pca dict (n_components); B/C's is a
    real DSP feature name list -- feature_dimension must be derived correctly
    from whichever real shape is present, never hard-coded to 15."""
    for name, artifact in REAL_REPRODUCIBILITY_ARTIFACTS.items():
        response = client.get(f"/api/experiments/{artifact['experiment_id']}")
        body = response.json()

        if isinstance(artifact["feature_set"], list):
            assert body["feature_dimension"] == len(artifact["feature_set"])
        else:
            assert body["feature_dimension"] == artifact["feature_set"]["n_components"]


# ---------------------------------------------------------------------------
# CRITICAL: proves the endpoints read from disk, not a hard-coded Python copy
# ---------------------------------------------------------------------------


def test_list_experiments_reflects_a_modified_temp_copy_of_the_real_artifact(tmp_path, monkeypatch) -> None:
    """Modifies a TEMP COPY (never the real project file) of
    central_experiment_results.json and confirms the service picks up the new
    value -- proof this is not `EXPERIMENTS = [...]` hard-coded in Python."""
    modified = copy.deepcopy(REAL_AGGREGATED_JSON)
    modified["matrix"][0]["precision"] = 0.123456789

    fake_path = tmp_path / "central_experiment_results.json"
    fake_path.write_text(json.dumps(modified), encoding="utf-8")
    monkeypatch.setattr(experiment_service, "AGGREGATED_JSON_PATH", fake_path)

    response = client.get("/api/experiments")

    assert response.json()[0]["precision"] == pytest.approx(0.123456789)


def test_get_experiment_detail_reflects_a_modified_temp_copy_of_the_real_artifact(tmp_path, monkeypatch) -> None:
    modified = copy.deepcopy(REAL_REPRODUCIBILITY_ARTIFACTS["B"])
    modified["threshold_value"] = 0.987654321
    modified["experiment_id"] = "EXP-B-TEST-ONLY"

    fake_path = tmp_path / "experiment_b_reproducibility.json"
    fake_path.write_text(json.dumps(modified), encoding="utf-8")

    fake_paths = dict(experiment_service.REPRODUCIBILITY_PATHS)
    fake_paths["B"] = fake_path
    monkeypatch.setattr(experiment_service, "REPRODUCIBILITY_PATHS", fake_paths)

    response = client.get("/api/experiments/EXP-B-TEST-ONLY")

    assert response.status_code == 200
    assert response.json()["threshold_value"] == pytest.approx(0.987654321)


def test_experiment_service_does_not_contain_a_hardcoded_matrix_literal() -> None:
    """Static confirmation: no module-level `EXPERIMENTS = [...]`-style
    literal exists in experiment_service.py -- every real value comes from a
    file read at call time."""
    import ast
    import inspect

    source = inspect.getsource(experiment_service)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Dict)):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    pytest.fail(
                        f"Found a module-level uppercase literal assignment {target.id!r} -- "
                        "experiment results must never be hard-coded in this module"
                    )


# ---------------------------------------------------------------------------
# NOT YET MEASURED convention (verified not applicable to current real state)
# ---------------------------------------------------------------------------


def test_real_artifacts_contain_no_not_yet_measured_placeholder() -> None:
    """This project's 'NOT YET MEASURED' convention is prose-only in
    docs/backlog.md, never implemented in code, and does not appear in any of
    TASK 9.5's real, already-measured artifacts -- confirmed directly rather
    than assumed."""
    serialized = json.dumps(REAL_AGGREGATED_JSON) + json.dumps(REAL_REPRODUCIBILITY_ARTIFACTS)

    assert "NOT YET MEASURED" not in serialized
