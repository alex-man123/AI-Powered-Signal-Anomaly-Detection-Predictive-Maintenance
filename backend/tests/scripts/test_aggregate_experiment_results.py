from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.core.database import get_engine
from app.ml.experiment_registry import ExperimentRecord, get_experiment_run
from scripts.aggregate_experiment_results import (
    AGGREGATED_JSON_PATH,
    CHART_PATH,
    COMPARISON_METRICS,
    EXPERIMENT_A_ID,
    EXPERIMENT_B_ID,
    EXPERIMENT_C_ID,
    REPORT_PATH,
    REPRODUCIBILITY_PATHS,
    AggregationError,
    build_comparative_matrix,
    build_reproducibility_artifact,
    evaluate_hypothesis,
    load_records,
    main,
    render_report,
    verify_comparability,
    verify_reproducibility,
    _metrics_match,
)


@pytest.fixture(scope="module")
def real_records() -> dict[str, ExperimentRecord]:
    """Loads the three REAL, already-registered runs from the project's own
    shared registry (mafaulda.db) -- never a mock, since this whole task is
    about aggregating genuinely measured results."""
    return load_records()


# --- load_records / comparability (AC1 + Section 5 comparability check) ---


def test_load_records_returns_the_three_real_registered_runs(real_records) -> None:
    assert real_records["A"].experiment_id == EXPERIMENT_A_ID
    assert real_records["B"].experiment_id == EXPERIMENT_B_ID
    assert real_records["C"].experiment_id == EXPERIMENT_C_ID
    assert real_records["A"].config["representation"] == "raw_pca"
    assert real_records["B"].config["representation"] == "dsp_features"
    assert real_records["C"].config["representation"] == "dsp_features"


def test_real_runs_are_comparable(real_records) -> None:
    assert verify_comparability(real_records) == []


def test_verify_comparability_detects_a_dataset_hash_mismatch(real_records) -> None:
    tampered = dict(real_records)
    tampered["B"] = replace(
        tampered["B"],
        config={**tampered["B"].config, "dataset_hash": "sha256:" + "0" * 64},
    )

    issues = verify_comparability(tampered)

    assert any("dataset_hash" in issue for issue in issues)


def test_verify_comparability_detects_a_test_files_mismatch(real_records) -> None:
    tampered = dict(real_records)
    tampered["C"] = replace(
        tampered["C"],
        config={**tampered["C"].config, "test_files": ["normal/does_not_exist.csv"]},
    )

    issues = verify_comparability(tampered)

    assert any("test_files" in issue for issue in issues)


# --- AC1: comparative matrix, all 15 real cells ---


def test_build_comparative_matrix_has_three_rows_and_real_measured_metrics(real_records) -> None:
    matrix = build_comparative_matrix(real_records)

    assert [row["experiment"] for row in matrix] == ["A", "B", "C"]

    for row in matrix:
        for metric in COMPARISON_METRICS:
            value = row[metric]
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0
        assert isinstance(row["confusion_matrix"], list) and len(row["confusion_matrix"]) == 2
        assert isinstance(row["inference_time"], float) and row["inference_time"] > 0.0


def test_matrix_rows_are_real_known_measurements_not_fabricated(real_records) -> None:
    """These expected values were independently reported by TASK 8.1/8.2/9.2/9.3/9.4 --
    copied here as a regression guard, not computed by this module's own logic."""
    matrix = {row["experiment"]: row for row in build_comparative_matrix(real_records)}

    assert matrix["B"]["precision"] == pytest.approx(0.6502463054187192)
    assert matrix["B"]["confusion_matrix"] == [[903, 71], [842, 132]]
    assert matrix["C"]["precision"] == pytest.approx(0.6243093922651933)
    assert matrix["C"]["confusion_matrix"] == [[906, 68], [861, 113]]


# --- AC4: reproducibility artifact construction ---


def test_reproducibility_artifact_for_a_represents_raw_pca_not_dsp_features(real_records) -> None:
    artifact = build_reproducibility_artifact("A", real_records["A"])

    assert artifact["feature_set"]["type"] == "raw_pca"
    assert "n_components" in artifact["feature_set"]
    assert not isinstance(artifact["feature_set"], list), (
        "Experiment A must never be represented with a DSP feature name list -- it uses PCA on raw windows"
    )


def test_reproducibility_artifact_for_b_derives_feature_set_from_the_live_dsp_registry(real_records) -> None:
    from app.features.registry import FEATURE_REGISTRY, FREQUENCY_FEATURE_REGISTRY

    artifact = build_reproducibility_artifact("B", real_records["B"])

    assert artifact["feature_set"] == list(FEATURE_REGISTRY.keys()) + list(FREQUENCY_FEATURE_REGISTRY.keys())
    assert len(artifact["feature_set"]) == 15


def test_reproducibility_artifact_for_c_matches_its_own_recorded_feature_names(real_records) -> None:
    artifact = build_reproducibility_artifact("C", real_records["C"])

    assert artifact["feature_set"] == real_records["C"].config["feature_names"]


def test_reproducibility_artifact_contains_every_required_field(real_records) -> None:
    required_top_level_fields = {
        "experiment_id",
        "representation",
        "model",
        "dataset_hash",
        "split_manifest_hash",
        "preprocessing_config",
        "feature_set",
        "model_config",
        "random_seed",
        "threshold_method",
        "threshold_value",
        "metrics",
        "timestamp",
    }
    required_metric_fields = {
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "confusion_matrix",
        "fpr",
        "fnr",
        "inference_time",
    }

    for name in ("A", "B", "C"):
        artifact = build_reproducibility_artifact(name, real_records[name])
        assert required_top_level_fields <= artifact.keys()
        assert required_metric_fields <= artifact["metrics"].keys()


def test_reproducibility_artifact_raises_if_feature_names_disagree_with_live_registry(real_records) -> None:
    tampered = replace(
        real_records["C"],
        config={**real_records["C"].config, "feature_names": ["not", "the", "real", "features"]},
    )

    with pytest.raises(AggregationError, match="feature_names"):
        build_reproducibility_artifact("C", tampered)


# --- AC2: hypothesis evaluation ---


def test_evaluate_hypothesis_returns_a_valid_conclusion_from_real_data(real_records) -> None:
    matrix = build_comparative_matrix(real_records)
    hypothesis = evaluate_hypothesis(matrix)

    assert hypothesis["conclusion"] in {"CONFIRMED", "REFUTED", "PARTIALLY SUPPORTED", "INCONCLUSIVE"}
    assert isinstance(hypothesis["ranking_metrics_favor_dsp"], bool)
    assert isinstance(hypothesis["threshold_metrics_favor_raw_pca"], bool)


@pytest.mark.parametrize(
    ("a_overrides", "b_overrides", "expected_conclusion"),
    [
        ({"roc_auc": 0.5, "pr_auc": 0.5}, {"roc_auc": 0.9, "pr_auc": 0.9}, "CONFIRMED"),
        (
            {"roc_auc": 0.9, "pr_auc": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9},
            {"roc_auc": 0.5, "pr_auc": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5},
            "REFUTED",
        ),
    ],
)
def test_evaluate_hypothesis_logic_on_synthetic_matrices(a_overrides, b_overrides, expected_conclusion) -> None:
    base_row = {
        "experiment": "A",
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "roc_auc": 0.5,
        "pr_auc": 0.5,
    }
    a_row = {**base_row, **a_overrides, "experiment": "A"}
    b_row = {**base_row, **b_overrides, "experiment": "B"}

    hypothesis = evaluate_hypothesis([a_row, b_row])

    assert hypothesis["conclusion"] == expected_conclusion


# --- _metrics_match: reproducibility comparison helper ---


def test_metrics_match_ignores_inference_time_differences() -> None:
    stored = {
        "precision": 0.5, "recall": 0.5, "f1": 0.5, "roc_auc": 0.5, "pr_auc": 0.5,
        "fpr": 0.1, "fnr": 0.1, "confusion_matrix": [[1, 2], [3, 4]], "inference_time": 0.001,
    }
    fresh = {**stored, "inference_time": 0.999}

    passed, mismatches = _metrics_match(stored, fresh)

    assert passed is True
    assert mismatches == []


def test_metrics_match_detects_a_confusion_matrix_difference() -> None:
    stored = {
        "precision": 0.5, "recall": 0.5, "f1": 0.5, "roc_auc": 0.5, "pr_auc": 0.5,
        "fpr": 0.1, "fnr": 0.1, "confusion_matrix": [[1, 2], [3, 4]], "inference_time": 0.001,
    }
    fresh = {**stored, "confusion_matrix": [[1, 2], [3, 5]]}

    passed, mismatches = _metrics_match(stored, fresh)

    assert passed is False
    assert any("confusion_matrix" in m for m in mismatches)


def test_metrics_match_detects_a_real_metric_difference_beyond_tolerance() -> None:
    stored = {
        "precision": 0.5, "recall": 0.5, "f1": 0.5, "roc_auc": 0.5, "pr_auc": 0.5,
        "fpr": 0.1, "fnr": 0.1, "confusion_matrix": [[1, 2], [3, 4]], "inference_time": 0.001,
    }
    fresh = {**stored, "roc_auc": 0.9}

    passed, mismatches = _metrics_match(stored, fresh)

    assert passed is False
    assert any("roc_auc" in m for m in mismatches)


# --- AC4 (CRITICAL): actual rerun reproducibility, demonstrated for A, B, and C separately ---


def test_verify_reproducibility_experiment_a_rerun_matches(
    real_records, tmp_path_factory: pytest.TempPathFactory
) -> None:
    tmp_path = tmp_path_factory.mktemp("verify_repro_a")
    engine = get_engine(f"sqlite:///{tmp_path / 'registry.db'}")

    result = verify_reproducibility("A", real_records["A"], models_dir=tmp_path / "models", registry_engine=engine)

    assert result.passed, result.mismatches


def test_verify_reproducibility_experiment_b_rerun_matches(
    real_records, tmp_path_factory: pytest.TempPathFactory
) -> None:
    tmp_path = tmp_path_factory.mktemp("verify_repro_b")
    engine = get_engine(f"sqlite:///{tmp_path / 'registry.db'}")

    result = verify_reproducibility("B", real_records["B"], registry_engine=engine)

    assert result.passed, result.mismatches


def test_verify_reproducibility_experiment_c_rerun_matches(
    real_records, tmp_path_factory: pytest.TempPathFactory
) -> None:
    tmp_path = tmp_path_factory.mktemp("verify_repro_c")
    engine = get_engine(f"sqlite:///{tmp_path / 'registry.db'}")

    result = verify_reproducibility("C", real_records["C"], registry_engine=engine)

    assert result.passed, result.mismatches


# --- report rendering: required section structure ---


def test_render_report_contains_all_fourteen_required_sections(real_records) -> None:
    matrix = build_comparative_matrix(real_records)
    hypothesis = evaluate_hypothesis(matrix)
    reproducibility = {
        name: type("R", (), {"passed": True, "mismatches": []})() for name in ("A", "B", "C")
    }

    report = render_report(matrix, [], hypothesis, reproducibility, CHART_PATH)

    required_headings = [
        "## 1. Executive Summary",
        "## 2. Initial Hypothesis",
        "## 3. Experimental Setup",
        "## 4. Experiment A",
        "## 5. Experiment B",
        "## 6. Experiment C",
        "## 7. Comparative Results",
        "## 8. A vs B",
        "## 9. B vs C",
        "## 10. Hypothesis Evaluation",
        "## 11. Alternative Interpretation",
        "## 12. Reproducibility",
        "## 13. Limitations",
        "## 14. Final Conclusion",
    ]
    for heading in required_headings:
        assert heading in report, f"Missing required section: {heading}"


def test_render_report_quotes_the_real_blueprint_hypothesis_faithfully(real_records) -> None:
    matrix = build_comparative_matrix(real_records)
    hypothesis = evaluate_hypothesis(matrix)
    reproducibility = {
        name: type("R", (), {"passed": True, "mismatches": []})() for name in ("A", "B", "C")
    }

    report = render_report(matrix, [], hypothesis, reproducibility, CHART_PATH)

    assert "separa mai bine normal/anomaly decat o reprezentare generica" in report
    assert hypothesis["conclusion"] in report


# --- full integration: main() produces every required deliverable ---


def test_main_produces_all_required_deliverables_and_all_three_reproduce(capsys) -> None:
    exit_code = main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert REPORT_PATH.exists()
    assert AGGREGATED_JSON_PATH.exists()
    assert CHART_PATH.exists()
    for path in REPRODUCIBILITY_PATHS.values():
        assert path.exists()

    assert "Experiment A reproducibility: PASS" in output
    assert "Experiment B reproducibility: PASS" in output
    assert "Experiment C reproducibility: PASS" in output

    aggregated = json.loads(AGGREGATED_JSON_PATH.read_text(encoding="utf-8"))
    assert aggregated["comparability_verified"] is True
    assert len(aggregated["matrix"]) == 3
    for row in aggregated["matrix"]:
        for metric in COMPARISON_METRICS:
            assert row[metric] is not None
