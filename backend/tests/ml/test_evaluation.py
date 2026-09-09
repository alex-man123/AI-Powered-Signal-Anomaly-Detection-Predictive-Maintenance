from __future__ import annotations

import ast
import inspect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from app.datasets.loader import DATASET_ROOT as REAL_DATASET_ROOT
from app.datasets.loader import load_all_splits
from app.datasets.validators import SAMPLING_RATE_HZ
from app.features.extractor import extract_feature_matrix
from app.ml import evaluation as evaluation_module
from app.ml.autoencoder import Autoencoder
from app.ml.evaluation import EvaluationError, evaluate
from app.ml.inference import predict as isolation_forest_predict
from app.ml.inference import reconstruction_error
from app.ml.isolation_forest import train_isolation_forest
from app.ml.scaling import apply_scaler, fit_scaler
from app.ml.scoring import calibrate, classify, normalize_scores, to_anomaly_score
from app.ml.training import TrainingConfig, train_autoencoder
from app.signal_processing.windowing import Window, create_windows


# --- Test 1/2/3: precision / recall / F1, manually verified ---


def test_precision_recall_f1_match_manual_calculation() -> None:
    """y_true=[0,0,1,1], y_pred=[0,1,0,1] -> TN=1,FP=1,FN=1,TP=1 (each pair
    appears exactly once). precision=recall=f1=TP/(TP+FP or FN)=1/2=0.5 -- a
    hand-verifiable case, not derived from this module's own implementation."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    scores = [0.1, 0.6, 0.4, 0.9]

    result = evaluate(y_true, y_pred, scores, inference_time=0.01)

    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


def test_precision_recall_perfect_predictions() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]

    result = evaluate(y_true, y_pred, scores, inference_time=0.0)

    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


# --- Test 4 (AC3): confusion matrix, exact TN/FP/FN/TP ---


def test_ac3_confusion_matrix_matches_manual_tn_fp_fn_tp() -> None:
    """8 samples with a deliberately asymmetric, easy-to-hand-count layout:
    y_true = [0,0,0,0,1,1,1,1], y_pred = [0,0,1,1,0,1,1,1]
    -> true negatives (true=0,pred=0): indices 0,1 -> TN=2
    -> false positives (true=0,pred=1): indices 2,3 -> FP=2
    -> false negatives (true=1,pred=0): index 4 -> FN=1
    -> true positives (true=1,pred=1): indices 5,6,7 -> TP=3
    """
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.6, 0.7, 0.3, 0.8, 0.85, 0.9]

    result = evaluate(y_true, y_pred, scores, inference_time=0.02)

    tn, fp = result["confusion_matrix"][0]
    fn, tp = result["confusion_matrix"][1]

    assert (tn, fp, fn, tp) == (2, 2, 1, 3)
    assert result["confusion_matrix"] == [[2, 2], [1, 3]]


# --- Test 5/6: FPR / FNR, manually verified ---


def test_fpr_fnr_match_manual_calculation() -> None:
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.6, 0.7, 0.3, 0.8, 0.85, 0.9]

    result = evaluate(y_true, y_pred, scores, inference_time=0.0)

    # FPR = FP / (FP + TN) = 2 / (2 + 2) = 0.5
    assert result["fpr"] == pytest.approx(0.5)
    # FNR = FN / (FN + TP) = 1 / (1 + 3) = 0.25
    assert result["fnr"] == pytest.approx(0.25)


def test_fpr_is_nan_when_no_actual_negatives() -> None:
    y_true = [1, 1, 1]
    y_pred = [1, 0, 1]
    scores = [0.9, 0.2, 0.8]

    result = evaluate(y_true, y_pred, scores, inference_time=0.0)

    assert math_isnan(result["fpr"])


def test_fnr_is_nan_when_no_actual_positives() -> None:
    y_true = [0, 0, 0]
    y_pred = [0, 1, 0]
    scores = [0.1, 0.6, 0.2]

    result = evaluate(y_true, y_pred, scores, inference_time=0.0)

    assert math_isnan(result["fnr"])


def math_isnan(value: float) -> bool:
    return value != value  # avoids importing math just for this helper in tests


# --- Test 7/8: ROC-AUC / PR-AUC with continuous scores ---


def test_roc_auc_is_one_for_perfectly_separable_scores() -> None:
    """All normal scores strictly below all anomaly scores -> perfect ranking ->
    ROC-AUC exactly 1.0, an analytically known extreme case."""
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

    result = evaluate(y_true, y_pred, scores, inference_time=0.0)

    assert result["roc_auc"] == pytest.approx(1.0)


def test_roc_auc_is_half_for_random_ranking() -> None:
    """Scores identical for both classes -> no ranking information -> ROC-AUC
    exactly 0.5, an analytically known case (every pair is a tie)."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    scores = [0.5, 0.5, 0.5, 0.5]

    result = evaluate(y_true, y_pred, scores, inference_time=0.0)

    assert result["roc_auc"] == pytest.approx(0.5)


def test_pr_auc_is_one_for_perfectly_separable_scores() -> None:
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

    result = evaluate(y_true, y_pred, scores, inference_time=0.0)

    assert result["pr_auc"] == pytest.approx(1.0)


def test_roc_auc_is_nan_for_single_class_y_true() -> None:
    y_true = [0, 0, 0, 0]
    y_pred = [0, 1, 0, 1]
    scores = [0.1, 0.6, 0.2, 0.7]

    result = evaluate(y_true, y_pred, scores, inference_time=0.0)

    assert math_isnan(result["roc_auc"])


def test_pr_auc_is_nan_not_fabricated_zero_for_no_positive_examples() -> None:
    """sklearn's own average_precision_score returns 0.0 here (verified
    empirically before writing this module) -- this module explicitly overrides
    that to NaN, since 0.0 would misleadingly look like a real, bad measurement
    rather than an undefined one (this task explicitly forbids inventing 0.5 for
    ROC-AUC; the same "no fabricated placeholder" principle applies here)."""
    y_true = [0, 0, 0, 0]
    y_pred = [0, 1, 0, 1]
    scores = [0.1, 0.6, 0.2, 0.7]

    result = evaluate(y_true, y_pred, scores, inference_time=0.0)

    assert math_isnan(result["pr_auc"])


def test_pr_auc_is_one_for_all_positive_y_true() -> None:
    """The symmetric all-positive case IS well-defined (precision trivially 1 at
    every threshold when there are no negatives) -- not overridden."""
    y_true = [1, 1, 1, 1]
    y_pred = [1, 1, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.2]

    result = evaluate(y_true, y_pred, scores, inference_time=0.0)

    assert result["pr_auc"] == pytest.approx(1.0)


# --- Test 9/10: ROC-AUC / PR-AUC use scores, not y_pred ---


def test_roc_auc_uses_scores_not_predictions() -> None:
    """y_pred is identical (useless, all-normal) for both rows below, but scores
    differ: in the first case scores rank anomalies correctly (ROC-AUC should be
    high), in the second they rank randomly/tied (ROC-AUC should be lower) --
    proving the function actually consumes anomaly_scores, not just re-deriving
    everything from y_pred."""
    y_true = [0, 0, 1, 1]
    useless_y_pred = [0, 0, 0, 0]  # identical in both cases -- always "no anomaly predicted"

    good_scores = [0.1, 0.2, 0.8, 0.9]
    result_good = evaluate(y_true, useless_y_pred, good_scores, inference_time=0.0)

    tied_scores = [0.5, 0.5, 0.5, 0.5]
    result_tied = evaluate(y_true, useless_y_pred, tied_scores, inference_time=0.0)

    assert result_good["roc_auc"] == pytest.approx(1.0)
    assert result_tied["roc_auc"] == pytest.approx(0.5)
    assert result_good["roc_auc"] != result_tied["roc_auc"]
    # Confirms precision/recall (from y_pred, identical in both cases) did NOT change.
    assert result_good["precision"] == result_tied["precision"] or (
        math_isnan(result_good["precision"]) and math_isnan(result_tied["precision"])
    )


def test_pr_auc_uses_scores_not_predictions() -> None:
    y_true = [0, 0, 1, 1]
    useless_y_pred = [1, 1, 1, 1]  # identical in both cases -- always "anomaly predicted"

    good_scores = [0.1, 0.2, 0.8, 0.9]
    result_good = evaluate(y_true, useless_y_pred, good_scores, inference_time=0.0)

    bad_scores = [0.9, 0.8, 0.2, 0.1]  # inverted ranking
    result_bad = evaluate(y_true, useless_y_pred, bad_scores, inference_time=0.0)

    assert result_good["pr_auc"] > result_bad["pr_auc"]
    assert result_good["f1"] == pytest.approx(result_bad["f1"])  # y_pred unchanged


# --- Test 11: input validation ---


def test_mismatched_lengths_raise_explicitly() -> None:
    with pytest.raises(EvaluationError, match="length"):
        evaluate([0, 1], [0, 1, 1], [0.1, 0.2, 0.3], inference_time=0.0)


def test_empty_input_raises_explicitly() -> None:
    with pytest.raises(EvaluationError, match="empty"):
        evaluate([], [], [], inference_time=0.0)


def test_invalid_binary_label_value_raises_explicitly() -> None:
    with pytest.raises(EvaluationError, match="y_true"):
        evaluate([0, 2, 1], [0, 1, 1], [0.1, 0.2, 0.3], inference_time=0.0)

    with pytest.raises(EvaluationError, match="y_pred"):
        evaluate([0, 1, 1], [0, -1, 1], [0.1, 0.2, 0.3], inference_time=0.0)


def test_nan_in_anomaly_scores_raises_explicitly() -> None:
    with pytest.raises(EvaluationError, match="NaN"):
        evaluate([0, 1], [0, 1], [0.1, float("nan")], inference_time=0.0)


def test_inf_in_anomaly_scores_raises_explicitly() -> None:
    with pytest.raises(EvaluationError, match="NaN"):
        evaluate([0, 1], [0, 1], [0.1, float("inf")], inference_time=0.0)


def test_negative_inference_time_raises_explicitly() -> None:
    with pytest.raises(EvaluationError, match="inference_time"):
        evaluate([0, 1], [0, 1], [0.1, 0.9], inference_time=-0.001)


def test_non_finite_inference_time_raises_explicitly() -> None:
    with pytest.raises(EvaluationError, match="inference_time"):
        evaluate([0, 1], [0, 1], [0.1, 0.9], inference_time=float("inf"))


# --- Test 12: inference time ---


def test_inference_time_is_returned_unchanged_when_valid() -> None:
    result = evaluate([0, 1], [0, 1], [0.1, 0.9], inference_time=0.01234)

    assert result["inference_time"] == pytest.approx(0.01234)
    assert result["inference_time"] >= 0
    assert math_isnan(result["inference_time"]) is False


def test_inference_time_zero_is_valid() -> None:
    result = evaluate([0, 1], [0, 1], [0.1, 0.9], inference_time=0.0)
    assert result["inference_time"] == 0.0


# --- Test 13 (AC1): the SAME evaluate() function for two different model outputs ---


def test_ac1_same_evaluate_function_used_for_two_different_model_style_outputs() -> None:
    """Simulates one 'Isolation-Forest-shaped' result set and one
    'Autoencoder-shaped' result set (different score scales/distributions,
    exactly as the two real models would actually produce) both being passed
    through the identical evaluate() call -- no branching, no per-model logic."""
    # "IF-like": normalized [0,1] scores, few false positives.
    if_y_true = [0, 0, 0, 1, 1]
    if_y_pred = [0, 0, 1, 1, 1]
    if_scores = [0.05, 0.10, 0.55, 0.80, 0.95]

    # "AE-like": raw (unnormalized, can exceed 1) reconstruction-error-scale scores.
    ae_y_true = [0, 0, 0, 1, 1]
    ae_y_pred = [0, 0, 0, 1, 1]
    ae_scores = [0.0004, 0.0006, 0.0009, 12.5, 18.3]

    if_result = evaluate(if_y_true, if_y_pred, if_scores, inference_time=0.003)
    ae_result = evaluate(ae_y_true, ae_y_pred, ae_scores, inference_time=0.021)

    assert set(if_result.keys()) == set(ae_result.keys())
    for metrics in (if_result, ae_result):
        for key in ("precision", "recall", "f1", "fpr", "fnr", "inference_time"):
            assert isinstance(metrics[key], float)
        assert isinstance(metrics["confusion_matrix"], list)


def test_evaluation_module_has_no_model_specific_dispatch_or_imports() -> None:
    """Static, source-level confirmation of AC1's architectural rule: no
    isolation_forest/autoencoder/training import, and no `model_type` branching
    anywhere in evaluation.py's actual CODE (the module docstring itself
    legitimately discusses/rules out `model_type` branching in prose, so it is
    excluded from the text-based checks below via `ast.get_docstring`)."""
    source = inspect.getsource(evaluation_module)
    tree = ast.parse(source)

    imported_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_names.append(node.module)
        elif isinstance(node, ast.Import):
            imported_names.extend(alias.name for alias in node.names)

    forbidden = ("isolation_forest", "autoencoder", "app.ml.training")
    for name in imported_names:
        for token in forbidden:
            assert token not in name, f"evaluation.py must not import {name!r}"

    module_docstring = ast.get_docstring(tree) or ""
    code_without_docstring = source.replace(module_docstring, "")

    assert "model_type" not in code_without_docstring
    assert "decision_function" not in code_without_docstring
    assert "reconstruction" not in code_without_docstring.lower()


# --- leakage: evaluate() never touches split/scaler/threshold/training ---


def test_evaluate_signature_has_no_dataset_or_model_parameters() -> None:
    params = list(inspect.signature(evaluate).parameters)
    assert params == ["y_true", "y_pred", "anomaly_scores", "inference_time"]
    for name in params:
        assert "test_labels" not in name or name == "y_true"  # sanity: no odd naming
        assert "split" not in name.lower()
        assert "scaler" not in name.lower()
        assert "threshold" not in name.lower()
        assert "model" not in name.lower()


def test_evaluate_does_not_modify_input_arrays() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    scores = np.array([0.1, 0.6, 0.4, 0.9])
    scores_snapshot = scores.copy()

    evaluate(y_true, y_pred, scores, inference_time=0.0)

    np.testing.assert_array_equal(scores, scores_snapshot)


# --- real end-to-end: same evaluate() applied to real Isolation Forest AND real
# Autoencoder, on the REAL test split (first legitimate use of it in this project) ---
#
# Train: 2 real normal recordings (same choice as TASK 6.2/6.4/7.2/7.3's own tests).
# Validation: 1 normal + 1 non-normal recording (same choice, used ONLY for
# threshold calibration, never for training/fitting).
# Test: 2 normal + 2 non-normal recordings, taken directly from data/processed/
# split_manifest.json's real "test" list -- genuinely new to this whole project
# (every earlier task's real-data test deliberately never touched "test").
REAL_TRAIN_NORMAL = ["normal/12.288.csv", "normal/16.1792.csv"]
REAL_VALIDATION_NORMAL = ["normal/14.336.csv"]
REAL_VALIDATION_NON_NORMAL = ["horizontal-misalignment/0.5mm/20.48.csv"]
REAL_TEST_NORMAL = ["normal/13.1072.csv", "normal/20.2752.csv"]
REAL_TEST_NON_NORMAL = [
    "horizontal-misalignment/0.5mm/18.8416.csv",
    "horizontal-misalignment/0.5mm/25.8048.csv",
]
REAL_WINDOW_SIZE = 1024
REAL_OVERLAP = 0.5
REAL_CHANNEL = 0
REAL_NPERSEG = 256
REAL_NOVERLAP = 128


def _write_manifest(tmp_path: Path, splits: dict[str, list[str]]) -> Path:
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps({"splits": splits}), encoding="utf-8")
    return manifest_path


def _windows_and_labels(loaded_split: list, dataset_root: Path) -> tuple[list[Window], list[str]]:
    windows: list[Window] = []
    labels: list[str] = []
    for recording in loaded_split:
        df = pd.read_csv(dataset_root / recording.relative_path, header=None)
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


def _binary_labels(labels: list[str]) -> list[int]:
    return [0 if label == "normal" else 1 for label in labels]


def test_real_data_same_evaluate_function_for_isolation_forest_and_autoencoder(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Real, honestly-measured results -- reported as-is, whatever they turn out
    to be, per this task's own "no invented results" rule. These models are
    trained on the same small 2-file real subset already used throughout TASK
    6.x/7.x's own tests (not retrained bigger/differently just to make TASK 8.1
    look good), so these numbers characterize that toy-scale training, not this
    project's eventual production model quality."""
    tmp_path = tmp_path_factory.mktemp("evaluation_real_data")
    manifest_path = _write_manifest(
        tmp_path,
        {
            "train": REAL_TRAIN_NORMAL,
            "validation": REAL_VALIDATION_NORMAL + REAL_VALIDATION_NON_NORMAL,
            "test": REAL_TEST_NORMAL + REAL_TEST_NON_NORMAL,
        },
    )

    loaded = load_all_splits(dataset_root=REAL_DATASET_ROOT, manifest_path=manifest_path)

    train_windows, train_labels = _windows_and_labels(loaded["train"], REAL_DATASET_ROOT)
    validation_windows, validation_labels = _windows_and_labels(loaded["validation"], REAL_DATASET_ROOT)
    test_windows, test_labels = _windows_and_labels(loaded["test"], REAL_DATASET_ROOT)

    assert "horizontal-misalignment" in test_labels and "normal" in test_labels, (
        "test setup error: expected both classes in the real test subset"
    )

    train_matrix = extract_feature_matrix(train_windows, SAMPLING_RATE_HZ, nperseg=REAL_NPERSEG, noverlap=REAL_NOVERLAP)
    validation_matrix = extract_feature_matrix(
        validation_windows, SAMPLING_RATE_HZ, nperseg=REAL_NPERSEG, noverlap=REAL_NOVERLAP
    )
    test_matrix = extract_feature_matrix(test_windows, SAMPLING_RATE_HZ, nperseg=REAL_NPERSEG, noverlap=REAL_NOVERLAP)

    scaler = fit_scaler(train_matrix)
    scaled_train = apply_scaler(scaler, train_matrix)
    scaled_validation = apply_scaler(scaler, validation_matrix)
    scaled_test = apply_scaler(scaler, test_matrix)

    y_true_test = _binary_labels(test_labels)

    # --- Isolation Forest ---
    if_model, if_scaler_unused = train_isolation_forest(train_matrix, train_labels, random_state=42)
    # NOTE: train_isolation_forest fits its OWN scaler internally (on normal-only
    # train, per TASK 6.2) -- reused as returned, never refit here.
    if_validation_raw = isolation_forest_predict(if_model, if_scaler_unused, validation_matrix)
    if_validation_anomaly = to_anomaly_score(if_validation_raw)
    if_calibration = calibrate(
        if_validation_anomaly, threshold_method="percentile", percentile_value=95, validation_labels=validation_labels
    )

    start = time.perf_counter()
    if_test_raw = isolation_forest_predict(if_model, if_scaler_unused, test_matrix)
    if_test_anomaly = to_anomaly_score(if_test_raw)
    if_test_normalized = normalize_scores(if_test_anomaly, if_calibration.normalization)
    if_test_predictions = classify(if_test_normalized, if_calibration).astype(int)
    if_inference_time = time.perf_counter() - start

    if_metrics = evaluate(y_true_test, if_test_predictions, if_test_normalized, if_inference_time)

    # --- Autoencoder ---
    torch.manual_seed(42)
    ae_model = Autoencoder(input_dim=scaled_train.shape[1], hidden_dim=10, bottleneck_dim=4)
    train_autoencoder(
        ae_model, scaled_train, train_labels, scaled_validation, config=TrainingConfig(max_epochs=10, seed=42)
    )

    ae_validation_errors = reconstruction_error(ae_model, scaled_validation)
    ae_calibration = calibrate(
        ae_validation_errors, threshold_method="percentile", percentile_value=95, validation_labels=validation_labels
    )

    start = time.perf_counter()
    ae_test_errors = reconstruction_error(ae_model, scaled_test)
    ae_test_normalized = normalize_scores(ae_test_errors, ae_calibration.normalization)
    ae_test_predictions = classify(ae_test_normalized, ae_calibration).astype(int)
    ae_inference_time = time.perf_counter() - start

    ae_metrics = evaluate(y_true_test, ae_test_predictions, ae_test_normalized, ae_inference_time)

    print("\n--- TASK 8.1 real evaluation results (small real-subset models) ---")
    print(f"Isolation Forest: {json.dumps(if_metrics, indent=2)}")
    print(f"Autoencoder: {json.dumps(ae_metrics, indent=2)}")

    # Structural checks only -- NOT asserting a specific "good" performance
    # number, since this task forbids inventing/assuming results.
    for metrics in (if_metrics, ae_metrics):
        assert set(metrics.keys()) == {
            "precision", "recall", "f1", "roc_auc", "pr_auc", "confusion_matrix",
            "fpr", "fnr", "inference_time",
        }
        assert metrics["inference_time"] >= 0
        total = sum(sum(row) for row in metrics["confusion_matrix"])
        assert total == len(y_true_test)
