"""TASK 13.1 -- tests for `scripts.run_cross_dataset_validation`.

The real CWRU Bearing Dataset is not present in this repository/environment.
These tests verify:
  1. The real, current, honest failure mode: calling the real entry point
     against the real (missing) CWRU root raises, producing no metrics.
  2. Structural "never retrains/refits on CWRU" guarantees, checked either by
     spying on the real, standing MAFAULDA model/scaler (`IsolationForest.
     fit`/`StandardScaler.fit`) across a run against a tiny SYNTHETIC
     CWRU-shaped fixture tree, or by confirming this script's own source
     never references a training entry point at all.

The synthetic fixture tree used below is built via `scipy.io.savemat` in this
test file only, to exercise the real MAFAULDA->CWRU scoring pipeline
end-to-end without a real CWRU download -- it is NEVER presented as evidence
about real cross-dataset generalization (see docs/results/
cross_dataset_validation.md for the actual, honest status of that question).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import scripts.run_cross_dataset_validation as cross_dataset_validation
from app.datasets.cwru_loader import DEFAULT_CWRU_ROOT, CWRULoaderError
from app.ml.inference import default_autoencoder_path, default_model_path
from scripts.run_cross_dataset_validation import (
    DEFAULT_NORMAL_SAMPLING_RATE_HZ,
    CrossDatasetValidationError,
    evaluate_model_on_cwru,
    run_cross_dataset_validation,
)

_MAFAULDA_MODEL_ARTIFACTS_PRESENT = default_model_path().exists() and default_autoencoder_path().exists()
_REAL_CWRU_DATASET_PRESENT = DEFAULT_CWRU_ROOT.exists()

pytestmark = pytest.mark.skipif(
    not _MAFAULDA_MODEL_ARTIFACTS_PRESENT,
    reason="Real, standing MAFAULDA model artifacts (models/isolation_forest_v1.pkl, "
    "models/autoencoder_v1.pt) are not present -- cannot test the scoring pipeline without them.",
)


def _write_synthetic_cwru_tree(root: Path, *, n_normal: int = 2, n_fault: int = 2) -> None:
    """A tiny, clearly-synthetic CWRU-shaped `.mat` tree (sine + noise, never
    real CWRU signal content) -- long enough (3000 samples) to produce
    several real MAFAULDA-configuration windows (WINDOW_SIZE=1024)."""
    rng = np.random.default_rng(seed=7)
    t = np.arange(3000)

    # Real directory-naming convention (verified against the actual,
    # downloaded CWRU mirror): "<N>k_..." for sampling rate, "B"/"IR"/"OR" for
    # fault location -- see app.datasets.cwru_loader's own docstring.
    (root / "12k_Drive_End_Bearing_Fault_Data/normal").mkdir(parents=True, exist_ok=True)
    (root / "12k_Drive_End_Bearing_Fault_Data/B").mkdir(parents=True, exist_ok=True)

    for i in range(n_normal):
        signal = np.sin(2 * np.pi * 50 * t / 12000) + 0.01 * rng.standard_normal(3000)
        savemat(str(root / f"12k_Drive_End_Bearing_Fault_Data/normal/{i}.mat"), {f"X{i:03d}_DE_time": signal})

    for i in range(n_fault):
        # A different, still-synthetic frequency/amplitude profile -- purely
        # to give the pipeline two distinct classes to compute a confusion
        # matrix from, not a simulation of any real fault signature.
        signal = 3 * np.sin(2 * np.pi * 200 * t / 12000) + 0.05 * rng.standard_normal(3000)
        savemat(str(root / f"12k_Drive_End_Bearing_Fault_Data/B/{i}.mat"), {f"X{i:03d}_DE_time": signal})


@pytest.fixture(scope="module")
def synthetic_cwru_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("synthetic_cwru")
    _write_synthetic_cwru_tree(root)
    return root


# --- real, current absence of the dataset ---


def test_run_cross_dataset_validation_raises_when_cwru_dataset_is_not_present(tmp_path: Path) -> None:
    with pytest.raises(CWRULoaderError):
        run_cross_dataset_validation(cwru_dataset_root=tmp_path / "does-not-exist")


# --- module never references a training entry point ---


def test_module_source_never_references_a_training_function() -> None:
    source = inspect.getsource(cross_dataset_validation)
    assert "train_isolation_forest" not in source
    assert "train_autoencoder" not in source
    assert "app.ml.training" not in source
    assert ".fit(" not in source


# --- spy-based: real scoring pipeline never fits/refits anything ---


def test_evaluate_isolation_forest_on_synthetic_cwru_never_calls_fit(
    synthetic_cwru_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _spy_fit(self, *args, **kwargs):
        raise AssertionError("IsolationForest.fit was called -- retraining is forbidden for TASK 13.1")

    monkeypatch.setattr(IsolationForest, "fit", _spy_fit)

    result = evaluate_model_on_cwru(model_path=default_model_path(), cwru_dataset_root=synthetic_cwru_root)

    assert result["model"] == "isolation_forest"
    assert result["retraining_performed"] is False
    assert result["recalibrated_on_cwru"] is False


def test_evaluate_autoencoder_on_synthetic_cwru_never_calls_scaler_fit(
    synthetic_cwru_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _spy_fit(self, *args, **kwargs):
        raise AssertionError("StandardScaler.fit was called -- the scaler must only ever be transformed with")

    monkeypatch.setattr(StandardScaler, "fit", _spy_fit)
    monkeypatch.setattr(StandardScaler, "fit_transform", _spy_fit)

    result = evaluate_model_on_cwru(model_path=default_autoencoder_path(), cwru_dataset_root=synthetic_cwru_root)

    assert result["model"] == "autoencoder"
    assert result["retraining_performed"] is False


# --- calibration is computed once, from MAFAULDA validation, never from CWRU ---


def test_calibration_is_computed_exactly_once_from_mafaulda_validation_not_cwru(
    synthetic_cwru_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real_calibrate = cross_dataset_validation.calibrate

    def _spy_calibrate(validation_scores, **kwargs):
        calls.append(len(validation_scores))
        return real_calibrate(validation_scores, **kwargs)

    monkeypatch.setattr(cross_dataset_validation, "calibrate", _spy_calibrate)

    evaluate_model_on_cwru(model_path=default_model_path(), cwru_dataset_root=synthetic_cwru_root)

    assert len(calls) == 1
    # The synthetic CWRU tree has far fewer windows than MAFAULDA's real
    # validation split -- a real, cheap way to confirm calibration saw
    # MAFAULDA-sized data, not the synthetic CWRU tree's data.
    assert calls[0] > 50


def test_threshold_mismatch_against_persisted_metadata_is_never_silently_accepted(
    synthetic_cwru_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a freshly-computed MAFAULDA-validation calibration ever disagreed
    with the model's persisted threshold_value, this must BLOCK, not proceed
    with either value silently."""
    real_calibrate = cross_dataset_validation.calibrate

    def _tampered_calibrate(validation_scores, **kwargs):
        calibration = real_calibrate(validation_scores, **kwargs)
        return calibration.__class__(
            normalization=calibration.normalization,
            threshold_method=calibration.threshold_method,
            percentile_value=calibration.percentile_value,
            threshold=calibration.threshold + 0.5,
        )

    monkeypatch.setattr(cross_dataset_validation, "calibrate", _tampered_calibrate)

    with pytest.raises(CrossDatasetValidationError, match="does not match"):
        evaluate_model_on_cwru(model_path=default_model_path(), cwru_dataset_root=synthetic_cwru_root)


# --- labels: 4 real CWRU classes reduced to the same binary target evaluate() needs ---


def test_result_preserves_real_fault_type_labels_alongside_the_binary_metric(synthetic_cwru_root: Path) -> None:
    result = evaluate_model_on_cwru(model_path=default_model_path(), cwru_dataset_root=synthetic_cwru_root)

    assert result["cwru_labels_present"] == ["ball", "normal"]
    assert set(result["metrics"].keys()) == {
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
