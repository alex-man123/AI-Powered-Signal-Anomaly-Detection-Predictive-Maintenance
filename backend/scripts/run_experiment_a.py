"""TASK 9.2 — Experiment A: Raw+PCA -> Isolation Forest.

Blueprint.md section 19's Experiment A: raw signal windows, dimensionality-reduced
via PCA to EXACTLY the same number of components as TASK 5.3's DSP feature vector
(TASK 9.1's `app.ml.dimensionality`), fed into an Isolation Forest trained with the
IDENTICAL methodology as TASK 6.2 (normal-labeled train windows only) -- the ONLY
difference from TASK 8.1's Isolation-Forest-on-DSP-features evaluation is the
representation the model consumes.

This script is an ORCHESTRATOR ONLY -- it re-implements none of the following,
reusing each exactly as already established:

    raw windows -> PCA              app.ml.dimensionality (TASK 9.1)
    normal-only IF training         app.ml.isolation_forest.train_isolation_forest (TASK 6.2)
    scaling (fit on train only)     performed internally by train_isolation_forest (TASK 5.4)
    raw score -> anomaly score      app.ml.scoring.to_anomaly_score (TASK 6.3)
    threshold calibration           app.ml.scoring.calibrate (TASK 6.3, validation only)
    metrics                         app.ml.evaluation.evaluate (TASK 8.1)
    experiment ID + persistence     app.ml.experiment_registry.register_experiment_run (TASK 8.3)
    model/scaler artifact I/O       app.ml.inference.save_model / app.ml.scaling.save_scaler (TASK 6.4/5.4)

PIPELINE ORDER (never reordered -- calibration always precedes test scoring, never
the reverse):

    1. load the real split manifest's train/validation/test file lists (a small,
       already-established real subset -- see CANONICAL TEST SET below)
    2. window each recording (TASK 2.4, unchanged)
    3. build raw window matrices for train/validation/test (no DSP applied)
    4. fit PCA on TRAIN ONLY (TASK 9.1's fit_pca), n_components = the REAL current
       DSP feature dimension (app.ml.dimensionality.dsp_feature_dimension())
    5. transform train/validation/test through that SAME fitted PCA
    6. train Isolation Forest on the normal-labeled subset of train's PCA
       representation (train_isolation_forest already does this filtering + its
       own internal scaler fit, exactly as TASK 6.2 established)
    7. score validation, calibrate threshold on validation ONLY
    8. score test (elapsed time measured here, nothing else), classify
    9. evaluate(y_true_test, y_pred_test, normalized_test_scores, inference_time) --
       TASK 8.1's function, unmodified
   10. register the run via TASK 8.3's registry (family "A") -- experiment_id is
       generated there, never chosen here
   11. persist the model/scaler/PCA artifacts so this run's pipeline can be
       reconstructed exactly
   12. round-trip-verify the registered run (get_experiment_run) before reporting

CANONICAL TEST SET (`TRAIN_FILES`/`VALIDATION_FILES`/`TEST_FILES` below): the same
small real MAFAULDA subset already used throughout TASK 6.x/7.x/8.x/9.1's own
real-data tests (2 normal train recordings, 1 normal + 1 horizontal-misalignment
validation recording, 2 normal + 2 horizontal-misalignment test recordings), all
drawn from `data/processed/split_manifest.json`'s real train/validation/test lists
(verified at runtime by `_verify_subset_matches_real_manifest`, not merely assumed).
Experiment B (a future task) MUST reuse these exact same file lists/windowing
config for a valid, apples-to-apples comparison on identical test windows -- that
is the entire point of fixing them here as named constants rather than
re-selecting files ad hoc.

`split_manifest_hash` in this run's registered config is the hash of the REAL,
authoritative `data/processed/split_manifest.json` (TASK 6.5's own
`compute_split_manifest_hash`) -- not a hash of the small temporary manifest this
script constructs to select its subset (that temporary file is a convenience
mechanism for avoiding a full-dataset load, not a new split; the real manifest is
what actually defines which split each file belongs to, and is what this run's
provenance should point back to).
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import Engine  # noqa: E402

from app.datasets.loader import DATASET_ROOT as REAL_DATASET_ROOT  # noqa: E402
from app.datasets.loader import LoadedRecording, load_all_splits  # noqa: E402
from app.ml.dimensionality import PCARepresentation, dsp_feature_dimension, fit_pca, transform_pca  # noqa: E402
from app.ml.evaluation import evaluate  # noqa: E402
from app.ml.experiment_registry import get_experiment_run, register_experiment_run  # noqa: E402
from app.ml.inference import predict as isolation_forest_predict  # noqa: E402
from app.ml.inference import save_model as save_isolation_forest_model  # noqa: E402
from app.ml.isolation_forest import train_isolation_forest  # noqa: E402
from app.ml.model_artifact import (  # noqa: E402
    DEFAULT_SPLIT_MANIFEST_PATH,
    compute_dataset_hash,
    compute_split_manifest_hash,
)
from app.ml.scaling import save_scaler  # noqa: E402
from app.ml.scoring import calibrate, classify, normalize_scores, to_anomaly_score  # noqa: E402
from app.signal_processing.windowing import Window, create_windows  # noqa: E402

# --- canonical Phase 9 test set -- see module docstring ---
TRAIN_FILES = ["normal/12.288.csv", "normal/16.1792.csv"]
VALIDATION_FILES = ["normal/14.336.csv", "horizontal-misalignment/0.5mm/20.48.csv"]
TEST_FILES = [
    "normal/13.1072.csv",
    "normal/20.2752.csv",
    "horizontal-misalignment/0.5mm/18.8416.csv",
    "horizontal-misalignment/0.5mm/25.8048.csv",
]
WINDOW_SIZE = 1024
OVERLAP = 0.5
CHANNEL = 0
RANDOM_SEED = 42
THRESHOLD_METHOD = "percentile"
PERCENTILE_VALUE = 95.0

DEFAULT_MODELS_DIR = REPO_ROOT / "models"
MODEL_ARTIFACT_FILENAME = "experiment_a_isolation_forest.pkl"
SCALER_ARTIFACT_FILENAME = "experiment_a_scaler.pkl"
PCA_ARTIFACT_FILENAME = "experiment_a_pca.pkl"


def _verify_subset_matches_real_manifest() -> None:
    """Confirms every file this script selected genuinely belongs to that split
    in the REAL, authoritative split_manifest.json -- this small subset is a true
    subset of the real split, not an accidental cross-split/invented selection."""
    real_manifest = json.loads(DEFAULT_SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    real_splits = real_manifest["splits"]

    for path in TRAIN_FILES:
        assert path in real_splits["train"], f"{path!r} is not listed under train in the real split manifest"
    for path in VALIDATION_FILES:
        assert path in real_splits["validation"], (
            f"{path!r} is not listed under validation in the real split manifest"
        )
    for path in TEST_FILES:
        assert path in real_splits["test"], f"{path!r} is not listed under test in the real split manifest"


def _write_temp_manifest(directory: Path) -> Path:
    manifest_path = directory / "experiment_a_manifest.json"
    manifest_path.write_text(
        json.dumps({"splits": {"train": TRAIN_FILES, "validation": VALIDATION_FILES, "test": TEST_FILES}}),
        encoding="utf-8",
    )
    return manifest_path


def _windows_and_labels(loaded_split: list[LoadedRecording], dataset_root: Path) -> tuple[list[Window], list[str]]:
    windows: list[Window] = []
    labels: list[str] = []
    for recording in loaded_split:
        df = pd.read_csv(dataset_root / recording.relative_path, header=None)
        values = df[CHANNEL].tolist()
        recording_windows = create_windows(
            values,
            recording_id=recording.relative_path,
            split=recording.split,
            window_size=WINDOW_SIZE,
            overlap=OVERLAP,
        )
        windows.extend(recording_windows)
        labels.extend([recording.label.value] * len(recording_windows))
    return windows, labels


def _binary_labels(labels: list[str]) -> list[int]:
    return [0 if label == "normal" else 1 for label in labels]


def run_experiment_a(
    *,
    dataset_root: Path = REAL_DATASET_ROOT,
    models_dir: Path | None = None,
    registry_engine: Engine | None = None,
) -> tuple[str, dict[str, Any]]:
    """Runs Experiment A end-to-end and returns `(experiment_id, metrics)`.

    Args:
        dataset_root: override only for testing against a synthetic dataset root;
            production runs use the real `data/raw/mafaulda/` (the default).
        models_dir: where to write the three artifacts this run produces; defaults
            to this repository's real `models/` directory. Overridable so tests
            can use an isolated `tmp_path`.
        registry_engine: SQLAlchemy engine for `app.ml.experiment_registry`;
            defaults to the project's shared database. Overridable so tests don't
            pollute/depend on the real experiment registry.
    """
    _verify_subset_matches_real_manifest()

    resolved_models_dir = models_dir or DEFAULT_MODELS_DIR
    resolved_models_dir.mkdir(parents=True, exist_ok=True)
    model_artifact_path = resolved_models_dir / MODEL_ARTIFACT_FILENAME
    scaler_artifact_path = resolved_models_dir / SCALER_ARTIFACT_FILENAME
    pca_artifact_path = resolved_models_dir / PCA_ARTIFACT_FILENAME

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        manifest_path = _write_temp_manifest(Path(tmp_dir_name))
        loaded = load_all_splits(dataset_root=dataset_root, manifest_path=manifest_path)

        train_windows, train_labels = _windows_and_labels(loaded["train"], dataset_root)
        validation_windows, validation_labels = _windows_and_labels(loaded["validation"], dataset_root)
        test_windows, test_labels = _windows_and_labels(loaded["test"], dataset_root)

        train_raw = np.array([w.values for w in train_windows], dtype=float)
        validation_raw = np.array([w.values for w in validation_windows], dtype=float)
        test_raw = np.array([w.values for w in test_windows], dtype=float)

    # --- step 4/5: PCA, fit on train only ---
    n_components = dsp_feature_dimension()
    pca_representation = fit_pca(train_raw, n_components=n_components)

    train_pca = transform_pca(pca_representation, train_raw)
    validation_pca = transform_pca(pca_representation, validation_raw)
    test_pca = transform_pca(pca_representation, test_raw)

    if not (train_pca.shape[1] == validation_pca.shape[1] == test_pca.shape[1] == n_components):
        raise RuntimeError(
            "Raw+PCA dimensionality is inconsistent across splits or does not match "
            f"the DSP feature dimension ({n_components}) -- BLOCKED, not proceeding."
        )

    # --- step 6: Isolation Forest, identical methodology to TASK 6.2 ---
    model, scaler = train_isolation_forest(train_pca, train_labels, random_state=RANDOM_SEED)

    # --- step 7: calibrate on validation only ---
    validation_raw_scores = isolation_forest_predict(model, scaler, validation_pca)
    validation_anomaly_scores = to_anomaly_score(validation_raw_scores)
    calibration = calibrate(
        validation_anomaly_scores,
        threshold_method=THRESHOLD_METHOD,
        percentile_value=PERCENTILE_VALUE,
        validation_labels=validation_labels,
    )

    # --- step 8: score test (inference_time measured around this only) ---
    y_true_test = _binary_labels(test_labels)

    start = time.perf_counter()
    test_raw_scores = isolation_forest_predict(model, scaler, test_pca)
    test_anomaly_scores = to_anomaly_score(test_raw_scores)
    test_normalized_scores = normalize_scores(test_anomaly_scores, calibration.normalization)
    test_predictions = classify(test_normalized_scores, calibration).astype(int)
    inference_time = time.perf_counter() - start

    # --- step 9: TASK 8.1 evaluation, unmodified ---
    metrics = evaluate(y_true_test, test_predictions, test_normalized_scores, inference_time)

    # --- step 11: persist artifacts (model + scaler + PCA, all needed to
    # reconstruct this run's pipeline) ---
    save_isolation_forest_model(model, model_artifact_path)
    save_scaler(scaler, scaler_artifact_path)
    joblib.dump(pca_representation, pca_artifact_path)

    # --- step 10: register the run (TASK 8.3 generates experiment_id) ---
    config = {
        "dataset_hash": compute_dataset_hash(),
        "split_manifest_hash": compute_split_manifest_hash(),
        "representation": "raw_pca",
        "model": "isolation_forest",
        "feature_dimension": n_components,
        "random_seed": RANDOM_SEED,
        "preprocessing_config": {
            "window_size": WINDOW_SIZE,
            "overlap": OVERLAP,
            "channel": CHANNEL,
            "dsp_preprocessing_applied": False,
        },
        "model_config": model.get_params(),
        "threshold": {"method": calibration.threshold_method, "value": calibration.threshold},
        "train_files": TRAIN_FILES,
        "validation_files": VALIDATION_FILES,
        "test_files": TEST_FILES,
        "model_artifact_path": str(model_artifact_path),
        "scaler_artifact_path": str(scaler_artifact_path),
        "pca_artifact_path": str(pca_artifact_path),
        "explained_variance": {
            "total": pca_representation.total_explained_variance,
            "per_component": list(pca_representation.explained_variance_ratio),
        },
    }

    experiment_id = register_experiment_run("A", config, metrics, engine=registry_engine)

    # --- step 12: round-trip verification before reporting ---
    retrieved = get_experiment_run(experiment_id, engine=registry_engine)
    if retrieved.config != config or retrieved.metrics != metrics:
        raise RuntimeError(
            f"Round-trip verification failed for {experiment_id} -- registered and "
            "retrieved run data do not match."
        )

    return experiment_id, metrics


def main() -> int:
    experiment_id, metrics = run_experiment_a()

    print(f"Experiment ID: {experiment_id}")
    print(json.dumps(metrics, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
