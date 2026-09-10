"""TASK 9.3 — Experiment B: DSP Features -> Isolation Forest.

Blueprint.md section 19's Experiment B: DSP feature vectors (TASK 5.3), scaled
(TASK 5.4), fed into Isolation Forest (TASK 6.2), scored/thresholded (TASK 6.3),
evaluated (TASK 8.1). This is EXACTLY what Phase 6/8 already built and measured --
TASK 9.3's own description says so explicitly: "Reutilizeaza direct modelul din
Phase 6 (deja e exact acest experiment)". This script's entire job is bookkeeping:
identify that existing model/artifact, run it (never retrain it) against this
project's canonical Phase 9 test set, and register the result as "Experiment B"
via TASK 8.3's registry -- it is NOT a new ML experiment.

CRITICAL -- NO RETRAINING ANYWHERE IN THIS SCRIPT: unlike `run_experiment_a.py`,
this script never imports or calls `app.ml.isolation_forest.train_isolation_
forest` (or anything else that calls `IsolationForest.fit`). It loads the REAL,
STANDING artifacts TASK 6.4 already produced (`app.ml.inference.load_model`/
`load_scaler`, defaulting to `models/isolation_forest_v1.pkl`/`scaler_v1.pkl`) and
only ever calls `.transform()`/`.decision_function()` on them via the existing
`app.ml.inference.predict` (which itself only transforms, per TASK 5.4/6.4's own
contract). Verified directly (not merely asserted) by this task's own tests, which
spy on `IsolationForest.fit` across a full run of this script and confirm it is
never called.

WHAT "REUSE" MEANS HERE, PRECISELY:
  - Model + scaler: LOADED from the real, standing artifacts (not retrained).
  - Feature extraction (TASK 5.3) and scaler APPLICATION (transform only, TASK
    5.4): re-run fresh against this run's real validation/test files -- this is
    normal, expected model *usage*, not model or pipeline re-implementation. A
    loaded model cannot score data that hasn't been extracted into features yet;
    computing those features is not "duplicate training".
  - Threshold calibration (TASK 6.3's `calibrate`): re-run fresh against the
    validation scores this run just computed. This is NOT model training (this
    project's own established vocabulary keeps "training" and "calibration"
    strictly separate everywhere -- e.g. TASK 6.2 trains, TASK 6.3 calibrates) --
    it is a deterministic statistical step needed to turn continuous scores into
    the binary predictions `evaluate()` (TASK 8.1) requires, and no persisted
    threshold artifact exists yet to load instead (TASK 6.4's own real-data test
    never persisted one via TASK 6.5's `save_model_artifact`; only the bare
    model/scaler were saved via the lower-level `save_model`/`save_scaler`).
  - Metrics (TASK 8.1's `evaluate`): computed fresh from this run's real
    predictions/scores -- never hand-copied from a previous report, never
    invented. Given the same model, same scaler, same validation/test files, and
    the same deterministic pipeline TASK 8.1's own real-data test already used,
    this necessarily reproduces that same measurement -- which is exactly the
    point: Experiment B has always effectively existed since Phase 6/8; this
    script only formalizes it with an `experiment_id` and a registry entry.

SAME TEST SET AS EXPERIMENT A (AC-critical for the comparative matrix): this
script imports `TRAIN_FILES`/`VALIDATION_FILES`/`TEST_FILES`/`WINDOW_SIZE`/
`OVERLAP`/`CHANNEL` DIRECTLY from `scripts.run_experiment_a` rather than
re-declaring parallel constants that could silently drift out of sync -- the same
literal file lists and windowing configuration are used for both experiments,
guaranteeing identical test windows/ground truth between A and B by construction,
not by coincidence or manual double-checking.

ARTIFACT PROVENANCE CAVEAT (disclosed, not hidden): `models/isolation_forest_v1.
pkl`/`scaler_v1.pkl` carry no embedded record of which files trained them -- this
script cannot mechanically PROVE they were trained on exactly `TRAIN_FILES` with
`RANDOM_SEED`. What IS directly verified here (`_verify_loaded_artifacts_match_
expected_representation`): the loaded model's `random_state` and the loaded
scaler's `feature_names_in_` (exact names AND order) match this project's real
DSP feature registry and this script's expected seed -- strong, checkable
evidence consistent with every one of this project's own real-data tests (TASK
6.2/6.4/7.2/7.3/8.1/9.1/9.2) having used exactly `TRAIN_FILES` and
`RANDOM_SEED=42` throughout. Full end-to-end provenance would require either
re-training (explicitly forbidden for this task) or a persisted training-
provenance record (which does not exist yet) -- this is disclosed as a known
limitation, not silently assumed away.

`dataset_hash`/`split_manifest_hash`: TASK 6.5's own real hashing mechanism
(`app.ml.model_artifact.compute_dataset_hash`/`compute_split_manifest_hash`),
unchanged -- not recomputed differently here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pandas as pd  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sqlalchemy import Engine  # noqa: E402

from app.datasets.loader import DATASET_ROOT as REAL_DATASET_ROOT  # noqa: E402
from app.datasets.loader import LoadedRecording, load_all_splits  # noqa: E402
from app.datasets.validators import SAMPLING_RATE_HZ  # noqa: E402
from app.features.extractor import extract_feature_matrix  # noqa: E402
from app.features.registry import FEATURE_REGISTRY, FREQUENCY_FEATURE_REGISTRY  # noqa: E402
from app.ml.evaluation import evaluate  # noqa: E402
from app.ml.experiment_registry import get_experiment_run, register_experiment_run  # noqa: E402
from app.ml.inference import default_model_path, default_scaler_path  # noqa: E402
from app.ml.inference import load_model, load_scaler  # noqa: E402
from app.ml.inference import predict as isolation_forest_predict  # noqa: E402
from app.ml.model_artifact import compute_dataset_hash, compute_split_manifest_hash  # noqa: E402
from app.ml.scoring import calibrate, classify, normalize_scores, to_anomaly_score  # noqa: E402
from app.signal_processing.windowing import Window, create_windows  # noqa: E402
from scripts.run_experiment_a import (  # noqa: E402
    CHANNEL,
    OVERLAP,
    RANDOM_SEED,
    TEST_FILES,
    THRESHOLD_METHOD,
    PERCENTILE_VALUE,
    TRAIN_FILES,
    VALIDATION_FILES,
    WINDOW_SIZE,
)

# Feature-extraction Welch PSD parameters -- same values TASK 8.1/9.1's own
# real-data tests already used for this exact representation.
NPERSEG = 256
NOVERLAP = 128


def _verify_loaded_artifacts_match_expected_representation(model: IsolationForest, scaler: StandardScaler) -> None:
    """Checkable evidence (not proof) that the loaded artifacts are genuinely
    Phase 6/8's DSP-features model -- see module docstring's provenance caveat."""
    expected_feature_names = list(FEATURE_REGISTRY.keys()) + list(FREQUENCY_FEATURE_REGISTRY.keys())

    if model.get_params().get("random_state") != RANDOM_SEED:
        raise RuntimeError(
            f"Loaded model's random_state ({model.get_params().get('random_state')}) does not match "
            f"this script's expected RANDOM_SEED ({RANDOM_SEED}) -- BLOCKED, refusing to proceed."
        )
    if not hasattr(scaler, "feature_names_in_") or list(scaler.feature_names_in_) != expected_feature_names:
        raise RuntimeError(
            "Loaded scaler's feature_names_in_ do not match this project's real DSP feature "
            "registry (names and/or order) -- BLOCKED, refusing to proceed."
        )


def _write_temp_manifest(directory: Path) -> Path:
    manifest_path = directory / "experiment_b_manifest.json"
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


def run_experiment_b(
    *,
    dataset_root: Path = REAL_DATASET_ROOT,
    model_path: Path | None = None,
    scaler_path: Path | None = None,
    registry_engine: Engine | None = None,
) -> tuple[str, dict[str, Any]]:
    """Runs Experiment B end-to-end (load existing model, score/calibrate/
    evaluate on the canonical Phase 9 test set) and returns `(experiment_id,
    metrics)`. Never trains anything.

    Args:
        dataset_root: override only for testing; production runs use the real
            `data/raw/mafaulda/` (the default).
        model_path/scaler_path: override only for testing against a
            test-specific artifact; production runs use the real, standing
            `models/isolation_forest_v1.pkl`/`scaler_v1.pkl` (the defaults).
        registry_engine: override only for testing (an isolated database);
            production runs use the project's shared experiment registry.
    """
    resolved_model_path = model_path or default_model_path()
    resolved_scaler_path = scaler_path or default_scaler_path()

    model = load_model(resolved_model_path)
    scaler = load_scaler(resolved_scaler_path)
    _verify_loaded_artifacts_match_expected_representation(model, scaler)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        manifest_path = _write_temp_manifest(Path(tmp_dir_name))
        loaded = load_all_splits(dataset_root=dataset_root, manifest_path=manifest_path)

        validation_windows, validation_labels = _windows_and_labels(loaded["validation"], dataset_root)
        test_windows, test_labels = _windows_and_labels(loaded["test"], dataset_root)

    validation_features = extract_feature_matrix(
        validation_windows, SAMPLING_RATE_HZ, nperseg=NPERSEG, noverlap=NOVERLAP
    )
    test_features = extract_feature_matrix(test_windows, SAMPLING_RATE_HZ, nperseg=NPERSEG, noverlap=NOVERLAP)

    feature_dimension = len(FEATURE_REGISTRY) + len(FREQUENCY_FEATURE_REGISTRY)
    if validation_features.shape[1] != feature_dimension or test_features.shape[1] != feature_dimension:
        raise RuntimeError(
            f"Extracted feature dimension does not match the expected DSP feature dimension "
            f"({feature_dimension}) -- BLOCKED, refusing to proceed."
        )

    # --- calibrate on validation only (never test) ---
    validation_raw_scores = isolation_forest_predict(model, scaler, validation_features)
    validation_anomaly_scores = to_anomaly_score(validation_raw_scores)
    calibration = calibrate(
        validation_anomaly_scores,
        threshold_method=THRESHOLD_METHOD,
        percentile_value=PERCENTILE_VALUE,
        validation_labels=validation_labels,
    )

    # --- score test (inference_time measured around this only) ---
    y_true_test = _binary_labels(test_labels)

    start = time.perf_counter()
    test_raw_scores = isolation_forest_predict(model, scaler, test_features)
    test_anomaly_scores = to_anomaly_score(test_raw_scores)
    test_normalized_scores = normalize_scores(test_anomaly_scores, calibration.normalization)
    test_predictions = classify(test_normalized_scores, calibration).astype(int)
    inference_time = time.perf_counter() - start

    # --- TASK 8.1 evaluation, unmodified ---
    metrics = evaluate(y_true_test, test_predictions, test_normalized_scores, inference_time)

    # --- register the run (TASK 8.3 generates experiment_id) ---
    config = {
        "dataset_hash": compute_dataset_hash(),
        "split_manifest_hash": compute_split_manifest_hash(),
        "representation": "dsp_features",
        "model": "isolation_forest",
        "feature_dimension": feature_dimension,
        "feature_names": list(FEATURE_REGISTRY.keys()) + list(FREQUENCY_FEATURE_REGISTRY.keys()),
        "random_seed": model.get_params()["random_state"],
        "preprocessing_config": {
            "window_size": WINDOW_SIZE,
            "overlap": OVERLAP,
            "channel": CHANNEL,
            "nperseg": NPERSEG,
            "noverlap": NOVERLAP,
        },
        "model_config": model.get_params(),
        "threshold": {"method": calibration.threshold_method, "value": calibration.threshold},
        "train_files": TRAIN_FILES,
        "validation_files": VALIDATION_FILES,
        "test_files": TEST_FILES,
        "model_artifact_path": str(resolved_model_path),
        "scaler_artifact_path": str(resolved_scaler_path),
        "reused_existing_artifact": True,
        "retraining_performed": False,
    }

    experiment_id = register_experiment_run("B", config, metrics, engine=registry_engine)

    # --- round-trip verification before reporting ---
    retrieved = get_experiment_run(experiment_id, engine=registry_engine)
    if retrieved.config != config or retrieved.metrics != metrics:
        raise RuntimeError(
            f"Round-trip verification failed for {experiment_id} -- registered and "
            "retrieved run data do not match."
        )

    return experiment_id, metrics


def main() -> int:
    experiment_id, metrics = run_experiment_b()

    print(f"Experiment ID: {experiment_id}")
    print(json.dumps(metrics, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
