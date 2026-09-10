"""TASK 9.4 — Experiment C: DSP Features -> Autoencoder.

Blueprint.md section 19's Experiment C: the same DSP feature vectors (TASK 5.3)
and scaling (TASK 5.4) as Experiment B, but scored via the Autoencoder's
reconstruction error (TASK 7.1-7.3) instead of Isolation Forest. Exactly like
TASK 9.3's Experiment B, this is bookkeeping over what Phase 7/8 already built and
measured, not a new ML experiment -- this script never trains anything.

CRITICAL -- NO RETRAINING ANYWHERE IN THIS SCRIPT: never imports or calls
`app.ml.training.train_autoencoder` (or anything that calls `.backward()`/
`optimizer.step()` on the Autoencoder). It loads the REAL, STANDING artifacts
TASK 7.3 already produced (`app.ml.model_artifact.load_model_artifact`, defaulting
to `models/autoencoder_v1.pt` + its `models/autoencoder_v1.json` metadata
sidecar) and only ever calls `.eval()`-mode forward passes on it via the existing
`app.ml.inference.reconstruction_error` (TASK 7.3's own function, unmodified).
Verified directly by this task's own tests, which spy across a full run of this
script and confirm no gradient-consuming/parameter-updating call occurs.

WHY THIS SCRIPT USES `load_model_artifact` (TASK 6.5) RATHER THAN THE LOWER-LEVEL
`load_model`/`load_scaler` PAIR run_experiment_b.py USES: unlike Isolation
Forest's artifact (bare `.pkl` + `.pkl`, no metadata sidecar was ever written for
it), TASK 7.3's real-data test DID persist a full `ModelArtifactMetadata` sidecar
alongside the Autoencoder (`models/autoencoder_v1.json`) -- that JSON is itself a
real, already-produced Phase 7/8 record containing `feature_names`,
`feature_dimension`, `scaler_artifact`, `threshold_method`, an already-calibrated
`threshold_value`, `score_direction`, `dataset_hash`, `split_manifest_hash`, and
`random_seed`. This script reads ALL of these directly from that metadata rather
than re-deriving them independently -- e.g. `random_seed` is read from the
metadata (a PyTorch `state_dict` carries no training-seed attribute the way
`sklearn`'s `IsolationForest.get_params()` does, so this is the only way to know
it without retraining).

ONE GENUINE GAP IN WHAT THE PERSISTED METADATA CAN CAPTURE (disclosed, not
hidden): TASK 6.5's `ModelArtifactMetadata` schema stores the CALIBRATED
`threshold_value` itself, but not the intermediate min/max normalization
parameters (`app.ml.scoring.ScoreNormalizationParams`) needed to correctly
normalize a NEW batch of test reconstruction errors into the same `[0,1]` scale
that threshold was calibrated in -- there is nowhere in that schema for them, and
adding one would be exactly the "new metadata schema" this task forbids
inventing. This script therefore recalibrates via `app.ml.scoring.calibrate`
(TASK 6.3, unmodified) on the SAME validation files the persisted metadata itself
was calibrated from -- deterministically reproducing the identical threshold
already recorded (verified as an explicit consistency check against `metadata.
threshold_value`, not assumed), while also obtaining the normalization parameters
needed to score test data. This is not a second/duplicate calibration
implementation, and it is not "re-training" (this project's own established
vocabulary keeps calibration strictly separate from training everywhere).

SAME TEST SET AS EXPERIMENTS A AND B: imports `TRAIN_FILES`/`VALIDATION_FILES`/
`TEST_FILES`/`CHANNEL`/`WINDOW_SIZE`/`OVERLAP`/`THRESHOLD_METHOD`/
`PERCENTILE_VALUE` directly from `scripts.run_experiment_a` (the same objects
`run_experiment_b.py` already imports), guaranteeing all three experiments score
against identical windows/ground truth by construction.

`dataset_hash`/`split_manifest_hash`: this script's OWN config records TASK 6.5's
real hashes (`app.ml.model_artifact.compute_dataset_hash`/
`compute_split_manifest_hash`), and additionally cross-checks them against the
values already recorded in the loaded metadata sidecar -- both must agree, or this
script refuses to proceed (a real, checkable consistency guard, not merely
copying the metadata's own numbers uncritically).
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

import pandas as pd  # noqa: E402
from sqlalchemy import Engine  # noqa: E402

from app.datasets.loader import DATASET_ROOT as REAL_DATASET_ROOT  # noqa: E402
from app.datasets.loader import LoadedRecording, load_all_splits  # noqa: E402
from app.datasets.validators import SAMPLING_RATE_HZ  # noqa: E402
from app.features.extractor import extract_feature_matrix  # noqa: E402
from app.features.registry import FEATURE_REGISTRY, FREQUENCY_FEATURE_REGISTRY  # noqa: E402
from app.ml.evaluation import evaluate  # noqa: E402
from app.ml.experiment_registry import get_experiment_run, register_experiment_run  # noqa: E402
from app.ml.inference import default_autoencoder_path, default_models_dir, reconstruction_error  # noqa: E402
from app.ml.inference import load_scaler as load_scaler_artifact  # noqa: E402
from app.ml.model_artifact import (  # noqa: E402
    ModelArtifactMetadata,
    compute_dataset_hash,
    compute_split_manifest_hash,
    load_model_artifact,
)
from app.ml.scoring import calibrate, classify, normalize_scores  # noqa: E402
from app.signal_processing.windowing import Window, create_windows  # noqa: E402
from scripts.run_experiment_a import (  # noqa: E402
    CHANNEL,
    OVERLAP,
    PERCENTILE_VALUE,
    TEST_FILES,
    THRESHOLD_METHOD,
    TRAIN_FILES,
    VALIDATION_FILES,
    WINDOW_SIZE,
)

# Feature-extraction Welch PSD parameters -- same values TASK 8.1/9.1/9.3's own
# real-data runs already used for this exact representation.
NPERSEG = 256
NOVERLAP = 128


def _verify_metadata_matches_expected_representation(metadata: ModelArtifactMetadata) -> None:
    """Checkable evidence the loaded artifact is genuinely Phase 7/8's
    DSP-features Autoencoder -- see module docstring."""
    expected_feature_names = list(FEATURE_REGISTRY.keys()) + list(FREQUENCY_FEATURE_REGISTRY.keys())

    if metadata.feature_names != expected_feature_names:
        raise RuntimeError(
            "Loaded Autoencoder metadata's feature_names do not match this project's "
            "real DSP feature registry (names and/or order) -- BLOCKED, refusing to proceed."
        )
    if metadata.feature_dimension != len(expected_feature_names):
        raise RuntimeError(
            f"Loaded Autoencoder metadata's feature_dimension ({metadata.feature_dimension}) does not "
            f"match the real DSP feature registry ({len(expected_feature_names)}) -- BLOCKED."
        )

    real_dataset_hash = compute_dataset_hash()
    real_split_manifest_hash = compute_split_manifest_hash()
    if metadata.dataset_hash != real_dataset_hash:
        raise RuntimeError(
            "Loaded Autoencoder metadata's dataset_hash does not match the current real "
            "dataset fingerprint -- BLOCKED, refusing to proceed with a possibly stale artifact."
        )
    if metadata.split_manifest_hash != real_split_manifest_hash:
        raise RuntimeError(
            "Loaded Autoencoder metadata's split_manifest_hash does not match the current real "
            "split manifest -- BLOCKED, refusing to proceed with a possibly stale artifact."
        )


def _write_temp_manifest(directory: Path) -> Path:
    manifest_path = directory / "experiment_c_manifest.json"
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


def run_experiment_c(
    *,
    dataset_root: Path = REAL_DATASET_ROOT,
    model_path: Path | None = None,
    registry_engine: Engine | None = None,
) -> tuple[str, dict[str, Any]]:
    """Runs Experiment C end-to-end (load existing Autoencoder + its TASK 6.5
    metadata, score/calibrate/evaluate on the canonical Phase 9 test set) and
    returns `(experiment_id, metrics)`. Never trains anything.

    Args:
        dataset_root: override only for testing; production runs use the real
            `data/raw/mafaulda/` (the default).
        model_path: override only for testing against a test-specific artifact;
            production runs use the real, standing `models/autoencoder_v1.pt`
            (the default, via `default_autoencoder_path()`).
        registry_engine: override only for testing (an isolated database);
            production runs use the project's shared experiment registry.
    """
    resolved_model_path = model_path or default_autoencoder_path()

    model, metadata = load_model_artifact(resolved_model_path)
    _verify_metadata_matches_expected_representation(metadata)

    scaler_path = default_models_dir() / metadata.scaler_artifact
    scaler = load_scaler_artifact(scaler_path)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        manifest_path = _write_temp_manifest(Path(tmp_dir_name))
        loaded = load_all_splits(dataset_root=dataset_root, manifest_path=manifest_path)

        validation_windows, validation_labels = _windows_and_labels(loaded["validation"], dataset_root)
        test_windows, test_labels = _windows_and_labels(loaded["test"], dataset_root)

    validation_features = extract_feature_matrix(
        validation_windows, SAMPLING_RATE_HZ, nperseg=NPERSEG, noverlap=NOVERLAP
    )
    test_features = extract_feature_matrix(test_windows, SAMPLING_RATE_HZ, nperseg=NPERSEG, noverlap=NOVERLAP)

    if validation_features.shape[1] != metadata.feature_dimension or test_features.shape[1] != metadata.feature_dimension:
        raise RuntimeError(
            f"Extracted feature dimension does not match the loaded Autoencoder's expected "
            f"feature_dimension ({metadata.feature_dimension}) -- BLOCKED, refusing to proceed."
        )

    from app.ml.scaling import apply_scaler

    scaled_validation = apply_scaler(scaler, validation_features)
    scaled_test = apply_scaler(scaler, test_features)

    # --- calibrate on validation only (never test); reconstruction_error is
    # already in the "higher = more anomalous" orientation -- no sign flip. ---
    validation_errors = reconstruction_error(model, scaled_validation)
    calibration = calibrate(
        validation_errors,
        threshold_method=THRESHOLD_METHOD,
        percentile_value=PERCENTILE_VALUE,
        validation_labels=validation_labels,
    )

    # Consistency check (not a silent assumption): recalibrating on the same
    # validation files/method the persisted metadata itself was calibrated from
    # must reproduce the SAME threshold already recorded there.
    if abs(calibration.threshold - metadata.threshold_value) > 1e-9:
        raise RuntimeError(
            f"Freshly-calibrated threshold ({calibration.threshold}) does not match the threshold "
            f"already persisted in the Autoencoder's metadata ({metadata.threshold_value}) -- "
            "BLOCKED: this indicates the loaded artifact/metadata may not correspond to the "
            "canonical validation set this script expects."
        )

    # --- score test (inference_time measured around this only) ---
    y_true_test = _binary_labels(test_labels)

    start = time.perf_counter()
    test_errors = reconstruction_error(model, scaled_test)
    test_normalized_scores = normalize_scores(test_errors, calibration.normalization)
    test_predictions = classify(test_normalized_scores, calibration).astype(int)
    inference_time = time.perf_counter() - start

    # --- TASK 8.1 evaluation, unmodified ---
    metrics = evaluate(y_true_test, test_predictions, test_normalized_scores, inference_time)

    # --- register the run (TASK 8.3 generates experiment_id) ---
    config = {
        "dataset_hash": metadata.dataset_hash,
        "split_manifest_hash": metadata.split_manifest_hash,
        "representation": "dsp_features",
        "model": "autoencoder",
        "feature_dimension": metadata.feature_dimension,
        "feature_names": metadata.feature_names,
        "random_seed": metadata.random_seed,
        "preprocessing_config": {
            "window_size": WINDOW_SIZE,
            "overlap": OVERLAP,
            "channel": CHANNEL,
            "nperseg": NPERSEG,
            "noverlap": NOVERLAP,
        },
        "model_config": {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "bottleneck_dim": model.bottleneck_dim,
        },
        "threshold": {"method": calibration.threshold_method, "value": calibration.threshold},
        "score_direction": metadata.score_direction.value,
        "train_files": TRAIN_FILES,
        "validation_files": VALIDATION_FILES,
        "test_files": TEST_FILES,
        "model_artifact_path": str(resolved_model_path),
        "scaler_artifact_path": str(scaler_path),
        "model_metadata_created_at": metadata.created_at,
        "reused_existing_artifact": True,
        "retraining_performed": False,
    }

    experiment_id = register_experiment_run("C", config, metrics, engine=registry_engine)

    # --- round-trip verification before reporting ---
    retrieved = get_experiment_run(experiment_id, engine=registry_engine)
    if retrieved.config != config or retrieved.metrics != metrics:
        raise RuntimeError(
            f"Round-trip verification failed for {experiment_id} -- registered and "
            "retrieved run data do not match."
        )

    return experiment_id, metrics


def main() -> int:
    experiment_id, metrics = run_experiment_c()

    print(f"Experiment ID: {experiment_id}")
    print(json.dumps(metrics, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
