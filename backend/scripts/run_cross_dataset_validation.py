"""TASK 13.1 -- cross-dataset validation: evaluate the real, standing
MAFAULDA-trained models (`models/isolation_forest_v1.pkl`,
`models/autoencoder_v1.pt` -- unchanged) on the real CWRU Bearing Dataset,
WITHOUT any retraining, refitting, or recalibration on CWRU. Mirrors
`scripts/run_experiment_b.py`/`run_experiment_c.py`'s exact "load the real
standing artifact, never train/fit anything" pattern -- only the TEST data
source changes (CWRU instead of MAFAULDA's own held-out test split).

WHAT IS REUSED, UNCHANGED, FROM MAFAULDA'S OWN PIPELINE:
  - `app.ml.model_artifact.load_model_artifact` -- loads the model + its real
    TASK 6.5 metadata sidecar (`feature_names`, `feature_dimension`,
    `scaler_artifact`, `threshold_method`, the already-CALIBRATED
    `threshold_value`, `score_direction`).
  - `app.ml.inference.load_scaler` -- loads the SAME scaler the model's
    metadata names (`scaler_v1.pkl`), fit exclusively on MAFAULDA train data.
    Only ever `.transform()`-ed (via `app.ml.scaling.apply_scaler`/
    `app.ml.inference.predict`) -- never refit here or anywhere in this file.
  - `app.features.registry`/`app.features.extractor.extract_feature_matrix` --
    the exact same 15 DSP features, same order, computed for CWRU windows too.
  - `app.ml.scoring` -- `to_anomaly_score`/`normalize_scores`/`classify`,
    unmodified.
  - `app.ml.evaluation.evaluate` -- unmodified, model-agnostic.

THRESHOLD: NEVER recalibrated on CWRU. `ModelArtifactMetadata` (TASK 6.5)
persists the calibrated `threshold_value` but NOT the intermediate min/max
normalization parameters needed to rescale a brand-new batch of scores into
the same `[0,1]` space that threshold lives in (same documented gap
`run_experiment_c.py` already discloses for the Autoencoder). This script
closes that gap exactly the way `run_experiment_c.py` already does: it
recalibrates via `app.ml.scoring.calibrate` on the SAME real MAFAULDA
validation files (`scripts.run_experiment_a.VALIDATION_FILES`) the persisted
metadata was itself calibrated from, and then asserts (does not merely hope)
that the freshly-computed threshold reproduces the ALREADY-PERSISTED
`metadata.threshold_value` before proceeding -- refusing to continue if they
disagree. CWRU data plays no role whatsoever in this calibration step; it is
used exclusively as final, held-out evaluation data, after calibration is
already fixed.

SAMPLING RATE: CWRU's real sampling rate (12 kHz or 48 kHz, resolved by
`app.datasets.cwru_loader` from the uploaded file tree) is passed to
`extract_feature_matrix` for CWRU windows -- NOT MAFAULDA's 50 kHz constant.
Window SIZE (in samples) and overlap FRACTION are reused unitlessly from
MAFAULDA's own configuration; the resulting physical time span per window
therefore differs between datasets, a genuine, disclosed methodological
difference discussed in docs/results/cross_dataset_validation.md, not
silently equalized here.

LABELS: CWRU's 4 real classes (`app.datasets.cwru_loader.CWRULabel`: normal,
ball, inner-race, outer-race) are reduced to the SAME binary target
`evaluate()` requires (`0 = normal`, `1 = anomaly`) exactly like MAFAULDA's
own 4 classes already are, everywhere else in this project (`scripts.
run_experiment_b/c._binary_labels`) -- fault-type information is not lost
(`cwru_labels_present` in this script's output), only aggregated for the
binary metric computation AC1 asks for.

GROUPING BY REAL SAMPLING RATE: the real, uploaded CWRU mirror
(`data/external/cwru/`) spans two real rates under one root -- 12 kHz
(`12k_Drive_End_Bearing_Fault_Data/`, `12k_Fan_End_Bearing_Fault_Data/`,
`Normal/` once its rate override is supplied) and 48 kHz
(`48k_Drive_End_Bearing_Fault_Data/`). `run_cross_dataset_validation` loads
every real recording once, groups them by each recording's own resolved
`sampling_rate`, and evaluates each group separately -- never mixing windows
computed at different frequency resolutions into one evaluation. The 48 kHz
group in this particular real download contains ONLY fault recordings (no
`Normal/` file was collected at 48 kHz in this mirror -- see
`app.datasets.cwru_loader`'s docstring for the real, disclosed ambiguity this
addresses) -- `evaluate()` (TASK 8.1) already handles a single-class test set
honestly (undefined metrics reported as `NaN`, never fabricated), so this is
reported as-is, not worked around.

STATUS: the real CWRU dataset (161 real `.mat` files) is present at
`data/external/cwru/` as of this writing. `run_cross_dataset_validation()`
(and this file's `main()`) can be run end-to-end against real data -- see
`docs/results/cross_dataset_validation.md` for the actual, real results this
produced.
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

from app.datasets.cwru_loader import CWRURecording, DEFAULT_CWRU_ROOT, load_all_recordings  # noqa: E402
from app.datasets.loader import DATASET_ROOT as MAFAULDA_DATASET_ROOT  # noqa: E402
from app.datasets.loader import load_all_splits  # noqa: E402
from app.datasets.validators import SAMPLING_RATE_HZ as MAFAULDA_SAMPLING_RATE_HZ  # noqa: E402
from app.features.extractor import extract_feature_matrix  # noqa: E402
from app.features.registry import FEATURE_REGISTRY, FREQUENCY_FEATURE_REGISTRY  # noqa: E402
from app.ml.evaluation import evaluate  # noqa: E402
from app.ml.inference import default_autoencoder_path, default_model_path, default_models_dir  # noqa: E402
from app.ml.inference import load_scaler  # noqa: E402
from app.ml.inference import predict as isolation_forest_predict  # noqa: E402
from app.ml.inference import reconstruction_error  # noqa: E402
from app.ml.model_artifact import ModelArtifactMetadata, ModelType, load_model_artifact  # noqa: E402
from app.ml.scaling import apply_scaler  # noqa: E402
from app.ml.scoring import ScoringCalibration  # noqa: E402
from app.ml.scoring import calibrate, classify, normalize_scores, to_anomaly_score  # noqa: E402
from app.signal_processing.windowing import Window, create_windows  # noqa: E402
from scripts.run_experiment_a import CHANNEL as MAFAULDA_CHANNEL  # noqa: E402
from scripts.run_experiment_a import OVERLAP as MAFAULDA_OVERLAP  # noqa: E402
from scripts.run_experiment_a import PERCENTILE_VALUE  # noqa: E402
from scripts.run_experiment_a import THRESHOLD_METHOD  # noqa: E402
from scripts.run_experiment_a import VALIDATION_FILES as MAFAULDA_VALIDATION_FILES  # noqa: E402
from scripts.run_experiment_a import WINDOW_SIZE as MAFAULDA_WINDOW_SIZE  # noqa: E402

# Feature-extraction Welch PSD parameters -- same values every real experiment
# script (B/C) and TASK 8.1/9.x's own real-data runs already use.
NPERSEG = 256
NOVERLAP = 128

# CWRU windowing reuses MAFAULDA's own window size (samples) / overlap
# (fraction) unchanged -- see module docstring's "SAMPLING RATE" note for why
# this is a disclosed, real difference rather than something to equalize.
CWRU_WINDOW_SIZE = MAFAULDA_WINDOW_SIZE
CWRU_OVERLAP = MAFAULDA_OVERLAP


class CrossDatasetValidationError(RuntimeError):
    """Raised when cross-dataset validation cannot proceed against real,
    consistent data (feature-registry mismatch, ambiguous CWRU sampling
    rates, a calibration consistency check that fails). Never produces
    fabricated/partial metrics when raised."""


def _expected_feature_names() -> list[str]:
    return list(FEATURE_REGISTRY.keys()) + list(FREQUENCY_FEATURE_REGISTRY.keys())


def _binary_labels(labels: list[str]) -> list[int]:
    return [0 if label == "normal" else 1 for label in labels]


def _mafaulda_validation_features() -> tuple[pd.DataFrame, list[str]]:
    """The exact same real MAFAULDA validation windows/features Experiments
    A/B/C already use. Used ONLY to reproduce the calibration those
    experiments already computed (and to verify it matches the persisted
    `threshold_value`) -- CWRU never participates in this step."""
    windows: list[Window] = []
    labels: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        manifest_path = Path(tmp_dir_name) / "manifest.json"
        manifest_path.write_text(
            json.dumps({"splits": {"validation": MAFAULDA_VALIDATION_FILES}}), encoding="utf-8"
        )
        loaded = load_all_splits(dataset_root=MAFAULDA_DATASET_ROOT, manifest_path=manifest_path)

        for recording in loaded["validation"]:
            df = pd.read_csv(MAFAULDA_DATASET_ROOT / recording.relative_path, header=None)
            values = df[MAFAULDA_CHANNEL].tolist()
            recording_windows = create_windows(
                values,
                recording_id=recording.relative_path,
                split=recording.split,
                window_size=MAFAULDA_WINDOW_SIZE,
                overlap=MAFAULDA_OVERLAP,
            )
            windows.extend(recording_windows)
            labels.extend([recording.label.value] * len(recording_windows))

    features = extract_feature_matrix(windows, MAFAULDA_SAMPLING_RATE_HZ, nperseg=NPERSEG, noverlap=NOVERLAP)
    return features, labels


def _cwru_windows_and_labels(recordings: list[CWRURecording]) -> tuple[list[Window], list[str], float]:
    """Windows every real, already-loaded CWRU recording/channel. Refuses to
    silently mix more than one real sampling rate into a single evaluation
    run (each real CWRU subset has ONE sampling rate; mixing them would make
    the frequency-domain features incomparable within the same run)."""
    sampling_rates = {recording.sampling_rate for recording in recordings}
    if len(sampling_rates) > 1:
        raise CrossDatasetValidationError(
            f"Loaded CWRU recordings span more than one real sampling rate ({sorted(sampling_rates)}) -- "
            "run cross-dataset validation once per sampling rate (point cwru_dataset_root at a single "
            "12khz/48khz subtree) rather than mixing them in one evaluation."
        )
    sampling_rate = sampling_rates.pop()

    windows: list[Window] = []
    labels: list[str] = []
    for recording in recordings:
        recording_windows = create_windows(
            list(recording.values),
            recording_id=f"{recording.relative_path}::{recording.channel_name}",
            split="cwru_external",
            window_size=CWRU_WINDOW_SIZE,
            overlap=CWRU_OVERLAP,
        )
        windows.extend(recording_windows)
        labels.extend([recording.label.value] * len(recording_windows))
    return windows, labels, sampling_rate


def _load_calibrated_model(
    model_path: Path,
) -> tuple[object, object, ModelArtifactMetadata, ScoringCalibration]:
    model, metadata = load_model_artifact(model_path)

    expected_feature_names = _expected_feature_names()
    if metadata.feature_names != expected_feature_names:
        raise CrossDatasetValidationError(
            f"{model_path}: metadata.feature_names do not match the real DSP feature registry -- "
            "refusing to proceed."
        )

    scaler_path = default_models_dir() / metadata.scaler_artifact
    scaler = load_scaler(scaler_path)

    validation_features, validation_labels = _mafaulda_validation_features()
    is_autoencoder = metadata.model_type == ModelType.AUTOENCODER

    if is_autoencoder:
        scaled_validation = apply_scaler(scaler, validation_features)
        validation_scores = reconstruction_error(model, scaled_validation)
    else:
        raw = isolation_forest_predict(model, scaler, validation_features)
        validation_scores = to_anomaly_score(raw)

    calibration = calibrate(
        validation_scores,
        threshold_method=THRESHOLD_METHOD,
        percentile_value=PERCENTILE_VALUE,
        validation_labels=validation_labels,
    )

    if abs(calibration.threshold - metadata.threshold_value) > 1e-9:
        raise CrossDatasetValidationError(
            f"{model_path}: freshly-calibrated threshold ({calibration.threshold}) does not match "
            f"the threshold already persisted in this model's metadata ({metadata.threshold_value}) -- "
            "BLOCKED, refusing to evaluate with a possibly-inconsistent calibration."
        )

    return model, scaler, metadata, calibration


def evaluate_model_on_cwru(
    *,
    model_path: Path,
    cwru_dataset_root: Path = DEFAULT_CWRU_ROOT,
    cwru_recordings: list[CWRURecording] | None = None,
) -> dict[str, Any]:
    """Evaluates ONE MAFAULDA-trained model (never retrained/refit here) on
    the real CWRU dataset.

    Args:
        cwru_recordings: pass a pre-loaded, already sampling-rate-homogeneous
            list of `CWRURecording`s (e.g. one group from
            `group_recordings_by_sampling_rate`) to evaluate exactly that
            subset. When omitted, every real recording under
            `cwru_dataset_root` is loaded via `load_all_recordings` -- which
            raises `CrossDatasetValidationError` below if that set spans more
            than one real sampling rate (this project's real CWRU mirror
            does: 12 kHz and 48 kHz subsets coexist under one root).

    Raises:
        CWRULoaderError: the real CWRU dataset (or a requested file within
            it) is not present -- see `app.datasets.cwru_loader`.
        CrossDatasetValidationError: a consistency check fails (feature
            registry mismatch, ambiguous sampling rates, threshold mismatch).
        ModelArtifactError / ModelPersistenceError: the MAFAULDA model/scaler
            artifact itself is missing or corrupt.
    """
    model, scaler, metadata, calibration = _load_calibrated_model(model_path)
    is_autoencoder = metadata.model_type == ModelType.AUTOENCODER

    resolved_recordings = (
        cwru_recordings if cwru_recordings is not None else load_all_recordings(dataset_root=cwru_dataset_root)
    )
    cwru_windows, cwru_labels, cwru_sampling_rate = _cwru_windows_and_labels(resolved_recordings)

    cwru_features = extract_feature_matrix(cwru_windows, cwru_sampling_rate, nperseg=NPERSEG, noverlap=NOVERLAP)
    if cwru_features.shape[1] != metadata.feature_dimension:
        raise CrossDatasetValidationError(
            f"CWRU feature dimension ({cwru_features.shape[1]}) does not match the loaded model's "
            f"expected feature_dimension ({metadata.feature_dimension}) -- BLOCKED."
        )

    y_true = _binary_labels(cwru_labels)

    start = time.perf_counter()
    if is_autoencoder:
        scaled_cwru = apply_scaler(scaler, cwru_features)
        raw_scores = reconstruction_error(model, scaled_cwru)
    else:
        raw = isolation_forest_predict(model, scaler, cwru_features)
        raw_scores = to_anomaly_score(raw)
    normalized_scores = normalize_scores(raw_scores, calibration.normalization)
    predictions = classify(normalized_scores, calibration).astype(int)
    inference_time = time.perf_counter() - start

    metrics = evaluate(y_true, predictions, normalized_scores, inference_time)

    return {
        "model": metadata.model_type.value,
        "model_artifact_path": str(model_path),
        "model_version": metadata.model_version,
        "threshold_value": metadata.threshold_value,
        "threshold_method": metadata.threshold_method,
        "score_direction": metadata.score_direction.value,
        "cwru_sampling_rate_hz": cwru_sampling_rate,
        "cwru_num_windows": len(cwru_windows),
        "cwru_labels_present": sorted(set(cwru_labels)),
        "retraining_performed": False,
        "recalibrated_on_cwru": False,
        "metrics": metrics,
    }


def _group_recordings_by_sampling_rate(
    recordings: list[CWRURecording],
) -> dict[float, list[CWRURecording]]:
    """Groups already-loaded real CWRU recordings by their OWN resolved
    `sampling_rate` (each `CWRURecording` already carries the real rate
    `app.datasets.cwru_loader` determined for it). This project's real,
    downloaded CWRU mirror spans two real rates (12 kHz: both
    `12k_Drive_End_...` and `12k_Fan_End_...`, plus `Normal/` once its rate
    override is supplied; 48 kHz: `48k_Drive_End_...` only) under one root --
    each group is evaluated separately (see `_cwru_windows_and_labels`'s own
    refusal to silently mix rates within one evaluation)."""
    groups: dict[float, list[CWRURecording]] = {}
    for recording in recordings:
        groups.setdefault(recording.sampling_rate, []).append(recording)
    return groups


def run_cross_dataset_validation(
    *, cwru_dataset_root: Path = DEFAULT_CWRU_ROOT, normal_sampling_rate_hz: float | None = None
) -> dict[str, Any]:
    """Evaluates BOTH real, standing MAFAULDA models on the real CWRU
    dataset, once per real sampling-rate subset found under
    `cwru_dataset_root` (this project's real download spans 12 kHz and 48 kHz
    -- see `_group_recordings_by_sampling_rate`). Raises (never fabricates a
    result) if the CWRU dataset is not present -- see module docstring.

    Args:
        normal_sampling_rate_hz: forwarded to `app.datasets.cwru_loader.
            load_all_recordings` for every file under `Normal/` -- required
            for `Normal/` to load at all (see that module's docstring for the
            real, disclosed ambiguity in CWRU's own documentation this
            addresses).

    Returns:
        `{"<rate_hz>": {"isolation_forest": {...}, "autoencoder": {...}}}` --
        one entry per real sampling rate actually present in the loaded data.
    """
    all_recordings = load_all_recordings(
        dataset_root=cwru_dataset_root, normal_sampling_rate_hz=normal_sampling_rate_hz
    )
    grouped = _group_recordings_by_sampling_rate(all_recordings)

    results: dict[str, Any] = {}
    for sampling_rate, recordings in sorted(grouped.items()):
        results[f"{sampling_rate:.0f}hz"] = {
            "isolation_forest": evaluate_model_on_cwru(
                model_path=default_model_path(), cwru_recordings=recordings
            ),
            "autoencoder": evaluate_model_on_cwru(
                model_path=default_autoencoder_path(), cwru_recordings=recordings
            ),
        }
    return results


# The real, uploaded CWRU mirror's `Normal/` files carry no sampling-rate
# directory segment (see app.datasets.cwru_loader's module docstring for the
# real ambiguity in CWRU's own "Apparatus and Procedures" documentation).
# 12,000 Hz was confirmed as the value to use for this run by the user
# directly (2026-09-11), based on that page's own wording: 48 kHz is
# documented as an ADDITIONAL rate specifically "for drive end bearing
# faults", implying the (non-fault) Normal baseline set uses the general,
# base 12,000 Hz rate. This is an explicit, disclosed, user-confirmed value --
# not silently assumed by this script.
DEFAULT_NORMAL_SAMPLING_RATE_HZ = 12000.0


def main() -> int:
    results = run_cross_dataset_validation(normal_sampling_rate_hz=DEFAULT_NORMAL_SAMPLING_RATE_HZ)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
