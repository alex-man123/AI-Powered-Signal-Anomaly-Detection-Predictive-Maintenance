"""TASK 10.5 -- orchestrates HTTP requests into Phase 5-8's existing artifacts,
registry, and ML pipeline for `GET /api/models`, `GET /api/models/{id}/performance`,
`POST /api/models/predict`. No feature extraction, scaling, model architecture,
scoring, or threshold-calibration ALGORITHM is reimplemented here -- everything
numeric is delegated to the real, already-tested Phase 5/6/7/8 modules.

REAL SOURCES OF TRUTH, reused directly (never duplicated):
  - Model artifact + its Model Artifact Contract metadata (TASK 6.5/7.3):
    `app.ml.model_artifact.load_model_artifact` on the real, standing
    `models/isolation_forest_v1.pkl` / `models/autoencoder_v1.pt`.
  - Real Phase 8 metrics: TASK 8.3's experiment registry
    (`app.ml.experiment_registry.get_experiment_run`) -- specifically
    `EXP-B-001` (Isolation Forest, DSP features) and `EXP-C-001` (Autoencoder,
    DSP features): the two DSP-features models Phase 6/7 actually trained and
    Phase 8 actually measured (TASK 9.3/9.4 already independently confirmed
    these registrations are bit-identical to Phase 8.1's own original
    real-data measurement -- this is the SAME real source, not a second one).
    Experiment A (raw+PCA) is a separate Phase 9 representation, not one of
    "Isolation Forest and Autoencoder" this task's AC1 asks for.
  - Feature extraction: `app.features.extractor.extract_features`/
    `extract_feature_matrix` (TASK 5.3, registry-driven).
  - Scaling: `app.ml.scaling.apply_scaler`, via `app.ml.inference.predict`
    (TASK 5.4/6.4) -- transform only, never refit.
  - Scoring/threshold: `app.ml.scoring` (TASK 6.3) -- `to_anomaly_score`,
    `calibrate`, `normalize_scores`, `classify`.
  - Autoencoder reconstruction error: `app.ml.inference.reconstruction_error`
    (TASK 7.3) -- already anomaly-oriented, no sign flip needed.

CALIBRATION GAP (disclosed, not silently patched, and NOT unique to this
task): TASK 6.5's `ModelArtifactMetadata` persists `threshold_value` but NOT
the validation min/max `normalize_scores` needs to turn a brand-new raw score
into `[0,1]`. TASK 9.2/9.3/9.4's own scripts face this exact same gap and all
three resolve it the same way: recalibrate fresh from the real validation set
(`app.ml.scoring.calibrate`) rather than inventing a persistence field Phase 6
never defined. This module follows that SAME established pattern. Recalibration
(and the real per-feature "normal" baseline used for explanations, see below)
is computed ONCE per process and cached module-level -- loading/windowing the
real validation recordings on every single `/predict` request would be the
`scripts/run_experiment_*.py`-style one-time cost repeated needlessly, not a
new algorithm.

The real validation-set configuration (`VALIDATION_FILES`/`WINDOW_SIZE`/
`OVERLAP`/`CHANNEL`/`THRESHOLD_METHOD`/`PERCENTILE_VALUE`) is imported DIRECTLY
from `scripts.run_experiment_a` (TASK 9.2) rather than re-declared -- the same
"import the canonical constants, never redeclare a parallel copy that can
drift" convention TASK 9.3/9.4 already established for exactly this reason.
Verified importable from this package's normal execution context (both
`uv run python`/`uv run pytest`), and this project's own existing test suite
(`backend/tests/experiments/test_experiment_b.py`, etc.) already imports from
`scripts.run_experiment_a` the same way.

STATUS THRESHOLDS (NORMAL/WARNING/ANOMALY): TASK 6.3 calibrates exactly ONE
threshold (the ANOMALY boundary -- `app.ml.scoring.classify`'s own `>=` cut).
No second, calibrated WARNING threshold exists anywhere in this project, and
blueprint.md's own illustrative `0.30`/`0.70` pair is explicitly framed there
as "puncte de plecare ilustrative... nu adevăr universal", not a calibrated
value this module could load. Rather than inventing an unrelated absolute
constant, WARNING's lower bound is derived PROPORTIONALLY from the one real
calibrated threshold that exists (`threshold * WARNING_THRESHOLD_RATIO`) --
disclosed here, tied to the model's own real calibration (if the threshold is
ever recalibrated to a different value, this boundary moves with it), rather
than a fixed number disconnected from it.

EXPLANATION: grounded in the real feature values `extract_features` computes
for the given signal, compared against a real per-feature mean/std baseline
computed from the SAME real normal-labeled validation windows used for
calibration (cached alongside it) -- never a generic string, never a feature
declared "abnormal" merely for having a large value. TASK 12.1 moved the
actual significance/deviation calculation into
`app.services.explanation_service` (this module now calls into it for both
the free-text `explanation` string and the structured `explanations` list --
one computation, two presentations, never two competing baseline/threshold
rules).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from app.api.schemas.models import (
    FeatureExplanation,
    ModelArtifactInfo,
    ModelMetrics,
    ModelResponse,
    PredictRequest,
    PredictResponse,
    PredictionStatus,
)
from app.datasets.loader import DATASET_ROOT as REAL_DATASET_ROOT
from app.datasets.loader import LoadedRecording, load_all_splits
from app.datasets.validators import SAMPLING_RATE_HZ
from app.features.extractor import extract_feature_matrix, extract_features
from app.ml.experiment_registry import ExperimentRegistryError, get_experiment_run
from app.ml.inference import ModelPersistenceError, default_autoencoder_path, default_model_path, default_scaler_path
from app.ml.inference import load_autoencoder as _load_autoencoder
from app.ml.inference import load_model as _load_isolation_forest_model
from app.ml.inference import load_scaler as _load_scaler
from app.ml.inference import predict as isolation_forest_raw_predict
from app.ml.inference import reconstruction_error as autoencoder_reconstruction_error
from app.ml.model_artifact import ModelArtifactError, ModelType, load_model_artifact
from app.ml.scoring import ScoringCalibration, calibrate, normalize_scores, to_anomaly_score
from app.services import explanation_service
from app.services.explanation_service import FeatureDeviation
from app.signal_processing.windowing import Window, create_windows
from scripts.run_experiment_a import (
    CHANNEL,
    OVERLAP,
    PERCENTILE_VALUE,
    THRESHOLD_METHOD,
    VALIDATION_FILES,
    WINDOW_SIZE,
)

# Same Welch PSD parameters TASK 8.1/9.1's own real-data DSP-feature runs used.
NPERSEG = 256
NOVERLAP = 128

ISOLATION_FOREST_EXPERIMENT_ID = "EXP-B-001"
AUTOENCODER_EXPERIMENT_ID = "EXP-C-001"

# See module docstring's "STATUS THRESHOLDS" section.
WARNING_THRESHOLD_RATIO = 0.5


class ModelServiceError(ValueError):
    """Raised for a model-service request this module cannot honestly satisfy
    (an unknown model id, a missing artifact, or missing real Phase 8 metrics).
    Never silently substituted with a placeholder."""


class ModelNotFoundError(ModelServiceError):
    """Raised for a model id/artifact/registered-metrics lookup that does not
    exist. The route layer (`app.api.routes.models`) converts this into HTTP
    404, never a 500."""


# --- artifact + real Phase 8 metrics (GET /api/models, GET /api/models/{id}/performance) ---


def _artifact_path_for(model_type: ModelType) -> Path:
    return default_model_path() if model_type == ModelType.ISOLATION_FOREST else default_autoencoder_path()


def _experiment_id_for(model_type: ModelType) -> str:
    return ISOLATION_FOREST_EXPERIMENT_ID if model_type == ModelType.ISOLATION_FOREST else AUTOENCODER_EXPERIMENT_ID


def get_model_response(model_type: ModelType) -> ModelResponse:
    """Real artifact metadata (TASK 6.5) + real Phase 8 metrics (TASK 8.3's
    registry) for one model. Raises `ModelNotFoundError` if either is missing
    -- never fabricates a placeholder metric or metadata field."""
    artifact_path = _artifact_path_for(model_type)
    try:
        _model, metadata = load_model_artifact(artifact_path)
    except ModelArtifactError as exc:
        raise ModelNotFoundError(f"model {model_type.value!r} artifact is not available: {exc}") from exc

    experiment_id = _experiment_id_for(model_type)
    try:
        record = get_experiment_run(experiment_id)
    except ExperimentRegistryError as exc:
        raise ModelNotFoundError(
            f"real Phase 8 metrics for {model_type.value!r} are not registered yet "
            f"(expected experiment {experiment_id!r}): {exc}"
        ) from exc

    return ModelResponse(
        model_type=model_type,
        artifact=ModelArtifactInfo(
            model_version=metadata.model_version,
            feature_names=metadata.feature_names,
            feature_dimension=metadata.feature_dimension,
            threshold_method=metadata.threshold_method,
            threshold_value=metadata.threshold_value,
            score_direction=metadata.score_direction,
            training_split=metadata.training_split,
            dataset_hash=metadata.dataset_hash,
            split_manifest_hash=metadata.split_manifest_hash,
            random_seed=metadata.random_seed,
            created_at=metadata.created_at,
        ),
        metrics=ModelMetrics(**record.metrics),
    )


def list_models() -> list[ModelResponse]:
    """`GET /api/models`: Isolation Forest and Autoencoder (AC1), each with its
    real Model Artifact Contract metadata and real Phase 8 metrics."""
    return [get_model_response(ModelType.ISOLATION_FOREST), get_model_response(ModelType.AUTOENCODER)]


def get_model_performance(model_id: str) -> ModelResponse:
    """`GET /api/models/{id}/performance`. `model_id` is one of `ModelType`'s
    own real values (`"isolation_forest"`/`"autoencoder"`) -- no separate
    numeric model id exists in this project (see TASK 10.1's schema docstring)."""
    try:
        model_type = ModelType(model_id)
    except ValueError as exc:
        raise ModelNotFoundError(f"model {model_id!r} does not exist") from exc
    return get_model_response(model_type)


# --- lazy, process-lifetime caches: real artifacts + real validation-derived calibration ---

_ISOLATION_FOREST_MODEL: Any = None
_SCALER: Any = None
_AUTOENCODER_MODEL: Any = None
_VALIDATION_FEATURES: pd.DataFrame | None = None
_VALIDATION_LABELS: list[str] | None = None
_ISOLATION_FOREST_CALIBRATION: ScoringCalibration | None = None
_AUTOENCODER_CALIBRATION: ScoringCalibration | None = None
_NORMAL_FEATURE_BASELINE: dict[str, tuple[float, float]] | None = None


def _get_isolation_forest_model():
    global _ISOLATION_FOREST_MODEL
    if _ISOLATION_FOREST_MODEL is None:
        try:
            _ISOLATION_FOREST_MODEL = _load_isolation_forest_model(default_model_path())
        except ModelPersistenceError as exc:
            raise ModelNotFoundError(f"model 'isolation_forest' artifact is not available: {exc}") from exc
    return _ISOLATION_FOREST_MODEL


def _get_scaler():
    global _SCALER
    if _SCALER is None:
        try:
            _SCALER = _load_scaler(default_scaler_path())
        except ModelPersistenceError as exc:
            raise ModelNotFoundError(f"model 'isolation_forest' scaler artifact is not available: {exc}") from exc
    return _SCALER


def _get_autoencoder_model():
    global _AUTOENCODER_MODEL
    if _AUTOENCODER_MODEL is None:
        try:
            _AUTOENCODER_MODEL = _load_autoencoder(default_autoencoder_path())
        except ModelPersistenceError as exc:
            raise ModelNotFoundError(f"model 'autoencoder' artifact is not available: {exc}") from exc
    return _AUTOENCODER_MODEL


def _validation_windows_and_labels() -> tuple[list[Window], list[str]]:
    """Loads and windows the real, canonical Phase 9 validation recordings
    (`scripts.run_experiment_a.VALIDATION_FILES`) -- same real files/windowing
    config Experiments A/B/C already use, imported directly, never redeclared."""
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        manifest_path = Path(tmp_dir_name) / "manifest.json"
        manifest_path.write_text(
            json.dumps({"splits": {"validation": VALIDATION_FILES}}), encoding="utf-8"
        )
        loaded = load_all_splits(dataset_root=REAL_DATASET_ROOT, manifest_path=manifest_path)

    windows: list[Window] = []
    labels: list[str] = []
    for recording in loaded["validation"]:
        recording: LoadedRecording
        df = pd.read_csv(REAL_DATASET_ROOT / recording.relative_path, header=None)
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


def get_sample_signal() -> dict[str, Any]:
    """TASK 11.2 (Dashboard) -- returns the first real, already-windowed
    validation recording (same real windows `_validation_windows_and_labels`
    already computes for calibration -- no second CSV-reading code path, no
    synthetic/random data). Lets a caller run `POST /api/models/predict`
    against a genuine signal without needing its own dataset file access."""
    windows, labels = _validation_windows_and_labels()
    window = windows[0]
    return {
        "recording_id": window.recording_id,
        "label": labels[0],
        "channel": CHANNEL,
        "sampling_rate": SAMPLING_RATE_HZ,
        "signal": list(window.values),
    }


def _validation_features_and_labels() -> tuple[pd.DataFrame, list[str]]:
    global _VALIDATION_FEATURES, _VALIDATION_LABELS
    if _VALIDATION_FEATURES is None:
        windows, labels = _validation_windows_and_labels()
        _VALIDATION_FEATURES = extract_feature_matrix(
            windows, SAMPLING_RATE_HZ, nperseg=NPERSEG, noverlap=NOVERLAP
        )
        _VALIDATION_LABELS = labels
    return _VALIDATION_FEATURES, _VALIDATION_LABELS


def _isolation_forest_calibration() -> ScoringCalibration:
    global _ISOLATION_FOREST_CALIBRATION
    if _ISOLATION_FOREST_CALIBRATION is None:
        features, labels = _validation_features_and_labels()
        raw_scores = isolation_forest_raw_predict(_get_isolation_forest_model(), _get_scaler(), features)
        anomaly_scores = to_anomaly_score(raw_scores)
        _ISOLATION_FOREST_CALIBRATION = calibrate(
            anomaly_scores,
            threshold_method=THRESHOLD_METHOD,
            percentile_value=PERCENTILE_VALUE,
            validation_labels=labels,
        )
    return _ISOLATION_FOREST_CALIBRATION


def _autoencoder_calibration() -> ScoringCalibration:
    global _AUTOENCODER_CALIBRATION
    if _AUTOENCODER_CALIBRATION is None:
        features, labels = _validation_features_and_labels()
        errors = autoencoder_reconstruction_error(_get_autoencoder_model(), features)
        _AUTOENCODER_CALIBRATION = calibrate(
            errors,
            threshold_method=THRESHOLD_METHOD,
            percentile_value=PERCENTILE_VALUE,
            validation_labels=labels,
        )
    return _AUTOENCODER_CALIBRATION


def _normal_feature_baseline() -> dict[str, tuple[float, float]]:
    """Real per-feature (mean, std) computed from the real normal-labeled
    validation windows -- see TASK 12.1's `app.services.explanation_service`
    for the actual calculation (moved there, not duplicated); this stays the
    single process-lifetime cache both the free-text `explanation` and the
    structured `explanations` share."""
    global _NORMAL_FEATURE_BASELINE
    if _NORMAL_FEATURE_BASELINE is None:
        features, labels = _validation_features_and_labels()
        _NORMAL_FEATURE_BASELINE = explanation_service.compute_feature_baseline(features, labels)
    return _NORMAL_FEATURE_BASELINE


def _status_for_score(score: float, threshold: float) -> PredictionStatus:
    if score >= threshold:
        return PredictionStatus.ANOMALY
    if score >= threshold * WARNING_THRESHOLD_RATIO:
        return PredictionStatus.WARNING
    return PredictionStatus.NORMAL


def _build_explanation(deviations: list[FeatureDeviation]) -> str:
    """The free-text reading of the SAME real deviations `run_prediction`
    also returns structured (TASK 12.1) -- one computation, two
    presentations, never two competing baseline/significance calculations."""
    if not deviations:
        return (
            "No individual feature deviates by more than "
            f"{explanation_service.SIGNIFICANT_DEVIATION_Z_SCORE:.0f} standard deviations from the real normal-"
            "validation baseline; the anomaly score reflects the model's overall "
            "multivariate assessment rather than any single feature."
        )

    parts = []
    for item in deviations:
        if item.deviation_percent is None:
            parts.append(f"{item.feature}={item.current_value:.4g} ({item.direction} a zero real normal-validation baseline)")
        else:
            parts.append(
                f"{item.feature}={item.current_value:.4g} ({item.deviation_percent:+.1f}% {item.direction} "
                "the real normal-validation baseline)"
            )
    return "Notable feature deviations from the real normal-validation baseline: " + "; ".join(parts) + "."


def run_prediction(request: PredictRequest) -> PredictResponse:
    """`POST /api/models/predict`: real feature extraction -> real scaling
    (Isolation Forest) -> real model inference -> real scoring/normalization ->
    threshold-derived status -> real-feature-based explanation (TASK 10.5
    free-text + TASK 12.1 structured deviations, from the same computation)."""
    features = extract_features(
        request.signal, request.sampling_rate, nperseg=NPERSEG, noverlap=NOVERLAP
    )
    features_df = pd.DataFrame([features])

    if request.model_type == ModelType.ISOLATION_FOREST:
        raw_scores = isolation_forest_raw_predict(_get_isolation_forest_model(), _get_scaler(), features_df)
        anomaly_scores = to_anomaly_score(raw_scores)
        calibration = _isolation_forest_calibration()
    else:
        anomaly_scores = autoencoder_reconstruction_error(_get_autoencoder_model(), features_df)
        calibration = _autoencoder_calibration()

    normalized_scores = normalize_scores(anomaly_scores, calibration.normalization)
    score = float(normalized_scores[0])

    status = _status_for_score(score, calibration.threshold)
    deviations = explanation_service.explain(features, _normal_feature_baseline())
    explanation = _build_explanation(deviations)

    return PredictResponse(
        anomaly_score=score,
        status=status,
        explanation=explanation,
        explanations=[
            FeatureExplanation(
                feature=item.feature,
                current_value=item.current_value,
                baseline_value=item.baseline_value,
                deviation=item.deviation,
                deviation_percent=item.deviation_percent,
                direction=item.direction,
                std_deviations=item.std_deviations,
            )
            for item in deviations
        ],
    )
