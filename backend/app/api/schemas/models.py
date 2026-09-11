"""TASK 10.1 -- request/response schemas for TASK 10.5's model endpoints
(`GET /api/models`, `GET /api/models/{id}/performance`, `POST /api/models/predict`).

`ModelType` is imported directly from `app.ml.model_artifact` (TASK 6.5/7.3) --
never re-declared -- since it is already the real, established identifier for
"which trained model" ("isolation_forest"/"autoencoder"). No separate numeric
model id is invented: Phase 6-8 never created a `models` database table (only
`experiment_runs`, TASK 8.3), so `ModelType`'s own two real values are used
directly as the identifier `GET /api/models/{id}` would receive.

`ModelMetrics` mirrors TASK 8.1's `evaluate()` return dict field-for-field
(`precision, recall, f1, roc_auc, pr_auc, confusion_matrix, fpr, fnr,
inference_time`) -- reused, not reinvented. Per that module's own documented
contract, `roc_auc`/`pr_auc`/`fpr`/`fnr` may legitimately be `float("nan")` for
an edge-case test set, so none of these fields carry a `ge=0, le=1` bound (a
bound that would incorrectly reject a real, valid NaN result).

`PredictionStatus` (`NORMAL`/`WARNING`/`ANOMALY`) mirrors blueprint.md section 30
("verde/galben/roșu pentru NORMAL/WARNING/ANOMALY") exactly. Per
`app.ml.scoring`'s own module docstring, the threshold logic that assigns this
3-tier status is explicitly NOT implemented anywhere yet (Phase 6 only
calibrates a single binary threshold) -- this schema defines the response SHAPE
only; TASK 10.5's service layer is responsible for actually computing `status`.

Updated by TASK 10.5 (minimal, justified extension -- not a redesign):
`ModelArtifactInfo` was added, mirroring TASK 6.5's real
`app.ml.model_artifact.ModelArtifactMetadata` field-for-field EXCEPT
`scaler_artifact` (a local artifact file path -- exactly the kind of internal
detail this schema layer already avoids exposing, per TASK 10.1's own design
note) and `model_type` (already present one level up on `ModelResponse`, so
not duplicated here). `GET /api/models`'s AC1 ("informațiile relevante din
Model Artifact Contract") needs this real metadata; `ModelResponse` did not
carry it before this task.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.ml.model_artifact import ModelType, ScoreDirection
from app.models.signal import SignalLabel


class ModelMetrics(BaseModel):
    """Mirrors `app.ml.evaluation.evaluate()`'s return dict exactly."""

    precision: float = Field(strict=True)
    recall: float = Field(strict=True)
    f1: float = Field(strict=True)
    roc_auc: float = Field(strict=True)
    pr_auc: float = Field(strict=True)
    confusion_matrix: list[list[int]] = Field(min_length=2, max_length=2)
    fpr: float = Field(strict=True)
    fnr: float = Field(strict=True)
    inference_time: float = Field(ge=0, strict=True)


class ModelArtifactInfo(BaseModel):
    """Mirrors `app.ml.model_artifact.ModelArtifactMetadata` (TASK 6.5/7.3) --
    everything except `scaler_artifact` (an internal file path) and
    `model_type` (already on the parent `ModelResponse`)."""

    model_config = ConfigDict(from_attributes=True)

    model_version: str = Field(min_length=1)
    feature_names: list[str] = Field(min_length=1)
    feature_dimension: int = Field(gt=0, strict=True)
    threshold_method: str
    threshold_value: float = Field(strict=True)
    score_direction: ScoreDirection
    training_split: str
    dataset_hash: str
    split_manifest_hash: str
    random_seed: int = Field(strict=True)
    created_at: str


class ModelResponse(BaseModel):
    """Shared shape for one row of `GET /api/models` and for
    `GET /api/models/{id}/performance` -- both need exactly the same real
    data (a model's identity + its Model Artifact Contract metadata + its real
    Phase 8 metrics), so this is intentionally ONE schema, not two
    near-duplicates."""

    model_config = ConfigDict(from_attributes=True)

    model_type: ModelType
    artifact: ModelArtifactInfo
    metrics: ModelMetrics


class PredictionStatus(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    ANOMALY = "ANOMALY"


class PredictRequest(BaseModel):
    model_type: ModelType
    signal: list[float] = Field(min_length=1)
    sampling_rate: float = Field(gt=0, strict=True)


class PredictResponse(BaseModel):
    anomaly_score: float = Field(ge=0, le=1, strict=True)
    status: PredictionStatus
    explanation: str = Field(min_length=1)


class SampleSignalResponse(BaseModel):
    """`GET /api/models/sample-signal` -- one real, already-windowed recording
    from the same real validation set TASK 10.5's own threshold calibration
    uses (`model_service._validation_windows_and_labels`), never synthetic/
    random data. Lets a client (e.g. the frontend Dashboard) demonstrate
    `POST /api/models/predict` against a genuine signal without needing its
    own dataset file access."""

    recording_id: str = Field(min_length=1)
    label: SignalLabel
    channel: int = Field(ge=0, strict=True)
    sampling_rate: float = Field(gt=0, strict=True)
    signal: list[float] = Field(min_length=1)
