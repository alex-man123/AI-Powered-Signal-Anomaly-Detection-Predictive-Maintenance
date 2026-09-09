"""TASK 6.5 — Model Artifact Contract: one common metadata schema for every saved
model (Isolation Forest now, Autoencoder later), so any prediction or audit can be
traced exactly to the configuration that produced the model, without assumptions.

Every model artifact (`.pkl` for Isolation Forest, `.pt` for Autoencoder -- Phase 7)
is accompanied by a JSON sidecar with the SAME basename
(`isolation_forest_v1.pkl` <-> `isolation_forest_v1.json`), containing exactly the
14 fields the backlog's own schema requires -- `ModelArtifactMetadata` below is that
ONE schema, shared by both model types; there is no separate
`IsolationForestMetadata`/`AutoencoderMetadata`.

HASH PROVENANCE (dataset_hash / split_manifest_hash) -- CRITICAL, real, never
invented:

- `split_manifest_hash`: SHA-256 of the real, current content of
  `data/processed/split_manifest.json` (TASK 1.5.7's own artifact) -- answers
  "which split configuration was used".
- `dataset_hash`: SHA-256 of the real, current content of
  `docs/dataset_audit/file_inventory.csv` (TASK 1.5.1's own artifact: the audited
  listing of all 880 real MAFAULDA files) -- answers "which dataset was used".
  This is a DELIBERATE choice over hashing the raw ~14 GB dataset directly: the
  inventory is the project's own already-existing, already-committed fingerprint
  of exactly which files constitute "the dataset" (its content changes if and
  only if that file listing changes), and hashing it costs milliseconds instead
  of hashing 14 GB on every model save. `compute_sha256()` itself is written to
  read any file incrementally/chunked regardless -- so hashing the raw dataset
  directly remains possible for a caller who explicitly asks for it (pass a
  different `path`), this module just does not do it by default.

Both hashes use the SAME generic `compute_sha256()` helper -- there is only one
hashing implementation in this module, applied to two different real files.
Format is always `sha256:<64 lowercase hex characters>` (validated by
`ModelArtifactMetadata` itself, not just produced correctly by convention).

SCORE DIRECTION -- CRITICAL, not hard-coded per model type: Isolation Forest's own
native `decision_function` follows sklearn's convention (lower = more anomalous),
but TASK 6.3's `app.ml.scoring` already flips this (`to_anomaly_score`) so the
PIPELINE's actual convention is `higher_is_more_anomalous` (blueprint.md's target
`0=normal/1=anomalous` scale) -- metadata must record the convention the pipeline
ACTUALLY consumes, not sklearn's raw internal one. A future Autoencoder's
reconstruction error is naturally `higher_is_more_anomalous` too (larger
reconstruction error = more anomalous, with no sign flip needed) -- but
`ScoreDirection` also defines `lower_is_more_anomalous` for a hypothetical future
model with the opposite native convention, so the schema does not silently assume
every future model shares Isolation Forest's (already-flipped) direction.

`threshold_value` is the ACTUAL calibrated threshold from
`app.ml.scoring.ScoringCalibration.threshold` -- never `percentile_value` (a
*parameter* to one calibration method, not the calibrated output itself). These
are explicitly different numbers and are never confused in this module.

`threshold_method` is validated against TASK 6.3's own
`app.ml.scoring.VALID_THRESHOLD_METHODS` -- not a separately invented list, so a
new method added there is automatically accepted here too.

`training_split` accepts only the literal `"train"` -- this project's methodology
(blueprint.md lines 39-41) trains exclusively on train's normal-labeled windows;
metadata must never claim a model was trained on `"validation"`/`"test"`.

TASK 7.3 update: `save_model_artifact`/`load_model_artifact` now also support
`model_type="autoencoder"`, dispatching to TASK 7.3's own
`app.ml.inference.save_autoencoder`/`load_autoencoder` (a `torch.save`-based
mechanism, distinct from Isolation Forest's joblib one -- see that module's
docstring) -- this was the one extension point TASK 6.5 explicitly deferred
("Autoencoder persistence is TASK 7.3's scope"), not a new architectural decision.
The metadata schema itself required no change at all: it already fully supported
`model_type="autoencoder"` (TASK 6.5's own tests already covered this), so this
update is purely about which serializer `save_model_artifact`/`load_model_artifact`
dispatch to based on `metadata.model_type` -- still exactly one metadata schema,
never a second `AutoencoderMetadata`.
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.ml.inference import load_autoencoder as _load_autoencoder_model
from app.ml.inference import load_model as _load_isolation_forest_model
from app.ml.inference import save_autoencoder as _save_autoencoder_model
from app.ml.inference import save_model as _save_isolation_forest_model
from app.ml.scoring import VALID_THRESHOLD_METHODS

_SHA256_HEX_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

# Repo-root-relative real artifact paths (backend/app/ml/model_artifact.py ->
# parents[3] == repo root), matching TASK 1.5.7's own split_manifest.json and TASK
# 1.5.1's own file_inventory.csv -- neither newly invented here.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLIT_MANIFEST_PATH = _REPO_ROOT / "data" / "processed" / "split_manifest.json"
DEFAULT_DATASET_FINGERPRINT_PATH = _REPO_ROOT / "docs" / "dataset_audit" / "file_inventory.csv"


class ModelArtifactError(ValueError):
    """Raised for invalid model artifact input: a missing model or sidecar file, a
    sidecar that fails validation (missing/malformed fields), or a `model_type`
    this module does not yet know how to serialize/deserialize. Never silently
    corrected, never falls back to a default/partial metadata object."""


class ModelType(str, Enum):
    ISOLATION_FOREST = "isolation_forest"
    AUTOENCODER = "autoencoder"


class ScoreDirection(str, Enum):
    HIGHER_IS_MORE_ANOMALOUS = "higher_is_more_anomalous"
    LOWER_IS_MORE_ANOMALOUS = "lower_is_more_anomalous"


class ModelArtifactMetadata(BaseModel):
    """The one common metadata schema for every model artifact (Isolation Forest
    now, Autoencoder later) -- see module docstring for the rationale behind each
    field's validation rule."""

    model_type: ModelType
    model_version: str = Field(min_length=1)
    feature_names: list[str] = Field(min_length=1)
    feature_dimension: int = Field(gt=0)
    scaler_artifact: str = Field(min_length=1)
    threshold_method: str
    threshold_value: float
    score_direction: ScoreDirection
    training_split: Literal["train"]
    dataset_hash: str
    split_manifest_hash: str
    random_seed: int
    created_at: str

    @field_validator("feature_names")
    @classmethod
    def _feature_names_must_be_non_empty_strings(cls, value: list[str]) -> list[str]:
        if any(not name.strip() for name in value):
            raise ValueError("feature_names must not contain empty/blank strings")
        return value

    @field_validator("threshold_method")
    @classmethod
    def _threshold_method_must_be_known(cls, value: str) -> str:
        if value not in VALID_THRESHOLD_METHODS:
            raise ValueError(
                f"threshold_method {value!r} is not one of {VALID_THRESHOLD_METHODS} "
                "(app.ml.scoring.VALID_THRESHOLD_METHODS)"
            )
        return value

    @field_validator("threshold_value")
    @classmethod
    def _threshold_value_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError(f"threshold_value must be finite, got {value}")
        return value

    @field_validator("dataset_hash", "split_manifest_hash")
    @classmethod
    def _hash_must_match_sha256_format(cls, value: str) -> str:
        if not _SHA256_HEX_PATTERN.match(value):
            raise ValueError(
                f"hash {value!r} does not match the required format 'sha256:<64 lowercase hex characters>'"
            )
        return value

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_iso8601(cls, value: str) -> str:
        try:
            datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"created_at {value!r} is not a valid ISO8601 timestamp: {exc}") from exc
        return value

    @model_validator(mode="after")
    def _feature_dimension_must_match_feature_names(self) -> "ModelArtifactMetadata":
        if self.feature_dimension != len(self.feature_names):
            raise ValueError(
                f"feature_dimension ({self.feature_dimension}) must equal "
                f"len(feature_names) ({len(self.feature_names)}) -- not auto-corrected"
            )
        return self


def compute_sha256(path: str | Path) -> str:
    """Computes `sha256:<64 hex chars>` over the REAL content of the file at
    `path`, read incrementally in fixed-size chunks (never loads the whole file
    into memory at once, regardless of file size)."""
    resolved_path = Path(path)
    hasher = hashlib.sha256()
    with open(resolved_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def compute_dataset_hash(path: str | Path = DEFAULT_DATASET_FINGERPRINT_PATH) -> str:
    """SHA-256 of the real dataset fingerprint file (TASK 1.5.1's
    `file_inventory.csv` by default) -- see module docstring for why this file,
    not the raw ~14 GB dataset, is hashed by default."""
    return compute_sha256(path)


def compute_split_manifest_hash(path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH) -> str:
    """SHA-256 of the real, current content of `split_manifest.json`."""
    return compute_sha256(path)


def _write_json_atomically(path: Path, content: str) -> None:
    """Writes `content` to a temp file in the same directory, then atomically
    renames it onto `path` -- avoids ever leaving a half-written/truncated sidecar
    behind if the process is interrupted mid-write (section 29's "avoid partial
    artifacts", without a transactional filesystem framework)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def save_model_artifact(
    model: object, metadata: ModelArtifactMetadata, path: str | Path
) -> tuple[Path, Path]:
    """Saves `model` and writes its JSON metadata sidecar (same basename, `.json`
    extension) -- the model file is written FIRST, and the sidecar only after that
    succeeds, so a sidecar never exists claiming a model that failed to save.

    Args:
        model: the trained model object -- an `IsolationForest` if
            `metadata.model_type == ModelType.ISOLATION_FOREST` (reuses TASK 6.4's
            `app.ml.inference.save_model`, joblib), or an `Autoencoder` if
            `metadata.model_type == ModelType.AUTOENCODER` (reuses TASK 7.3's
            `app.ml.inference.save_autoencoder`, `torch.save`) -- see module
            docstring.
        metadata: an already-validated `ModelArtifactMetadata` (Pydantic validates
            at construction time, so an invalid metadata object cannot exist to be
            passed here in the first place).
        path: the model artifact path, e.g. `models/isolation_forest_v1.pkl` or
            `models/autoencoder_v1.pt`.

    Returns:
        `(model_path, sidecar_path)`.
    """
    model_path = Path(path)
    sidecar_path = model_path.with_suffix(".json")

    if metadata.model_type == ModelType.ISOLATION_FOREST:
        _save_isolation_forest_model(model, model_path)
    else:
        _save_autoencoder_model(model, model_path)

    _write_json_atomically(sidecar_path, metadata.model_dump_json(indent=2))

    return model_path, sidecar_path


def load_model_artifact(path: str | Path) -> tuple[object, ModelArtifactMetadata]:
    """Loads a model together with its metadata sidecar. NEVER returns a model
    without validated metadata, and never fills in missing metadata fields with
    defaults -- traceability is the entire point of this contract.

    Args:
        path: the model artifact path (same path `save_model_artifact` was given).

    Returns:
        `(model, metadata)`.

    Raises:
        ModelArtifactError: if the model file is missing; if the sidecar is
            missing (even when the model file itself exists); or if the sidecar
            fails to parse as JSON or fails `ModelArtifactMetadata` validation
            (missing/malformed/inconsistent fields).
    """
    model_path = Path(path)
    sidecar_path = model_path.with_suffix(".json")

    if not model_path.exists():
        raise ModelArtifactError(f"Model artifact not found: {model_path}")
    if not sidecar_path.exists():
        raise ModelArtifactError(
            f"Model artifact exists at {model_path} but its metadata sidecar is "
            f"missing: {sidecar_path} -- refusing to load a model without its "
            "provenance contract"
        )

    try:
        metadata = ModelArtifactMetadata.model_validate_json(
            sidecar_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ModelArtifactError(
            f"Invalid or incomplete metadata sidecar at {sidecar_path}: {exc}"
        ) from exc

    if metadata.model_type == ModelType.ISOLATION_FOREST:
        model = _load_isolation_forest_model(model_path)
    else:
        model = _load_autoencoder_model(model_path)

    return model, metadata
