"""TASK 6.4 — model persistence (save/load) + inference entry point.

Responsibility, precisely scoped: given a TRAINED Isolation Forest (TASK 6.2) and
its TRAIN-FITTED scaler (TASK 5.4), persist them to disk and reload them such that
inference on the reloaded objects is identical to inference on the originals --
nothing else. This module trains nothing, fits nothing, calibrates nothing.

    trained model + train-fitted scaler
            |
            v
       save_model() / save_scaler()      <- thin wrappers, see below
            |
            v
    models/isolation_forest_v1.pkl + models/scaler_v1.pkl
            |
            v
       load_model() / load_scaler()      <- explicit errors, never a silent None
            |
            v
    reloaded model + reloaded scaler
            |
            v
       predict()                         <- reuses TASK 6.2's score() unchanged
            |
            v
    raw decision_function scores (identical to pre-save scores -- AC1)

Reuse, not reimplementation:
- `save_model`/`load_model` here are NOT a second model-serialization mechanism --
  they call TASK 6.2's own `app.ml.isolation_forest.save_model`/`load_model`
  (joblib) directly. This module adds exactly one thing those lower-level
  functions deliberately don't: explicit, clearly-typed errors for a missing or
  corrupted artifact (`ModelPersistenceError`), and a refusal to hand back an
  object of the wrong type -- appropriate at this higher-level "production
  inference" entry point, without changing TASK 6.2's own lower-level contract.
- `save_scaler`/`load_scaler` here are the same kind of thin, error-wrapping
  wrapper around TASK 5.4's own `app.ml.scaling.save_scaler`/`load_scaler`.
- `predict()` is a documented alias for TASK 6.2's own `app.ml.isolation_forest.
  score` (which itself already calls TASK 5.4's `apply_scaler` -- transform only,
  never refit -- before `IsolationForest.decision_function`). No scaling logic is
  duplicated here.
- Normalization/threshold calibration (TASK 6.3's `app.ml.scoring`) are
  DELIBERATELY NOT wrapped or re-exported here: this module's own contract stops at
  raw decision_function scores, matching `isolation_forest.score()`'s own contract.
  A caller who wants normalized/classified scores calls `app.ml.scoring`'s
  functions directly on this module's output -- folding that in here would make
  `inference.py` implicitly depend on one specific, already-calibrated
  `ScoringCalibration`, which is exactly the "duplicate scoring logic elsewhere"
  this task explicitly forbids. Wiring model + scaler + calibration into a single
  combined artifact is TASK 6.5's Model Artifact Contract, not implemented here.

Default artifact location: `Settings().model_dir` (this project's own existing
config field for exactly this purpose -- see `.env.example`'s `MODEL_DIR`) when
configured; otherwise the repository's own top-level `models/` directory
(blueprint.md line 160: "models/ (fișiere .pkl / .pt salvate pe disc)"), which
already exists (`models/.gitkeep`) and is already `.gitignore`d for `*.pkl`/`*.pt`
-- neither path is a newly invented convention. Resolved at call time (not frozen
into a module-level constant at import time), so it reflects whatever `Settings()`
currently returns and remains independent of the process's working directory.

Trust boundary: `load_model`/`load_scaler` deserialize via `joblib` (itself backed
by `pickle`), which executes arbitrary code for a maliciously crafted file. This
module never loads a path it did not receive explicitly from its caller -- there is
no automatic scanning of a directory for "any .pkl found", no accepting a path from
an untrusted request body here (that boundary belongs to a future API task, which
must itself decide what paths it is willing to pass in). Callers of this module are
responsible for only ever pointing it at artifacts this application itself produced
via `save_model`/`save_scaler`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.core.config import get_settings
from app.ml.isolation_forest import DEFAULT_MODEL_FILENAME
from app.ml.isolation_forest import load_model as _load_isolation_forest_model
from app.ml.isolation_forest import save_model as _save_isolation_forest_model
from app.ml.isolation_forest import score as _score_with_isolation_forest
from app.ml.scaling import DEFAULT_SCALER_FILENAME
from app.ml.scaling import load_scaler as _load_scaler_artifact
from app.ml.scaling import save_scaler as _save_scaler_artifact


class ModelPersistenceError(ValueError):
    """Raised for invalid model/scaler persistence: a missing artifact path, or an
    artifact that deserializes to something other than the expected type (a
    corrupted or wrong-kind file). Never silently returns `None` or swallows the
    underlying exception -- the original error is always preserved as `__cause__`.
    """


def default_models_dir() -> Path:
    """Resolves the artifact directory: `Settings().model_dir` if configured,
    else this repository's own top-level `models/` directory. Computed fresh on
    each call (not cached into a constant) so it always reflects the current
    `Settings()` and is independent of the caller's working directory."""
    configured = get_settings().model_dir
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3] / "models"


def default_model_path() -> Path:
    """Default path for the trained Isolation Forest artifact -- exactly
    `models/isolation_forest_v1.pkl` under `default_models_dir()`, matching this
    task's own required filename (`app.ml.isolation_forest.DEFAULT_MODEL_FILENAME`,
    unchanged)."""
    return default_models_dir() / DEFAULT_MODEL_FILENAME


def default_scaler_path() -> Path:
    """Default path for the train-fitted scaler artifact -- `scaler_v1.pkl` under
    `default_models_dir()`, matching TASK 5.4's own
    `app.ml.scaling.DEFAULT_SCALER_FILENAME`, unchanged."""
    return default_models_dir() / DEFAULT_SCALER_FILENAME


def save_model(model: IsolationForest, path: str | Path | None = None) -> Path:
    """Persists a trained Isolation Forest. Thin wrapper around TASK 6.2's own
    `app.ml.isolation_forest.save_model` (joblib) -- the serialization mechanism
    itself is not reimplemented here.

    Args:
        model: an already-trained `IsolationForest` (e.g. from
            `app.ml.isolation_forest.train_isolation_forest`) -- never fit here.
        path: where to write the artifact; defaults to `default_model_path()`.

    Returns:
        The resolved `Path` the artifact was written to.
    """
    resolved_path = Path(path) if path is not None else default_model_path()
    return _save_isolation_forest_model(model, resolved_path)


def load_model(path: str | Path | None = None) -> IsolationForest:
    """Deserializes an already-trained Isolation Forest. NEVER creates a new model
    or calls `.fit()` -- this is deserialization only.

    Args:
        path: artifact to load; defaults to `default_model_path()`.

    Returns:
        The deserialized `IsolationForest`, ready for `predict()`/
        `decision_function()` -- identical to the model that was saved (AC1).

    Raises:
        ModelPersistenceError: if `path` does not exist, fails to deserialize, or
            deserializes into something other than an `IsolationForest`. The
            original exception (if any) is preserved as `__cause__` -- never
            silently swallowed or replaced with `None`.
    """
    resolved_path = Path(path) if path is not None else default_model_path()

    if not resolved_path.exists():
        raise ModelPersistenceError(f"Model artifact not found: {resolved_path}")

    try:
        model = _load_isolation_forest_model(resolved_path)
    except Exception as exc:
        raise ModelPersistenceError(
            f"Failed to deserialize model artifact at {resolved_path}: {exc}"
        ) from exc

    if not isinstance(model, IsolationForest):
        raise ModelPersistenceError(
            f"Artifact at {resolved_path} did not deserialize into an IsolationForest "
            f"(got {type(model).__name__} instead) -- refusing to use it for inference"
        )

    return model


def save_scaler(scaler: StandardScaler, path: str | Path | None = None) -> Path:
    """Persists a train-fitted scaler. Thin wrapper around TASK 5.4's own
    `app.ml.scaling.save_scaler` -- not a second scaler-serialization mechanism.

    Args:
        scaler: an already-fitted `StandardScaler` (e.g. from
            `app.ml.isolation_forest.train_isolation_forest`'s second return
            value) -- never (re)fit here.
        path: where to write the artifact; defaults to `default_scaler_path()`.

    Returns:
        The resolved `Path` the artifact was written to.
    """
    resolved_path = Path(path) if path is not None else default_scaler_path()
    return _save_scaler_artifact(scaler, resolved_path)


def load_scaler(path: str | Path | None = None) -> StandardScaler:
    """Deserializes an already-fitted scaler. NEVER fits/refits it here.

    Args:
        path: artifact to load; defaults to `default_scaler_path()`.

    Returns:
        The deserialized `StandardScaler`, ready to be passed to `predict()` --
        identical to the scaler that was saved.

    Raises:
        ModelPersistenceError: if `path` does not exist, fails to deserialize, or
            deserializes into something other than a `StandardScaler`. The
            original exception (if any) is preserved as `__cause__`.
    """
    resolved_path = Path(path) if path is not None else default_scaler_path()

    if not resolved_path.exists():
        raise ModelPersistenceError(f"Scaler artifact not found: {resolved_path}")

    try:
        scaler = _load_scaler_artifact(resolved_path)
    except Exception as exc:
        raise ModelPersistenceError(
            f"Failed to deserialize scaler artifact at {resolved_path}: {exc}"
        ) from exc

    if not isinstance(scaler, StandardScaler):
        raise ModelPersistenceError(
            f"Artifact at {resolved_path} did not deserialize into a StandardScaler "
            f"(got {type(scaler).__name__} instead) -- refusing to use it for inference"
        )

    return scaler


def predict(model: IsolationForest, scaler: StandardScaler, features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Raw Isolation Forest `decision_function` scores for `features`.

    A documented alias for TASK 6.2's own `app.ml.isolation_forest.score` --
    `features` is transformed through `scaler` (transform only, via TASK 5.4's
    `apply_scaler` -- never refit) before `model.decision_function(...)`. No
    scaling or scoring logic is duplicated in this function's body.

    Args:
        model: a trained (or `load_model`-reloaded) `IsolationForest`.
        scaler: its matching train-fitted (or `load_scaler`-reloaded) scaler.
        features: one or more windows' feature rows, same column contract as
            `scaler` was fit on.

    Returns:
        A 1D `numpy.ndarray` of length `len(features)`.

    Raises:
        app.ml.scaling.ScalingError: propagated unchanged from `apply_scaler` for
            a feature-count/name mismatch -- not re-wrapped, since TASK 5.4
            already defines this failure mode clearly.
    """
    return _score_with_isolation_forest(model, scaler, features)
