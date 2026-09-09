"""TASK 6.4 — model persistence (save/load) + inference entry point.
TASK 7.3 extends this module with the equivalent Isolation-Forest-style
save/load/inference functions for the Autoencoder (TASK 7.1/7.2) -- see the
"Autoencoder (TASK 7.3)" section further down this docstring.

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

Autoencoder (TASK 7.3): `save_autoencoder`/`load_autoencoder` are the Autoencoder
equivalent of `save_model`/`load_model` above, and `reconstruct`/
`reconstruction_error` are the Autoencoder equivalent of `predict` -- same
`ModelPersistenceError` contract (missing artifact, corrupted artifact, wrong
type), same "never fit/train here" rule.

One real difference from Isolation Forest's joblib-based artifact: `joblib.dump`
pickles the WHOLE `IsolationForest` object (architecture and all), so
`joblib.load` alone is self-sufficient. A PyTorch `state_dict()` carries only
tensor VALUES, not the constructor arguments (`input_dim`/`hidden_dim`/
`bottleneck_dim`) needed to build a fresh `Autoencoder` before `load_state_dict`
can even be called -- so `save_autoencoder` saves those three integers alongside
the state dict in the same `.pt` file (via `torch.save` on a plain dict), and
`load_autoencoder` reads them back to reconstruct the exact same architecture.
This is NOT the same artifact as TASK 7.2's own `save_checkpoint`/
`load_checkpoint` (`app.ml.training`), which additionally carries optimizer state
+ epoch + best-validation-loss for mid-training resume -- a training-time
artifact, not this module's final "ready for inference" one.

Reconstruction error convention (`reconstruction_error`): per-sample MSE between
`x` and the model's reconstruction of it, `mean((x - x_hat)**2, axis=1)` -- HIGHER
error means MORE anomalous (the opposite of Isolation Forest's raw
`decision_function`, but exactly TASK 6.3/6.5's target pipeline convention,
`score_direction="higher_is_more_anomalous"`). Unlike `predict()` (Isolation
Forest), no sign flip is ever applied to a reconstruction error anywhere in this
module -- it is already in the pipeline's target orientation the moment it is
computed, so it can be handed directly to `app.ml.scoring.fit_score_normalizer`/
`normalize_scores`/`calibrate_threshold`/`classify` (TASK 6.3), none of which are
modified or duplicated here -- those functions never assumed Isolation Forest in
the first place; only `app.ml.scoring.compute_normalized_scores` is Isolation-
Forest-specific (it calls `to_anomaly_score`), which is why this module adds its
own trivial Autoencoder equivalent, `compute_autoencoder_normalized_scores`,
rather than reusing that one function.

`model.eval()` + `torch.no_grad()` for every inference call here (`reconstruct`,
`reconstruction_error`) -- never computes gradients, never mutates `model`'s
weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.core.config import get_settings
from app.ml.autoencoder import Autoencoder
from app.ml.isolation_forest import DEFAULT_MODEL_FILENAME
from app.ml.isolation_forest import load_model as _load_isolation_forest_model
from app.ml.isolation_forest import save_model as _save_isolation_forest_model
from app.ml.isolation_forest import score as _score_with_isolation_forest
from app.ml.scaling import DEFAULT_SCALER_FILENAME
from app.ml.scaling import load_scaler as _load_scaler_artifact
from app.ml.scaling import save_scaler as _save_scaler_artifact
from app.ml.scoring import ScoringCalibration, normalize_scores

DEFAULT_AUTOENCODER_FILENAME = "autoencoder_v1.pt"


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


def default_autoencoder_path() -> Path:
    """Default path for the trained Autoencoder artifact -- exactly
    `models/autoencoder_v1.pt` under `default_models_dir()`, matching this task's
    own required filename."""
    return default_models_dir() / DEFAULT_AUTOENCODER_FILENAME


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


# --- Autoencoder (TASK 7.3) ---


def save_autoencoder(model: Autoencoder, path: str | Path | None = None) -> Path:
    """Persists a trained Autoencoder's `state_dict` plus the architecture
    hyperparameters needed to reconstruct it (`input_dim`/`hidden_dim`/
    `bottleneck_dim` -- see module docstring for why these must travel with the
    state dict, unlike Isolation Forest's joblib artifact). NOT TASK 7.2's own
    `save_checkpoint` (a different, training-time artifact).

    Args:
        model: an already-trained `Autoencoder` (TASK 7.1/7.2) -- never fit here.
        path: where to write the artifact; defaults to `default_autoencoder_path()`.

    Returns:
        The resolved `Path` the artifact was written to.
    """
    resolved_path = Path(path) if path is not None else default_autoencoder_path()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "bottleneck_dim": model.bottleneck_dim,
        },
        resolved_path,
    )
    return resolved_path


def load_autoencoder(path: str | Path | None = None) -> Autoencoder:
    """Deserializes an Autoencoder previously written by `save_autoencoder`:
    reconstructs a fresh `Autoencoder` with the saved architecture hyperparameters,
    loads the saved `state_dict` into it, and sets it to `eval()` mode. NEVER
    trains/fits anything here.

    Args:
        path: artifact to load; defaults to `default_autoencoder_path()`.

    Returns:
        The deserialized `Autoencoder`, in `eval()` mode, ready for `reconstruct`/
        `reconstruction_error` -- identical to the model that was saved (AC3).

    Raises:
        ModelPersistenceError: if `path` does not exist, fails to deserialize, or
            is missing the expected keys (`state_dict`/`input_dim`/`hidden_dim`/
            `bottleneck_dim`) or a `state_dict` incompatible with them. The
            original exception (if any) is preserved as `__cause__`.
    """
    resolved_path = Path(path) if path is not None else default_autoencoder_path()

    if not resolved_path.exists():
        raise ModelPersistenceError(f"Autoencoder artifact not found: {resolved_path}")

    try:
        checkpoint = torch.load(resolved_path, map_location="cpu")
    except Exception as exc:
        raise ModelPersistenceError(
            f"Failed to deserialize Autoencoder artifact at {resolved_path}: {exc}"
        ) from exc

    try:
        model = Autoencoder(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            bottleneck_dim=checkpoint["bottleneck_dim"],
        )
        model.load_state_dict(checkpoint["state_dict"])
    except (KeyError, TypeError, RuntimeError) as exc:
        raise ModelPersistenceError(
            f"Artifact at {resolved_path} is missing required Autoencoder fields "
            f"or has an incompatible state_dict: {exc}"
        ) from exc

    model.eval()
    return model


def _validate_autoencoder_features(features: pd.DataFrame | np.ndarray, input_dim: int) -> torch.Tensor:
    array = np.asarray(features, dtype=np.float32)

    if array.ndim != 2:
        raise ModelPersistenceError(
            f"features must be 2-dimensional (n_samples, n_features), got shape {array.shape}"
        )
    if array.shape[0] == 0:
        raise ModelPersistenceError("features must have at least one row (one window/sample)")
    if not np.isfinite(array).all():
        raise ModelPersistenceError(
            "features contains NaN/Inf values -- refusing to silently mask them"
        )
    if array.shape[1] != input_dim:
        raise ModelPersistenceError(
            f"features has {array.shape[1]} columns, but model.input_dim is {input_dim}"
        )

    return torch.from_numpy(array)


def reconstruct(model: Autoencoder, features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Reconstructs `features` through `model` -- `model.eval()` + `torch.no_grad()`,
    no gradients computed, `model` never mutated.

    Args:
        model: a trained (or `load_autoencoder`-reloaded) `Autoencoder`.
        features: one or more windows' feature rows, shape `(N, model.input_dim)`.

    Returns:
        The reconstruction as a `numpy.ndarray`, shape `(N, model.input_dim)` --
        identical shape to `features`.

    Raises:
        ModelPersistenceError: if `features` is empty, not 2-dimensional, contains
            NaN/Inf, or has a different column count than `model.input_dim`.
    """
    tensor = _validate_autoencoder_features(features, model.input_dim)
    model.eval()
    with torch.no_grad():
        reconstruction = model(tensor)
    return reconstruction.numpy()


def reconstruction_error(model: Autoencoder, features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Per-sample reconstruction error: `mean((x - x_hat)**2, axis=1)` -- HIGHER
    means MORE anomalous (see module docstring's "Reconstruction error convention").

    Args:
        model: a trained (or `load_autoencoder`-reloaded) `Autoencoder`.
        features: one or more windows' feature rows, shape `(N, model.input_dim)`.

    Returns:
        A 1D `numpy.ndarray` of length `N` -- one finite error per row, never a
        single scalar for the whole batch.

    Raises:
        ModelPersistenceError: same conditions as `reconstruct`.
    """
    tensor = _validate_autoencoder_features(features, model.input_dim)
    model.eval()
    with torch.no_grad():
        reconstruction = model(tensor)
        errors = torch.mean((tensor - reconstruction) ** 2, dim=1)
    return errors.numpy()


def compute_autoencoder_normalized_scores(
    model: Autoencoder, features: pd.DataFrame | np.ndarray, calibration: ScoringCalibration
) -> np.ndarray:
    """End-to-end scoring of new data using an ALREADY-CALIBRATED
    `ScoringCalibration` (TASK 6.3) -- never refits the normalizer. Unlike
    Isolation Forest's `app.ml.scoring.compute_normalized_scores`, no
    `to_anomaly_score` sign flip is applied: `reconstruction_error` is already in
    the pipeline's target orientation (higher = more anomalous).

    Returns:
        Normalized `[0,1]` anomaly scores, one per row of `features`.
    """
    errors = reconstruction_error(model, features)
    return normalize_scores(errors, calibration.normalization)
