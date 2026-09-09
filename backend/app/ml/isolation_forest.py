"""TASK 6.2 — Isolation Forest baseline training, exclusively on normal-labeled train
windows.

Methodology (blueprint.md line 39-41, "Formulare metodologică per secțiunea 5"):
"Antrenare unsupervised/semi-supervised (modelul învață doar din date etichetate
'normal'), evaluare supervizată (folosim label-urile reale ale dataset-ului ca ground
truth la testare)." This module implements exactly the training half of that
methodology -- evaluation against real labels is a later task's scope (TASK 6.3/6.4).

Pipeline (this task's own required flow -- filter BEFORE fitting the scaler, not
after):

    train feature matrix (TASK 5.3) + train labels (per-row metadata, NOT a feature
    column)
        -> select_normal_samples(): keep only rows whose label == "normal"
        -> app.ml.scaling.fit_scaler() (TASK 5.4) fit on THIS normal-only subset
           (not on all of train -- fitting on anomalous rows would leak their
           distribution into what should be a purely "normal" reference frame)
        -> app.ml.scaling.apply_scaler() on that same subset
        -> sklearn.ensemble.IsolationForest.fit()

Reuses TASK 5.4's `scaling.py` directly (`fit_scaler`/`apply_scaler`) -- there is no
second scaler implementation here, and this module never fits/mutates a scaler on
anything but the normal-only training subset. New data (validation/test, for
`decision_function` scoring) is transformed with `apply_scaler` using that SAME
train-fitted scaler -- never refit.

Labels are metadata used only to select which rows to train on. They are never
appended to the feature matrix as a numeric column -- `IsolationForest.fit()` only
ever sees the (scaled) DSP feature columns from TASK 5.3/5.4, never a label-derived
value.

Out of scope for this task (later phases, not implemented here): anomaly score
normalization/thresholding (TASK 6.3), evaluation against real labels (TASK 6.4), the
shared Model Artifact Contract (TASK 6.5) -- this module only guarantees that a
trained model + its train-fitted scaler can already be saved and reloaded correctly
(via the same `joblib` pattern TASK 5.4 already established for `scaler_v1.pkl`), so
TASK 6.5 has something concrete to wire a `scaler_artifact`/model artifact reference
onto later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.ml.scaling import apply_scaler, fit_scaler
from app.models.signal import SignalLabel

DEFAULT_MODEL_FILENAME = "isolation_forest_v1.pkl"

# Matches the seed already used throughout this project (data/processed/
# split_manifest.json's own split seed, and the fixed seeds used across this
# codebase's own tests) -- not a newly-invented value.
DEFAULT_RANDOM_STATE = 42


class IsolationForestTrainingError(ValueError):
    """Raised for invalid Isolation Forest training input: a features/labels length
    mismatch, or zero normal-labeled samples available to train on. Never silently
    corrected -- there is no fallback that trains on a different label set or an
    empty matrix."""


def select_normal_samples(
    features: pd.DataFrame | np.ndarray, labels: Sequence[str]
) -> pd.DataFrame | np.ndarray:
    """Filters `features` down to exactly the rows labeled `"normal"`
    (`app.models.signal.SignalLabel.NORMAL.value`), preserving row order and (for a
    DataFrame) original index.

    Args:
        features: a feature matrix (e.g. TASK 5.3's `extract_feature_matrix`
            output), one row per window.
        labels: one label string per row of `features`, aligned positionally (e.g.
            built by the caller alongside the same window list passed to
            `extract_feature_matrix` -- labels are never a column inside
            `features` itself).

    Returns:
        The subset of `features` whose corresponding label is `"normal"` -- same
        type (DataFrame stays a DataFrame with its original index; ndarray stays an
        ndarray) and column set as the input, fewer (or, in principle, the same
        number of) rows.

    Raises:
        IsolationForestTrainingError: if `len(features) != len(labels)`, or if no
            row's label is `"normal"` (Isolation Forest cannot be trained on zero
            samples).
    """
    if len(features) != len(labels):
        raise IsolationForestTrainingError(
            f"features has {len(features)} rows but labels has {len(labels)} entries -- "
            "they must align 1:1, exactly one label per row/window"
        )

    mask = np.array([label == SignalLabel.NORMAL.value for label in labels])

    if isinstance(features, pd.DataFrame):
        normal_features = features.loc[mask]
    else:
        normal_features = np.asarray(features)[mask]

    if len(normal_features) == 0:
        raise IsolationForestTrainingError(
            f"No samples labeled {SignalLabel.NORMAL.value!r} found among {len(labels)} "
            "provided labels -- Isolation Forest requires at least one normal training sample."
        )

    return normal_features


def train_isolation_forest(
    train_features: pd.DataFrame | np.ndarray,
    train_labels: Sequence[str],
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    **isolation_forest_kwargs,
) -> tuple[IsolationForest, StandardScaler]:
    """Trains an Isolation Forest EXCLUSIVELY on normal-labeled train windows.

    Args:
        train_features: the TRAIN split's feature matrix (TASK 5.3 output) --
            may still contain non-normal rows; this function does the filtering.
        train_labels: one label per row of `train_features`, aligned positionally
            (see `select_normal_samples`).
        random_state: explicit, deterministic seed forwarded to `IsolationForest`
            (defaults to 42, matching this project's own established seed
            convention). Fixing this and reusing the same `train_features`/
            `train_labels` produces bit-reproducible `decision_function` scores.
        **isolation_forest_kwargs: any other `sklearn.ensemble.IsolationForest`
            constructor argument this project's blueprint does not itself
            constrain (e.g. `n_estimators`, `contamination`) -- sklearn's own
            defaults apply for anything not passed here; none are invented by this
            function.

    Returns:
        `(model, scaler)`: the fitted `IsolationForest` AND the `StandardScaler`
        it was scored through (TASK 5.4's `fit_scaler`, fit on the normal-only
        subset of `train_features`). Both are needed to score new data later --
        `score()` below requires the exact same scaler, never a freshly-fit one.

    Raises:
        IsolationForestTrainingError: if `train_features`/`train_labels` lengths
            differ, or if no row is labeled `"normal"` -- see
            `select_normal_samples`.
    """
    normal_features = select_normal_samples(train_features, train_labels)

    scaler = fit_scaler(normal_features)
    scaled_normal_features = apply_scaler(scaler, normal_features)

    model = IsolationForest(random_state=random_state, **isolation_forest_kwargs)
    model.fit(scaled_normal_features)

    return model, scaler


def score(
    model: IsolationForest, scaler: StandardScaler, features: pd.DataFrame | np.ndarray
) -> np.ndarray:
    """Computes `IsolationForest.decision_function` scores for new data (validation,
    test, or any other window's features) -- higher is "more normal", lower/negative
    is "more anomalous" (sklearn's own convention, not redefined here).

    `features` is transformed with `scaler` via `apply_scaler` -- `.transform()`
    only, never `.fit()`/`.fit_transform()` -- the exact same scaler `
    train_isolation_forest` returned, fit on train's normal-only subset.

    Args:
        model: an `IsolationForest` returned by `train_isolation_forest` (or
            `load_model`).
        scaler: the `StandardScaler` returned alongside it (or `load_scaler`'d from
            the same artifact) -- must have been fit on data with the same number
            (and, for a DataFrame, the same names/order) of columns as `features`.
        features: one or more windows' feature rows to score -- same column
            contract as the matrix `scaler` was fit on.

    Returns:
        A 1D `numpy.ndarray` of length `len(features)`, one finite score per row.

    Raises:
        app.ml.scaling.ScalingError: if `features` has a different number of
            columns (or, for a DataFrame, different column names/order) than
            `scaler` was fit on -- propagated unchanged from `apply_scaler`, not
            re-wrapped, since TASK 5.4 already defines this failure mode clearly.
    """
    scaled_features = apply_scaler(scaler, features)
    return model.decision_function(scaled_features)


def save_model(model: IsolationForest, path: str | Path) -> Path:
    """Serializes a trained model to `path` via `joblib.dump` -- the same pattern
    TASK 5.4's `save_scaler` already established (joblib being scikit-learn's own
    documented recommendation for persisting fitted estimators). Creates parent
    directories if needed. The scaler is saved SEPARATELY via
    `app.ml.scaling.save_scaler` -- this function never bundles the two into one
    artifact, per this task's own instruction to keep them as two distinct
    artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path) -> IsolationForest:
    """Deserializes a model previously written by `save_model` (via `joblib.load`).
    The reloaded model scores identically to the original for the same
    (identically-scaled) input."""
    return joblib.load(Path(path))
