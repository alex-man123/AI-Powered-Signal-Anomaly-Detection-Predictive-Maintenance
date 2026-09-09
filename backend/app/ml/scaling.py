"""TASK 5.4 — feature scaling: common infrastructure shared by every model.

This is NOT TASK 3.1's raw-signal preprocessing (`app.signal_processing.
preprocessing`'s `fit_standardizer`/`transform_standardizer`, which standardizes one
RAW SIGNAL window before FFT/PSD/filtering). This module standardizes TASK 5.3's
output instead -- a FEATURE MATRIX (rows = windows, columns = named features, e.g. a
`pandas.DataFrame` from `app.features.extractor.extract_feature_matrix`). The two
operate at different pipeline stages, on different shaped data, and are never
combined into one component (per this task's own explicit instruction).

Architecture: `scaling.py` has no knowledge of, and no import from, any concrete
model (`isolation_forest.py`, `autoencoder.py`, `training.py`). The dependency
direction is strictly `models -> scaling`, never the reverse -- so this module can be
imported and used identically by Isolation Forest (TASK 6.2) and the Autoencoder
(TASK 7.2) without either depending on the other's code. As of this task, neither
`app/ml/isolation_forest.py` nor `app/ml/autoencoder.py`/`training.py` exist yet
(Phase 6/7 have not started) -- there is therefore no duplicated scaling logic in
those files to remove, and no artificial integration is added here to pre-empt that
future work. AC2 is satisfied structurally: exactly one scaling implementation exists
in the codebase, in the one place a future model module would import it from.

Fit/apply separation, mirroring TASK 3.1's own fit/transform convention (blueprint.md
section 8's leakage rule, restated here for feature vectors rather than raw signals):
`fit_scaler()` computes scaling parameters from EXACTLY one feature matrix (the
caller's train matrix) and returns a fitted `StandardScaler`; `apply_scaler()`
requires that fitted scaler as an explicit argument and only ever calls its
`.transform()` -- never `.fit()`, `.fit_transform()`, or `.partial_fit()`. There is
deliberately no `fit_apply`/`fit_transform` convenience function here, for the same
reason TASK 3.1 has none: a caller cannot accidentally re-fit on validation/test data
by reaching for the "wrong" one-call helper, because no such helper exists.

Uses `sklearn.preprocessing.StandardScaler` directly (already a project dependency)
-- the scaling algorithm itself is never reimplemented. Zero-variance/constant
features are handled entirely by StandardScaler's own well-defined behavior (verified
empirically: `scale_` is set to `1.0`, not `0.0`, for such a feature, so `transform`
never divides by zero) -- no custom workaround is introduced for this case.

Serialization uses `joblib` (already available transitively via scikit-learn, and
scikit-learn's own documented recommendation for persisting fitted estimators, since
it handles the NumPy arrays inside a fitted `StandardScaler` more efficiently than raw
`pickle`) via `save_scaler`/`load_scaler`. `DEFAULT_SCALER_FILENAME = "scaler_v1.pkl"`
is exported as the exact artifact name this task specifies; WHERE that file is
written (e.g. the project's existing top-level `models/` directory, or
`Settings().model_dir`, both established by earlier tasks but not yet wired to any
concrete artifact) is deliberately left to the caller -- this module has no
filesystem-path opinion of its own, keeping it independent of `app.core.config` and
directly unit-testable against a `tmp_path`. Wiring this into TASK 6.5's full Model
Artifact Contract (a `scaler_artifact` reference consumed by both models) is that
later task's responsibility; this task only guarantees the artifact itself
(`save_scaler`/`load_scaler`) already works correctly and losslessly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DEFAULT_SCALER_FILENAME = "scaler_v1.pkl"

FeatureMatrix = TypeVar("FeatureMatrix", pd.DataFrame, np.ndarray)


class ScalingError(ValueError):
    """Raised for invalid feature-scaling input (empty matrix, wrong dimensionality,
    non-finite values, or a feature-count/name mismatch between the matrix a scaler
    was fit on and the one it is asked to transform). Never silently corrected --
    scaling never reshapes, reorders, or drops features/rows to make a mismatched
    input "fit"."""


def _validate_matrix(features: FeatureMatrix) -> np.ndarray:
    array = np.asarray(features, dtype=float)

    if array.ndim != 2:
        raise ScalingError(
            f"features must be 2-dimensional (n_samples, n_features), got shape {array.shape}"
        )
    if array.shape[0] == 0:
        raise ScalingError("features must have at least one row (one window/sample)")
    if array.shape[1] == 0:
        raise ScalingError("features must have at least one column (one feature)")
    if not np.isfinite(array).all():
        raise ScalingError(
            "features contains NaN/Inf values -- scaling refuses to silently propagate "
            "or mask them (e.g. via nan_to_num); fix the upstream feature matrix instead"
        )

    return array


def fit_scaler(train_features: FeatureMatrix) -> StandardScaler:
    """Fits a `StandardScaler` EXCLUSIVELY on `train_features`.

    This function must never be called with validation/test data -- `apply_scaler`
    takes the resulting fitted scaler as an explicit argument specifically so that
    isn't possible by accident (the same anti-leakage pattern as TASK 3.1's
    `fit_normalizer`/`fit_standardizer`).

    Args:
        train_features: the TRAIN split's feature matrix only (e.g. TASK 5.3's
            `extract_feature_matrix` output for train windows) -- a
            `pandas.DataFrame` (column names are then remembered as
            `scaler.feature_names_in_` and checked again in `apply_scaler`) or a 2D
            `numpy.ndarray`, shape `(n_samples, n_features)`.

    Returns:
        A `StandardScaler` fitted on `train_features` -- `scaler.mean_`/
        `scaler.scale_`/`scaler.var_` reflect train's statistics only.

    Raises:
        ScalingError: if `train_features` is empty, not 2-dimensional, or contains
            NaN/Inf.
    """
    array = _validate_matrix(train_features)

    scaler = StandardScaler()
    if isinstance(train_features, pd.DataFrame):
        scaler.fit(train_features)
    else:
        scaler.fit(array)

    return scaler


def apply_scaler(scaler: StandardScaler, features: FeatureMatrix) -> FeatureMatrix:
    """Applies an ALREADY-FITTED scaler to `features` -- transform only, never fit.

    Semantically `apply_scaler = transform`, never `fit + transform`: this function
    calls `scaler.transform(...)` and nothing else. `scaler` itself is never
    mutated by this call -- its `mean_`/`scale_`/`var_` are unchanged before and
    after, regardless of what `features` contains (verified directly by this
    module's own tests).

    Args:
        scaler: a `StandardScaler` returned by `fit_scaler` (fitted on train only).
        features: the feature matrix to transform -- train, validation, test, or
            any other split; same shape convention as `fit_scaler`'s
            `train_features`. If this is a `pandas.DataFrame` and `scaler` was fit
            on one too, its column names AND order must match
            `scaler.feature_names_in_` exactly.

    Returns:
        The scaled feature matrix, same type/shape/row-count/column-order as
        `features` -- a `pandas.DataFrame` in, `pandas.DataFrame` out (same columns
        and index, values replaced), a `numpy.ndarray` in, `numpy.ndarray` out.
        Only `fit_scaler`'s train matrix is guaranteed `mean~=0`/`std~=1` after this;
        `features` from a different distribution (validation/test) is NOT expected
        to end up centered at zero -- that is the correct, anti-leakage-consistent
        behavior, not a bug.

    Raises:
        ScalingError: if `features` is empty, not 2-dimensional, contains NaN/Inf,
            has a different number of columns than `scaler` was fit on, or (for a
            DataFrame fit on a DataFrame) has different column names/order than
            `scaler.feature_names_in_`.
    """
    array = _validate_matrix(features)

    if array.shape[1] != scaler.n_features_in_:
        raise ScalingError(
            f"features has {array.shape[1]} columns, but scaler was fit on "
            f"{scaler.n_features_in_} columns"
        )

    if isinstance(features, pd.DataFrame) and hasattr(scaler, "feature_names_in_"):
        actual_columns = list(features.columns)
        expected_columns = list(scaler.feature_names_in_)
        if actual_columns != expected_columns:
            raise ScalingError(
                f"features columns {actual_columns} do not match the columns "
                f"scaler was fit on {expected_columns} (name and/or order differs)"
            )

    if isinstance(features, pd.DataFrame):
        transformed = scaler.transform(features)
        return pd.DataFrame(transformed, columns=features.columns, index=features.index)

    transformed = scaler.transform(array)
    return transformed


def save_scaler(scaler: StandardScaler, path: str | Path) -> Path:
    """Serializes a fitted scaler to `path` via `joblib.dump`. Creates parent
    directories if they do not already exist. Returns `path` as a `Path`, for the
    caller's convenience (e.g. logging/further reference)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    return path


def load_scaler(path: str | Path) -> StandardScaler:
    """Deserializes a scaler previously written by `save_scaler` (via
    `joblib.load`). The reloaded scaler transforms identically to the original --
    same `mean_`/`scale_`/`var_`, same `feature_names_in_` if the original had it."""
    return joblib.load(Path(path))
