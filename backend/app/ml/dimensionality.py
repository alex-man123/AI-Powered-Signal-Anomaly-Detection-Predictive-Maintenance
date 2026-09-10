"""TASK 9.1 — Raw + PCA representation, dimensionality-controlled to exactly match
TASK 5.3's DSP feature vector -- Phase 9's central DSP-vs-Raw experiment
(blueprint.md section 19).

THE METHODOLOGICAL PROBLEM THIS MODULE SOLVES (blueprint.md section 19, lines
280-301): a raw signal window has hundreds/thousands of samples; a DSP feature
vector has ~15-30 values. Comparing `raw (thousands of dims) -> model` against
`DSP features (~15-30 dims) -> model` directly confounds two different variables
-- REPRESENTATION (raw vs. engineered) and DIMENSIONALITY (thousands vs. ~15-30) --
so any performance difference could come from either one, making the comparison
meaningless. Blueprint's own adopted protocol (its "Decizie", section 19): PCA on
the raw waveform, reduced to EXACTLY the same number of components as the DSP
feature vector, so the ONLY variable left between the two arms is representation
type -- "features generice (PCA) vs. features inginerite cu semnificatie fizica
(DSP)". This module IS that PCA-on-raw step; nothing more.

RAW MEANS RAW: this module performs NO detrending, filtering, normalization, FFT,
PSD, STFT, or DSP feature extraction on its input -- only `sklearn.decomposition.
PCA` (which centers by subtracting the per-column mean internally, PCA's own
mathematical definition, not an extra preprocessing decision) is applied, directly
to whatever raw window values the caller passes in. This is DELIBERATELY separate
from TASK 5.4's `app.ml.scaling` (`StandardScaler` for DSP FEATURE vectors, a
different pipeline stage on different data) -- this module does not import
`app.ml.scaling` and does not standardize its raw input before PCA, since neither
blueprint.md nor this task mandates it, and doing so unasked would blur exactly
the representation-vs-dimensionality distinction this module exists to preserve.

CRITICAL anti-leakage rule (AC1), enforced by the API shape itself, not just by
correct internal code: `fit_pca(train_raw_windows, n_components)` and
`transform_pca(representation, raw_windows)` are two SEPARATE functions.
`transform_pca` calls only `PCA.transform(...)` internally -- there is no
`fit_transform` anywhere in this module, and no code path through which a caller
passing validation/test data to `transform_pca` could cause a re-fit; the
`PCARepresentation` it requires can only have been produced by `fit_pca` in the
first place (the same fit/transform separation pattern already established by
TASK 3.1's `fit_normalizer`/`transform_normalizer` and TASK 5.4's own `fit_scaler`/
`apply_scaler`).

DIMENSION SOURCE OF TRUTH: `n_components` is never hard-coded by this module.
`dsp_feature_dimension()` computes the real, current DSP feature count directly
from TASK 5.3's own registries (`len(FEATURE_REGISTRY) + len(FREQUENCY_FEATURE_
REGISTRY)`) -- a caller wiring up Experiment A (TASK 9.4) is expected to pass THAT
value as `n_components`, so if the DSP feature set ever changes, this module's
target dimensionality changes with it automatically, without editing this file.

MATHEMATICAL LIMIT (section 6): `n_components` cannot exceed
`min(n_train_samples, n_features)` -- if the real DSP feature dimension is larger
than what the available raw training data can support, `fit_pca` raises
`DimensionalityError` explicitly rather than silently reducing `n_components` to
whatever PCA CAN produce (which would break AC2's exact-equality requirement
without saying so).

EXPLAINED VARIANCE (AC3): `fit_pca`'s returned `PCARepresentation` always carries
`explained_variance_ratio` and `total_explained_variance` (`sum` of the former) --
reported as measured, never hidden, never inflated, even when small.

Persistence: not implemented here -- neither blueprint.md nor this task defines an
artifact contract for a fitted PCA object, and TASK 6.5's Model Artifact Contract
is for trained anomaly-detection models (Isolation Forest/Autoencoder), not this
representation-construction step. A future task (TASK 9.4) that needs to reuse a
fitted `PCARepresentation` across a process boundary can serialize the
`PCARepresentation.pca` object itself (a plain scikit-learn estimator) with
whatever mechanism it needs at that time -- inventing one now would be scope creep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class DimensionalityError(ValueError):
    """Raised for invalid PCA representation input/configuration: a non-2D or
    empty raw-window matrix, NaN/Inf values, a non-positive `n_components`, an
    `n_components` exceeding `min(n_samples, n_features)`, or a `transform_pca`
    call whose column count doesn't match the fitted representation. Never
    silently corrected -- there is no automatic reduction of `n_components` to
    whatever PCA happens to be able to produce."""


@dataclass(frozen=True)
class PCARepresentation:
    """The result of `fit_pca` -- a fitted PCA plus its explicit, never-hidden
    explained-variance report (AC3). `pca` is a plain, unmodified
    `sklearn.decomposition.PCA` instance; `transform_pca` is the only supported
    way to apply it to new data."""

    pca: PCA
    n_components: int
    explained_variance_ratio: tuple[float, ...]
    total_explained_variance: float


def dsp_feature_dimension() -> int:
    """The real, current DSP feature vector dimension from TASK 5.3's own
    registries -- `len(FEATURE_REGISTRY) + len(FREQUENCY_FEATURE_REGISTRY)`.
    Computed fresh on each call (never cached into a module-level constant), so it
    always reflects whatever features are currently registered."""
    from app.features.registry import FEATURE_REGISTRY, FREQUENCY_FEATURE_REGISTRY

    return len(FEATURE_REGISTRY) + len(FREQUENCY_FEATURE_REGISTRY)


def _validate_matrix(raw_windows: pd.DataFrame | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(raw_windows, dtype=float)

    if array.ndim != 2:
        raise DimensionalityError(
            f"{name} must be 2-dimensional (n_windows, window_size), got shape {array.shape}"
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise DimensionalityError(f"{name} must have at least one row and one column, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise DimensionalityError(
            f"{name} contains NaN/Inf values -- refusing to silently mask them; fix the "
            "upstream raw window data instead"
        )

    return array


def fit_pca(train_raw_windows: pd.DataFrame | np.ndarray, n_components: int) -> PCARepresentation:
    """Fits PCA EXCLUSIVELY on `train_raw_windows`. Must never be called with
    validation/test data -- `transform_pca` takes the resulting `PCARepresentation`
    as an explicit argument specifically so that isn't possible by accident (the
    same anti-leakage pattern as TASK 3.1/5.4's fit/transform separation).

    Args:
        train_raw_windows: the TRAIN split's raw window values only, shape
            `(n_train_windows, window_size)` -- one row per window, each column a
            raw sample position. No detrending/filtering/normalization/feature
            extraction is applied here or expected to have been applied already
            (see module docstring's "raw means raw").
        n_components: target dimensionality -- for Experiment A (TASK 9.4), this
            must be `dsp_feature_dimension()`'s current value, never a literal
            number chosen independently.

    Returns:
        A `PCARepresentation` with the fitted `PCA`, `n_components`, and the
        explicit `explained_variance_ratio`/`total_explained_variance` (AC3).

    Raises:
        DimensionalityError: if `train_raw_windows` is empty, not 2-dimensional,
            contains NaN/Inf; if `n_components <= 0`; or if `n_components` exceeds
            `min(n_train_windows, window_size)` (PCA cannot produce that many
            components from this data -- never silently reduced).
    """
    array = _validate_matrix(train_raw_windows, name="train_raw_windows")

    if n_components <= 0:
        raise DimensionalityError(f"n_components must be > 0, got {n_components}")

    max_components = min(array.shape[0], array.shape[1])
    if n_components > max_components:
        raise DimensionalityError(
            f"n_components ({n_components}) exceeds min(n_train_windows={array.shape[0]}, "
            f"window_size={array.shape[1]}) = {max_components} -- PCA cannot produce this many "
            "components from this training data. Not silently reduced -- provide more training "
            "windows, or treat this as a blocking condition for the experiment."
        )

    # svd_solver="full" is the exact (non-randomized) LAPACK SVD -- deterministic
    # across processes/runs by construction. Left as "auto", sklearn silently
    # switches to the randomized solver once max(array.shape) is large (as with
    # real 1024-sample raw windows), which is NOT run-to-run deterministic
    # without pinning a random_state -- discovered via TASK 9.5's reproducibility
    # verification. n_components here is always small (the DSP feature
    # dimension), so the exact solver's extra cost is negligible.
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(array)

    return PCARepresentation(
        pca=pca,
        n_components=n_components,
        explained_variance_ratio=tuple(float(v) for v in pca.explained_variance_ratio_),
        total_explained_variance=float(np.sum(pca.explained_variance_ratio_)),
    )


def transform_pca(representation: PCARepresentation, raw_windows: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Applies an ALREADY-FITTED `representation` to `raw_windows` -- transform
    only, never fit. Calls `PCA.transform(...)` and nothing else; `representation.
    pca` is never mutated by this call (its `components_`/`mean_`/
    `explained_variance_ratio_` are unchanged before and after, regardless of
    what `raw_windows` contains).

    Args:
        representation: a `PCARepresentation` from `fit_pca` (fitted on train
            only).
        raw_windows: the raw window matrix to transform -- train, validation,
            test, or any other split; same column contract as `fit_pca`'s
            `train_raw_windows` (same `window_size`).

    Returns:
        The PCA-transformed representation, shape `(n_windows, representation.
        n_components)` -- for Experiment A, this equals `(n_windows,
        dsp_feature_dimension())` when `fit_pca` was called with that value.

    Raises:
        DimensionalityError: if `raw_windows` is empty, not 2-dimensional,
            contains NaN/Inf, or has a different column count (`window_size`)
            than `representation` was fit on.
    """
    array = _validate_matrix(raw_windows, name="raw_windows")

    if array.shape[1] != representation.pca.n_features_in_:
        raise DimensionalityError(
            f"raw_windows has {array.shape[1]} columns, but the PCA representation "
            f"was fit on {representation.pca.n_features_in_} columns"
        )

    return representation.pca.transform(array)
