"""TASK 3.1 — raw signal preprocessing: detrending, min-max normalization, z-score
standardization.

Operates on a single 1D raw signal (`(samples,)`) — matching
`app.signal_processing.windowing.Window.values`, which is already one channel's
samples for one window (TASK 2.4 established that this project handles one channel
at a time; there is no stacked multi-channel array anywhere upstream of this module).

Distinct from TASK 5.4 (feature-vector scaling, which runs on an extractor's output,
not on signal samples) — see backlog TASK 3.1's own architecture note. This module
never imports from a features module and never implements feature scaling.

Fit/transform separation, per blueprint.md section 8's leakage rule ("Normalizarea...
se fit-uieste doar pe train, apoi se aplica pe val/test -- niciodata invers"):
`fit_*` functions compute parameters from exactly one signal (the caller's train
signal) and return an immutable parameters object; `transform_*` functions require
that object as an argument and never recompute it. There is deliberately no
`fit_transform` convenience function, so a caller cannot accidentally re-fit on
validation/test data by using the "wrong" one-call helper.

`detrend()` needs no fit/transform split -- it has no cross-signal parameters to
reuse, it is applied independently to whichever signal it's given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import detrend as _scipy_detrend


class PreprocessingError(ValueError):
    """Raised for preprocessing configuration/data problems -- e.g. fitting a min-max
    normalizer or standardizer on a constant signal, which would divide by zero.
    Never silently handled (no NaN produced, no arbitrary magic-number substitution)."""


@dataclass(frozen=True)
class MinMaxParams:
    min: float
    max: float


@dataclass(frozen=True)
class StandardizeParams:
    mean: float
    std: float


def detrend(signal: Sequence[float]) -> np.ndarray:
    """Removes a linear trend (`scipy.signal.detrend(..., type="linear")`). Does not
    normalize/standardize/filter/FFT, and does not modify the input -- always works
    on its own copy, shape/length preserved exactly."""
    array = np.array(signal, dtype=float, copy=True)
    return _scipy_detrend(array, type="linear")


def fit_normalizer(train_signal: Sequence[float]) -> MinMaxParams:
    """Computes min/max from `train_signal` only. Must never be called with
    validation/test data -- transform_normalizer takes the resulting params as an
    explicit argument specifically so that isn't possible by accident."""
    array = np.asarray(train_signal, dtype=float)
    min_value = float(array.min())
    max_value = float(array.max())

    if max_value == min_value:
        raise PreprocessingError(
            f"Cannot fit min-max normalizer: signal is constant (min=max={min_value}); "
            "this would divide by zero. Not silently handled -- the caller must decide "
            "how to treat a constant window (e.g. exclude it, per "
            "docs/dataset_audit/data_quality_report.md's own EXCLUDE policy for "
            "all-constant signals)."
        )

    return MinMaxParams(min=min_value, max=max_value)


def transform_normalizer(signal: Sequence[float], params: MinMaxParams) -> np.ndarray:
    """`x_normalized = (x - min) / (max - min)`, using `params` exactly as given --
    never recomputed from `signal` itself. Values outside `[0, 1]` are expected and
    correct when `signal` isn't the same data `params` was fit on (e.g. transforming
    test with train's parameters) -- that is not a bug, it is the point."""
    array = np.asarray(signal, dtype=float)
    return (array - params.min) / (params.max - params.min)


def fit_standardizer(train_signal: Sequence[float]) -> StandardizeParams:
    """Computes mean/std from `train_signal` only — same non-refit guarantee as
    `fit_normalizer`."""
    array = np.asarray(train_signal, dtype=float)
    mean = float(array.mean())
    std = float(array.std())

    if std == 0:
        raise PreprocessingError(
            f"Cannot fit standardizer: signal has zero standard deviation (mean={mean}); "
            "this would divide by zero."
        )

    return StandardizeParams(mean=mean, std=std)


def transform_standardizer(signal: Sequence[float], params: StandardizeParams) -> np.ndarray:
    """`z = (x - mean) / std`, using `params` exactly as given. Transforming a signal
    whose own mean/std differ from `params` (e.g. test transformed with train's
    parameters) is expected to NOT produce mean~0/std~1 — that is the anti-leakage
    signature this task's AC2 checks for, not an error."""
    array = np.asarray(signal, dtype=float)
    return (array - params.mean) / params.std
