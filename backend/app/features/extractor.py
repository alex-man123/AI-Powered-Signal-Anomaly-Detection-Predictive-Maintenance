"""TASK 5.3 — feature extractor: signal window -> feature vector -> feature matrix.

Registry-driven only. This module never names an individual feature -- there is no
`features["mean"] = mean(signal)` anywhere here, and no import of an individual
feature function from `app.features.time_domain` / `app.features.frequency_domain`.
It only imports the two registries (`app.features.registry.FEATURE_REGISTRY`,
`FREQUENCY_FEATURE_REGISTRY`) and iterates them:

    for name, func in FEATURE_REGISTRY.items():
        features[name] = func(signal)
    for name, func in FREQUENCY_FEATURE_REGISTRY.items():
        features[name] = func(frequencies, power)

Adding an entry to either registry makes it appear in every output of this module
automatically -- nothing here needs to change. The output column/key count is always
`len(FEATURE_REGISTRY) + len(FREQUENCY_FEATURE_REGISTRY)`, computed at call time, never
a literal number.

Two distinct, real calling conventions exist among registered features (this is a
genuine architectural fact, documented already in TASK 5.2's own module docstring, not
papered over): `FEATURE_REGISTRY` functions take `f(signal)`; `FREQUENCY_FEATURE_
REGISTRY` functions take `f(frequencies, power)` -- an already-computed spectrum. This
extractor is the "caller" that TASK 5.2's own docstring anticipates: it computes
exactly ONE Welch PSD (`app.signal_processing.psd.compute_welch_psd`, TASK 4.2) per
window, then feeds that SAME `(frequencies, power)` pair to every one of
`FREQUENCY_FEATURE_REGISTRY`'s functions -- the PSD is never recomputed per feature.

Sampling rate: MAFAULDA is dataset-wide exactly 50,000 Hz
(`app.datasets.validators.SAMPLING_RATE_HZ` -- the same validated constant TASK
2.1-2.5's pipeline already uses; `Window` objects (TASK 2.4) carry no `fs` field of
their own, since every recording in this dataset shares one sampling rate). `fs` is a
parameter here defaulting to that constant, never a new independent literal.

Determinism (AC1): every step here -- NumPy array construction, the 9 time-domain
formulas, `scipy.signal.welch`, the 6 frequency-domain formulas -- is a pure,
side-effect-free computation with no random number generation, no wall-clock/thread
dependence, and no mutable shared state. The same input signal always produces
bit-identical output.

Feature ordering: Python dicts preserve insertion order (guaranteed since Python
3.7), and so does iterating `FEATURE_REGISTRY.items()` then `FREQUENCY_FEATURE_
REGISTRY.items()` -- the resulting column order is always
`list(FEATURE_REGISTRY) + list(FREQUENCY_FEATURE_REGISTRY)`, never re-sorted or
randomized.

Provenance (recording_id/split/window bounds): deliberately NOT attached as extra
columns on the feature matrix, so that `matrix.shape[1]` stays EXACTLY
`len(FEATURE_REGISTRY) + len(FREQUENCY_FEATURE_REGISTRY)`, per AC3's own exact-equality
wording. Instead, `extract_feature_matrix`'s row `i` corresponds exactly to
`windows[i]` (same order, one row per window, never reordered/shuffled/deduplicated)
-- a caller that needs to join recording_id/split back onto the matrix does so via
that positional correspondence (e.g. `pd.concat([provenance_df, matrix], axis=1)`),
not by asking this function to carry a mixed feature/metadata schema.

No leakage: this module extracts features from whatever windows it is given, in the
order given, one row per window -- it never combines/concatenates windows from
different recordings or different splits, never reshuffles across splits, and never
fits any parameter on the data (no scaling, no normalization is performed here --
that is a later task's scope, see TASK 5.4/module warnings below).

Not implemented here (out of this task's scope): feature scaling/normalization,
feature selection, PCA, anomaly detection, ML, persistence -- extract_feature_matrix
returns raw feature values only.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from app.datasets.validators import SAMPLING_RATE_HZ
from app.features.registry import FEATURE_REGISTRY, FREQUENCY_FEATURE_REGISTRY
from app.signal_processing.psd import compute_welch_psd
from app.signal_processing.windowing import Window


class FeatureExtractionError(ValueError):
    """Raised when a registered feature function fails on a given window's signal.
    Always identifies which feature name failed and preserves the original exception
    as `__cause__` (`raise ... from exc`) -- never a bare `except Exception: value =
    <default>` that would silently produce an incomplete or fabricated feature
    vector."""


def extract_features(
    signal: Sequence[float],
    fs: float = SAMPLING_RATE_HZ,
    *,
    nperseg: int,
    noverlap: int,
) -> dict[str, float]:
    """Extracts every registered feature (TASK 5.1's `FEATURE_REGISTRY` plus TASK
    5.2's `FREQUENCY_FEATURE_REGISTRY`) from a single signal window.

    Args:
        signal: 1D real-valued signal for exactly one window (e.g. a
            `Window.values` from `app.signal_processing.windowing`, TASK 2.4).
        fs: sampling rate in Hz, used only for the Welch PSD that feeds the
            frequency-domain features. Defaults to this dataset's own validated
            constant (`app.datasets.validators.SAMPLING_RATE_HZ` = 50,000 Hz) --
            every MAFAULDA recording shares this one sampling rate, so `Window`
            itself carries no `fs` field to read it from.
        nperseg: Welch PSD segment length (TASK 4.2's `compute_welch_psd` -- see
            that module for its own validation rules, e.g. `nperseg <=
            len(signal)`). Required, not defaulted, since the right value depends on
            the window size the caller is using -- this module does not invent one.
        noverlap: Welch PSD segment overlap (TASK 4.2's own contract), required for
            the same reason.

    Returns:
        A dict mapping feature name -> value, in a fixed, deterministic order:
        every `FEATURE_REGISTRY` name (registry insertion order), then every
        `FREQUENCY_FEATURE_REGISTRY` name (registry insertion order). Always exactly
        `len(FEATURE_REGISTRY) + len(FREQUENCY_FEATURE_REGISTRY)` entries -- never a
        hard-coded count.

    Raises:
        FeatureExtractionError: if any registered feature function raises while
            processing `signal` (e.g. `app.features.time_domain.FeatureError` on a
            constant/all-zero window, or `app.signal_processing.psd.PSDError` /
            `app.features.frequency_domain.FrequencyFeatureError` for an invalid
            spectral configuration) -- identifies the failing feature by name and
            keeps the original exception as `__cause__`. Never caught and replaced
            with a default value.
    """
    values = np.asarray(signal, dtype=float)

    features: dict[str, float] = {}

    for name, func in FEATURE_REGISTRY.items():
        try:
            features[name] = func(values)
        except Exception as exc:
            raise FeatureExtractionError(
                f"Feature extraction failed for feature {name!r}: {exc}"
            ) from exc

    try:
        frequencies, power = compute_welch_psd(values, fs, nperseg=nperseg, noverlap=noverlap)
    except Exception as exc:
        raise FeatureExtractionError(
            f"Welch PSD computation failed (needed for frequency-domain features): {exc}"
        ) from exc

    for name, func in FREQUENCY_FEATURE_REGISTRY.items():
        try:
            features[name] = func(frequencies, power)
        except Exception as exc:
            raise FeatureExtractionError(
                f"Feature extraction failed for feature {name!r}: {exc}"
            ) from exc

    return features


def extract_feature_matrix(
    windows: Sequence[Window],
    fs: float = SAMPLING_RATE_HZ,
    *,
    nperseg: int,
    noverlap: int,
) -> pd.DataFrame:
    """Extracts features from multiple windows into one feature matrix.

    Args:
        windows: a sequence of `Window` objects (TASK 2.4), e.g. all windows from
            one split. Each window is processed independently -- windows are never
            combined/concatenated/reordered/shuffled; row `i` of the result
            corresponds exactly to `windows[i]`.
        fs, nperseg, noverlap: forwarded to `extract_features` for every window --
            see its docstring.

    Returns:
        A `pandas.DataFrame` with one row per window (in the same order as
        `windows`) and one column per registered feature, named and ordered exactly
        as `list(FEATURE_REGISTRY) + list(FREQUENCY_FEATURE_REGISTRY)`. Column count
        is always `len(FEATURE_REGISTRY) + len(FREQUENCY_FEATURE_REGISTRY)`, derived
        from the registries at call time -- never hard-coded. For an empty
        `windows`, returns a 0-row DataFrame with that same column set (not an
        error -- there is simply nothing to extract from).

        Deliberately carries NO recording_id/split/window-bounds columns (see
        module docstring's "Provenance" note) -- only feature values, so that
        `matrix.shape[1]` always equals the registries' combined size exactly.

    Raises:
        FeatureExtractionError: if any window fails feature extraction -- see
            `extract_features`. Processing stops at the first failing window (no
            partial/silently-skipped matrix is returned).
    """
    columns = list(FEATURE_REGISTRY.keys()) + list(FREQUENCY_FEATURE_REGISTRY.keys())

    rows = [
        extract_features(window.values, fs, nperseg=nperseg, noverlap=noverlap) for window in windows
    ]

    return pd.DataFrame(rows, columns=columns)
