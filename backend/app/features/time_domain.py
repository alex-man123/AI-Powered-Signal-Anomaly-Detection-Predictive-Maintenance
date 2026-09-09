"""TASK 5.1 — time-domain vibration features.

Nine pure, independent functions, each `f(signal: Sequence[float]) -> float`. None
have side effects, none modify the input, none depend on each other's internal state
(`crest_factor` composes `peak`/`rms` because that IS its mathematical definition, not
as a shared-state convenience).

Operates on a single 1D signal (`(samples,)`) -- same one-channel-at-a-time convention
established by TASK 2.4's `windowing.Window.values` and followed by every
`app.signal_processing` module since (TASK 3.1/3.2/4.1/4.2/4.3). This module computes
scalar summary statistics over such a signal (or window); it does not select, scale,
or persist features -- that is TASK 5.3/5.4's scope.

Statistical conventions (must be read before interpreting any result):

- `std`/`variance` use **population** convention (`ddof=0`), matching TASK 3.1's
  `fit_standardizer`, which also uses `array.std()` (NumPy's `ddof=0` default) rather
  than the unbiased sample estimator (`ddof=1`). Chosen for consistency with that
  existing convention rather than reintroducing a different one here.
- `peak` is **maximum absolute amplitude** (`max(abs(x))`), not `max(x)`, so a large
  negative excursion (as common in vibration signals) is correctly captured.
- `skewness` is the **population third standardized moment**
  (`E[(X-mean)^3] / std^3`), with no sample-bias correction applied (none requested).
- `kurtosis` uses the **non-excess (Pearson) convention**
  (`E[(X-mean)^4] / std^4`), for which a Gaussian distribution's kurtosis is
  approximately 3 -- NOT the excess convention (`... - 3`, Gaussian ~ 0). Chosen
  because it is the more intuitive of the two documented options, per this task's own
  instruction.

Divide-by-zero is never silently produced (no NaN, no arbitrary substituted value):
`skewness`/`kurtosis` raise `FeatureError` when the signal has zero standard deviation
(a constant signal, including an all-zero one) since the standardized moment is then
mathematically undefined (0/0); `crest_factor` raises `FeatureError` when RMS is zero
(only possible for an all-zero signal) for the same reason -- matching the existing
divide-by-zero convention already used by TASK 3.1's `fit_normalizer`/
`fit_standardizer` (`PreprocessingError` on a constant signal).

No explicit NaN/Inf handling is added here, matching every existing
`app.signal_processing` module (fft.py/psd.py/spectrogram.py/filtering.py): NaN/Inf in
the input propagate through NumPy's own arithmetic exactly as NumPy defines, and are
not separately validated or substituted.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class FeatureError(ValueError):
    """Raised for invalid feature input (empty signal) or a mathematically undefined
    result (a standardized moment or crest factor computed on a zero-variance
    signal, which would divide by zero). Never silently corrected."""


def _validate_signal(signal: Sequence[float]) -> np.ndarray:
    array = np.asarray(signal, dtype=float)
    if array.size == 0:
        raise FeatureError("signal must not be empty")
    return array


def mean(signal: Sequence[float]) -> float:
    """Arithmetic mean of the signal.

    Formula:
        mean = sum(x_i) / N

    Physical interpretation:
        The signal's DC offset / average level.

    Fault relevance:
        A shift in mean can indicate sensor drift or a DC bias fault; on its own it is
        a weak fault indicator for vibration (most faults show up in variance/shape),
        but it is a standard baseline feature.
    """
    array = _validate_signal(signal)
    return float(np.mean(array))


def std(signal: Sequence[float]) -> float:
    """Population standard deviation of the signal (`ddof=0`).

    Formula:
        std = sqrt(mean((x - mean(x))^2))

    Physical interpretation:
        Spread of the signal's amplitude around its mean.

    Fault relevance:
        Increased spread often accompanies increased vibration energy, a common
        symptom of imbalance/misalignment faults.
    """
    array = _validate_signal(signal)
    return float(np.std(array, ddof=0))


def variance(signal: Sequence[float]) -> float:
    """Population variance of the signal (`ddof=0`).

    Formula:
        variance = mean((x - mean(x))^2)

    Physical interpretation:
        Average squared deviation from the mean -- the squared-unit counterpart of
        `std`.

    Fault relevance:
        Same as `std` (they carry the same information, in different units); included
        separately since some downstream consumers expect variance directly rather
        than re-squaring `std`.
    """
    array = _validate_signal(signal)
    return float(np.var(array, ddof=0))


def rms(signal: Sequence[float]) -> float:
    """Root mean square of the signal.

    Formula:
        RMS = sqrt(mean(x^2))

    Physical interpretation:
        The effective magnitude/energy level of the signal (for a pure sinusoid of
        amplitude A, RMS = A / sqrt(2)).

    Fault relevance:
        One of the most widely used vibration health indicators -- an increased RMS
        over time is a classic symptom of developing imbalance/misalignment/wear
        faults.
    """
    array = _validate_signal(signal)
    return float(np.sqrt(np.mean(array**2)))


def peak(signal: Sequence[float]) -> float:
    """Maximum absolute amplitude of the signal.

    Formula:
        peak = max(abs(x))

    Physical interpretation:
        The single largest excursion from zero, in either direction -- a large
        negative excursion is a genuine vibration peak, not something to be
        discarded, hence `abs()` rather than `max(x)`.

    Fault relevance:
        Sensitive to isolated sharp events (impacts, impulsive faults such as
        bearing defects) that RMS, being an average, can dilute.
    """
    array = _validate_signal(signal)
    return float(np.max(np.abs(array)))


def peak_to_peak(signal: Sequence[float]) -> float:
    """Peak-to-peak amplitude of the signal.

    Formula:
        peak_to_peak = max(x) - min(x)

    Physical interpretation:
        The full excursion range of the signal. Not `2 * peak`, since that identity
        only holds for signals symmetric about zero -- real vibration signals
        (especially faulty ones) are frequently asymmetric.

    Fault relevance:
        A widely used vibration severity indicator, standard in condition-monitoring
        practice alongside RMS.
    """
    array = _validate_signal(signal)
    return float(np.max(array) - np.min(array))


def skewness(signal: Sequence[float]) -> float:
    """Population skewness (third standardized moment) of the signal.

    Formula:
        skewness = E[(X - mean)^3] / std^3

    Physical interpretation:
        Asymmetry of the signal's amplitude distribution around its mean. Zero for a
        symmetric distribution (e.g. a pure sinusoid or Gaussian noise); nonzero
        indicates a lopsided distribution.

    Fault relevance:
        Some fault mechanisms (e.g. certain rubbing or one-sided impact conditions)
        produce an asymmetric vibration waveform, which skewness can flag even when
        RMS is unchanged.

    Raises:
        FeatureError: if the signal has zero standard deviation (constant signal) --
            the standardized moment is then 0/0, mathematically undefined.
    """
    array = _validate_signal(signal)
    mean_value = float(np.mean(array))
    std_value = float(np.std(array, ddof=0))
    if std_value == 0:
        raise FeatureError(
            f"Cannot compute skewness: signal has zero standard deviation (mean={mean_value}); "
            "the standardized third moment would be 0/0, which is mathematically undefined."
        )
    third_moment = float(np.mean((array - mean_value) ** 3))
    return third_moment / std_value**3


def kurtosis(signal: Sequence[float]) -> float:
    """Population kurtosis (fourth standardized moment) of the signal, using the
    **non-excess (Pearson) convention** -- a Gaussian distribution has kurtosis
    approximately equal to 3 under this convention (NOT 0, which would be the excess
    convention).

    Formula:
        kurtosis = E[(X - mean)^4] / std^4

    Physical interpretation:
        "Peakedness"/tail-heaviness of the signal's amplitude distribution. A value
        near 3 resembles a Gaussian; a value well above 3 indicates a distribution
        with heavier tails / more extreme outliers than Gaussian.

    Fault relevance:
        A classic bearing-fault indicator: localized bearing defects produce sharp,
        intermittent impulses that raise kurtosis well above 3, even when RMS barely
        changes -- kurtosis is often more sensitive to early-stage impulsive faults
        than RMS.

    Raises:
        FeatureError: if the signal has zero standard deviation (constant signal) --
            the standardized moment is then 0/0, mathematically undefined.
    """
    array = _validate_signal(signal)
    mean_value = float(np.mean(array))
    std_value = float(np.std(array, ddof=0))
    if std_value == 0:
        raise FeatureError(
            f"Cannot compute kurtosis: signal has zero standard deviation (mean={mean_value}); "
            "the standardized fourth moment would be 0/0, which is mathematically undefined."
        )
    fourth_moment = float(np.mean((array - mean_value) ** 4))
    return fourth_moment / std_value**4


def crest_factor(signal: Sequence[float]) -> float:
    """Crest factor of the signal: ratio of peak amplitude to RMS.

    Formula:
        crest_factor = peak(signal) / rms(signal)

    Physical interpretation:
        How "spiky" the signal is relative to its overall energy level. A pure
        sinusoid has crest factor sqrt(2) ~= 1.414; a signal with rare sharp impulses
        riding on a low overall energy level has a much higher crest factor.

    Fault relevance:
        Directly targets impulsive faults (e.g. localized bearing defects) that
        produce sharp, infrequent peaks without necessarily raising RMS much --
        crest factor can catch these when RMS alone would not.

    Raises:
        FeatureError: if RMS is zero (only possible for an all-zero signal) -- the
            ratio would divide by zero, so it is never silently substituted with an
            arbitrary value.
    """
    peak_value = peak(signal)
    rms_value = rms(signal)
    if rms_value == 0:
        raise FeatureError(
            "Cannot compute crest factor: RMS is zero (signal is all-zero); "
            "peak / RMS would divide by zero."
        )
    return peak_value / rms_value
