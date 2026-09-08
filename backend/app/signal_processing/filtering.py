"""TASK 3.2 — Butterworth filtering (low-pass, high-pass, band-pass), zero-phase.

Strict filtering only: no detrending/normalization/standardization (TASK 3.1's
concern, not repeated or auto-chained here) and no FFT/feature extraction (later
tasks). `butter_filter()` does exactly one thing.

Zero-phase via `scipy.signal.filtfilt` (forward-backward filtering) — never
`scipy.signal.lfilter`, which would introduce a phase shift. No manual phase
compensation is implemented; filtfilt's own zero-phase property is relied on directly.

Operates on a single 1D signal (`(samples,)`) — same convention as TASK 3.1's
preprocessing.py and TASK 2.4's Window.values (one channel at a time; no
multi-channel axis is invented here since nothing upstream produces one).
"""

from __future__ import annotations

from typing import Literal, Sequence, Union

import numpy as np
from scipy.signal import butter, filtfilt

FilterType = Literal["lowpass", "highpass", "bandpass"]
Cutoff = Union[float, tuple[float, float]]


class FilteringError(ValueError):
    """Raised for invalid filter configuration (non-positive fs/order, an unknown
    btype, or a cutoff that does not satisfy 0 < cutoff < Nyquist). Never silently
    corrected, never left to produce an unclear error from a lower-level SciPy call."""


def _normalized_frequency(cutoff: Cutoff, btype: FilterType, nyquist: float) -> float | tuple[float, float]:
    if btype in ("lowpass", "highpass"):
        if isinstance(cutoff, (tuple, list)):
            raise FilteringError(
                f"{btype} requires a single scalar cutoff, got {cutoff!r} (use a "
                f"(low, high) pair only for btype='bandpass')"
            )
        if not (0 < cutoff < nyquist):
            raise FilteringError(
                f"cutoff must satisfy 0 < cutoff < nyquist (fs/2={nyquist}), got cutoff={cutoff}"
            )
        return cutoff / nyquist

    # bandpass
    if not (isinstance(cutoff, (tuple, list)) and len(cutoff) == 2):
        raise FilteringError(f"bandpass requires cutoff=(low, high), got {cutoff!r}")
    low, high = cutoff
    if not (0 < low < high < nyquist):
        raise FilteringError(
            f"bandpass requires 0 < low < high < nyquist (fs/2={nyquist}), got low={low}, high={high}"
        )
    return (low / nyquist, high / nyquist)


def butter_filter(
    signal: Sequence[float],
    cutoff: Cutoff,
    fs: float,
    order: int,
    btype: FilterType,
) -> np.ndarray:
    """Zero-phase Butterworth filtering via `scipy.signal.butter` + `filtfilt`.

    Args:
        signal: 1D input signal.
        cutoff: a single frequency in Hz for `btype in {"lowpass", "highpass"}`, or a
            `(low, high)` pair in Hz for `btype="bandpass"`.
        fs: sampling rate in Hz, must be > 0.
        order: Butterworth filter order, must be > 0.
        btype: one of "lowpass", "highpass", "bandpass".

    Returns:
        The filtered signal as a new array (input is never modified in place),
        same length as `signal`.

    Raises:
        FilteringError: if `fs <= 0`, `order <= 0`, `btype` is not recognized, or
            `cutoff` does not satisfy `0 < cutoff < fs/2` (resp. `0 < low < high < fs/2`
            for bandpass) -- a cutoff at or beyond the Nyquist frequency is always
            rejected explicitly, never passed through to SciPy to fail unclearly.
    """
    if fs <= 0:
        raise FilteringError(f"fs must be > 0, got {fs}")
    if order <= 0:
        raise FilteringError(f"order must be > 0, got {order}")
    if btype not in ("lowpass", "highpass", "bandpass"):
        raise FilteringError(f"btype must be one of 'lowpass', 'highpass', 'bandpass', got {btype!r}")

    nyquist = fs / 2
    wn = _normalized_frequency(cutoff, btype, nyquist)

    b, a = butter(order, wn, btype=btype)

    array = np.array(signal, dtype=float, copy=True)
    return filtfilt(b, a, array)
