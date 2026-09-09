"""TASK 4.2 — Welch power spectral density (PSD).

Thin wrapper over `scipy.signal.welch` — the algorithm itself (segmenting, windowing,
overlap-averaging) is never reimplemented here. Distinct from TASK 4.1's
`compute_fft`/magnitude spectrum: this module always produces a genuine Welch PSD via
SciPy, never `numpy.abs(fft)` relabeled as PSD.

Operates on a single 1D real signal (`(samples,)`) — same convention as TASK 3.1's
preprocessing.py, TASK 3.2's filtering.py, and TASK 4.1's fft.py. No multi-channel
axis is invented here.

No automatic detrending/normalization/standardization/filtering is applied to the
input — TASK 3.1/3.2 remain separate, explicit steps a caller chooses to run (or not)
before calling this function. `scipy.signal.welch`'s own defaults (Hann window,
`detrend="constant"` per-segment, `scaling="density"`) are used as-is; blueprint.md
does not specify an override for any of them, so none is introduced.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import welch as _scipy_welch


class PSDError(ValueError):
    """Raised for invalid Welch PSD configuration (non-positive fs/nperseg, an
    invalid noverlap, or an empty signal). Never silently corrected."""


def compute_welch_psd(
    signal: Sequence[float],
    fs: float,
    nperseg: int,
    noverlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the Welch power spectral density of a real 1D signal.

    Args:
        signal: 1D real-valued input signal.
        fs: sampling rate in Hz, must be > 0.
        nperseg: length of each segment (samples), must be > 0 and <= len(signal).
        noverlap: number of overlapping samples between segments, must satisfy
            `0 <= noverlap < nperseg`.

    Returns:
        `(frequencies, psd)`: non-negative frequencies from 0 Hz up to at most the
        Nyquist frequency (`fs/2`), and the corresponding Welch PSD estimate
        (`scipy.signal.welch`'s own output, un-modified) — same length as
        `frequencies`, non-negative.

    Raises:
        PSDError: if `fs <= 0`, `signal` is empty, `nperseg <= 0`,
            `nperseg > len(signal)`, or `noverlap` does not satisfy
            `0 <= noverlap < nperseg`.
    """
    if fs <= 0:
        raise PSDError(f"fs must be > 0, got {fs}")

    array = np.asarray(signal, dtype=float)
    if array.size == 0:
        raise PSDError("signal must not be empty")

    if nperseg <= 0:
        raise PSDError(f"nperseg must be > 0, got {nperseg}")
    if nperseg > array.size:
        raise PSDError(f"nperseg ({nperseg}) must not exceed signal length ({array.size})")

    if not (0 <= noverlap < nperseg):
        raise PSDError(f"noverlap must satisfy 0 <= noverlap < nperseg ({nperseg}), got {noverlap}")

    frequencies, psd = _scipy_welch(array, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return frequencies, psd
