"""TASK 4.3 — STFT / spectrogram.

Thin wrapper over `scipy.signal.spectrogram` — the short-time FFT itself is never
reimplemented (no manual time-frequency matrix construction, no `numpy.fft` in a
loop). Distinct from TASK 4.1 (single FFT, no time axis) and TASK 4.2 (Welch PSD,
segments averaged into ONE spectrum, no time axis) — this is the only one of the
three that preserves a time axis.

API uses `window_size`/`hop_length` (matching blueprint.md section 13's own
terminology: "window size (rezolutie frecventa) vs. hop length (rezolutie timp)"),
converted to SciPy's `nperseg`/`noverlap` via `noverlap = window_size - hop_length`
-- never the other way around.

Operates on a single 1D real signal (`(samples,)`) — same convention as TASK
3.1/3.2/4.1/4.2. No multi-channel axis is invented here.

`scipy.signal.spectrogram`'s own defaults (window="tukey", `mode="psd"`) are used
as-is; blueprint.md does not specify these parameters explicitly, only the general
UI intent ("magnitudine ca si culoare" in section 20, which describes the heatmap's
visual appearance, not a mandated `mode="magnitude"` at this computation layer) — so
`Sxx` here is honestly what SciPy's default actually returns (power spectral density
per time-frequency bin), not relabeled as "magnitude" to match that casual UI phrase.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import spectrogram as _scipy_spectrogram


class SpectrogramError(ValueError):
    """Raised for invalid spectrogram configuration (non-positive fs/window_size/
    hop_length, hop_length > window_size, window_size larger than the signal, or an
    empty signal). Never silently corrected."""


def compute_spectrogram(
    signal: Sequence[float],
    fs: float,
    window_size: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the STFT spectrogram of a real 1D signal.

    Args:
        signal: 1D real-valued input signal.
        fs: sampling rate in Hz, must be > 0.
        window_size: STFT segment length in samples (`nperseg`), must be > 0 and
            <= len(signal). Larger values improve frequency resolution
            (`Delta_f = fs / window_size`) at the cost of temporal resolution.
        hop_length: number of samples between consecutive segment starts, must
            satisfy `0 < hop_length <= window_size`. Converted to SciPy's
            `noverlap = window_size - hop_length` (never the reverse).

    Returns:
        `(frequencies, times, Sxx)`: `frequencies` are non-negative, up to at most
        Nyquist (`fs/2`); `times` is strictly increasing; `Sxx.shape ==
        (len(frequencies), len(times))`, so `Sxx[:, i]` is the spectrum at `times[i]`.

    Raises:
        SpectrogramError: if `fs <= 0`, `signal` is empty, `window_size <= 0`,
            `window_size > len(signal)`, or `hop_length` does not satisfy
            `0 < hop_length <= window_size`.
    """
    if fs <= 0:
        raise SpectrogramError(f"fs must be > 0, got {fs}")

    array = np.asarray(signal, dtype=float)
    if array.size == 0:
        raise SpectrogramError("signal must not be empty")

    if window_size <= 0:
        raise SpectrogramError(f"window_size must be > 0, got {window_size}")
    if window_size > array.size:
        raise SpectrogramError(f"window_size ({window_size}) must not exceed signal length ({array.size})")

    if not (0 < hop_length <= window_size):
        raise SpectrogramError(
            f"hop_length must satisfy 0 < hop_length <= window_size ({window_size}), got {hop_length}"
        )

    noverlap = window_size - hop_length
    frequencies, times, sxx = _scipy_spectrogram(array, fs=fs, nperseg=window_size, noverlap=noverlap)
    return frequencies, times, sxx
