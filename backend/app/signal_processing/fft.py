"""TASK 4.1 — FFT, frequency axis, magnitude spectrum, dominant frequency.

Strict scope: `compute_fft()` and `dominant_frequency()` only. No STFT, PSD/Welch,
spectrogram, feature extraction, filtering, or plotting happens here (later tasks).

Operates on a single 1D real signal (`(samples,)`) — same convention as TASK 3.1's
preprocessing.py and TASK 3.2's filtering.py (one channel at a time; no multi-channel
axis is invented here since nothing upstream produces one).

Uses `numpy.fft.rfft`/`numpy.fft.rfftfreq` (real-input FFT) rather than the general
complex `numpy.fft.fft`, since the signal is always real — this returns only the
non-negative frequencies up to Nyquist (`fs/2`), with no negative-frequency mirror to
discard manually.

No amplitude calibration/single-sided scaling is applied beyond `np.abs()` on the
complex spectrum — blueprint.md does not call for one, and this task's scope is the
magnitude spectrum itself, not an amplitude-calibration convention.

No frequency-estimation refinement (interpolation, zero-crossing, curve fitting) is
implemented — `dominant_frequency()` returns exactly the frequency of the bin with
the largest magnitude, at FFT bin resolution, per this task's explicit instruction.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class FFTError(ValueError):
    """Raised for invalid FFT input (non-positive fs, empty signal). Never silently
    corrected -- an empty or misconfigured signal has no meaningful spectrum."""


def compute_fft(signal: Sequence[float], fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Computes the single-sided FFT of a real 1D signal.

    Args:
        signal: 1D real-valued input signal, length N.
        fs: sampling rate in Hz, must be > 0.

    Returns:
        `(frequencies, magnitude)`:
        - `frequencies`: `numpy.fft.rfftfreq(N, d=1/fs)` -- non-negative frequencies
          from 0 Hz up to at most the Nyquist frequency (`fs/2`), length `N // 2 + 1`.
        - `magnitude`: `numpy.abs(numpy.fft.rfft(signal))`, the same length as
          `frequencies`.

    Raises:
        FFTError: if `fs <= 0` or `signal` is empty.
    """
    if fs <= 0:
        raise FFTError(f"fs must be > 0, got {fs}")

    array = np.asarray(signal, dtype=float)
    if array.size == 0:
        raise FFTError("signal must not be empty")

    spectrum = np.fft.rfft(array)
    frequencies = np.fft.rfftfreq(array.size, d=1 / fs)
    magnitude = np.abs(spectrum)

    return frequencies, magnitude


def dominant_frequency(freqs: np.ndarray, magnitude: np.ndarray) -> float:
    """Returns the frequency (from `freqs`) at the index of the largest value in
    `magnitude`. `freqs` and `magnitude` must correspond element-wise (as returned
    together by `compute_fft`). No interpolation beyond FFT bin resolution."""
    if len(freqs) != len(magnitude):
        raise FFTError(
            f"freqs and magnitude must have the same length, got {len(freqs)} and {len(magnitude)}"
        )
    if len(magnitude) == 0:
        raise FFTError("magnitude must not be empty")

    index = int(np.argmax(magnitude))
    return float(freqs[index])
