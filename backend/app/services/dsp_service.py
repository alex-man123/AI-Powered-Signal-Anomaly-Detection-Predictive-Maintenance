"""TASK 10.3 -- orchestrates HTTP requests into Phase 3/4's existing,
already-tested DSP functions. No FFT/Welch/spectrogram/Butterworth algorithm is
reimplemented here -- every numeric computation is delegated directly to:

    - `app.signal_processing.fft.compute_fft` / `dominant_frequency` (TASK 4.1)
    - `app.signal_processing.psd.compute_welch_psd` (TASK 4.2)
    - `app.signal_processing.spectrogram.compute_spectrogram` (TASK 4.3)
    - `app.signal_processing.filtering.butter_filter` (TASK 3.2)

This module's only real job is translating between TASK 10.1's Pydantic
request/response schemas (plain Python lists/floats, JSON-serializable) and
these functions' own numpy-based signatures/return values (`.tolist()` on every
`numpy.ndarray` before it reaches a response schema -- FastAPI's JSON encoder
cannot serialize a raw `numpy.ndarray`/`np.float64` without this). Every
DSP-level `ValueError` subclass these functions raise (`FFTError`/`PSDError`/
`SpectrogramError`/`FilteringError` -- e.g. a `cutoff >= Nyquist`) is left to
propagate unchanged; `app.api.routes.signals` is what turns it into an
explicit HTTP 422, never a 500 -- this module does not catch or reinterpret it.
"""

from __future__ import annotations

from app.api.schemas.signal_processing import (
    FFTRequest,
    FFTResponse,
    FilterRequest,
    FilterResponse,
    PSDRequest,
    PSDResponse,
    SpectrogramRequest,
    SpectrogramResponse,
)
from app.signal_processing.fft import compute_fft, dominant_frequency
from app.signal_processing.filtering import butter_filter
from app.signal_processing.psd import compute_welch_psd
from app.signal_processing.spectrogram import compute_spectrogram


def run_fft(request: FFTRequest) -> FFTResponse:
    frequencies, magnitude = compute_fft(request.signal, request.sampling_rate)
    dominant = dominant_frequency(frequencies, magnitude)
    return FFTResponse(
        frequencies=frequencies.tolist(),
        magnitude=magnitude.tolist(),
        dominant_frequency=dominant,
    )


def run_psd(request: PSDRequest) -> PSDResponse:
    frequencies, psd = compute_welch_psd(
        request.signal, request.sampling_rate, request.nperseg, request.noverlap
    )
    return PSDResponse(frequencies=frequencies.tolist(), psd=psd.tolist())


def run_spectrogram(request: SpectrogramRequest) -> SpectrogramResponse:
    frequencies, times, sxx = compute_spectrogram(
        request.signal, request.sampling_rate, request.window_size, request.hop_length
    )
    return SpectrogramResponse(
        frequencies=frequencies.tolist(),
        times=times.tolist(),
        values=sxx.tolist(),
    )


def run_filter(request: FilterRequest) -> FilterResponse:
    filtered = butter_filter(
        request.signal, request.cutoff, request.sampling_rate, request.order, request.btype
    )
    return FilterResponse(filtered_signal=filtered.tolist())
