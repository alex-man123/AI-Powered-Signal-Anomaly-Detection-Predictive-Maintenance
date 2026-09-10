"""TASK 10.3 -- `POST /api/fft`, `/api/psd`, `/api/spectrogram`, `/api/dsp/filter`.

Thin routes only: `app.services.dsp_service` delegates every computation to
Phase 3/4's existing, already-tested DSP functions (no algorithm lives here).
The one thing every route does beyond that delegation is convert a DSP-level
`ValueError` (`FFTError`/`PSDError`/`SpectrogramError`/`FilteringError` -- e.g.
`cutoff >= Nyquist`, raised by `butter_filter` itself, never re-checked here)
into an explicit HTTP 422 with the DSP function's own message -- `ValueError`
specifically, never a bare `except Exception`, so an unrelated bug still
surfaces as a 500 rather than being silently reported as a client error.

TASK 10.8: added `summary`/`description` reflecting this real, already-
implemented behavior -- no 404 exists on any of these four routes (none of
them look up an id), so none is declared. `tags=["Signal Processing"]` is
applied once at `include_router()` time in `app.main`.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

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
from app.services import dsp_service

router = APIRouter()


@router.post(
    "/fft",
    response_model=FFTResponse,
    summary="Compute FFT",
    description="Single-sided FFT (frequencies, magnitude, dominant frequency) via app.signal_processing.fft.",
)
def post_fft(request: FFTRequest) -> FFTResponse:
    try:
        return dsp_service.run_fft(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/psd",
    response_model=PSDResponse,
    summary="Compute Welch PSD",
    description="Welch power spectral density via app.signal_processing.psd.compute_welch_psd.",
)
def post_psd(request: PSDRequest) -> PSDResponse:
    try:
        return dsp_service.run_psd(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/spectrogram",
    response_model=SpectrogramResponse,
    summary="Compute spectrogram",
    description="STFT spectrogram (frequencies, times, Sxx) via app.signal_processing.spectrogram.",
)
def post_spectrogram(request: SpectrogramRequest) -> SpectrogramResponse:
    try:
        return dsp_service.run_spectrogram(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/dsp/filter",
    response_model=FilterResponse,
    summary="Apply a Butterworth filter",
    description="Zero-phase Butterworth low/high/band-pass filtering via app.signal_processing.filtering.butter_filter.",
)
def post_filter(request: FilterRequest) -> FilterResponse:
    try:
        return dsp_service.run_filter(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
