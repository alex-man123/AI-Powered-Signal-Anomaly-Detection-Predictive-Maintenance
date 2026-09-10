"""TASK 10.1 -- request/response schemas for TASK 10.3's signal-processing
endpoints (`POST /api/signals/fft`, `/psd`, `/spectrogram`, `POST /api/dsp/filter`).

Field names and constraints mirror the REAL, already-tested Phase 3/4 function
signatures exactly (never invented independently):
    - `app.signal_processing.fft.compute_fft(signal, fs) -> (freqs, magnitude)`
    - `app.signal_processing.psd.compute_welch_psd(signal, fs, nperseg, noverlap)
      -> (frequencies, psd)`
    - `app.signal_processing.spectrogram.compute_spectrogram(signal, fs,
      window_size, hop_length) -> (frequencies, times, Sxx)`
    - `app.signal_processing.filtering.butter_filter(signal, cutoff, fs, order,
      btype) -> filtered_signal`

No FFT/PSD/spectrogram/filtering computation happens in this module -- these are
request/response shapes only. The one thing validated here beyond plain types is
that `cutoff`'s SHAPE matches `btype` (scalar for lowpass/highpass, a `(low,
high)` pair for bandpass) -- a structural check already implied by
`butter_filter`'s own `Cutoff = Union[float, tuple[float, float]]` contract, not a
new DSP rule invented for this schema. The Nyquist-relative check (`0 < cutoff <
fs/2`) is intentionally NOT duplicated here -- that is `butter_filter`'s own
`FilteringError`, to be surfaced by TASK 10.3's route/service (TASK 10.7), not
reimplemented at the schema layer.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from app.signal_processing.filtering import FilterType

SamplingRate = Annotated[float, Field(gt=0, strict=True)]


class FFTRequest(BaseModel):
    signal: list[float] = Field(min_length=1)
    sampling_rate: SamplingRate


class FFTResponse(BaseModel):
    frequencies: list[float]
    magnitude: list[float]
    dominant_frequency: float


class PSDRequest(BaseModel):
    signal: list[float] = Field(min_length=1)
    sampling_rate: SamplingRate
    nperseg: int = Field(gt=0, strict=True)
    noverlap: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def _noverlap_must_be_less_than_nperseg(self) -> "PSDRequest":
        if self.noverlap >= self.nperseg:
            raise ValueError(f"noverlap ({self.noverlap}) must be < nperseg ({self.nperseg})")
        return self


class PSDResponse(BaseModel):
    frequencies: list[float]
    psd: list[float]


class SpectrogramRequest(BaseModel):
    signal: list[float] = Field(min_length=1)
    sampling_rate: SamplingRate
    window_size: int = Field(gt=0, strict=True)
    hop_length: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def _hop_length_must_not_exceed_window_size(self) -> "SpectrogramRequest":
        if self.hop_length > self.window_size:
            raise ValueError(f"hop_length ({self.hop_length}) must be <= window_size ({self.window_size})")
        return self


class SpectrogramResponse(BaseModel):
    frequencies: list[float]
    times: list[float]
    values: list[list[float]] = Field(
        description="Sxx, shape (len(frequencies), len(times)) -- same orientation as "
        "app.signal_processing.spectrogram.compute_spectrogram's own return contract."
    )


class FilterRequest(BaseModel):
    signal: list[float] = Field(min_length=1)
    sampling_rate: SamplingRate
    cutoff: float | tuple[float, float]
    order: int = Field(gt=0, strict=True)
    btype: FilterType

    @model_validator(mode="after")
    def _cutoff_shape_must_match_btype(self) -> "FilterRequest":
        is_pair = isinstance(self.cutoff, tuple)
        if self.btype == "bandpass" and not is_pair:
            raise ValueError("btype='bandpass' requires cutoff=(low, high)")
        if self.btype != "bandpass" and is_pair:
            raise ValueError(f"btype={self.btype!r} requires a single scalar cutoff, not a (low, high) pair")
        return self


class FilterResponse(BaseModel):
    filtered_signal: list[float]
