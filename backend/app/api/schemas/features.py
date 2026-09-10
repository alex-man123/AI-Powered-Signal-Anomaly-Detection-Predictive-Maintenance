"""TASK 10.1 -- request/response schemas for TASK 10.4's feature-extraction
endpoint (`POST /api/features/extract`).

No feature computation happens here (that stays in `app.features.registry`,
TASK 5.x) -- this module only shapes the request/response. The one real,
non-DSP invariant enforced here is that a response's `feature_names` and
`values` must correspond 1:1 (same length) -- a structural contract, not a DSP
rule. Backlog TASK 10.4's AC1 ("exact numărul de features din registry") is a
TASK 10.4 service-layer responsibility (populating this schema from the live
`FEATURE_REGISTRY`/`FREQUENCY_FEATURE_REGISTRY`), not something this schema can
enforce by itself without importing DSP/feature logic into the schema layer.

Updated by TASK 10.4 (minimal, justified extension -- not a redesign): TASK
5.3's real `app.features.extractor.extract_features(signal, fs, *, nperseg,
noverlap)` requires `nperseg`/`noverlap` with no default (the frequency-domain
features need a Welch PSD, TASK 4.2, which cannot pick these for the caller) --
TASK 10.1's original request omitted them. Added here with the same validation
already used for `PSDRequest` in `signal_processing.py` (`noverlap < nperseg`),
since this is the same real Welch-PSD constraint, not a new rule invented for
this schema.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class FeatureExtractionRequest(BaseModel):
    signal: list[float] = Field(min_length=1)
    sampling_rate: float = Field(gt=0, strict=True)
    nperseg: int = Field(gt=0, strict=True)
    noverlap: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def _noverlap_must_be_less_than_nperseg(self) -> "FeatureExtractionRequest":
        if self.noverlap >= self.nperseg:
            raise ValueError(f"noverlap ({self.noverlap}) must be < nperseg ({self.nperseg})")
        return self


class FeatureExtractionResponse(BaseModel):
    feature_names: list[str] = Field(min_length=1)
    values: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def _names_and_values_must_have_the_same_length(self) -> "FeatureExtractionResponse":
        if len(self.feature_names) != len(self.values):
            raise ValueError(
                f"feature_names (len={len(self.feature_names)}) and values "
                f"(len={len(self.values)}) must have the same length"
            )
        return self
