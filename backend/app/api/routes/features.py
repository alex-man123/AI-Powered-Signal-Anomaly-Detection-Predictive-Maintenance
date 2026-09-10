"""TASK 10.4 -- `POST /api/features/extract`.

Thin route only: `app.services.feature_service` delegates to Phase 5.3's
existing, registry-driven `extract_features` (no feature formula lives here).
`FeatureExtractionError` (raised by the extractor itself -- e.g. an invalid
Welch PSD configuration for the frequency-domain features) is a `ValueError`
subclass and is converted to an explicit HTTP 422 here, the same pattern
already used in TASK 10.3's `signals.py` -- never a bare `except Exception`.

TASK 10.8: added `summary`/`description` reflecting this real, already-
implemented behavior -- no 404 exists on this route. `tags=["Features"]` is
applied once at `include_router()` time in `app.main`.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas.features import FeatureExtractionRequest, FeatureExtractionResponse
from app.services import feature_service

router = APIRouter()


@router.post(
    "/features/extract",
    response_model=FeatureExtractionResponse,
    summary="Extract DSP features",
    description="Extracts every feature currently registered in "
    "app.features.registry (FEATURE_REGISTRY + FREQUENCY_FEATURE_REGISTRY) from one signal window.",
)
def post_extract_features(request: FeatureExtractionRequest) -> FeatureExtractionResponse:
    try:
        return feature_service.run_feature_extraction(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
