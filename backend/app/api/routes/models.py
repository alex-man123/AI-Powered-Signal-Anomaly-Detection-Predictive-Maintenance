"""TASK 10.5 -- `GET /api/models`, `GET /api/models/{id}/performance`,
`POST /api/models/predict`.

Thin routes only: `app.services.model_service` delegates every real
computation to Phase 5-8's existing artifacts/registry/ML pipeline (no
feature/scaling/scoring/model logic lives here). `ModelNotFoundError` (an
unknown model id, or a missing artifact/registered-metrics record) is
converted to an explicit HTTP 404; any other `ValueError` raised while running
a prediction (e.g. a DSP-level error from feature extraction) is converted to
HTTP 422 -- the same pattern already used in TASK 10.3/10.4's routes -- never a
bare `except Exception`.

TASK 10.8: added `summary`/`description` + explicit `404` documentation on all
three routes. `tags=["Models"]` is applied once at `include_router()` time in
`app.main`.

FIXED (post-10.8, on request): `post_predict`'s `except ModelNotFoundError`
clause used to be dead code -- `model_service.run_prediction` only called the
cached `_get_isolation_forest_model`/`_get_autoencoder_model`/`_get_scaler`
loaders, which raised `ModelPersistenceError` (a *different* `ValueError`
subclass) for a missing artifact, so it was actually caught by the generic
`except ValueError` below and surfaced as 422, not 404 -- a genuinely missing
model artifact is a server-side/infrastructure condition, not a client input
error. Fixed minimally in `model_service.py`: those three cached loaders now
catch `ModelPersistenceError` and re-raise `ModelNotFoundError`, so a missing
artifact now correctly reaches this route's existing `except
ModelNotFoundError` branch (already written, previously unreachable) and
returns 404 -- no change was needed here, in this route file, or in
`run_prediction`'s own logic; the fix lives entirely at the two loader call
sites that were mis-translating the exception type.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas.errors import ErrorResponse
from app.api.schemas.models import ModelResponse, PredictRequest, PredictResponse, SampleSignalResponse
from app.services import model_service
from app.services.model_service import ModelNotFoundError

router = APIRouter()


@router.get(
    "/models/sample-signal",
    response_model=SampleSignalResponse,
    summary="Get a real sample signal to run through prediction",
    description="One real, already-windowed recording from the same real validation set "
    "TASK 10.5's own threshold calibration uses -- never synthetic/random data. "
    "`POST /api/models/predict` requires a real signal from its caller; this endpoint is "
    "what lets a client (e.g. the frontend Dashboard) demonstrate that route without "
    "needing its own dataset file access.",
)
def get_sample_signal() -> SampleSignalResponse:
    return SampleSignalResponse(**model_service.get_sample_signal())


@router.get(
    "/models",
    response_model=list[ModelResponse],
    summary="List models",
    description="Isolation Forest and Autoencoder, each with its real Model Artifact Contract "
    "metadata (TASK 6.5/7.3) and real Phase 8 metrics (TASK 8.3's registry).",
    responses={
        404: {
            "model": ErrorResponse,
            "description": "A model's artifact or its registered Phase 8 metrics are not available.",
        }
    },
)
def get_models() -> list[ModelResponse]:
    try:
        return model_service.list_models()
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/models/{model_id}/performance",
    response_model=ModelResponse,
    summary="Get model performance",
    description="Real Model Artifact Contract metadata + real Phase 8 metrics for one model "
    "('isolation_forest' or 'autoencoder' -- this project's only two real model ids).",
    responses={404: {"model": ErrorResponse, "description": "No model exists with the given id."}},
)
def get_model_performance(model_id: str) -> ModelResponse:
    try:
        return model_service.get_model_performance(model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/models/predict",
    response_model=PredictResponse,
    summary="Predict anomaly score for a signal",
    description="Real feature extraction -> scaling -> model inference -> scoring/normalization -> "
    "threshold-derived NORMAL/WARNING/ANOMALY status -> explanation grounded in the computed features.",
    responses={404: {"model": ErrorResponse, "description": "The requested model's artifact is not available."}},
)
def post_predict(request: PredictRequest) -> PredictResponse:
    try:
        return model_service.run_prediction(request)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
