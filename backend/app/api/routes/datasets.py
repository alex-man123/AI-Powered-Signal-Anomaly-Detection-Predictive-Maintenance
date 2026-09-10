"""TASK 10.2 -- `GET /api/datasets`, `GET /api/datasets/{id}`.

Thin routes only: all real metadata comes from `app.services.dataset_service`
(see that module's docstring for exactly where each value is sourced from).
`dataset_id` is a plain `int` path parameter -- FastAPI itself rejects a
non-integer id with HTTP 422 before this function ever runs, and the only
integer this service currently recognizes is compared against directly (no
path/filename is ever built from user input), so no arbitrary local file
access is possible through this id.

TASK 10.8: added `summary`/`description` (reflecting this route's real,
already-implemented behavior, nothing new) and an explicit `404` response
declaration for `GET /api/datasets/{id}` (the one real, reachable error case
this route raises -- verified against `test_datasets.py`'s own existing
404 test). The `tags=["Datasets"]` grouping is applied once, at
`include_router()` time in `app.main` -- not repeated per route here.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas.datasets import DatasetDetailResponse, DatasetSummaryResponse
from app.api.schemas.errors import ErrorResponse
from app.services.dataset_service import DatasetNotFoundError, get_dataset_detail, list_datasets

router = APIRouter()


@router.get(
    "/datasets",
    response_model=list[DatasetSummaryResponse],
    summary="List datasets",
    description="Lists every dataset this project currently has (real-only -- MAFAULDA).",
)
def get_datasets() -> list[DatasetSummaryResponse]:
    return [DatasetSummaryResponse(**row) for row in list_datasets()]


@router.get(
    "/datasets/{dataset_id}",
    response_model=DatasetDetailResponse,
    summary="Get dataset detail",
    description="Real dataset metadata (signal count, sampling rate, channels, "
    "samples per signal, labels) sourced from the split manifest and audit-confirmed constants.",
    responses={404: {"model": ErrorResponse, "description": "No dataset exists with the given id."}},
)
def get_dataset(dataset_id: int) -> DatasetDetailResponse:
    try:
        detail = get_dataset_detail(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return DatasetDetailResponse(**detail)
