"""TASK 10.6 -- `GET /api/experiments`, `GET /api/experiments/{id}`.

Thin routes only: `app.services.experiment_service` reads TASK 9.5's real,
already-generated artifacts (no aggregation/comparison logic lives here).
`ExperimentNotFoundError` is converted to an explicit HTTP 404.

TASK 10.8: added `summary`/`description` + explicit `404` documentation for
`GET /experiments/{id}` (real, reachable -- e.g. the disclosed, now-superseded
`EXP-A-001` id). `tags=["Experiments"]` is applied once at `include_router()`
time in `app.main`.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas.errors import ErrorResponse
from app.api.schemas.experiments import ExperimentDetailResponse, ExperimentResponse
from app.services import experiment_service
from app.services.experiment_service import ExperimentNotFoundError

router = APIRouter()


@router.get(
    "/experiments",
    response_model=list[ExperimentResponse],
    summary="List experiments",
    description="The real A/B/C central-experiment comparison matrix TASK 9.5 already generated "
    "(docs/results/central_experiment_results.json).",
)
def get_experiments() -> list[ExperimentResponse]:
    return experiment_service.list_experiments()


@router.get(
    "/experiments/{experiment_id}",
    response_model=ExperimentDetailResponse,
    summary="Get experiment detail",
    description="The real, per-experiment reproducibility record TASK 9.5 already generated "
    "(docs/results/experiment_{a,b,c}_reproducibility.json), matched by its own experiment_id.",
    responses={404: {"model": ErrorResponse, "description": "No experiment exists with the given id."}},
)
def get_experiment(experiment_id: str) -> ExperimentDetailResponse:
    try:
        return experiment_service.get_experiment_detail(experiment_id)
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
