"""TASK 10.6 -- `GET /api/experiments`, `GET /api/experiments/{id}`.

Thin routes only: `app.services.experiment_service` reads TASK 9.5's real,
already-generated artifacts (no aggregation/comparison logic lives here).
`ExperimentNotFoundError` is converted to an explicit HTTP 404.

TASK 10.8: added `summary`/`description` + explicit `404` documentation for
`GET /experiments/{id}` (real, reachable -- e.g. the disclosed, now-superseded
`EXP-A-001` id). `tags=["Experiments"]` is applied once at `include_router()`
time in `app.main`.

TASK 12.2 -- added `GET /experiments/pca-visualization` (real per-window PC1/
PC2/PC3 + real class label, `app.services.pca_service`, itself reusing TASK
9.1/9.2's already-fitted PCA -- see that module's own docstring). Declared
BEFORE `/experiments/{experiment_id}` so FastAPI's first-match routing
resolves this literal path instead of treating "pca-visualization" as an
`{experiment_id}` path parameter.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas.errors import ErrorResponse
from app.api.schemas.experiments import ExperimentDetailResponse, ExperimentResponse
from app.api.schemas.pca_visualization import PCAVisualizationResponse
from app.services import experiment_service, pca_service
from app.services.experiment_service import ExperimentNotFoundError
from app.services.pca_service import PCAVisualizationError

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
    "/experiments/pca-visualization",
    response_model=PCAVisualizationResponse,
    summary="Get real per-window PCA coordinates for 3D visualization",
    description="Real PC1/PC2/PC3 coordinates and real class label per real window, produced by "
    "projecting real MAFAULDA windows through Experiment A's already-fitted PCA (TASK 9.1/9.2) -- "
    "for dimensionality-reduction visualization only, never a claim of guaranteed class separability.",
    responses={
        500: {
            "model": ErrorResponse,
            "description": "The real, persisted Experiment A PCA artifact is unavailable.",
        }
    },
)
def get_pca_visualization() -> PCAVisualizationResponse:
    try:
        points, explained_variance_ratio, total_components = pca_service.get_pca_visualization()
    except PCAVisualizationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PCAVisualizationResponse(
        points=points,
        explained_variance_ratio=explained_variance_ratio,
        total_components=total_components,
    )


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
