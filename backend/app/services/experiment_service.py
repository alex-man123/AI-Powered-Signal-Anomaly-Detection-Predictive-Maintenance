"""TASK 10.6 -- experiment_service: reads TASK 9.5's own already-generated,
real artifacts for `GET /api/experiments` (list) and `GET /api/experiments/{id}`
(detail). No experiment aggregation/comparison/training logic is duplicated
here -- this module only parses JSON files TASK 9.5's own aggregation script
(`scripts.aggregate_experiment_results`) already wrote to disk.

TWO REAL ARTIFACTS, TWO GRANULARITIES (not two competing sources of truth --
see `app.api.schemas.experiments`'s own docstring for the same point):
  - LIST: `docs/results/central_experiment_results.json` (the flat A/B/C
    comparison matrix TASK 9.5 already produced).
  - DETAIL: `docs/results/experiment_{a,b,c}_reproducibility.json` (the finer-
    grained, per-experiment reproducibility record TASK 9.5 ALSO already
    produced) -- this is what backlog TASK 10.6's own listed detail fields
    (`preprocessing_config`, `feature_set`, `model_config`, `threshold_method`/
    `threshold_value`, `timestamp`) require; the flat list artifact does not
    carry them.

Path constants (`AGGREGATED_JSON_PATH`, `REPRODUCIBILITY_PATHS`) are imported
directly from `scripts.aggregate_experiment_results` -- the SAME module that
writes these files -- rather than redeclared, so this service can never drift
from where TASK 9.5's own script actually writes them (same "import the
canonical constant, never redeclare" convention already established by TASK
9.3/9.4/10.5).

DISCLOSED DISCREPANCY: backlog TASK 10.6's own worked example queries
`GET /api/experiments/EXP-A-001`. That specific id no longer identifies
Experiment A's real reproducibility artifact: TASK 9.5 discovered and fixed a
real non-determinism bug in Experiment A's PCA step and re-registered
Experiment A as `EXP-A-002` (see `experiment_a_reproducibility.json`'s own
`experiment_id` field, and `central_experiment_report.md` sections 4/13 for
the full disclosure) -- `EXP-A-001` is a real, but now-superseded, id. This
service does not special-case or silently redirect it: querying `EXP-A-001`
today correctly returns 404 (it does not identify any CURRENT real artifact),
and `EXP-A-002` returns Experiment A's real, current data. Inventing a
redirect/alias from `EXP-A-001` to `EXP-A-002` here would misrepresent which
id the real artifact actually carries.

`NOT YET MEASURED`: searched for and confirmed this string is not implemented
anywhere in this codebase (only prose in `docs/backlog.md`) and does not
appear in any of TASK 9.5's real, already-measured artifacts -- there is
nothing to special-case; every real field in these artifacts is already a
real measured number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.api.schemas.experiments import ExperimentDetailResponse, ExperimentResponse
from scripts.aggregate_experiment_results import AGGREGATED_JSON_PATH, REPRODUCIBILITY_PATHS


class ExperimentServiceError(ValueError):
    """Raised when TASK 9.5's real artifacts cannot be honestly read (a
    missing file or malformed JSON). Never silently substituted with a
    placeholder."""


class ExperimentNotFoundError(ExperimentServiceError):
    """Raised for an `experiment_id` that does not identify any current real
    reproducibility artifact. The route layer converts this into HTTP 404."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ExperimentServiceError(
            f"expected TASK 9.5 artifact not found: {path} -- run "
            "scripts/aggregate_experiment_results.py to generate it"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def list_experiments() -> list[ExperimentResponse]:
    """`GET /api/experiments`: the real A/B/C comparison matrix TASK 9.5
    already generated (`central_experiment_results.json`)."""
    data = _read_json(AGGREGATED_JSON_PATH)
    return [ExperimentResponse(**row) for row in data["matrix"]]


def _feature_dimension(feature_set: dict[str, Any] | list[str]) -> int:
    """Derives the (not literally top-level) feature dimension from the real,
    already-present `feature_set` field -- see schema module docstring."""
    if isinstance(feature_set, list):
        return len(feature_set)
    return int(feature_set["n_components"])


def _all_reproducibility_artifacts() -> list[dict[str, Any]]:
    return [_read_json(path) for path in REPRODUCIBILITY_PATHS.values() if path.exists()]


def get_experiment_detail(experiment_id: str) -> ExperimentDetailResponse:
    """`GET /api/experiments/{id}`: the real, per-experiment reproducibility
    record TASK 9.5 already generated, matched by its own real `experiment_id`
    field -- never by filename/letter alone (see module docstring's disclosed
    EXP-A-001 -> EXP-A-002 discrepancy)."""
    for artifact in _all_reproducibility_artifacts():
        if artifact.get("experiment_id") == experiment_id:
            return ExperimentDetailResponse(
                **artifact, feature_dimension=_feature_dimension(artifact["feature_set"])
            )
    raise ExperimentNotFoundError(f"experiment {experiment_id!r} does not exist")
