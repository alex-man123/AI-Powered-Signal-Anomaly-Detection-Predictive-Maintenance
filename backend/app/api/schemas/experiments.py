"""TASK 10.1 -- request/response schemas for TASK 10.6's experiment endpoints
(`GET /api/experiments`, `GET /api/experiments/{id}`).

`ExperimentResponse`/`ExperimentListResponse` mirror TASK 9.5's own already-
produced, real contract EXACTLY -- `docs/results/central_experiment_results.
json`'s flat per-experiment row shape (`experiment, experiment_id,
representation, model, dataset_hash, split_manifest_hash, precision, recall,
f1, roc_auc, pr_auc, fpr, fnr, confusion_matrix, inference_time`) -- rather
than inventing a differently nested response shape. TASK 10.6's service layer
is expected to read that real JSON (or `app.ml.experiment_registry.
get_experiment_run`) and populate this schema directly; no experiment
aggregation/comparison logic is duplicated here.

Updated by TASK 10.6 (minimal, justified extension -- not a redesign):
`ExperimentDetailResponse` was added for `GET /api/experiments/{id}`, whose own
required field set (`preprocessing_config`, `feature_set`, `model_config`,
`threshold_method`/`threshold_value`, `timestamp`, the FULL `metrics` dict) is
a strict superset of `ExperimentResponse`'s flat list-row shape and mirrors
TASK 9.5's OTHER real artifact instead: `docs/results/experiment_{a,b,c}_
reproducibility.json` (the per-experiment reproducibility record TASK 9.5
already generates) -- not a second, competing source of truth, but the SAME
task's finer-grained real artifact, used for the finer-grained endpoint.
`metrics` reuses `app.api.schemas.models.ModelMetrics` (the exact same
`evaluate()`-shaped schema TASK 10.5 already defined) rather than a duplicate.

`feature_set` is `dict[str, Any] | list[str]`: Experiment A's real artifact
represents it as `{"type": "raw_pca", "n_components": ...}` (never a DSP
feature name list -- TASK 9.5's own explicit contract), while B/C's real
artifacts represent it as the real DSP feature name list -- both are read
as-is, never normalized into one invented shape.

`model_config` is aliased to `model_configuration` internally: a field
literally named `model_config` collides with Pydantic v2's own reserved
`BaseModel.model_config` class attribute (verified interactively -- assigning
a field of that exact name silently does not create a normal field). The
alias (`populate_by_name=True`, `protected_namespaces=()`) makes the real JSON
key parse in and serialize back out unchanged, so the wire contract's field
name is still exactly `model_config`.

`feature_dimension` is not literally a top-level key in the real
reproducibility artifact (it is implicit -- `len(feature_set)` for B/C,
`feature_set["n_components"]` for A) but IS one of TASK 10.6's explicitly
requested detail fields; the service layer derives it directly from that same
real, already-present data (never invented, never hard-coded) rather than
omitting it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.api.schemas.models import ModelMetrics


class ExperimentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    experiment: Literal["A", "B", "C"]
    experiment_id: str = Field(min_length=1)
    representation: str = Field(min_length=1)
    model: str = Field(min_length=1)
    dataset_hash: str = Field(min_length=1)
    split_manifest_hash: str = Field(min_length=1)
    precision: float = Field(strict=True)
    recall: float = Field(strict=True)
    f1: float = Field(strict=True)
    roc_auc: float = Field(strict=True)
    pr_auc: float = Field(strict=True)
    fpr: float = Field(strict=True)
    fnr: float = Field(strict=True)
    confusion_matrix: list[list[int]] = Field(min_length=2, max_length=2)
    inference_time: float = Field(ge=0, strict=True)


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentResponse]


class ExperimentDetailResponse(BaseModel):
    """`GET /api/experiments/{id}` -- mirrors `docs/results/experiment_{a,b,c}_
    reproducibility.json` (TASK 9.5) field-for-field. See module docstring for
    the `model_config` alias and `feature_dimension` derivation notes."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True, protected_namespaces=())

    experiment_id: str = Field(min_length=1)
    representation: str = Field(min_length=1)
    model: str = Field(min_length=1)
    dataset_hash: str = Field(min_length=1)
    split_manifest_hash: str = Field(min_length=1)
    feature_dimension: int = Field(gt=0, strict=True)
    preprocessing_config: dict[str, Any]
    feature_set: dict[str, Any] | list[str]
    model_configuration: dict[str, Any] = Field(alias="model_config")
    random_seed: int = Field(strict=True)
    threshold_method: str
    threshold_value: float = Field(strict=True)
    metrics: ModelMetrics
    timestamp: str = Field(min_length=1)
