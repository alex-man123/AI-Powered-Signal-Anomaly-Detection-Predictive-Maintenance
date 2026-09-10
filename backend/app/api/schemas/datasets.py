"""TASK 10.1 -- request/response schemas for TASK 10.2's dataset endpoints
(`GET /api/datasets`, `GET /api/datasets/{id}`).

No dataset-loading/aggregation logic lives here (that is TASK 10.2's service-layer
job) -- these are pure data shapes, grounded in the REAL fields already established
by `app.models.signal` (Phase 2: `Dataset`/`Signal` ORM models) and
`docs/dataset_audit/AUDIT_REPORT.md` (the audit-confirmed facts: 8 channels per
file, one `SignalLabel` per recording, a positive `sampling_rate`).

Field-naming decision (disclosed, not silently assumed): backlog TASK 10.2's AC1
asks for "samples, sampling rate, channels" in the detail response, but neither
`Dataset` nor `Signal` (Phase 2) defines a dataset-level aggregate for any of these
-- `Dataset` itself only has `id`/`name`; every other fact lives per-`Signal` row.
This schema therefore exposes `signal_count` (an unambiguous `COUNT(*)`-style
count of a dataset's real recordings), `sampling_rate` (mirrors
`Signal.sampling_rate`'s own type/constraint), `channels` (distinct `Signal.
channel` values present, each bounded exactly like `Signal.channel`), and
`labels` (distinct `SignalLabel` values present) -- each field traceable to an
existing, real column, rather than inventing a new aggregate semantic. The
aggregation itself is TASK 10.2's job, not this task's.

Updated by TASK 10.2 (minimal, justified extension -- not a redesign): AC1's
"samples" is genuinely ambiguous between "how many recordings" (`signal_count`,
already defined above) and "how many raw time-domain samples per recording"
(audit-confirmed as exactly 250,000 for all 880 real files, zero variance --
`app.datasets.validators.MINIMUM_SIGNAL_LENGTH`). Rather than silently picking
one reading, `samples_per_signal` was added to `DatasetDetailResponse` so both
real, audit-traceable facts are exposed.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from app.models.signal import SignalLabel

# Every real MAFAULDA file has exactly 8 channel columns (docs/dataset_audit/
# AUDIT_REPORT.md Sec.5/9) -- the same bound already enforced on `Signal.channel`
# by `app.models.signal.SignalRecord`.
ChannelIndex = Annotated[int, Field(ge=0, le=7, strict=True)]


class DatasetSummaryResponse(BaseModel):
    """One row of `GET /api/datasets`."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str = Field(min_length=1)
    signal_count: int = Field(ge=0, strict=True)


class DatasetDetailResponse(BaseModel):
    """`GET /api/datasets/{id}`. See module docstring for the field-naming
    decision behind `signal_count`/`sampling_rate`/`channels`/`labels`."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str = Field(min_length=1)
    signal_count: int = Field(ge=0, strict=True)
    sampling_rate: float = Field(gt=0, strict=True)
    channels: list[ChannelIndex] = Field(min_length=1)
    samples_per_signal: int = Field(gt=0, strict=True)
    labels: list[SignalLabel] = Field(min_length=1)
