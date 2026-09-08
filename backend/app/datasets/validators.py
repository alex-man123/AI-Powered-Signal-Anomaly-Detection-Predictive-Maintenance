"""TASK 2.3 — dataset validators: sampling rate, channel count, shape, minimum signal
length.

Independent of FastAPI, UI, and ML — pure functions over already-available metadata
(sampling rate value, shape tuple), never reading or re-parsing a file themselves.

Expected values all come from the local audit, never from external MAFAULDA docs:

- Sampling rate: 50,000 Hz. Source: docs/dataset_audit/signal_structure.md §5 — an
  ADOPTED project decision (`sampling_rate_source: official_documentation`), not a
  value independently measured from local file content (no file has a header,
  metadata, or time column — confirmed in the same section). Defined once here as
  `SAMPLING_RATE_HZ`; `app.datasets.loader` imports it from this module (not the other
  way around, to avoid a circular import between the two).
- Channel count: 8. Source: docs/dataset_audit/signal_structure.md §3 (4-file sample),
  re-confirmed dataset-wide (all 880 real files) by
  backend/scripts/run_data_quality_audit.py during TASK 1.5.8. Defined once here as
  `NUM_CHANNELS`; imported by `app.datasets.loader`.
- Minimum signal length: 250,000 samples. Source: docs/dataset_audit/signal_structure.md
  §3 (4-file sample: exactly 250,000 rows each); re-confirmed for all 880 real files by
  the same TASK 1.5.8 full audit run (zero variance observed — every file has exactly
  250,000 rows). `data_quality_report.md`'s own prose only states column-count
  consistency explicitly, not row-count consistency in so many words — the row-count
  fact is reproducible directly from that already-committed script + the dataset, not
  invented here.

No tolerance is defined anywhere in blueprint.md/backlog.md/the audit for sampling rate
or channel count — and none is needed, since their real-data variance is exactly zero.
Both are therefore validated by exact equality. Signal length uses a minimum-bound
check (`>=`, not `==`), per this task's own explicit `validate_signal_length` design —
250,000 is used as that floor precisely because it is the only value ever observed
locally, not an arbitrary guess.
"""

from __future__ import annotations

# Canonical definitions (validators.py has no dependency on loader.py -- loader.py
# imports these from here, not the other way around, to avoid a circular import while
# still letting both modules share one source of truth for each constant).
NUM_CHANNELS = 8
SAMPLING_RATE_HZ = 50000.0

EXPECTED_SAMPLING_RATE_HZ = SAMPLING_RATE_HZ
EXPECTED_CHANNEL_COUNT = NUM_CHANNELS
MINIMUM_SIGNAL_LENGTH = 250_000


class DatasetValidationError(ValueError):
    """Shared across all dataset validators. `loader.py`'s `DatasetLoaderError`
    inherits from this (see loader.py) rather than duplicating a parallel exception
    hierarchy, per this task's explicit instruction to reuse an existing suitable
    error mechanism instead of creating a second one."""


def validate_sampling_rate(actual: float, expected: float = EXPECTED_SAMPLING_RATE_HZ) -> None:
    if actual <= 0:
        raise DatasetValidationError(f"Sampling rate must be positive, got {actual!r}")
    if actual != expected:
        raise DatasetValidationError(f"Sampling rate mismatch: expected={expected}, actual={actual}")


def validate_channel_count(actual: int, expected: int = EXPECTED_CHANNEL_COUNT) -> None:
    if actual != expected:
        raise DatasetValidationError(f"Channel count mismatch: expected={expected}, actual={actual}")


def validate_signal_length(actual_length: int, minimum_length: int = MINIMUM_SIGNAL_LENGTH) -> None:
    if actual_length < minimum_length:
        raise DatasetValidationError(
            f"Signal length below minimum: expected>={minimum_length}, actual={actual_length}"
        )


def validate_signal_shape(
    shape: tuple[int, int],
    *,
    expected_channels: int = EXPECTED_CHANNEL_COUNT,
    minimum_length: int = MINIMUM_SIGNAL_LENGTH,
) -> None:
    """`shape` is `(rows, columns)` = `(samples, channels)` — matching how every real
    MAFAULDA file and `app.datasets.data_quality.FileQualityReport.rows`/`.columns`
    already represent shape (pandas' own `DataFrame.shape` convention: rows first).
    Never `(channels, samples)` — this project's data is samples-major, never
    transposed anywhere in the pipeline."""
    if len(shape) != 2:
        raise DatasetValidationError(f"Signal shape must be 2-dimensional (rows, columns), got {shape!r}")

    rows, columns = shape
    validate_channel_count(columns, expected_channels)
    validate_signal_length(rows, minimum_length)


def validate_signal(
    *,
    sampling_rate: float,
    shape: tuple[int, int],
    expected_sampling_rate: float = EXPECTED_SAMPLING_RATE_HZ,
    expected_channels: int = EXPECTED_CHANNEL_COUNT,
    minimum_length: int = MINIMUM_SIGNAL_LENGTH,
) -> None:
    """Runs all dataset validators together; raises on the first failure."""
    validate_sampling_rate(sampling_rate, expected_sampling_rate)
    validate_signal_shape(shape, expected_channels=expected_channels, minimum_length=minimum_length)
