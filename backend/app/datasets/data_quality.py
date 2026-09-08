"""TASK 1.5.8 — data quality audit: missing values, constant/zero channels, read errors.

Read-only with respect to data/raw/mafaulda/: reads one file at a time (never all 880
simultaneously), classifies it, and discards the DataFrame before moving to the next.
No NaN handling, no interpolation, no row/column removal, no file modification — this
module only detects and classifies, per TASK 1.5.8's own scope.

No column is ever assumed to represent a specific sensor (per TASK 1.5.3: files have no
header) — channels are always referred to positionally (column 0, column 1, ...).

Thresholds used are objective, not arbitrary: "constant" means max == min for the
present (non-missing) values of that column; "zero" means that constant value is
exactly 0. No variance/std threshold is used to flag anything as corrupted — per the
task's explicit instruction, low variance is not the same as a constant signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

STATUS_OK = "OK"
STATUS_READ_ERROR = "READ_ERROR"
STATUS_INVALID_STRUCTURE = "INVALID_STRUCTURE"
STATUS_NON_NUMERIC_DATA = "NON_NUMERIC_DATA"

CLASS_OK = "OK"
CLASS_CONSTANT_NONZERO = "CONSTANT_NONZERO"
CLASS_ZERO_SIGNAL = "ZERO_SIGNAL"


@dataclass(frozen=True)
class ChannelQuality:
    column: int
    min: float
    max: float
    std: float
    classification: str


@dataclass(frozen=True)
class FileQualityReport:
    relative_path: str
    status: str
    rows: int | None
    columns: int | None
    missing_values: int | None
    missing_percentage: float | None
    channels: tuple[ChannelQuality, ...]
    error_message: str | None = None

    @property
    def has_missing_values(self) -> bool:
        return bool(self.missing_values)

    @property
    def constant_channels(self) -> tuple[ChannelQuality, ...]:
        return tuple(c for c in self.channels if c.classification == CLASS_CONSTANT_NONZERO)

    @property
    def zero_channels(self) -> tuple[ChannelQuality, ...]:
        return tuple(c for c in self.channels if c.classification == CLASS_ZERO_SIGNAL)

    @property
    def all_channels_constant_or_zero(self) -> bool:
        return bool(self.channels) and all(c.classification != CLASS_OK for c in self.channels)

    @property
    def is_problematic(self) -> bool:
        return self.status != STATUS_OK or self.has_missing_values or bool(
            self.constant_channels or self.zero_channels
        )


def classify_channel(series: pd.Series) -> str:
    """Constant means max == min among the present (non-missing) values. Zero means
    that constant value is exactly 0. A channel that is all-NaN (no present values at
    all) falls back to OK here -- it is still fully surfaced via the file's own
    missing_values/missing_percentage, so it is never silently dropped, just not
    double-counted as "constant" too."""
    min_v = series.min()
    max_v = series.max()
    if pd.isna(min_v) or pd.isna(max_v):
        return CLASS_OK
    if min_v == max_v:
        return CLASS_ZERO_SIGNAL if min_v == 0 else CLASS_CONSTANT_NONZERO
    return CLASS_OK


def analyze_file_quality(path: Path, relative_path: str) -> FileQualityReport:
    try:
        df = pd.read_csv(path, header=None)
    except Exception as exc:  # noqa: BLE001 - audit must continue past any unreadable file
        return FileQualityReport(
            relative_path=relative_path,
            status=STATUS_READ_ERROR,
            rows=None,
            columns=None,
            missing_values=None,
            missing_percentage=None,
            channels=(),
            error_message=f"{type(exc).__name__}: {exc}",
        )

    if df.shape[0] == 0 or df.shape[1] == 0:
        return FileQualityReport(
            relative_path=relative_path,
            status=STATUS_INVALID_STRUCTURE,
            rows=df.shape[0],
            columns=df.shape[1],
            missing_values=None,
            missing_percentage=None,
            channels=(),
            error_message=f"empty structure: shape={df.shape}",
        )

    non_numeric_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric_cols:
        return FileQualityReport(
            relative_path=relative_path,
            status=STATUS_NON_NUMERIC_DATA,
            rows=df.shape[0],
            columns=df.shape[1],
            missing_values=None,
            missing_percentage=None,
            channels=(),
            error_message=f"non-numeric columns at position(s): {non_numeric_cols}",
        )

    rows, columns = df.shape
    missing = int(df.isna().sum().sum())
    total_values = rows * columns
    missing_percentage = (missing / total_values * 100) if total_values else 0.0

    channels = tuple(
        ChannelQuality(
            column=int(col),
            min=float(df[col].min()) if not pd.isna(df[col].min()) else float("nan"),
            max=float(df[col].max()) if not pd.isna(df[col].max()) else float("nan"),
            std=float(df[col].std()) if not pd.isna(df[col].std()) else float("nan"),
            classification=classify_channel(df[col]),
        )
        for col in df.columns
    )

    return FileQualityReport(
        relative_path=relative_path,
        status=STATUS_OK,
        rows=rows,
        columns=columns,
        missing_values=missing,
        missing_percentage=missing_percentage,
        channels=channels,
    )
