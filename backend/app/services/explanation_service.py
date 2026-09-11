"""TASK 12.1 -- structured per-feature deviation explanations: given a
signal's real extracted feature values and the real per-feature (mean, std)
normal baseline, decides which features deviate significantly from that
baseline and by how much (as a real, computed percentage) -- never an
invented baseline, threshold, or percentage.

BASELINE (section 3 of the task): `compute_feature_baseline` is the SAME
calculation `app.services.model_service._normal_feature_baseline` used before
this task -- moved here, not duplicated, so `model_service` now calls into
this module instead of computing it inline (one source of truth for both the
free-text `explanation` string TASK 10.5 already returns and this task's new
structured `explanations` list). It is the mean/std of each feature across
the real normal-labeled windows of the SAME real validation set TASK 10.5's
own threshold calibration already uses -- never the anomalous signal being
explained, never a hardcoded number, never data that includes anomalies.

SIGNIFICANCE (sections 2 and 7): a feature is included only if it is
>= `SIGNIFICANT_DEVIATION_Z_SCORE` standard deviations from that real
baseline -- the EXACT threshold `model_service`'s own `_build_explanation`
already established and disclosed before this task (not a new number invented
here; this task's own instructions say to reuse an existing project threshold
when one already exists, and this one does). A feature whose baseline std is
0 has no defined z-score and is skipped, same as before this task.

PERCENTAGE (sections 4-5): `deviation_percent = (current - baseline_mean) /
abs(baseline_mean) * 100`. When `baseline_mean == 0` this ratio is undefined
(would be +/-Infinity, or NaN if `current` is also 0) -- `deviation_percent`
is `None` in that case rather than a fabricated number; `deviation` (the
real, signed absolute difference) is always present so the caller still has
something real to show.

SORTING (section 8): by `abs(deviation_percent)` descending -- falling back to
`abs(std_deviations)` for the rare zero-baseline-mean case where no
percentage exists, so that case still gets a sensible, real, deterministic
ranking position instead of being dropped or randomly placed.

This module never re-implements feature extraction, model inference, or
scoring -- it only compares two sets of already-real numbers (current
features, real baseline) that its caller (`model_service`) provides.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.models.signal import SignalLabel

# Reused from `model_service`'s own pre-existing, disclosed rule -- see module
# docstring. Kept as an explicit, named constant (not inlined) so it can be
# tuned in one place without touching the selection/sorting logic itself.
SIGNIFICANT_DEVIATION_Z_SCORE = 2.0

# Same cap `model_service`'s previous free-text explanation already used, to
# keep a ranked list of "what's notable" short and readable rather than
# dumping every feature that crosses the significance threshold.
MAX_EXPLANATIONS = 3


@dataclass(frozen=True)
class FeatureDeviation:
    """One feature's real, computed deviation from the real normal baseline."""

    feature: str
    current_value: float
    baseline_value: float
    deviation: float
    deviation_percent: float | None
    direction: str  # "above" | "below"
    std_deviations: float  # |z-score| vs. the real baseline -- why this feature was selected


def compute_feature_baseline(features: pd.DataFrame, labels: list[str]) -> dict[str, tuple[float, float]]:
    """Real per-feature (mean, std), computed ONLY from the real
    normal-labeled rows of `features` -- never from the signal being
    explained, never from anomalous rows. `features`/`labels` are expected to
    be the same real, already-extracted validation feature matrix and label
    list `model_service` uses for calibration (row-aligned)."""
    normal_mask = [label == SignalLabel.NORMAL.value for label in labels]
    normal_features = features[normal_mask]
    return {
        column: (float(normal_features[column].mean()), float(normal_features[column].std()))
        for column in normal_features.columns
    }


def explain(
    current_features: dict[str, float],
    baseline: dict[str, tuple[float, float]],
    *,
    z_score_threshold: float = SIGNIFICANT_DEVIATION_Z_SCORE,
    max_explanations: int = MAX_EXPLANATIONS,
) -> list[FeatureDeviation]:
    """Real `current_features` vs. the real `baseline` -> only the features
    that deviate by >= `z_score_threshold` standard deviations, sorted by
    |deviation_percent| descending (falling back to |std_deviations| when no
    percentage is defined), capped at `max_explanations`."""
    candidates: list[FeatureDeviation] = []

    for name, value in current_features.items():
        if name not in baseline:
            continue
        mean, std = baseline[name]
        if std <= 0:
            continue

        z_score = (value - mean) / std
        if abs(z_score) < z_score_threshold:
            continue

        deviation = value - mean
        deviation_percent = None if mean == 0 else (deviation / abs(mean)) * 100

        candidates.append(
            FeatureDeviation(
                feature=name,
                current_value=value,
                baseline_value=mean,
                deviation=deviation,
                deviation_percent=deviation_percent,
                direction="above" if deviation > 0 else "below",
                std_deviations=abs(z_score),
            )
        )

    candidates.sort(
        key=lambda item: abs(item.deviation_percent) if item.deviation_percent is not None else item.std_deviations,
        reverse=True,
    )
    return candidates[:max_explanations]
