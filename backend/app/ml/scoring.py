"""TASK 6.3 — anomaly scoring: raw score orientation, [0,1] normalization,
configurable threshold calibration.

Separation of responsibility from TASK 6.2 (blueprint.md's own pipeline, line 226):

    normal windows -> feature extraction -> feature matrix -> IsolationForest.fit()
        -> anomaly score (raw) -> normalizare -> threshold -> NORMAL/WARNING/ANOMALY

`app.ml.isolation_forest` owns "model training / model inference" (`train_isolation_
forest`, `score` = a thin wrapper around `IsolationForest.decision_function`, reused
here UNCHANGED, never reimplemented). This module owns everything downstream of that
raw score: sign orientation, normalization, threshold calibration, and the final
binary decision -- it never fits a scaler or a model itself.

SCORE ORIENTATION (verified empirically, not assumed -- see below):
`sklearn.ensemble.IsolationForest.decision_function` follows sklearn's own
documented convention, "the lower, the more abnormal": a clearly anomalous point
(mean far outside a fitted normal distribution) produces a MORE NEGATIVE
`decision_function` value than a normal point, confirmed directly against a
synthetic outlier before writing this module. Blueprint.md (lines 256-262) targets
the OPPOSITE, product-facing convention: `0 = normal`, `1 = highly anomalous`
("Anomaly score = cât de neobișnuit este semnalul ... nu o măsură calibrată de
severitate fizică"). `to_anomaly_score()` below performs the one explicit,
documented sign flip (`-decision_function`) that bridges these two conventions --
every function past that point in this module operates on the flipped,
anomaly-oriented convention (higher = more anomalous), never on raw sklearn output
directly.

NORMALIZATION (blueprint.md line 257): "min-max ... pe baza distribuției scorurilor
din validation set (nu din test set -- altfel leakage), mapat la [0, 1]." This
module implements min-max normalization freshly here (`fit_score_normalizer`/
`normalize_scores`) rather than reusing TASK 3.1's `preprocessing.fit_normalizer`/
`transform_normalizer` (same underlying formula, deliberately NOT the same function):
those operate on raw SIGNAL windows and raise on a constant signal, a different
domain with a different, already-committed edge-case contract; anomaly SCORES are
a different kind of data with a different, deliberately more permissive contract
for the degenerate case (see below) -- reusing that function would either violate
its existing contract or force an unrelated domain to accommodate this one.
`normalize_scores` ALWAYS clips its output to `[0, 1]` (AC2) -- any finite raw score
outside the validation-derived `[validation_min, validation_max]` range (including
extreme synthetic values) is clamped, never extrapolated past the guaranteed range.

Degenerate validation distribution (`validation_min == validation_max`, e.g. every
validation window happened to score identically): dividing by zero is never
attempted. Chosen, documented convention: with a single-point reference
distribution, there is no meaningful notion of "how far" a new score is in relative
terms -- only whether it matches that single known point or not. A new raw score
exactly equal to that point normalizes to `0.0` (matches the entire known
"reference" exactly -- not anomalous relative to what's known); any other raw score
normalizes to `1.0` (any deviation from the only known reference point is maximally
out-of-distribution, since there is zero known spread to express a smaller degree
of deviation against). This is an explicit, simple, tested choice -- not a
universal truth, and not `nan_to_num`-style masking of an actual division by zero
(no division is attempted in this branch at all).

THRESHOLD CALIBRATION operates on already-NORMALIZED `[0,1]` validation scores (not
raw scores) -- consistent with blueprint.md's own threshold examples being stated on
a `[0,1]`-like scale ("pragurile NORMAL/WARNING/ANOMALY (0.30/0.70)"). Min-max
normalization is a monotonic transform, so a percentile computed on normalized
validation scores lands at the same relative position as on raw validation scores;
expressing it in `[0,1]` keeps this module's threshold values on the same scale as
the scores they are compared against at inference time.

Two selectable strategies (backlog TASK 6.3), neither hard-coded as "the" threshold:

- `"percentile"` (default -- see rationale below): threshold = the
  `percentile_value`-th percentile (default 95, blueprint.md's own illustrative
  example) of validation scores. Per blueprint.md line 258's more precise wording
  ("percentila scorurilor pe date normale" -- percentile of NORMAL-labeled scores
  specifically, not all validation scores indiscriminately, since a fault-
  contaminated validation set would otherwise inflate the percentile and raise the
  threshold above where it should sit): when `validation_labels` is provided, only
  the normal-labeled rows are used; when `validation_labels` is omitted (`None`),
  falls back to the percentile of every score given (documented, not hidden).
- `"validation_f1_optimal"`: sweeps every candidate threshold from
  `sklearn.metrics.precision_recall_curve` and returns the one maximizing F1 against
  `validation_labels` converted to a binary target (`SignalLabel.NORMAL.value -> 0`,
  any other label -> 1 -- this project's only labels are the 4 `SignalLabel` values,
  see `app.models.signal`). REQUIRES `validation_labels` with at least one normal AND
  one non-normal example; raises explicitly otherwise (this method is only
  meaningful when validation actually contains fault examples -- no fallback is
  invented here, since neither the blueprint nor the backlog defines one).

DEFAULT METHOD: `"percentile"`. Rationale (not presented as universally correct --
see blueprint.md line 258's own framing, "puncte de plecare ilustrative ... nu adevăr
universal"): it needs no fault examples in validation at all to produce a threshold
(only knowledge of which rows are normal), so it remains usable even when
validation happens to contain zero fault windows -- unlike `"validation_f1_optimal"`,
which requires both classes to be present and fails otherwise.

TEST DATA NEVER PARTICIPATES: no function in this module accepts a "test_scores" or
"test_labels" argument anywhere -- `fit_score_normalizer`/`calibrate_threshold` only
ever see whatever the caller labels "validation". This is a structural property of
the API surface itself, not merely a documented intention.

Not implemented here (later tasks, out of this task's scope): a 3-tier
NORMAL/WARNING/ANOMALY classification (blueprint's UI concept, needing a SECOND
threshold -- TASK 6.3's own ACs only ever mention ONE calibrated threshold; adding a
second, uncalibrated one here would be inventing scope this task never asked for),
TASK 6.5's full model artifact contract (this module's `ScoringCalibration` dataclass
only groups the values that contract will eventually need to persist -- it is not
itself a serialization mechanism), evaluation metrics beyond the F1 used internally
for threshold search, ROC/PR curve plotting, or any API/UI wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler

from app.ml.isolation_forest import score as raw_isolation_forest_score
from app.models.signal import SignalLabel

DEFAULT_THRESHOLD_METHOD = "percentile"
DEFAULT_PERCENTILE_VALUE = 95.0
VALID_THRESHOLD_METHODS = ("percentile", "validation_f1_optimal")


class ScoringError(ValueError):
    """Raised for invalid scoring/calibration input (empty or non-finite scores,
    scores/labels length mismatch, an out-of-range `percentile_value`, an unknown
    `threshold_method`, or `"validation_f1_optimal"` requested without the
    validation labels/classes it requires). Never silently corrected -- there is no
    fallback that guesses a threshold or masks NaN/Inf."""


@dataclass(frozen=True)
class ScoreNormalizationParams:
    """Min-max parameters derived EXCLUSIVELY from a validation set's
    anomaly-oriented raw scores (see `fit_score_normalizer`). Never fit on test."""

    validation_min: float
    validation_max: float


@dataclass(frozen=True)
class ScoringCalibration:
    """Everything TASK 6.5's future model artifact will need to persist for this
    task's part of the pipeline (grouped here for convenience -- this dataclass
    does not itself serialize anything). `percentile_value` is `None` when
    `threshold_method != "percentile"` (it was not used to produce `threshold`)."""

    normalization: ScoreNormalizationParams
    threshold_method: str
    percentile_value: float | None
    threshold: float


def _validate_scores(array: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    if array.size == 0:
        raise ScoringError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ScoringError(
            f"{name} contains NaN/Inf values -- refusing to silently mask them (e.g. via "
            "nan_to_num); fix the upstream scores instead"
        )
    return array


def _validate_labels_match(scores: np.ndarray, labels: Sequence[str], *, name: str) -> None:
    if len(labels) != len(scores):
        raise ScoringError(
            f"{name} has {len(scores)} scores but {len(labels)} labels -- they must align 1:1"
        )


def to_anomaly_score(raw_scores: np.ndarray) -> np.ndarray:
    """Flips `IsolationForest.decision_function`'s convention (lower = more
    anomalous) into this project's anomaly-oriented convention (higher = more
    anomalous), matching blueprint.md's target `0 = normal, 1 = highly anomalous`
    scale. The ONLY sign transformation in this module -- every other function
    here operates on already-flipped scores."""
    return -np.asarray(raw_scores, dtype=float)


def fit_score_normalizer(validation_raw_scores: np.ndarray) -> ScoreNormalizationParams:
    """Computes min/max EXCLUSIVELY from `validation_raw_scores` (anomaly-oriented,
    i.e. already passed through `to_anomaly_score` -- never sklearn's raw
    `decision_function` output directly, and never test scores).

    Args:
        validation_raw_scores: anomaly-oriented raw scores for the validation set
            only (see `to_anomaly_score`).

    Returns:
        `ScoreNormalizationParams` with `validation_min`/`validation_max` --
        `normalize_scores` uses these exactly, never recomputed from new data.

    Raises:
        ScoringError: if `validation_raw_scores` is empty or contains NaN/Inf.
    """
    array = _validate_scores(validation_raw_scores, name="validation_raw_scores")
    return ScoreNormalizationParams(validation_min=float(array.min()), validation_max=float(array.max()))


def normalize_scores(raw_scores: np.ndarray, params: ScoreNormalizationParams) -> np.ndarray:
    """Min-max normalizes anomaly-oriented `raw_scores` into `[0, 1]` using `params`
    exactly as given (never refit from `raw_scores` itself -- the same anti-leakage
    pattern as TASK 3.1/5.4's own fit/apply separation).

    Args:
        raw_scores: anomaly-oriented raw scores (see `to_anomaly_score`) for ANY
            split -- validation (to calibrate a threshold) or new/test data (to
            score it against an already-calibrated threshold).
        params: a `ScoreNormalizationParams` from `fit_score_normalizer`, fit on
            validation only.

    Returns:
        A same-shape array, every value in `[0, 1]` (AC2), guaranteed even for
        extreme/out-of-validation-range finite inputs via clipping. If
        `params.validation_min == params.validation_max` (degenerate validation
        distribution), a score exactly equal to that value normalizes to `0.0`;
        any other (finite) score normalizes to `1.0` -- see module docstring for
        the rationale (no division by zero is attempted in this branch).

    Raises:
        ScoringError: if `raw_scores` is empty or contains NaN/Inf.
    """
    array = _validate_scores(raw_scores, name="raw_scores")

    if params.validation_min == params.validation_max:
        return np.where(array == params.validation_min, 0.0, 1.0)

    normalized = (array - params.validation_min) / (params.validation_max - params.validation_min)
    return np.clip(normalized, 0.0, 1.0)


def calibrate_threshold(
    validation_normalized_scores: np.ndarray,
    *,
    threshold_method: str = DEFAULT_THRESHOLD_METHOD,
    percentile_value: float = DEFAULT_PERCENTILE_VALUE,
    validation_labels: Sequence[str] | None = None,
) -> float:
    """Calibrates a single anomaly-decision threshold from VALIDATION data only --
    this function has no `test_scores`/`test_labels` parameter anywhere in its
    signature; it is structurally impossible to pass test data into calibration
    through this API.

    Args:
        validation_normalized_scores: validation's NORMALIZED (`[0,1]`,
            `normalize_scores`'s output) anomaly scores -- not raw scores.
        threshold_method: `"percentile"` (default) or `"validation_f1_optimal"`.
        percentile_value: used only when `threshold_method="percentile"`; must be
            in `[0, 100]` (default 95, blueprint.md's own illustrative example).
        validation_labels: one label per row of `validation_normalized_scores`
            (`app.models.signal.SignalLabel` values, e.g. `"normal"`). Optional for
            `"percentile"` (see below); REQUIRED for `"validation_f1_optimal"`.

    `"percentile"` behavior: if `validation_labels` is given, the percentile is
    computed over only the normal-labeled scores (blueprint.md line 258's precise
    wording -- a fault-contaminated validation set would otherwise inflate the
    percentile). If `validation_labels` is omitted, falls back to the percentile of
    every score given.

    `"validation_f1_optimal"` behavior: converts `validation_labels` to a binary
    target (`SignalLabel.NORMAL.value -> 0`, anything else -> 1), sweeps every
    candidate threshold from `sklearn.metrics.precision_recall_curve`, and returns
    the one maximizing F1. Requires both classes present in `validation_labels`.

    Returns:
        The calibrated threshold (a `[0,1]`-scale value, comparable directly
        against `normalize_scores`' output at inference time).

    Raises:
        ScoringError: unknown `threshold_method`; `validation_normalized_scores`
            empty/non-finite; `percentile_value` outside `[0, 100]`;
            `validation_labels` length mismatch; `"percentile"` with labels given
            but zero normal-labeled rows; `"validation_f1_optimal"` with
            `validation_labels=None` or fewer than 2 classes present.
    """
    if threshold_method not in VALID_THRESHOLD_METHODS:
        raise ScoringError(
            f"Unknown threshold_method {threshold_method!r}; expected one of {VALID_THRESHOLD_METHODS}"
        )

    scores = _validate_scores(validation_normalized_scores, name="validation_normalized_scores")

    if validation_labels is not None:
        _validate_labels_match(scores, validation_labels, name="validation_normalized_scores")

    if threshold_method == "percentile":
        return _percentile_threshold(scores, percentile_value, validation_labels)

    return _f1_optimal_threshold(scores, validation_labels)


def _percentile_threshold(
    scores: np.ndarray, percentile_value: float, validation_labels: Sequence[str] | None
) -> float:
    if not (0 <= percentile_value <= 100):
        raise ScoringError(f"percentile_value must be in [0, 100], got {percentile_value}")

    if validation_labels is not None:
        normal_mask = np.array([label == SignalLabel.NORMAL.value for label in validation_labels])
        scores = scores[normal_mask]
        if scores.size == 0:
            raise ScoringError(
                f"No samples labeled {SignalLabel.NORMAL.value!r} found in validation_labels -- "
                "cannot compute a normal-only percentile threshold"
            )

    return float(np.percentile(scores, percentile_value))


def _f1_optimal_threshold(scores: np.ndarray, validation_labels: Sequence[str] | None) -> float:
    if validation_labels is None:
        raise ScoringError(
            "threshold_method='validation_f1_optimal' requires validation_labels "
            "(needs both normal and fault examples to compute F1) -- none were provided"
        )

    binary_targets = np.array(
        [0 if label == SignalLabel.NORMAL.value else 1 for label in validation_labels]
    )

    unique_classes = set(binary_targets.tolist())
    if len(unique_classes) < 2:
        raise ScoringError(
            "threshold_method='validation_f1_optimal' requires validation_labels to contain "
            f"BOTH normal and fault/anomaly examples; only found class(es) {unique_classes} -- "
            "no fault examples are present in this validation set"
        )

    precision, recall, thresholds = precision_recall_curve(binary_targets, scores)
    denominator = precision[:-1] + recall[:-1]
    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    best_index = int(np.argmax(f1_scores))
    return float(thresholds[best_index])


def calibrate(
    validation_raw_scores: np.ndarray,
    *,
    threshold_method: str = DEFAULT_THRESHOLD_METHOD,
    percentile_value: float = DEFAULT_PERCENTILE_VALUE,
    validation_labels: Sequence[str] | None = None,
) -> ScoringCalibration:
    """Convenience wrapper running the full validation-only calibration flow:
    `fit_score_normalizer` -> `normalize_scores` (on validation itself) ->
    `calibrate_threshold`. See those functions for individual argument/error
    semantics. `validation_raw_scores` must be anomaly-oriented (see
    `to_anomaly_score`), not sklearn's raw `decision_function` output.
    """
    normalization = fit_score_normalizer(validation_raw_scores)
    validation_normalized_scores = normalize_scores(validation_raw_scores, normalization)
    threshold = calibrate_threshold(
        validation_normalized_scores,
        threshold_method=threshold_method,
        percentile_value=percentile_value,
        validation_labels=validation_labels,
    )

    return ScoringCalibration(
        normalization=normalization,
        threshold_method=threshold_method,
        percentile_value=percentile_value if threshold_method == "percentile" else None,
        threshold=threshold,
    )


def compute_normalized_scores(
    model: IsolationForest,
    scaler: StandardScaler,
    features: object,
    calibration: ScoringCalibration,
) -> np.ndarray:
    """End-to-end scoring of new data (validation-for-calibration-checks, test, or
    any future window) using an ALREADY-CALIBRATED `ScoringCalibration` -- never
    refits the normalizer. Reuses TASK 6.2's `score()` unchanged for the raw
    sklearn call (which itself reuses TASK 5.4's `apply_scaler`, transform-only).

    Returns:
        Normalized `[0,1]` anomaly scores, one per row of `features`.
    """
    raw = raw_isolation_forest_score(model, scaler, features)
    anomaly_raw = to_anomaly_score(raw)
    return normalize_scores(anomaly_raw, calibration.normalization)


def classify(normalized_scores: np.ndarray, calibration: ScoringCalibration) -> np.ndarray:
    """Binary anomaly decision: `normalized_score >= calibration.threshold`. Returns
    a boolean array, `True` meaning anomalous. This task implements only this single
    calibrated threshold -- not the blueprint's later 3-tier NORMAL/WARNING/ANOMALY
    UI classification, which needs a second threshold this task's ACs never ask
    for."""
    array = _validate_scores(normalized_scores, name="normalized_scores")
    return array >= calibration.threshold
