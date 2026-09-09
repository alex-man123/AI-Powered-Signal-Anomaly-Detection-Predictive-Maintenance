"""TASK 8.1 — common model evaluation framework: ONE `evaluate()` function used
identically for Isolation Forest AND Autoencoder, on the same test set.

Blueprint.md line 270: "Metrici: Precision, Recall, F1, ROC-AUC, PR-AUC ... confusion
matrix, FPR, FNR, timp de antrenare, timp de inferenta." (training time is tracked
elsewhere -- TASK 7.2's own `TrainingHistory` -- this task's own backlog scope is
inference time only, not training time).

CRITICAL architectural rule (AC1): this module is completely MODEL-AGNOSTIC. It has
no knowledge of `IsolationForest`, `Autoencoder`, `decision_function`, or
reconstruction error -- no import from `app.ml.isolation_forest`/`app.ml.
autoencoder`/`app.ml.training` anywhere in this file, and no `if model_type ==
...`/`elif model_type == ...` branch anywhere. Both models feed the SAME `evaluate()`
by first being reduced, by their OWN model-specific code (already implemented in
TASK 6.2-6.4 for Isolation Forest and TASK 7.1-7.3 for Autoencoder), to the same
three generic inputs:

    Isolation Forest                              Autoencoder
    (app.ml.isolation_forest.score +               (app.ml.inference.
     app.ml.scoring.to_anomaly_score/               reconstruction_error --
     normalize_scores/classify)                     already the target
            |                                        orientation, no flip)
            v                                              v
      y_true, y_pred, anomaly_scores, inference_time (IDENTICAL SHAPE/CONTRACT)
            |                                              |
            +----------------------+-----------------------+
                                   v
                             evaluate()   <-- this module, called identically

Label convention (`y_true`): `0 = normal`, `1 = anomaly` -- this project's only
labels are the 4 `app.models.signal.SignalLabel` values; aggregating them down to
this binary target (`SignalLabel.NORMAL.value -> 0`, any other label -> 1) is the
CALLER's responsibility (already established in `app.ml.isolation_forest.
select_normal_samples`/`app.ml.scoring`'s own binary-target conversion for F1-optimal
calibration) -- `evaluate()` itself never touches label semantics, it only validates
that `y_true`/`y_pred` are already `{0, 1}`-valued.

Score convention (`anomaly_scores`): continuous, HIGHER = MORE ANOMALOUS -- this
project's established target orientation (TASK 6.3's `to_anomaly_score`/
`normalize_scores`, TASK 7.3's `reconstruction_error`, both already produce scores in
this orientation before `evaluate()` ever sees them). `evaluate()` never flips a
sign -- it trusts the orientation is already correct, exactly like TASK 6.3/7.3's own
downstream consumers (`calibrate_threshold`/`classify`) do.

Predictions vs. scores (section 7's own critical distinction): precision/recall/F1/
confusion-matrix/FPR/FNR are computed from `(y_true, y_pred)` (binary); ROC-AUC/PR-AUC
are computed from `(y_true, anomaly_scores)` (continuous) -- never the reverse. This
module's own tests explicitly construct a case where `anomaly_scores` carries strictly
more information than `y_pred` and confirms ROC-AUC/PR-AUC actually respond to that
extra information (ruling out an implementation that quietly discretizes scores back
to `y_pred` internally).

Inference time: `evaluate()` does NOT measure it itself -- it would need to know how
to run inference for whichever model produced `y_pred`/`anomaly_scores`, which would
break model-agnosticism (AC1's whole point). The CALLER measures elapsed wall-clock
time around their own inference call (e.g. `time.perf_counter()` around
`app.ml.inference.predict`/`reconstruction_error`, excluding model loading/training/
threshold calibration/dataset loading) and passes the resulting float in.

Mathematically undefined metrics are reported as `float("nan")`, deliberately and
explicitly, never silently fabricated as an arbitrary "safe-looking" number (this task
explicitly forbids inventing `0.5` for a single-class ROC-AUC, and the same "no
fabrication" principle is applied consistently to every other metric with an
undefined edge case in this module):

- FPR (`FP / (FP + TN)`): NaN when there are zero actual negatives in `y_true` (no
  normal examples at all in the test set) -- the denominator is 0.
- FNR (`FN / (FN + TP)`): NaN when there are zero actual positives (no anomaly
  examples at all) -- the denominator is 0.
- precision/recall/F1: `sklearn`'s own `zero_division=np.nan` (verified available in
  the installed sklearn version) -- NaN, not sklearn's older `0`/`1` defaults, which
  would silently look like a real (bad or good) score rather than "undefined".
- ROC-AUC: `sklearn.metrics.roc_auc_score` itself already returns `nan` (with a
  warning) when `y_true` has only one class present -- verified directly against the
  installed sklearn version before relying on this, not assumed from documentation.
- PR-AUC: `sklearn.metrics.average_precision_score` does NOT already do this
  symmetrically -- verified empirically that it returns `0.0` (not NaN) when `y_true`
  has zero positive examples (its own "recall is set to one for all thresholds"
  convention, not a real measurement), while it correctly returns `1.0` when `y_true`
  is all-positive (a genuinely well-defined value: precision is trivially 1 at every
  threshold when there are no negatives to misclassify). This module explicitly
  overrides the zero-positive-examples case to `nan` for consistency with every other
  "undefined" metric here, but leaves the all-positive `1.0` alone since that one is
  real, not a fabricated placeholder.

Not implemented here (later tasks, out of this task's scope): model training,
loading, feature extraction, scaling, threshold calibration, reconstruction, ROC/PR
curve plotting, experiment persistence (SQLite/JSON), API/UI wiring, Phase 9.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class EvaluationError(ValueError):
    """Raised for invalid evaluation input: mismatched lengths, empty arrays, a
    `y_true`/`y_pred` value outside `{0, 1}`, non-finite `anomaly_scores`, or a
    negative/non-finite `inference_time`. Never silently corrected."""


def _validate_binary_labels(array: np.ndarray, *, name: str) -> np.ndarray:
    unique_values = set(np.unique(array).tolist())
    if not unique_values <= {0, 1}:
        raise EvaluationError(f"{name} must contain only 0/1 values, got {sorted(unique_values)}")
    return array.astype(int)


def evaluate(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    anomaly_scores: Sequence[float],
    inference_time: float,
) -> dict:
    """Computes the common evaluation metrics for ANY anomaly-detection model
    already reduced to `(y_true, y_pred, anomaly_scores, inference_time)` -- see
    module docstring for the exact input conventions and the "no fabricated
    undefined values" rule each metric follows.

    Args:
        y_true: ground-truth binary labels, `0 = normal`, `1 = anomaly`, length N.
        y_pred: binary predictions (e.g. `app.ml.scoring.classify`'s output cast
            to int), same convention, length N.
        anomaly_scores: continuous anomaly scores, higher = more anomalous
            (already in this project's target orientation -- see module
            docstring), length N.
        inference_time: wall-clock seconds the caller measured for producing
            `y_pred`/`anomaly_scores` (model inference only -- see module
            docstring for what to exclude). Must be `>= 0` and finite.

    Returns:
        A plain `dict` with keys `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`,
        `confusion_matrix` (`[[TN, FP], [FN, TP]]`, a 2x2 list of native ints),
        `fpr`, `fnr`, `inference_time`. Any metric that is mathematically
        undefined for this particular `y_true`/`y_pred`/`anomaly_scores` (e.g. a
        single-class test set) is `float("nan")` -- never a fabricated
        placeholder value.

    Raises:
        EvaluationError: if `y_true`/`y_pred`/`anomaly_scores` have different
            lengths, are empty, if `y_true`/`y_pred` contain a value outside
            `{0, 1}`, if `anomaly_scores` contains NaN/Inf, or if
            `inference_time` is negative or non-finite.
    """
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    scores_array = np.asarray(anomaly_scores, dtype=float)

    if not (len(y_true_array) == len(y_pred_array) == len(scores_array)):
        raise EvaluationError(
            f"y_true ({len(y_true_array)}), y_pred ({len(y_pred_array)}), and "
            f"anomaly_scores ({len(scores_array)}) must all have the same length"
        )
    if len(y_true_array) == 0:
        raise EvaluationError("y_true/y_pred/anomaly_scores must not be empty")
    if not np.isfinite(scores_array).all():
        raise EvaluationError("anomaly_scores contains NaN/Inf values -- refusing to silently mask them")
    if not math.isfinite(inference_time) or inference_time < 0:
        raise EvaluationError(f"inference_time must be finite and >= 0, got {inference_time}")

    y_true_array = _validate_binary_labels(y_true_array, name="y_true")
    y_pred_array = _validate_binary_labels(y_pred_array, name="y_pred")

    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred_array, labels=[0, 1]).ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else float("nan")

    precision = float(precision_score(y_true_array, y_pred_array, pos_label=1, zero_division=np.nan))
    recall = float(recall_score(y_true_array, y_pred_array, pos_label=1, zero_division=np.nan))
    f1 = float(f1_score(y_true_array, y_pred_array, pos_label=1, zero_division=np.nan))

    roc_auc = float(roc_auc_score(y_true_array, scores_array))

    if not np.any(y_true_array == 1):
        # sklearn's own average_precision_score returns 0.0 here (its "recall is
        # set to one for all thresholds" convention) -- not a real measurement,
        # so this overrides it to NaN for consistency with every other
        # undefined-metric case in this module (see module docstring).
        pr_auc = float("nan")
    else:
        pr_auc = float(average_precision_score(y_true_array, scores_array))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "fpr": fpr,
        "fnr": fnr,
        "inference_time": float(inference_time),
    }
