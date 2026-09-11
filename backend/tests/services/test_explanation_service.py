"""TASK 12.1 -- tests for `app.services.explanation_service`.

Pure unit tests (no HTTP, no model artifacts): `explain`/`compute_feature_
baseline` take plain, explicit inputs, so every real-vs-invented-data
requirement from the task ("no fabricated baseline, no fabricated percentage,
correct sign, correct sort order") can be checked directly against
hand-computed expected values, rather than indirectly through a full
`/predict` request.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from app.services.explanation_service import (
    MAX_EXPLANATIONS,
    SIGNIFICANT_DEVIATION_Z_SCORE,
    compute_feature_baseline,
    explain,
)


def test_disclosed_constants_match_the_project_documented_values() -> None:
    """These are the exact values `model_service`'s own docstring already
    disclosed before this task moved the calculation here -- not new numbers
    invented for TASK 12.1."""
    assert SIGNIFICANT_DEVIATION_Z_SCORE == 2.0
    assert MAX_EXPLANATIONS == 3


# --- correct calculation ---


def test_positive_deviation_percent_is_computed_correctly() -> None:
    baseline = {"rms": (10.0, 1.0)}
    result = explain({"rms": 13.2}, baseline)

    assert len(result) == 1
    item = result[0]
    assert item.feature == "rms"
    assert item.current_value == 13.2
    assert item.baseline_value == 10.0
    assert item.deviation == pytest.approx(3.2)
    assert item.deviation_percent == pytest.approx(32.0)
    assert item.direction == "above"


def test_negative_deviation_percent_is_computed_correctly() -> None:
    baseline = {"rms": (10.0, 1.0)}
    result = explain({"rms": 8.0}, baseline)

    assert len(result) == 1
    item = result[0]
    assert item.deviation == pytest.approx(-2.0)
    assert item.deviation_percent == pytest.approx(-20.0)
    assert item.direction == "below"


# --- zero baseline ---


def test_zero_baseline_mean_never_produces_nan_or_infinity() -> None:
    """baseline mean == 0 -> deviation_percent is None (an honest "undefined"),
    never +-Infinity or NaN -- the real, signed deviation is still present."""
    baseline = {"mean_feature": (0.0, 1.0)}
    result = explain({"mean_feature": 5.0}, baseline)

    assert len(result) == 1
    item = result[0]
    assert item.deviation_percent is None
    assert item.deviation == pytest.approx(5.0)
    assert item.direction == "above"
    assert not (isinstance(item.deviation_percent, float) and math.isnan(item.deviation_percent))


def test_zero_baseline_std_is_skipped_not_a_division_by_zero() -> None:
    """A feature with zero variance in the real baseline has no defined
    z-score -- excluded, never crashes, never fabricates a score."""
    baseline = {"constant_feature": (5.0, 0.0)}
    result = explain({"constant_feature": 999.0}, baseline)

    assert result == []


# --- significance filtering ---


def test_only_significant_deviations_are_included() -> None:
    baseline = {"a": (10.0, 1.0), "b": (10.0, 1.0)}
    # a: z = 0.5 (not significant); b: z = 3.0 (significant)
    result = explain({"a": 10.5, "b": 13.0}, baseline)

    assert [item.feature for item in result] == ["b"]


def test_deviation_exactly_at_the_threshold_is_included() -> None:
    """`>= threshold`, not `> threshold` -- matches the real, disclosed rule."""
    baseline = {"a": (10.0, 1.0)}
    result = explain({"a": 12.0}, baseline)  # z == 2.0 exactly

    assert len(result) == 1


def test_a_feature_missing_from_the_baseline_is_ignored() -> None:
    baseline = {"a": (10.0, 1.0)}
    result = explain({"a": 13.0, "unregistered_feature": 999.0}, baseline)

    assert [item.feature for item in result] == ["a"]


# --- sorting ---


def test_results_are_sorted_by_absolute_deviation_percent_descending() -> None:
    baseline = {"a": (10.0, 1.0), "b": (10.0, 1.0), "c": (10.0, 1.0)}
    # a: +30% (z=3), b: +50% (z=5), c: -20% (z=2, exactly at threshold, |pct|=20)
    result = explain({"a": 13.0, "b": 15.0, "c": 8.0}, baseline)

    assert [item.feature for item in result] == ["b", "a", "c"]
    assert [item.deviation_percent for item in result] == [pytest.approx(50.0), pytest.approx(30.0), pytest.approx(-20.0)]


def test_zero_baseline_case_sorts_by_std_deviations_fallback() -> None:
    """A feature with no defined percentage still gets a real, deterministic
    sort position (via |z-score|, its own sort key) instead of being dropped
    or crashing the sort -- here its z-score (2.0) places it below the other
    feature's real 30% deviation."""
    baseline = {"zero_mean": (0.0, 1.0), "normal_mean": (10.0, 1.0)}
    # zero_mean: z=2 (no percent, sort key=2.0); normal_mean: z=3, pct=+30% (sort key=30.0)
    result = explain({"zero_mean": 2.0, "normal_mean": 13.0}, baseline)

    assert [item.feature for item in result] == ["normal_mean", "zero_mean"]
    assert result[1].deviation_percent is None
    assert result[1].std_deviations == pytest.approx(2.0)


def test_max_explanations_caps_the_result_length() -> None:
    baseline = {name: (10.0, 1.0) for name in ["a", "b", "c", "d"]}
    current = {"a": 13.0, "b": 15.0, "c": 20.0, "d": 25.0}  # all significant

    result = explain(current, baseline, max_explanations=2)

    assert len(result) == 2
    # still the two largest |deviation_percent| among the four
    assert [item.feature for item in result] == ["d", "c"]


# --- no invented values ---


def test_all_values_are_traceable_to_the_real_inputs_not_invented() -> None:
    baseline = {"x": (5.0, 2.0)}
    result = explain({"x": 11.0}, baseline)

    item = result[0]
    assert item.current_value == 11.0
    assert item.baseline_value == 5.0
    assert item.deviation == pytest.approx(6.0)
    assert item.deviation_percent == pytest.approx(120.0)
    assert item.std_deviations == pytest.approx(3.0)


# --- normal signal ---


def test_a_signal_with_no_significant_deviations_produces_an_empty_result() -> None:
    baseline = {"a": (10.0, 1.0), "b": (5.0, 0.5)}
    result = explain({"a": 10.1, "b": 5.05}, baseline)

    assert result == []


# --- compute_feature_baseline: real normal-only data, never anomalous rows ---


def test_compute_feature_baseline_uses_only_normal_labeled_rows() -> None:
    features = pd.DataFrame(
        {"rms": [1.0, 3.0, 100.0]},
    )
    labels = ["normal", "normal", "imbalance"]

    baseline = compute_feature_baseline(features, labels)

    # Mean/sample-std (pandas' own default, ddof=1) of [1.0, 3.0] only -- the
    # anomalous 100.0 row must never contribute to the baseline.
    assert baseline["rms"][0] == pytest.approx(2.0)
    assert baseline["rms"][1] == pytest.approx(pd.Series([1.0, 3.0]).std())


def test_compute_feature_baseline_covers_every_feature_column() -> None:
    features = pd.DataFrame({"rms": [1.0, 3.0], "kurtosis": [4.0, 6.0]})
    labels = ["normal", "normal"]

    baseline = compute_feature_baseline(features, labels)

    assert set(baseline.keys()) == {"rms", "kurtosis"}
