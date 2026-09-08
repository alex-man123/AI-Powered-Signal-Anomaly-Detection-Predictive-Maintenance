import numpy as np
import pytest

from app.signal_processing.preprocessing import (
    MinMaxParams,
    PreprocessingError,
    StandardizeParams,
    detrend,
    fit_normalizer,
    fit_standardizer,
    transform_normalizer,
    transform_standardizer,
)

# Justified per the task's own example ("ex. panta reziduala < 1e-3"): scipy's linear
# detrend performs an exact least-squares fit, so the residual slope on a clean
# synthetic signal should be many orders of magnitude below this -- 1e-3 is a generous,
# safe bound, not a tight one chosen to make the test barely pass.
RESIDUAL_SLOPE_TOLERANCE = 1e-3


def _residual_slope(signal: np.ndarray) -> float:
    """Measures the actual linear slope remaining in `signal` via least-squares
    regression against sample index -- not just checking mean~0, which would not
    prove the linear trend itself was removed."""
    t = np.arange(len(signal))
    slope, _intercept = np.polyfit(t, signal, deg=1)
    return float(slope)


# --- detrend (AC1) ---


def test_detrend_removes_known_linear_trend_within_tolerance() -> None:
    t = np.arange(1000)
    true_slope = 0.05
    intercept = 3.0
    sinusoid = np.sin(2 * np.pi * 0.01 * t)
    signal = sinusoid + true_slope * t + intercept

    # Sanity: the synthetic signal really does have a strong trend before detrending.
    assert abs(_residual_slope(signal)) > 0.01

    detrended = detrend(signal)

    assert abs(_residual_slope(detrended)) < RESIDUAL_SLOPE_TOLERANCE


def test_detrend_preserves_length() -> None:
    signal = np.linspace(0, 10, 337)

    result = detrend(signal)

    assert len(result) == len(signal)


def test_detrend_does_not_modify_input_in_place_list() -> None:
    original_list = [1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 8.0]
    snapshot = list(original_list)

    detrend(original_list)

    assert original_list == snapshot


def test_detrend_does_not_modify_input_in_place_ndarray() -> None:
    original_array = np.array([1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 8.0])
    snapshot = original_array.copy()

    detrend(original_array)

    assert np.array_equal(original_array, snapshot)


# --- min-max normalization ---


def test_fit_normalizer_computes_train_min_and_max() -> None:
    params = fit_normalizer([0.0, 5.0, 10.0])

    assert params == MinMaxParams(min=0.0, max=10.0)


def test_transform_normalizer_on_the_fitted_signal_produces_zero_one_range() -> None:
    train = [0.0, 5.0, 10.0]
    params = fit_normalizer(train)

    result = transform_normalizer(train, params)

    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


def test_transform_normalizer_on_test_with_train_params_can_exceed_zero_one() -> None:
    """AC2-equivalent for min-max: fitting only on train and applying those exact
    parameters to a differently-distributed test signal must NOT silently produce a
    freshly-rescaled [0, 1] result -- values outside that range are the expected,
    correct signature that no refit happened."""
    train = [0.0, 10.0]
    test = [100.0, 110.0]

    params = fit_normalizer(train)
    result = transform_normalizer(test, params)

    # If test had been independently re-fit, this would be [0, 1]. It is not.
    assert result[0] == pytest.approx(10.0)
    assert result[1] == pytest.approx(11.0)
    assert not np.all((result >= 0.0) & (result <= 1.0))


def test_normalizer_params_are_unchanged_after_being_used_to_transform_test() -> None:
    train = [0.0, 10.0]
    params = fit_normalizer(train)
    snapshot = MinMaxParams(min=params.min, max=params.max)

    transform_normalizer([100.0, 110.0], params)

    assert params == snapshot


def test_fit_normalizer_rejects_constant_signal_explicitly() -> None:
    with pytest.raises(PreprocessingError, match="constant"):
        fit_normalizer([5.0, 5.0, 5.0])


def test_transform_normalizer_does_not_modify_input() -> None:
    signal = [1.0, 2.0, 3.0]
    snapshot = list(signal)
    params = fit_normalizer([0.0, 10.0])

    transform_normalizer(signal, params)

    assert signal == snapshot


# --- z-score standardization (AC2, the critical anti-leakage test) ---


def test_fit_standardizer_computes_train_mean_and_std() -> None:
    train = [1.0, 2.0, 3.0, 4.0, 5.0]

    params = fit_standardizer(train)

    assert params.mean == pytest.approx(3.0)
    assert params.std == pytest.approx(np.std(train))


def test_transform_standardizer_on_fitted_train_signal_is_mean_zero_std_one() -> None:
    train = np.array([10.0, 12.0, 14.0, 11.0, 13.0, 9.0, 15.0])
    params = fit_standardizer(train)

    result = transform_standardizer(train, params)

    assert np.mean(result) == pytest.approx(0.0, abs=1e-9)
    assert np.std(result) == pytest.approx(1.0, abs=1e-9)


def test_ac2_test_transformed_with_train_params_is_not_mean_zero() -> None:
    """The critical anti-leakage test: train and test have deliberately different
    distributions. Fit ONLY on train, then transform test with train's parameters --
    the result must NOT be mean~0, which would only happen if test had been
    (incorrectly) refit on itself."""
    train = np.array([0.0, 1.0, -1.0, 0.5, -0.5, 0.2, -0.2])  # centered near 0
    test = np.array([100.0, 101.0, 99.0, 100.5, 99.5, 100.2, 99.8])  # centered near 100

    params = fit_standardizer(train)
    test_transformed = transform_standardizer(test, params)

    assert np.mean(test_transformed) != pytest.approx(0.0, abs=1.0)
    # Concretely: the result must match applying train's exact params to test's mean
    # -- computed independently here, not assumed -- and that value is nowhere near 0.
    expected_mean = (np.mean(test) - params.mean) / params.std
    assert np.mean(test_transformed) == pytest.approx(expected_mean)
    assert abs(expected_mean) > 50  # train's params put test's mean far from zero


def test_standardizer_params_are_unchanged_after_transforming_a_different_signal() -> None:
    train = [0.0, 1.0, -1.0, 2.0, -2.0]
    params = fit_standardizer(train)
    snapshot = StandardizeParams(mean=params.mean, std=params.std)

    transform_standardizer([500.0, 501.0, 499.0], params)

    assert params == snapshot


def test_an_accidental_refit_on_test_would_be_caught_by_ac2() -> None:
    """Demonstrates that AC2's test actually discriminates correct vs. buggy
    behavior: simulating the bug (fitting AGAIN on test, i.e. what a broken
    implementation might do internally) produces mean~0 -- which the correct-usage
    test above explicitly asserts is NOT the case for real transform_standardizer."""
    test = np.array([100.0, 101.0, 99.0, 100.5, 99.5, 100.2, 99.8])

    buggy_params = fit_standardizer(test)  # the bug: fitting on test itself
    buggy_result = transform_standardizer(test, buggy_params)

    assert np.mean(buggy_result) == pytest.approx(0.0, abs=1e-9)  # this is the leakage signature


def test_fit_standardizer_rejects_zero_variance_signal_explicitly() -> None:
    with pytest.raises(PreprocessingError, match="zero standard deviation"):
        fit_standardizer([7.0, 7.0, 7.0, 7.0])


def test_transform_standardizer_does_not_modify_input() -> None:
    signal = [1.0, 2.0, 3.0]
    snapshot = list(signal)
    params = fit_standardizer([0.0, 1.0, -1.0])

    transform_standardizer(signal, params)

    assert signal == snapshot


def test_transform_standardizer_preserves_length() -> None:
    train = [0.0, 1.0, -1.0, 2.0]
    params = fit_standardizer(train)
    test = [5.0] * 250

    result = transform_standardizer(test, params)

    assert len(result) == len(test)


# --- real-pipeline sanity check (small, not the full dataset) ---


def test_small_real_window_through_detrend_and_standardize() -> None:
    """One real MAFAULDA window (via TASK 2.4's windowing.py) piped through
    detrend -> standardize, confirming the two modules compose without error on real
    data -- not a full-dataset integration test."""
    from pathlib import Path

    import pandas as pd

    from app.signal_processing.windowing import create_windows

    dataset_root = Path(__file__).resolve().parents[3] / "data" / "raw" / "mafaulda"
    relative_path = "normal/12.288.csv"
    df = pd.read_csv(dataset_root / relative_path, header=None)
    channel_values = df[0].tolist()

    windows = create_windows(
        channel_values, recording_id=relative_path, split="train", window_size=1024, overlap=0.0
    )
    real_window = windows[0].values

    detrended = detrend(real_window)
    params = fit_standardizer(detrended)
    standardized = transform_standardizer(detrended, params)

    assert len(standardized) == 1024
    assert np.mean(standardized) == pytest.approx(0.0, abs=1e-9)
