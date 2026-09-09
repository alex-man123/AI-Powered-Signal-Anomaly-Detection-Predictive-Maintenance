import numpy as np
import pytest

from app.features.registry import FEATURE_REGISTRY
from app.features.time_domain import (
    FeatureError,
    crest_factor,
    kurtosis,
    mean,
    peak,
    peak_to_peak,
    rms,
    skewness,
    std,
    variance,
)

FS = 1000.0  # Hz


# --- 1. mean ---


def test_mean_of_known_vector_is_arithmetic_average() -> None:
    assert mean([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(3.0)


# --- 2/3. std / variance (population convention, ddof=0) ---


def test_std_uses_population_ddof0_not_sample_ddof1() -> None:
    """Explicit check of the documented convention -- population std (ddof=0) must
    differ from the unbiased sample estimator (ddof=1) for this vector, and this
    module's `std()` must match ddof=0, not ddof=1."""
    signal = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]

    population_std = np.std(signal, ddof=0)
    sample_std = np.std(signal, ddof=1)
    assert population_std != pytest.approx(sample_std), "test setup error: ddof=0 and ddof=1 must differ here"

    assert std(signal) == pytest.approx(population_std)
    assert std(signal) != pytest.approx(sample_std)


def test_variance_of_known_vector_matches_population_formula() -> None:
    signal = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    expected = float(np.mean((np.array(signal) - np.mean(signal)) ** 2))

    assert variance(signal) == pytest.approx(expected)
    # std/variance must be consistent with each other (variance = std^2).
    assert std(signal) == pytest.approx(np.sqrt(expected))


# --- 4. RMS (AC1) ---


def test_ac1_rms_of_sine_is_amplitude_over_sqrt2_within_1_percent() -> None:
    n = 10_000
    amplitude = 3.0
    t = np.arange(n) / FS
    signal = amplitude * np.sin(2 * np.pi * 50 * t)

    expected = amplitude / np.sqrt(2)
    measured = rms(signal)

    assert measured == pytest.approx(expected, rel=0.01)


# --- 5. peak ---


def test_peak_uses_max_absolute_value_including_negative_amplitude() -> None:
    signal = [-2.0, 1.0, 3.0, -5.0, 2.0]
    assert peak(signal) == pytest.approx(5.0)


# --- 6. peak_to_peak ---


def test_peak_to_peak_of_known_vector() -> None:
    signal = [-2.0, 1.0, 3.0, -5.0, 2.0]
    # max - min = 3 - (-5) = 8, NOT 2 * peak (which would incorrectly give 10 here).
    assert peak_to_peak(signal) == pytest.approx(8.0)


# --- 7. skewness ---


def test_skewness_of_symmetric_signal_is_zero() -> None:
    """A signal symmetric about its mean has zero skewness by definition -- an
    analytic ground truth independent of this module's own formula."""
    signal = [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert skewness(signal) == pytest.approx(0.0, abs=1e-9)


def test_skewness_matches_manual_third_moment_formula() -> None:
    """Cross-checked against the formula computed directly with NumPy primitives in
    the test itself, not by calling this module's own implementation."""
    signal = np.array([1.0, 2.0, 2.0, 3.0, 10.0])
    m = np.mean(signal)
    sigma = np.std(signal, ddof=0)
    expected = np.mean((signal - m) ** 3) / sigma**3

    assert skewness(signal) == pytest.approx(expected)


def test_skewness_raises_explicitly_for_constant_signal() -> None:
    with pytest.raises(FeatureError, match="standard deviation"):
        skewness([5.0, 5.0, 5.0])


# --- 8. kurtosis (AC2, non-excess convention ~= 3 for Gaussian) ---


def test_kurtosis_of_symmetric_bimodal_signal_matches_analytic_value() -> None:
    """mean=0, variance=E[x^2]=1, fourth moment=E[x^4]=1 -> non-excess kurtosis =
    1/1^2 = 1 -- an exact analytic ground truth, not dependent on random sampling."""
    signal = [-1.0, -1.0, 1.0, 1.0]
    assert kurtosis(signal) == pytest.approx(1.0)


def test_ac2_kurtosis_of_gaussian_noise_is_approximately_3() -> None:
    """Non-excess (Pearson) kurtosis of a true Gaussian distribution is exactly 3;
    a large, fixed-seed sample's kurtosis fluctuates only slightly around it."""
    rng = np.random.default_rng(42)
    signal = rng.normal(0, 1, size=200_000)

    measured = kurtosis(signal)

    assert measured == pytest.approx(3.0, abs=0.1)


def test_kurtosis_raises_explicitly_for_constant_signal() -> None:
    with pytest.raises(FeatureError, match="standard deviation"):
        kurtosis([5.0, 5.0, 5.0])


# --- 9. crest factor (AC3) ---


def test_ac3_crest_factor_is_higher_for_impulsive_signal_than_sinusoidal() -> None:
    n = 1000
    t = np.arange(n) / FS
    sinusoid = np.sin(2 * np.pi * 50 * t)

    impulsive = np.zeros(n)
    impulsive[::100] = 5.0  # 10 rare, sharp spikes; mostly zero elsewhere

    cf_sine = crest_factor(sinusoid)
    cf_impulsive = crest_factor(impulsive)

    print(f"\ncrest_factor sine={cf_sine:.4f} impulsive={cf_impulsive:.4f}")

    assert cf_impulsive > cf_sine
    # Not just marginally larger -- a comfortable, non-tight margin.
    assert cf_impulsive > 2 * cf_sine


def test_crest_factor_raises_explicitly_for_zero_signal() -> None:
    with pytest.raises(FeatureError, match="RMS"):
        crest_factor([0.0, 0.0, 0.0])


# --- edge cases shared across the non-divide-by-zero features ---


def test_mean_std_variance_rms_peak_ptp_on_constant_nonzero_signal() -> None:
    signal = [4.0, 4.0, 4.0]

    assert mean(signal) == pytest.approx(4.0)
    assert std(signal) == pytest.approx(0.0)
    assert variance(signal) == pytest.approx(0.0)
    assert rms(signal) == pytest.approx(4.0)
    assert peak(signal) == pytest.approx(4.0)
    assert peak_to_peak(signal) == pytest.approx(0.0)


def test_mean_std_variance_rms_peak_ptp_on_all_zero_signal() -> None:
    signal = [0.0, 0.0, 0.0]

    assert mean(signal) == pytest.approx(0.0)
    assert std(signal) == pytest.approx(0.0)
    assert variance(signal) == pytest.approx(0.0)
    assert rms(signal) == pytest.approx(0.0)
    assert peak(signal) == pytest.approx(0.0)
    assert peak_to_peak(signal) == pytest.approx(0.0)


def test_mean_std_variance_rms_peak_ptp_on_mixed_sign_signal() -> None:
    signal = [-10.0, -1.0, 0.0, 1.0, 10.0]

    assert mean(signal) == pytest.approx(0.0)
    assert peak(signal) == pytest.approx(10.0)
    assert peak_to_peak(signal) == pytest.approx(20.0)
    assert rms(signal) == pytest.approx(np.sqrt(np.mean(np.array(signal) ** 2)))


def test_crest_factor_of_single_nonzero_sample_is_one() -> None:
    # peak == rms == |x| for a single sample -> crest factor exactly 1.
    assert crest_factor([7.0]) == pytest.approx(1.0)


# --- empty signal: explicit failure for every registered feature ---


@pytest.mark.parametrize("name,func", list(FEATURE_REGISTRY.items()))
def test_all_features_raise_explicitly_for_empty_signal(name: str, func) -> None:
    with pytest.raises(FeatureError, match="empty"):
        func([])


# --- input immutability ---


def test_features_do_not_modify_input_signal() -> None:
    signal = [1.0, -2.0, 3.0, -4.0, 5.0]
    snapshot = list(signal)

    for func in FEATURE_REGISTRY.values():
        func(signal)

    assert signal == snapshot


# --- registry ---


def test_registry_contains_all_nine_features() -> None:
    expected_names = {
        "mean",
        "std",
        "variance",
        "rms",
        "peak",
        "peak_to_peak",
        "skewness",
        "kurtosis",
        "crest_factor",
    }
    assert set(FEATURE_REGISTRY.keys()) == expected_names


def test_registry_values_are_callable() -> None:
    for func in FEATURE_REGISTRY.values():
        assert callable(func)


def test_registry_functions_accept_ndarray_input_and_return_float() -> None:
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    for name, func in FEATURE_REGISTRY.items():
        result = func(signal)
        assert isinstance(result, float), f"{name} did not return a float"


def test_registry_is_extensible_without_modifying_dispatch_logic() -> None:
    """Demonstrates the required extension mechanism: adding a single entry, no
    if/elif chain to touch. Cleans up after itself so it doesn't leak into other
    tests (registry is process-global module state)."""

    def dummy_feature(signal: np.ndarray) -> float:
        return 0.0

    FEATURE_REGISTRY["__test_dummy__"] = dummy_feature
    try:
        assert FEATURE_REGISTRY["__test_dummy__"] is dummy_feature
        assert FEATURE_REGISTRY["__test_dummy__"](np.array([1.0])) == 0.0
    finally:
        del FEATURE_REGISTRY["__test_dummy__"]
