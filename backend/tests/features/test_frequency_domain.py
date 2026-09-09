import numpy as np
import pytest

from app.features.frequency_domain import (
    FrequencyFeatureError,
    band_energy,
    dominant_frequency,
    spectral_bandwidth,
    spectral_centroid,
    spectral_energy,
    spectral_entropy,
)
from app.signal_processing.fft import FFTError
from app.signal_processing.psd import compute_welch_psd

FS = 1000.0  # Hz
N = 8192  # samples -> 8.192 s, plenty of Welch segments
NPERSEG = 1024
NOVERLAP = 512


def _sine_psd(freq: float, amplitude: float = 1.0, n: int = N, fs: float = FS):
    t = np.arange(n) / fs
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    return compute_welch_psd(signal, fs, nperseg=NPERSEG, noverlap=NOVERLAP)


def _noise_psd(seed: int = 42, noise_std: float = 1.0, n: int = N, fs: float = FS):
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, noise_std, n)
    return compute_welch_psd(signal, fs, nperseg=NPERSEG, noverlap=NOVERLAP)


# --- 1. dominant_frequency ---


def test_dominant_frequency_of_sine_matches_known_frequency() -> None:
    freqs, psd = _sine_psd(100.0)
    detected = dominant_frequency(freqs, psd)

    # Spectral resolution here is fs/nperseg ~= 0.977 Hz -- a couple of bins of slack.
    delta_f = freqs[1] - freqs[0]
    assert abs(detected - 100.0) <= 2 * delta_f


def test_dominant_frequency_delegates_to_task_4_1_fft_error() -> None:
    """Confirms this is a literal reuse of TASK 4.1's own `dominant_frequency` --
    its own `FFTError` (not this module's `FrequencyFeatureError`) propagates for
    mismatched-length input, since no wrapping/reimplementation happens here."""
    with pytest.raises(FFTError, match="same length"):
        dominant_frequency(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0]))


# --- 2. spectral_centroid (AC2) ---


def test_ac2_spectral_centroid_of_concentrated_energy_matches_known_frequency() -> None:
    freqs, psd = _sine_psd(100.0)
    centroid = spectral_centroid(freqs, psd)

    # Measured empirically at ~99.999 Hz for this exact setup; tolerance is
    # justified by spectral resolution (fs/nperseg ~= 0.977 Hz), not an arbitrary
    # wide margin.
    delta_f = freqs[1] - freqs[0]
    assert abs(centroid - 100.0) <= delta_f


def test_spectral_centroid_raises_explicitly_for_zero_power() -> None:
    freqs = np.array([0.0, 1.0, 2.0, 3.0])
    power = np.zeros(4)
    with pytest.raises(FrequencyFeatureError, match="zero"):
        spectral_centroid(freqs, power)


def test_spectral_centroid_rejects_negative_power() -> None:
    freqs = np.array([0.0, 1.0, 2.0])
    power = np.array([1.0, -1.0, 2.0])
    with pytest.raises(FrequencyFeatureError, match="non-negative"):
        spectral_centroid(freqs, power)


# --- 3. spectral_bandwidth ---


def test_spectral_bandwidth_is_smaller_for_concentrated_than_distributed_signal() -> None:
    freqs_sine, psd_sine = _sine_psd(100.0)
    freqs_noise, psd_noise = _noise_psd()

    bandwidth_sine = spectral_bandwidth(freqs_sine, psd_sine)
    bandwidth_noise = spectral_bandwidth(freqs_noise, psd_noise)

    print(f"\nbandwidth sine={bandwidth_sine:.4f} Hz, bandwidth noise={bandwidth_noise:.4f} Hz")

    assert bandwidth_sine < bandwidth_noise
    # Not marginal -- measured ~0.64 Hz (sine) vs. ~143 Hz (broadband noise) for this
    # exact setup, so a 10x margin is a safe, non-tuned threshold.
    assert bandwidth_noise > 10 * bandwidth_sine


def test_spectral_bandwidth_of_two_equal_symmetric_spikes_matches_analytic_value() -> None:
    """Independent analytic ground truth, not derived from this module's own
    formula: two equal-power bins at f=90 and f=110 -> centroid=100 exactly, and
    bandwidth = sqrt(((90-100)^2 + (110-100)^2)/2) = sqrt(100) = 10 exactly."""
    freqs = np.array([90.0, 100.0, 110.0])
    power = np.array([1.0, 0.0, 1.0])

    assert spectral_bandwidth(freqs, power) == pytest.approx(10.0)


# --- 4. spectral_energy ---


def test_spectral_energy_of_sine_matches_theoretical_average_power() -> None:
    """Independent ground truth: a pure sinusoid of amplitude A has average power
    A^2/2 (Rayleigh/Parseval), a fact derived from signal theory, not from this
    module's own implementation."""
    amplitude = 2.0
    freqs, psd = _sine_psd(100.0, amplitude=amplitude)

    expected_power = amplitude**2 / 2
    measured = spectral_energy(freqs, psd)

    # Measured empirically within ~0.01% for this exact clean synthetic setup.
    assert measured == pytest.approx(expected_power, rel=0.01)


def test_spectral_energy_is_near_invariant_to_nperseg() -> None:
    """A bare sum(power) would scale with the number of frequency bins (and thus
    with nperseg); the trapezoidal-integral definition should not."""
    amplitude = 2.0
    n = N
    t = np.arange(n) / FS
    signal = amplitude * np.sin(2 * np.pi * 100 * t)

    energies = []
    for nperseg in (256, 512, 1024, 2048):
        freqs, psd = compute_welch_psd(signal, FS, nperseg=nperseg, noverlap=nperseg // 2)
        energies.append(spectral_energy(freqs, psd))

    print(f"\nspectral_energy across nperseg={[256, 512, 1024, 2048]}: {energies}")

    expected = amplitude**2 / 2
    for energy in energies:
        assert energy == pytest.approx(expected, rel=0.02)


# --- 5. spectral_entropy (AC1) ---


def test_ac1_spectral_entropy_of_sine_is_clearly_lower_than_white_noise() -> None:
    freqs_sine, psd_sine = _sine_psd(100.0)
    freqs_noise, psd_noise = _noise_psd()

    entropy_sine = spectral_entropy(freqs_sine, psd_sine)
    entropy_noise = spectral_entropy(freqs_noise, psd_noise)

    print(f"\nentropy sine={entropy_sine:.4f}, entropy noise={entropy_noise:.4f}")

    assert 0.0 <= entropy_sine <= 1.0
    assert 0.0 <= entropy_noise <= 1.0
    assert entropy_sine < entropy_noise
    # Not marginal -- measured ~0.156 (sine) vs. ~0.993 (noise) for this exact setup.
    assert entropy_sine < 0.5
    assert entropy_noise > 0.9


def test_spectral_entropy_of_single_bin_spectrum_is_zero() -> None:
    freqs = np.array([50.0])
    power = np.array([3.0])
    assert spectral_entropy(freqs, power) == pytest.approx(0.0)


def test_spectral_entropy_raises_explicitly_for_zero_power() -> None:
    freqs = np.array([0.0, 1.0, 2.0])
    power = np.zeros(3)
    with pytest.raises(FrequencyFeatureError, match="zero"):
        spectral_entropy(freqs, power)


# --- 6. band_energy ---


def test_band_energy_isolates_each_components_energy_in_a_two_tone_signal() -> None:
    """Independent ground truth: each sinusoidal component contributes A^2/2 average
    power, and (per Parseval, for well-separated frequencies) that power should be
    recoverable by integrating the PSD over a band around just that component."""
    n = N
    t = np.arange(n) / FS
    amplitude_1, amplitude_2 = 2.0, 3.0
    signal = amplitude_1 * np.sin(2 * np.pi * 50 * t) + amplitude_2 * np.sin(2 * np.pi * 200 * t)

    freqs, psd = compute_welch_psd(signal, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    energy_band_1 = band_energy(freqs, psd, 40.0, 60.0)
    energy_band_2 = band_energy(freqs, psd, 190.0, 210.0)

    assert energy_band_1 == pytest.approx(amplitude_1**2 / 2, rel=0.02)
    assert energy_band_2 == pytest.approx(amplitude_2**2 / 2, rel=0.02)


def test_band_energy_of_band_with_no_signal_content_is_near_zero() -> None:
    n = N
    t = np.arange(n) / FS
    signal = 2.0 * np.sin(2 * np.pi * 50 * t) + 3.0 * np.sin(2 * np.pi * 200 * t)

    freqs, psd = compute_welch_psd(signal, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    energy_empty_band = band_energy(freqs, psd, 300.0, 320.0)

    assert energy_empty_band == pytest.approx(0.0, abs=1e-6)


def test_band_energy_low_freq_negative_raises() -> None:
    freqs, psd = _sine_psd(100.0)
    with pytest.raises(FrequencyFeatureError, match="low_freq"):
        band_energy(freqs, psd, -10.0, 50.0)


def test_band_energy_high_freq_not_greater_than_low_freq_raises() -> None:
    freqs, psd = _sine_psd(100.0)
    with pytest.raises(FrequencyFeatureError, match="high_freq"):
        band_energy(freqs, psd, 100.0, 100.0)
    with pytest.raises(FrequencyFeatureError, match="high_freq"):
        band_energy(freqs, psd, 100.0, 50.0)


def test_band_energy_high_freq_above_nyquist_bound_raises() -> None:
    freqs, psd = _sine_psd(100.0)
    max_freq = float(freqs.max())
    with pytest.raises(FrequencyFeatureError, match="Nyquist"):
        band_energy(freqs, psd, 0.0, max_freq + 100.0)


def test_band_energy_returns_zero_for_band_narrower_than_one_bin() -> None:
    """A band landing strictly between two adjacent bins (fewer than 2 bins
    selected) is defined to return 0.0 -- an honest 'no measured energy here'
    answer, not an error."""
    freqs = np.array([0.0, 10.0, 20.0, 30.0])
    power = np.array([1.0, 1.0, 1.0, 1.0])
    # Band [11, 19] contains no bin at all (10 and 20 are excluded).
    assert band_energy(freqs, power, 11.0, 19.0) == pytest.approx(0.0)


# --- shared validation across all five power-based features ---


@pytest.mark.parametrize(
    "func",
    [
        spectral_centroid,
        spectral_bandwidth,
        spectral_energy,
        spectral_entropy,
    ],
)
def test_power_based_features_raise_explicitly_for_empty_input(func) -> None:
    with pytest.raises(FrequencyFeatureError, match="empty"):
        func(np.array([]), np.array([]))


@pytest.mark.parametrize(
    "func",
    [
        spectral_centroid,
        spectral_bandwidth,
        spectral_energy,
        spectral_entropy,
    ],
)
def test_power_based_features_raise_explicitly_for_mismatched_lengths(func) -> None:
    with pytest.raises(FrequencyFeatureError, match="shape"):
        func(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0]))


@pytest.mark.parametrize(
    "func",
    [
        spectral_centroid,
        spectral_bandwidth,
        spectral_energy,
        spectral_entropy,
    ],
)
def test_power_based_features_reject_negative_power(func) -> None:
    with pytest.raises(FrequencyFeatureError, match="non-negative"):
        func(np.array([0.0, 1.0, 2.0]), np.array([1.0, -1.0, 2.0]))


# --- input immutability ---


def test_frequency_domain_features_do_not_modify_input_arrays() -> None:
    freqs, psd = _sine_psd(100.0)
    freqs_snapshot = freqs.copy()
    psd_snapshot = psd.copy()

    dominant_frequency(freqs, psd)
    spectral_centroid(freqs, psd)
    spectral_bandwidth(freqs, psd)
    spectral_energy(freqs, psd)
    spectral_entropy(freqs, psd)
    band_energy(freqs, psd, 90.0, 110.0)

    assert np.array_equal(freqs, freqs_snapshot)
    assert np.array_equal(psd, psd_snapshot)
