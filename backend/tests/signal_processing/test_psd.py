import numpy as np
import pytest

from app.signal_processing.fft import compute_fft
from app.signal_processing.psd import PSDError, compute_welch_psd

FS = 1000.0  # Hz
N = 2000  # samples -> 2 s, enough for multiple Welch segments at nperseg=256
NPERSEG = 256
NOVERLAP = 128


def _noisy_50hz_signal(n: int = N, fs: float = FS, noise_std: float = 0.5) -> np.ndarray:
    """Deterministic (fixed seed) 50 Hz sinusoid + white noise -- same signal used for
    both the FFT and Welch comparison, per the task's own instruction not to generate
    two different signals for the two methods."""
    t = np.arange(n) / fs
    clean = np.sin(2 * np.pi * 50 * t)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, noise_std, n)
    return clean + noise


def _coefficient_of_variation(freqs: np.ndarray, values: np.ndarray, lo: float, hi: float) -> float:
    """std/mean of the spectrum values within [lo, hi] Hz -- a scale-independent
    variability metric, needed because FFT magnitude and Welch PSD have different
    absolute scales and cannot be compared as raw values. The [100, 400] Hz region
    used by the tests below deliberately excludes the 50 Hz signal component (and a
    safety margin around it) and the Nyquist edge, so it measures noise-floor
    variability only, not variability caused by the signal itself."""
    mask = (freqs >= lo) & (freqs <= hi)
    region = values[mask]
    assert region.size > 5, "test setup error: too few bins in the comparison region"
    return float(np.std(region) / np.mean(region))


# --- AC1: dominant frequency at 50 Hz +-1 Hz from noisy signal ---


def test_ac1_welch_dominant_frequency_of_noisy_50hz_signal_within_1hz() -> None:
    signal = _noisy_50hz_signal()

    freqs, psd = compute_welch_psd(signal, FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    dominant = freqs[int(np.argmax(psd))]

    assert abs(dominant - 50) <= 1


# --- AC1: Welch spectral variability is visibly lower than simple FFT's ---


def test_ac1_welch_has_lower_spectral_variability_than_simple_fft() -> None:
    """Explicit, reproducible methodology (same noisy signal for both):
    1. Compute simple FFT magnitude (TASK 4.1's compute_fft) and Welch PSD for the
       identical noisy signal.
    2. Restrict both to the same [100, 400] Hz region -- away from the 50 Hz signal
       peak (so the comparison measures noise variability, not signal content) and
       away from the Nyquist edge.
    3. Compute the coefficient of variation (std/mean) of each spectrum's values in
       that region -- scale-independent, so FFT magnitude and PSD (different units)
       can be compared fairly.
    4. Welch's averaging over overlapping segments is expected to produce a visibly
       smaller coefficient of variation than the single-shot FFT.
    """
    signal = _noisy_50hz_signal()

    freqs_fft, magnitude = compute_fft(signal, FS)
    freqs_psd, psd = compute_welch_psd(signal, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    cv_fft = _coefficient_of_variation(freqs_fft, magnitude, lo=100, hi=400)
    cv_welch = _coefficient_of_variation(freqs_psd, psd, lo=100, hi=400)

    print(f"\ncv_fft={cv_fft:.4f} cv_welch={cv_welch:.4f} ratio={cv_welch / cv_fft:.3f}")

    assert cv_welch < cv_fft
    # Not just marginally smaller -- a comfortable, non-tight margin (empirically the
    # measured ratio for this fixed-seed signal is ~0.49; 0.7 leaves generous room
    # without being a suspiciously precise/tuned threshold).
    assert cv_welch < 0.7 * cv_fft


# --- Nyquist ---


def test_max_frequency_does_not_exceed_nyquist() -> None:
    signal = _noisy_50hz_signal()

    freqs, _psd = compute_welch_psd(signal, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    assert freqs.max() <= FS / 2


def test_frequencies_and_psd_have_equal_length() -> None:
    signal = _noisy_50hz_signal()

    freqs, psd = compute_welch_psd(signal, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    assert len(freqs) == len(psd)


def test_no_negative_frequencies_returned() -> None:
    signal = _noisy_50hz_signal()

    freqs, _psd = compute_welch_psd(signal, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    assert np.all(freqs >= 0)


# --- PSD non-negative ---


def test_psd_is_non_negative() -> None:
    signal = _noisy_50hz_signal()

    _freqs, psd = compute_welch_psd(signal, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    assert np.all(psd >= 0)


# --- input immutability ---


def test_compute_welch_psd_does_not_modify_input_signal() -> None:
    signal = list(_noisy_50hz_signal())
    snapshot = list(signal)

    compute_welch_psd(signal, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    assert signal == snapshot


# --- validation: fs ---


def test_fs_zero_or_negative_raises() -> None:
    signal = _noisy_50hz_signal()
    with pytest.raises(PSDError, match="fs"):
        compute_welch_psd(signal, 0, nperseg=NPERSEG, noverlap=NOVERLAP)
    with pytest.raises(PSDError, match="fs"):
        compute_welch_psd(signal, -1000.0, nperseg=NPERSEG, noverlap=NOVERLAP)


# --- validation: nperseg ---


def test_nperseg_zero_or_negative_raises() -> None:
    signal = _noisy_50hz_signal()
    with pytest.raises(PSDError, match="nperseg"):
        compute_welch_psd(signal, FS, nperseg=0, noverlap=0)
    with pytest.raises(PSDError, match="nperseg"):
        compute_welch_psd(signal, FS, nperseg=-10, noverlap=0)


def test_nperseg_greater_than_signal_length_raises() -> None:
    short_signal = [1.0, 2.0, 3.0]
    with pytest.raises(PSDError, match="nperseg"):
        compute_welch_psd(short_signal, FS, nperseg=256, noverlap=0)


# --- validation: noverlap ---


def test_noverlap_negative_raises() -> None:
    signal = _noisy_50hz_signal()
    with pytest.raises(PSDError, match="noverlap"):
        compute_welch_psd(signal, FS, nperseg=NPERSEG, noverlap=-1)


def test_noverlap_equal_to_nperseg_raises() -> None:
    signal = _noisy_50hz_signal()
    with pytest.raises(PSDError, match="noverlap"):
        compute_welch_psd(signal, FS, nperseg=NPERSEG, noverlap=NPERSEG)


def test_noverlap_greater_than_nperseg_raises() -> None:
    signal = _noisy_50hz_signal()
    with pytest.raises(PSDError, match="noverlap"):
        compute_welch_psd(signal, FS, nperseg=NPERSEG, noverlap=NPERSEG + 50)


# --- validation: empty signal ---


def test_empty_signal_raises() -> None:
    with pytest.raises(PSDError, match="empty"):
        compute_welch_psd([], FS, nperseg=NPERSEG, noverlap=NOVERLAP)
