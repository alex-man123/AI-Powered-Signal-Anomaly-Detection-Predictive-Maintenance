import numpy as np
import pytest

from app.signal_processing.fft import FFTError, compute_fft, dominant_frequency

FS = 1000.0  # Hz
N = 1000  # samples -> Delta_f = fs/N = 1 Hz/bin; 50 Hz falls exactly on bin 50


def _sine(freq: float, n: int = N, fs: float = FS, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(n) / fs
    return amplitude * np.sin(2 * np.pi * freq * t)


# --- AC1: 50 Hz synthetic sinusoid -> dominant frequency within +-1 Hz ---


def test_ac1_dominant_frequency_of_50hz_sine_is_within_1hz() -> None:
    signal = _sine(50)

    freqs, magnitude = compute_fft(signal, FS)
    detected = dominant_frequency(freqs, magnitude)

    assert abs(detected - 50) <= 1


# --- AC2: max frequency returned never exceeds Nyquist ---


def test_ac2_max_frequency_does_not_exceed_nyquist() -> None:
    signal = _sine(50)

    freqs, _magnitude = compute_fft(signal, FS)

    nyquist = FS / 2
    assert freqs.max() <= nyquist


def test_ac2_rfftfreq_reaches_nyquist_for_even_n() -> None:
    """For even N, numpy.fft.rfftfreq's last bin lands exactly on Nyquist -- verified
    against the real function output, not asserted from a hand-built axis."""
    signal = _sine(50, n=1000)

    freqs, _magnitude = compute_fft(signal, FS)

    assert freqs[-1] == pytest.approx(FS / 2)


# --- shape / length consistency ---


def test_frequencies_and_magnitude_have_equal_length() -> None:
    signal = _sine(50)

    freqs, magnitude = compute_fft(signal, FS)

    assert len(freqs) == len(magnitude)


def test_output_length_matches_rfft_convention_for_even_n() -> None:
    signal = _sine(50, n=1000)

    freqs, magnitude = compute_fft(signal, FS)

    assert len(freqs) == 1000 // 2 + 1
    assert len(magnitude) == 1000 // 2 + 1


def test_output_length_matches_rfft_convention_for_odd_n() -> None:
    n = 999
    signal = _sine(50, n=n)

    freqs, magnitude = compute_fft(signal, FS)

    assert len(freqs) == n // 2 + 1
    assert len(magnitude) == n // 2 + 1


def test_no_negative_frequencies_are_returned() -> None:
    signal = _sine(50)

    freqs, _magnitude = compute_fft(signal, FS)

    assert np.all(freqs >= 0)


# --- magnitude is real, non-negative, matching frequencies ---


def test_magnitude_is_real_and_non_negative() -> None:
    signal = _sine(50)

    _freqs, magnitude = compute_fft(signal, FS)

    assert not np.iscomplexobj(magnitude)
    assert np.all(magnitude >= 0)


# --- DC / constant signal ---


def test_constant_signal_has_dominant_frequency_at_dc() -> None:
    signal = np.full(N, 3.0)

    freqs, magnitude = compute_fft(signal, FS)
    detected = dominant_frequency(freqs, magnitude)

    assert detected == pytest.approx(0.0)


# --- dominant_frequency correctness on a known spectrum ---


def test_dominant_frequency_returns_frequency_of_largest_magnitude_bin() -> None:
    # A 120 Hz component with larger amplitude than a 50 Hz component -> 120 Hz wins.
    signal = _sine(50, amplitude=1.0) + _sine(120, amplitude=5.0)

    freqs, magnitude = compute_fft(signal, FS)
    detected = dominant_frequency(freqs, magnitude)

    assert abs(detected - 120) <= 1


def test_dominant_frequency_rejects_mismatched_lengths() -> None:
    with pytest.raises(FFTError, match="same length"):
        dominant_frequency(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0]))


def test_dominant_frequency_rejects_empty_input() -> None:
    with pytest.raises(FFTError, match="empty"):
        dominant_frequency(np.array([]), np.array([]))


# --- input immutability ---


def test_compute_fft_does_not_modify_input_signal() -> None:
    signal = list(_sine(50))
    snapshot = list(signal)

    compute_fft(signal, FS)

    assert signal == snapshot


# --- validation ---


def test_fs_zero_or_negative_raises() -> None:
    with pytest.raises(FFTError, match="fs"):
        compute_fft(_sine(50), 0)
    with pytest.raises(FFTError, match="fs"):
        compute_fft(_sine(50), -1000.0)


def test_empty_signal_raises() -> None:
    with pytest.raises(FFTError, match="empty"):
        compute_fft([], FS)
