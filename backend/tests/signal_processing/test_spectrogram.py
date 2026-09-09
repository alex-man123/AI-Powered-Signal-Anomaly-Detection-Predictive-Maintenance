import numpy as np
import pytest

from app.signal_processing.spectrogram import SpectrogramError, compute_spectrogram

FS = 1000.0  # Hz
N = 2000  # samples -> 2 s


def _chirp_signal(low_freq: float = 50.0, high_freq: float = 150.0, n: int = N, fs: float = FS) -> np.ndarray:
    """First half at `low_freq`, second half at `high_freq`, split at the midpoint
    (t=1.0 s for n=2000, fs=1000) -- deterministic, no randomness."""
    t = np.arange(n) / fs
    midpoint = t[n // 2]
    return np.where(t < midpoint, np.sin(2 * np.pi * low_freq * t), np.sin(2 * np.pi * high_freq * t))


# --- Test 1: shape consistency ---


def test_sxx_shape_matches_frequency_and_time_axis_lengths() -> None:
    signal = _chirp_signal()

    freqs, times, sxx = compute_spectrogram(signal, FS, window_size=256, hop_length=128)

    assert sxx.shape == (len(freqs), len(times))


# --- Test 2 (AC1): dominant frequency shifts from 50 Hz to 150 Hz ---


def test_ac1_dominant_frequency_shifts_from_50hz_to_150hz() -> None:
    """Boundary-column-safe methodology: a segment is only used as "early" if its
    entire time window ends before the 50->150 Hz transition, and "late" only if its
    entire window starts after it -- computed from window_size/fs and the real
    `times` axis, not by assuming an even column split. Columns whose window straddles
    the transition are excluded entirely, per the task's explicit instruction."""
    window_size = 256
    hop_length = 128
    signal = _chirp_signal()
    transition_time = 1.0  # seconds, for N=2000/fs=1000

    freqs, times, sxx = compute_spectrogram(signal, FS, window_size=window_size, hop_length=hop_length)

    half_width = (window_size / FS) / 2
    early_mask = times + half_width <= transition_time
    late_mask = times - half_width >= transition_time

    assert early_mask.sum() >= 3, "test setup error: not enough clean early columns"
    assert late_mask.sum() >= 3, "test setup error: not enough clean late columns"

    dominant_per_column = freqs[np.argmax(sxx, axis=0)]

    early_dominant = dominant_per_column[early_mask]
    late_dominant = dominant_per_column[late_mask]

    # Delta_f = fs/window_size ~= 3.9 Hz here; allow a couple of bins of tolerance.
    tolerance_hz = 2 * (FS / window_size)
    assert np.all(np.abs(early_dominant - 50) <= tolerance_hz), early_dominant
    assert np.all(np.abs(late_dominant - 150) <= tolerance_hz), late_dominant


# --- Test 3: frequency axis ---


def test_frequency_axis_is_non_negative_and_bounded_by_nyquist() -> None:
    signal = _chirp_signal()

    freqs, _times, _sxx = compute_spectrogram(signal, FS, window_size=256, hop_length=128)

    assert np.all(freqs >= 0)
    assert freqs.max() <= FS / 2


# --- Test 4: time axis ---


def test_time_axis_is_strictly_increasing() -> None:
    signal = _chirp_signal()

    _freqs, times, _sxx = compute_spectrogram(signal, FS, window_size=256, hop_length=128)

    assert np.all(np.diff(times) > 0)


# --- Test 5: hop_length changes the number of time columns as expected ---


def test_smaller_hop_length_produces_more_time_columns() -> None:
    signal = _chirp_signal()

    _f1, times_hop_large, _s1 = compute_spectrogram(signal, FS, window_size=256, hop_length=128)
    _f2, times_hop_small, _s2 = compute_spectrogram(signal, FS, window_size=256, hop_length=32)

    # A smaller hop_length means more (closer-together) analysis positions.
    assert len(times_hop_small) > len(times_hop_large)


# --- Test 6 (AC2): window size trade-off — frequency resolution vs. time resolution ---


def test_ac2_larger_window_improves_frequency_resolution_and_reduces_time_resolution() -> None:
    """Same signal, same hop_length, two window sizes. Frequency resolution measured
    directly from the real returned frequency axis spacing (freqs[1]-freqs[0]), not
    just the Delta_f=fs/window_size formula in the abstract. Temporal resolution
    measured as the number of time bins produced for the same signal duration."""
    signal = _chirp_signal(low_freq=100.0, high_freq=100.0)  # stationary; only resolution matters here
    hop_length = 64
    small_window = 128
    large_window = 512

    freqs_small, times_small, _sxx_small = compute_spectrogram(
        signal, FS, window_size=small_window, hop_length=hop_length
    )
    freqs_large, times_large, _sxx_large = compute_spectrogram(
        signal, FS, window_size=large_window, hop_length=hop_length
    )

    delta_f_small = freqs_small[1] - freqs_small[0]
    delta_f_large = freqs_large[1] - freqs_large[0]

    print(
        f"\nsmall window={small_window}: delta_f={delta_f_small:.4f} Hz, n_times={len(times_small)}\n"
        f"large window={large_window}: delta_f={delta_f_large:.4f} Hz, n_times={len(times_large)}"
    )

    # Frequency resolution: larger window -> smaller (better) bin spacing.
    assert delta_f_large < delta_f_small
    # Matches the theoretical Delta_f = fs / window_size relationship.
    assert delta_f_small == pytest.approx(FS / small_window)
    assert delta_f_large == pytest.approx(FS / large_window)

    # Time resolution: larger window -> fewer analysis positions for the same signal.
    assert len(times_large) < len(times_small)


# --- validation ---


def test_fs_zero_or_negative_raises() -> None:
    signal = _chirp_signal()
    with pytest.raises(SpectrogramError, match="fs"):
        compute_spectrogram(signal, 0, window_size=256, hop_length=128)
    with pytest.raises(SpectrogramError, match="fs"):
        compute_spectrogram(signal, -100.0, window_size=256, hop_length=128)


def test_window_size_zero_or_negative_raises() -> None:
    signal = _chirp_signal()
    with pytest.raises(SpectrogramError, match="window_size"):
        compute_spectrogram(signal, FS, window_size=0, hop_length=1)
    with pytest.raises(SpectrogramError, match="window_size"):
        compute_spectrogram(signal, FS, window_size=-10, hop_length=1)


def test_window_size_greater_than_signal_length_raises() -> None:
    short_signal = [1.0, 2.0, 3.0]
    with pytest.raises(SpectrogramError, match="window_size"):
        compute_spectrogram(short_signal, FS, window_size=256, hop_length=64)


def test_hop_length_zero_or_negative_raises() -> None:
    signal = _chirp_signal()
    with pytest.raises(SpectrogramError, match="hop_length"):
        compute_spectrogram(signal, FS, window_size=256, hop_length=0)
    with pytest.raises(SpectrogramError, match="hop_length"):
        compute_spectrogram(signal, FS, window_size=256, hop_length=-5)


def test_hop_length_greater_than_window_size_raises() -> None:
    signal = _chirp_signal()
    with pytest.raises(SpectrogramError, match="hop_length"):
        compute_spectrogram(signal, FS, window_size=256, hop_length=300)


def test_empty_signal_raises() -> None:
    with pytest.raises(SpectrogramError, match="empty"):
        compute_spectrogram([], FS, window_size=256, hop_length=128)


# --- input immutability ---


def test_compute_spectrogram_does_not_modify_input_signal() -> None:
    signal = list(_chirp_signal())
    snapshot = list(signal)

    compute_spectrogram(signal, FS, window_size=256, hop_length=128)

    assert signal == snapshot
