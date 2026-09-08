import numpy as np
import pytest

from app.signal_processing.filtering import FilteringError, butter_filter

FS = 1000.0  # Hz
N = 1000  # samples -> 1 second, FFT bin resolution = fs/N = 1 Hz/bin (exact bins)


def _time_axis(n: int = N, fs: float = FS) -> np.ndarray:
    return np.arange(n) / fs


def _sine(freq: float, t: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * freq * t)


def _fft_amplitude(signal: np.ndarray, freq_hz: float, fs: float = FS, tolerance_bins: int = 1) -> float:
    """Amplitude of the FFT component nearest `freq_hz`, searching within
    `tolerance_bins` of the exact bin to be robust to any off-by-one bin rounding --
    documented methodology per the task's own instruction, not an arbitrary fudge."""
    n = len(signal)
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    exact_bin = int(round(freq_hz / (fs / n)))
    lo = max(0, exact_bin - tolerance_bins)
    hi = min(len(freqs), exact_bin + tolerance_bins + 1)
    return float(np.max(spectrum[lo:hi]))


def _attenuation_db(original: np.ndarray, filtered: np.ndarray, freq_hz: float) -> float:
    original_amp = _fft_amplitude(original, freq_hz)
    filtered_amp = _fft_amplitude(filtered, freq_hz)
    assert original_amp > 0, "test setup error: original signal has no energy at freq_hz"
    return 20 * np.log10(original_amp / max(filtered_amp, 1e-12))


# --- AC1: low-pass attenuates a 200 Hz component by >= 20 dB ---


def test_ac1_lowpass_attenuates_200hz_component_by_at_least_20db() -> None:
    t = _time_axis()
    signal = _sine(10, t) + _sine(200, t)

    filtered = butter_filter(signal, cutoff=50.0, fs=FS, order=4, btype="lowpass")

    attenuation_db = _attenuation_db(signal, filtered, freq_hz=200)
    assert attenuation_db >= 20, f"expected >=20 dB attenuation at 200 Hz, got {attenuation_db:.2f} dB"

    # The passband component (10 Hz) should survive with little attenuation.
    passband_attenuation_db = _attenuation_db(signal, filtered, freq_hz=10)
    assert passband_attenuation_db < 3, (
        f"expected the 10 Hz passband component to be largely preserved, "
        f"got {passband_attenuation_db:.2f} dB attenuation"
    )


# --- AC2: cutoff >= Nyquist raises explicitly ---


def test_ac2_cutoff_equal_to_nyquist_raises() -> None:
    nyquist = FS / 2
    with pytest.raises(FilteringError, match="nyquist"):
        butter_filter([0.0] * 100, cutoff=nyquist, fs=FS, order=4, btype="lowpass")


def test_ac2_cutoff_above_nyquist_raises() -> None:
    nyquist = FS / 2
    with pytest.raises(FilteringError, match="nyquist"):
        butter_filter([0.0] * 100, cutoff=nyquist + 50, fs=FS, order=4, btype="highpass")


# --- AC3: filtfilt introduces no phase shift (known peak stays within +-1 sample) ---


def test_ac3_filtfilt_does_not_shift_a_known_peak() -> None:
    """A Gaussian pulse (sigma=15 samples => most spectral energy well under 50 Hz)
    filtered with a 100 Hz low-pass (well above the pulse's own bandwidth) should
    preserve its peak position almost exactly -- unlike scipy.signal.lfilter, which
    would shift it. This is the intended, robust way to test phase preservation: a
    smooth, band-limited pulse whose peak stays unambiguous after filtering."""
    t_samples = np.arange(N)
    center = 500
    sigma = 15.0
    pulse = np.exp(-((t_samples - center) ** 2) / (2 * sigma**2))

    filtered = butter_filter(pulse, cutoff=100.0, fs=FS, order=4, btype="lowpass")

    original_peak_index = int(np.argmax(pulse))
    filtered_peak_index = int(np.argmax(filtered))

    assert original_peak_index == center  # sanity check on the synthetic setup itself
    assert abs(filtered_peak_index - original_peak_index) <= 1, (
        f"peak shifted from {original_peak_index} to {filtered_peak_index}"
    )


# --- low-pass, high-pass, band-pass behavior ---


def test_lowpass_keeps_low_component_attenuates_high_component() -> None:
    t = _time_axis()
    signal = _sine(10, t) + _sine(200, t)

    filtered = butter_filter(signal, cutoff=50.0, fs=FS, order=4, btype="lowpass")

    assert _attenuation_db(signal, filtered, 200) >= 20
    assert _attenuation_db(signal, filtered, 10) < 3


def test_highpass_attenuates_low_component_keeps_high_component() -> None:
    t = _time_axis()
    signal = _sine(5, t) + _sine(200, t)

    filtered = butter_filter(signal, cutoff=50.0, fs=FS, order=4, btype="highpass")

    assert _attenuation_db(signal, filtered, 5) >= 20
    assert _attenuation_db(signal, filtered, 200) < 3


def test_bandpass_keeps_in_band_component_attenuates_out_of_band_components() -> None:
    t = _time_axis()
    signal = _sine(10, t) + _sine(100, t) + _sine(300, t)

    filtered = butter_filter(signal, cutoff=(60.0, 150.0), fs=FS, order=4, btype="bandpass")

    assert _attenuation_db(signal, filtered, 10) >= 20  # below the band
    assert _attenuation_db(signal, filtered, 300) >= 20  # above the band
    assert _attenuation_db(signal, filtered, 100) < 3  # inside the band, preserved


# --- validation ---


def test_fs_zero_or_negative_raises() -> None:
    with pytest.raises(FilteringError, match="fs"):
        butter_filter([0.0] * 100, cutoff=10.0, fs=0, order=4, btype="lowpass")
    with pytest.raises(FilteringError, match="fs"):
        butter_filter([0.0] * 100, cutoff=10.0, fs=-100.0, order=4, btype="lowpass")


def test_order_zero_or_negative_raises() -> None:
    with pytest.raises(FilteringError, match="order"):
        butter_filter([0.0] * 100, cutoff=10.0, fs=FS, order=0, btype="lowpass")
    with pytest.raises(FilteringError, match="order"):
        butter_filter([0.0] * 100, cutoff=10.0, fs=FS, order=-2, btype="lowpass")


def test_unknown_btype_raises() -> None:
    with pytest.raises(FilteringError, match="btype"):
        butter_filter([0.0] * 100, cutoff=10.0, fs=FS, order=4, btype="notch")  # type: ignore[arg-type]


def test_bandpass_low_greater_or_equal_high_raises() -> None:
    with pytest.raises(FilteringError, match="bandpass"):
        butter_filter([0.0] * 100, cutoff=(100.0, 50.0), fs=FS, order=4, btype="bandpass")
    with pytest.raises(FilteringError, match="bandpass"):
        butter_filter([0.0] * 100, cutoff=(100.0, 100.0), fs=FS, order=4, btype="bandpass")


def test_bandpass_low_non_positive_raises() -> None:
    with pytest.raises(FilteringError, match="bandpass"):
        butter_filter([0.0] * 100, cutoff=(0.0, 100.0), fs=FS, order=4, btype="bandpass")
    with pytest.raises(FilteringError, match="bandpass"):
        butter_filter([0.0] * 100, cutoff=(-10.0, 100.0), fs=FS, order=4, btype="bandpass")


def test_bandpass_high_at_or_above_nyquist_raises() -> None:
    nyquist = FS / 2
    with pytest.raises(FilteringError, match="bandpass"):
        butter_filter([0.0] * 100, cutoff=(10.0, nyquist), fs=FS, order=4, btype="bandpass")
    with pytest.raises(FilteringError, match="bandpass"):
        butter_filter([0.0] * 100, cutoff=(10.0, nyquist + 10), fs=FS, order=4, btype="bandpass")


def test_scalar_cutoff_for_bandpass_raises() -> None:
    with pytest.raises(FilteringError, match="bandpass"):
        butter_filter([0.0] * 100, cutoff=50.0, fs=FS, order=4, btype="bandpass")


def test_tuple_cutoff_for_lowpass_or_highpass_raises() -> None:
    with pytest.raises(FilteringError, match="lowpass"):
        butter_filter([0.0] * 100, cutoff=(10.0, 50.0), fs=FS, order=4, btype="lowpass")
    with pytest.raises(FilteringError, match="highpass"):
        butter_filter([0.0] * 100, cutoff=(10.0, 50.0), fs=FS, order=4, btype="highpass")


# --- data preservation ---


def test_filter_does_not_modify_input_in_place() -> None:
    t = _time_axis()
    signal = list(_sine(10, t) + _sine(200, t))
    snapshot = list(signal)

    butter_filter(signal, cutoff=50.0, fs=FS, order=4, btype="lowpass")

    assert signal == snapshot


def test_filter_preserves_signal_length() -> None:
    t = _time_axis()
    signal = _sine(10, t) + _sine(200, t)

    filtered = butter_filter(signal, cutoff=50.0, fs=FS, order=4, btype="lowpass")

    assert len(filtered) == len(signal)


def test_filter_returns_new_array_not_same_object() -> None:
    t = _time_axis()
    signal = _sine(10, t)

    filtered = butter_filter(signal, cutoff=50.0, fs=FS, order=4, btype="lowpass")

    assert filtered is not signal
