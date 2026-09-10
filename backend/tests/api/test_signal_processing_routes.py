"""TASK 10.3 -- tests for `POST /api/fft`, `/api/psd`, `/api/spectrogram`,
`/api/dsp/filter`.

Uses the real `app.main.app` through `TestClient` (HTTP -> Pydantic schema ->
`dsp_service` -> Phase 3/4 DSP implementation -> response schema -> JSON) --
never a mock of the DSP layer. The CRITICAL reuse requirement is proven by
monkeypatching `app.services.dsp_service`'s imported references to the real
Phase 3/4 functions with a spy and confirming the route actually calls them
(not a second, parallel implementation) -- on top of behavioral assertions
that already match Phase 3/4's own known properties (dominant frequency of a
known sine, cutoff/Nyquist rejection, filtered-signal length).
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import dsp_service

client = TestClient(app)

FS = 1000.0  # Hz
N = 1000  # samples -> Delta_f = fs/N = 1 Hz/bin


def _sine(freq: float, n: int = N, fs: float = FS, amplitude: float = 1.0) -> list[float]:
    t = np.arange(n) / fs
    return (amplitude * np.sin(2 * np.pi * freq * t)).tolist()


# ---------------------------------------------------------------------------
# FFT
# ---------------------------------------------------------------------------


def test_fft_valid_request_returns_200_and_matches_response_schema() -> None:
    response = client.post("/api/fft", json={"signal": _sine(50), "sampling_rate": FS})

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"frequencies", "magnitude", "dominant_frequency"}
    assert len(body["frequencies"]) == len(body["magnitude"])


def test_fft_dominant_frequency_of_50hz_sine_is_within_1hz() -> None:
    """Same property already verified in Phase 4.1's own tests -- proves the
    route reaches the real `compute_fft`/`dominant_frequency`, not a stub."""
    response = client.post("/api/fft", json={"signal": _sine(50), "sampling_rate": FS})

    assert abs(response.json()["dominant_frequency"] - 50) <= 1


def test_fft_frequencies_never_exceed_nyquist() -> None:
    response = client.post("/api/fft", json={"signal": _sine(50), "sampling_rate": FS})

    assert max(response.json()["frequencies"]) <= FS / 2


def test_fft_rejects_sampling_rate_as_string_with_422() -> None:
    response = client.post("/api/fft", json={"signal": _sine(50), "sampling_rate": "1000"})

    assert response.status_code == 422


def test_fft_rejects_empty_signal_with_422() -> None:
    response = client.post("/api/fft", json={"signal": [], "sampling_rate": FS})

    assert response.status_code == 422


def test_fft_rejects_non_positive_sampling_rate_with_422() -> None:
    response = client.post("/api/fft", json={"signal": _sine(50), "sampling_rate": -1.0})

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# PSD
# ---------------------------------------------------------------------------


def test_psd_valid_request_returns_200_and_matches_response_schema() -> None:
    response = client.post(
        "/api/psd", json={"signal": _sine(50), "sampling_rate": FS, "nperseg": 256, "noverlap": 128}
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"frequencies", "psd"}
    assert len(body["frequencies"]) == len(body["psd"])


def test_psd_peak_is_near_the_known_synthetic_frequency() -> None:
    """Reuses the same synthetic-signal property Phase 4.2 already verifies:
    the Welch PSD's peak frequency should land near the injected 50 Hz tone."""
    response = client.post(
        "/api/psd", json={"signal": _sine(50), "sampling_rate": FS, "nperseg": 256, "noverlap": 128}
    )
    body = response.json()

    peak_index = int(np.argmax(body["psd"]))
    assert abs(body["frequencies"][peak_index] - 50) <= 5


def test_psd_rejects_noverlap_greater_or_equal_to_nperseg_with_422() -> None:
    response = client.post(
        "/api/psd", json={"signal": _sine(50), "sampling_rate": FS, "nperseg": 256, "noverlap": 256}
    )

    assert response.status_code == 422


def test_psd_rejects_nperseg_as_string_with_422() -> None:
    response = client.post(
        "/api/psd", json={"signal": _sine(50), "sampling_rate": FS, "nperseg": "256", "noverlap": 128}
    )

    assert response.status_code == 422


def test_psd_rejects_nperseg_larger_than_signal_length_with_422() -> None:
    """A request-shape-valid but DSP-invalid parameter (Pydantic cannot check
    this without knowing len(signal)) -- must be caught by the real PSDError
    and surfaced as 422, not 500."""
    response = client.post(
        "/api/psd", json={"signal": _sine(50, n=10), "sampling_rate": FS, "nperseg": 256, "noverlap": 128}
    )

    assert response.status_code == 422
    assert "nperseg" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Spectrogram
# ---------------------------------------------------------------------------


def test_spectrogram_valid_request_returns_200_and_matches_response_schema() -> None:
    response = client.post(
        "/api/spectrogram",
        json={"signal": _sine(50), "sampling_rate": FS, "window_size": 256, "hop_length": 128},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"frequencies", "times", "values"}
    assert len(body["values"]) == len(body["frequencies"])
    assert all(len(row) == len(body["times"]) for row in body["values"])


def test_spectrogram_times_are_strictly_increasing() -> None:
    response = client.post(
        "/api/spectrogram",
        json={"signal": _sine(50), "sampling_rate": FS, "window_size": 256, "hop_length": 128},
    )
    times = response.json()["times"]

    assert all(earlier < later for earlier, later in zip(times, times[1:]))


def test_spectrogram_rejects_hop_length_greater_than_window_size_with_422() -> None:
    response = client.post(
        "/api/spectrogram",
        json={"signal": _sine(50), "sampling_rate": FS, "window_size": 128, "hop_length": 256},
    )

    assert response.status_code == 422


def test_spectrogram_rejects_window_size_larger_than_signal_with_422() -> None:
    response = client.post(
        "/api/spectrogram",
        json={"signal": _sine(50, n=10), "sampling_rate": FS, "window_size": 256, "hop_length": 128},
    )

    assert response.status_code == 422
    assert "window_size" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


def test_filter_valid_request_returns_200_with_same_length_output() -> None:
    signal = _sine(50)
    response = client.post(
        "/api/dsp/filter",
        json={"signal": signal, "sampling_rate": FS, "cutoff": 100.0, "order": 4, "btype": "lowpass"},
    )

    assert response.status_code == 200
    assert len(response.json()["filtered_signal"]) == len(signal)


def test_filter_lowpass_attenuates_a_high_frequency_component() -> None:
    """Same property Phase 3.2 already verifies (a low-pass filter should
    strongly attenuate a component above cutoff) -- confirms the real
    `butter_filter` ran, not a passthrough."""
    signal = (np.array(_sine(20)) + np.array(_sine(300, amplitude=1.0))).tolist()

    response = client.post(
        "/api/dsp/filter",
        json={"signal": signal, "sampling_rate": FS, "cutoff": 50.0, "order": 4, "btype": "lowpass"},
    )
    filtered = np.array(response.json()["filtered_signal"])

    freqs, magnitude = np.fft.rfftfreq(len(filtered), d=1 / FS), np.abs(np.fft.rfft(filtered))
    mag_at_300hz = magnitude[np.argmin(np.abs(freqs - 300))]
    mag_at_20hz = magnitude[np.argmin(np.abs(freqs - 20))]
    assert mag_at_300hz < 0.1 * mag_at_20hz


def test_filter_zero_phase_does_not_shift_a_known_peak() -> None:
    """Same zero-phase property already verified in Phase 3.2's own tests
    (`filtfilt`, never `lfilter`)."""
    signal = np.zeros(N)
    peak_index = 500
    signal[peak_index] = 10.0

    response = client.post(
        "/api/dsp/filter",
        json={"signal": signal.tolist(), "sampling_rate": FS, "cutoff": 100.0, "order": 4, "btype": "lowpass"},
    )
    filtered = np.array(response.json()["filtered_signal"])

    detected_peak_index = int(np.argmax(np.abs(filtered)))
    assert abs(detected_peak_index - peak_index) <= 2


def test_filter_bandpass_accepts_a_cutoff_pair() -> None:
    signal = _sine(50)
    response = client.post(
        "/api/dsp/filter",
        json={"signal": signal, "sampling_rate": FS, "cutoff": [10.0, 100.0], "order": 4, "btype": "bandpass"},
    )

    assert response.status_code == 200
    assert len(response.json()["filtered_signal"]) == len(signal)


# --- AC2 (CRITICAL): cutoff >= Nyquist -> 422, explicit message, never 500 ---


def test_filter_rejects_cutoff_equal_to_nyquist_with_422_and_explicit_message() -> None:
    nyquist = FS / 2
    response = client.post(
        "/api/dsp/filter",
        json={"signal": _sine(50), "sampling_rate": FS, "cutoff": nyquist, "order": 4, "btype": "lowpass"},
    )

    assert response.status_code == 422
    assert response.status_code != 500
    detail = response.json()["detail"]
    assert "nyquist" in detail.lower()
    assert str(nyquist) in detail


def test_filter_rejects_cutoff_above_nyquist_with_422() -> None:
    response = client.post(
        "/api/dsp/filter",
        json={"signal": _sine(50), "sampling_rate": FS, "cutoff": FS, "order": 4, "btype": "lowpass"},
    )

    assert response.status_code == 422
    assert response.status_code != 500


def test_filter_rejects_fs_non_positive_with_422() -> None:
    response = client.post(
        "/api/dsp/filter",
        json={"signal": _sine(50), "sampling_rate": -1.0, "cutoff": 100.0, "order": 4, "btype": "lowpass"},
    )

    assert response.status_code == 422


def test_filter_rejects_order_non_positive_with_422() -> None:
    response = client.post(
        "/api/dsp/filter",
        json={"signal": _sine(50), "sampling_rate": FS, "cutoff": 100.0, "order": 0, "btype": "lowpass"},
    )

    assert response.status_code == 422


def test_filter_rejects_bandpass_with_a_scalar_cutoff_with_422() -> None:
    response = client.post(
        "/api/dsp/filter",
        json={"signal": _sine(50), "sampling_rate": FS, "cutoff": 100.0, "order": 4, "btype": "bandpass"},
    )

    assert response.status_code == 422


def test_filter_rejects_invalid_bandpass_low_greater_than_high_with_422() -> None:
    """Request-shape-valid (a (low, high) pair, matching btype) but DSP-invalid
    -- must be caught by the real FilteringError, not accepted or 500'd."""
    response = client.post(
        "/api/dsp/filter",
        json={"signal": _sine(50), "sampling_rate": FS, "cutoff": [200.0, 100.0], "order": 4, "btype": "bandpass"},
    )

    assert response.status_code == 422
    assert response.status_code != 500


def test_filter_rejects_empty_signal_with_422() -> None:
    response = client.post(
        "/api/dsp/filter",
        json={"signal": [], "sampling_rate": FS, "cutoff": 100.0, "order": 4, "btype": "lowpass"},
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# CRITICAL: proves the API reuses Phase 3/4's real functions, not a duplicate
# ---------------------------------------------------------------------------


def test_fft_route_calls_the_real_compute_fft_and_dominant_frequency(monkeypatch) -> None:
    calls: list[str] = []
    real_compute_fft = dsp_service.compute_fft
    real_dominant_frequency = dsp_service.dominant_frequency

    def spy_compute_fft(*args, **kwargs):
        calls.append("compute_fft")
        return real_compute_fft(*args, **kwargs)

    def spy_dominant_frequency(*args, **kwargs):
        calls.append("dominant_frequency")
        return real_dominant_frequency(*args, **kwargs)

    monkeypatch.setattr(dsp_service, "compute_fft", spy_compute_fft)
    monkeypatch.setattr(dsp_service, "dominant_frequency", spy_dominant_frequency)

    response = client.post("/api/fft", json={"signal": _sine(50), "sampling_rate": FS})

    assert response.status_code == 200
    assert calls == ["compute_fft", "dominant_frequency"]


def test_psd_route_calls_the_real_compute_welch_psd(monkeypatch) -> None:
    calls: list[str] = []
    real_fn = dsp_service.compute_welch_psd

    def spy(*args, **kwargs):
        calls.append("compute_welch_psd")
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(dsp_service, "compute_welch_psd", spy)

    response = client.post(
        "/api/psd", json={"signal": _sine(50), "sampling_rate": FS, "nperseg": 256, "noverlap": 128}
    )

    assert response.status_code == 200
    assert calls == ["compute_welch_psd"]


def test_spectrogram_route_calls_the_real_compute_spectrogram(monkeypatch) -> None:
    calls: list[str] = []
    real_fn = dsp_service.compute_spectrogram

    def spy(*args, **kwargs):
        calls.append("compute_spectrogram")
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(dsp_service, "compute_spectrogram", spy)

    response = client.post(
        "/api/spectrogram",
        json={"signal": _sine(50), "sampling_rate": FS, "window_size": 256, "hop_length": 128},
    )

    assert response.status_code == 200
    assert calls == ["compute_spectrogram"]


def test_filter_route_calls_the_real_butter_filter(monkeypatch) -> None:
    calls: list[str] = []
    real_fn = dsp_service.butter_filter

    def spy(*args, **kwargs):
        calls.append("butter_filter")
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(dsp_service, "butter_filter", spy)

    response = client.post(
        "/api/dsp/filter",
        json={"signal": _sine(50), "sampling_rate": FS, "cutoff": 100.0, "order": 4, "btype": "lowpass"},
    )

    assert response.status_code == 200
    assert calls == ["butter_filter"]


def test_dsp_service_does_not_reimplement_fft_with_raw_numpy_calls() -> None:
    """Static confirmation: `dsp_service.py` never calls `np.fft.*` directly --
    every FFT computation must go through `compute_fft`, the one already-tested
    Phase 4.1 implementation."""
    import ast
    import inspect

    source = inspect.getsource(dsp_service)
    tree = ast.parse(source)
    module_docstring = ast.get_docstring(tree) or ""
    code_without_docstring = source.replace(module_docstring, "")

    assert "np.fft" not in code_without_docstring
    assert "numpy.fft" not in code_without_docstring
    assert "scipy.signal.welch" not in code_without_docstring
    assert "scipy.signal.spectrogram" not in code_without_docstring
    assert "scipy.signal.butter" not in code_without_docstring
    assert "filtfilt" not in code_without_docstring
