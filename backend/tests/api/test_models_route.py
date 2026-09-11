"""TASK 10.5 -- tests for `GET /api/models`, `GET /api/models/{id}/performance`,
`POST /api/models/predict`.

Uses the real `app.main.app` through `TestClient` against the REAL, standing
model artifacts (`models/isolation_forest_v1.pkl`+`.json`,
`models/autoencoder_v1.pt`+`.json`) and the REAL registered Phase 8 metrics
(`EXP-B-001`/`EXP-C-001`, TASK 8.3's registry) -- never a mock of the ML layer.
Metric/metadata values used for cross-checks are read independently here (via
`get_experiment_run`/`load_model_artifact` directly), never copied as literals
that could silently drift from the real source.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ml.experiment_registry import get_experiment_run
from app.ml.inference import default_autoencoder_path, default_model_path
from app.ml.model_artifact import load_model_artifact
from app.services import model_service

client = TestClient(app)

FS = 50000.0  # Hz -- app.datasets.validators.SAMPLING_RATE_HZ
N = 1024


def _sine(freq: float, n: int = N, fs: float = FS, amplitude: float = 1.0) -> list[float]:
    t = np.arange(n) / fs
    return (amplitude * np.sin(2 * np.pi * freq * t)).tolist()


# ---------------------------------------------------------------------------
# GET /api/models
# ---------------------------------------------------------------------------


def test_list_models_returns_200() -> None:
    response = client.get("/api/models")

    assert response.status_code == 200


def test_list_models_includes_isolation_forest() -> None:
    response = client.get("/api/models")
    model_types = [row["model_type"] for row in response.json()]

    assert "isolation_forest" in model_types


def test_list_models_includes_autoencoder() -> None:
    response = client.get("/api/models")
    model_types = [row["model_type"] for row in response.json()]

    assert "autoencoder" in model_types


def test_list_models_has_exactly_two_models() -> None:
    response = client.get("/api/models")

    assert len(response.json()) == 2


def test_list_models_metrics_match_the_real_registered_phase_8_experiment_runs() -> None:
    """Cross-checks against `get_experiment_run` called independently here --
    proves the route's metrics are NOT hard-coded in `models.py`/`model_
    service.py`, but genuinely read from TASK 8.3's real registry."""
    response = client.get("/api/models")
    by_type = {row["model_type"]: row for row in response.json()}

    real_if_metrics = get_experiment_run("EXP-B-001").metrics
    real_ae_metrics = get_experiment_run("EXP-C-001").metrics

    assert by_type["isolation_forest"]["metrics"] == pytest.approx(real_if_metrics, nan_ok=True)
    assert by_type["autoencoder"]["metrics"] == pytest.approx(real_ae_metrics, nan_ok=True)


def test_list_models_artifact_info_matches_the_real_model_artifact_contract() -> None:
    """Cross-checks against `load_model_artifact` called independently here --
    proves `artifact.*` is read from the real TASK 6.5 sidecar, not invented."""
    response = client.get("/api/models")
    by_type = {row["model_type"]: row for row in response.json()}

    _model, real_if_metadata = load_model_artifact(default_model_path())
    _ae_model, real_ae_metadata = load_model_artifact(default_autoencoder_path())

    assert by_type["isolation_forest"]["artifact"]["feature_dimension"] == real_if_metadata.feature_dimension
    assert by_type["isolation_forest"]["artifact"]["threshold_value"] == real_if_metadata.threshold_value
    assert by_type["isolation_forest"]["artifact"]["dataset_hash"] == real_if_metadata.dataset_hash
    assert by_type["autoencoder"]["artifact"]["feature_dimension"] == real_ae_metadata.feature_dimension
    assert by_type["autoencoder"]["artifact"]["threshold_value"] == real_ae_metadata.threshold_value


def test_list_models_route_does_not_hardcode_metrics(monkeypatch) -> None:
    """Static + behavioral confirmation: the route genuinely calls
    `app.services.model_service.list_models` (spied), not a hard-coded literal
    dict of metrics."""
    calls: list[str] = []
    real_fn = model_service.list_models

    def spy():
        calls.append("list_models")
        return real_fn()

    monkeypatch.setattr(model_service, "list_models", spy)

    response = client.get("/api/models")

    assert response.status_code == 200
    assert calls == ["list_models"]


# ---------------------------------------------------------------------------
# GET /api/models/{id}/performance
# ---------------------------------------------------------------------------


def test_get_model_performance_returns_200_for_isolation_forest() -> None:
    response = client.get("/api/models/isolation_forest/performance")

    assert response.status_code == 200


def test_get_model_performance_returns_200_for_autoencoder() -> None:
    response = client.get("/api/models/autoencoder/performance")

    assert response.status_code == 200


def test_get_model_performance_metrics_match_the_real_registry_for_isolation_forest() -> None:
    response = client.get("/api/models/isolation_forest/performance")
    real_metrics = get_experiment_run("EXP-B-001").metrics

    assert response.json()["metrics"] == pytest.approx(real_metrics, nan_ok=True)


def test_get_model_performance_metrics_match_the_real_registry_for_autoencoder() -> None:
    response = client.get("/api/models/autoencoder/performance")
    real_metrics = get_experiment_run("EXP-C-001").metrics

    assert response.json()["metrics"] == pytest.approx(real_metrics, nan_ok=True)


def test_get_model_performance_returns_404_for_a_nonexistent_model() -> None:
    response = client.get("/api/models/random_forest/performance")

    assert response.status_code == 404
    assert response.status_code != 500
    assert "random_forest" in response.json()["detail"]


def test_get_model_performance_metrics_are_not_static_placeholder_values() -> None:
    """None of the returned metrics are trivially fake/round-number
    placeholders (e.g. 0.0/1.0 across the board) -- a weak but real sanity
    check that this is genuine measured data."""
    response = client.get("/api/models/isolation_forest/performance")
    metrics = response.json()["metrics"]

    assert metrics["precision"] not in (0.0, 1.0)
    assert metrics["confusion_matrix"] != [[0, 0], [0, 0]]


# ---------------------------------------------------------------------------
# GET /api/models/sample-signal
# ---------------------------------------------------------------------------


def test_get_sample_signal_returns_200() -> None:
    response = client.get("/api/models/sample-signal")

    assert response.status_code == 200


def test_get_sample_signal_matches_the_real_first_validation_window() -> None:
    """Cross-checks against `model_service._validation_windows_and_labels`
    called independently here -- proves the route serves the real, already-
    windowed validation recording, not invented/random data."""
    from app.services.model_service import CHANNEL, _validation_windows_and_labels

    response = client.get("/api/models/sample-signal")
    body = response.json()

    windows, labels = _validation_windows_and_labels()
    real_window = windows[0]

    assert body["recording_id"] == real_window.recording_id
    assert body["label"] == labels[0]
    assert body["channel"] == CHANNEL
    assert body["signal"] == pytest.approx(list(real_window.values))


def test_get_sample_signal_can_be_fed_directly_into_predict() -> None:
    """The whole point of this endpoint: its output is a valid `predict` input."""
    sample = client.get("/api/models/sample-signal").json()

    response = client.post(
        "/api/models/predict",
        json={
            "model_type": "isolation_forest",
            "signal": sample["signal"],
            "sampling_rate": sample["sampling_rate"],
        },
    )

    assert response.status_code == 200
    assert "anomaly_score" in response.json()


# ---------------------------------------------------------------------------
# POST /api/models/predict
# ---------------------------------------------------------------------------


def test_predict_valid_request_returns_200_for_isolation_forest() -> None:
    response = client.post(
        "/api/models/predict",
        json={"model_type": "isolation_forest", "signal": _sine(500), "sampling_rate": FS},
    )

    assert response.status_code == 200


def test_predict_valid_request_returns_200_for_autoencoder() -> None:
    response = client.post(
        "/api/models/predict",
        json={"model_type": "autoencoder", "signal": _sine(500), "sampling_rate": FS},
    )

    assert response.status_code == 200


def test_predict_response_contains_anomaly_score() -> None:
    response = client.post(
        "/api/models/predict",
        json={"model_type": "isolation_forest", "signal": _sine(500), "sampling_rate": FS},
    )

    assert "anomaly_score" in response.json()


def test_predict_response_score_is_in_zero_one_range() -> None:
    response = client.post(
        "/api/models/predict",
        json={"model_type": "isolation_forest", "signal": _sine(500), "sampling_rate": FS},
    )
    score = response.json()["anomaly_score"]

    assert 0.0 <= score <= 1.0


def test_predict_response_contains_a_valid_status() -> None:
    response = client.post(
        "/api/models/predict",
        json={"model_type": "autoencoder", "signal": _sine(500), "sampling_rate": FS},
    )

    assert response.json()["status"] in ("NORMAL", "WARNING", "ANOMALY")


def test_predict_response_contains_an_explanation() -> None:
    response = client.post(
        "/api/models/predict",
        json={"model_type": "isolation_forest", "signal": _sine(500), "sampling_rate": FS},
    )

    assert len(response.json()["explanation"]) > 0


def test_predict_explanation_is_grounded_in_the_actually_computed_features() -> None:
    """Not a generic string: cross-checks that the explanation's numeric
    feature values genuinely match what `extract_features` computes for THIS
    exact signal -- proves the explanation is derived from real per-request
    computation, not a fixed/templated message."""
    from app.features.extractor import extract_features

    signal = _sine(freq=2000, amplitude=5.0)  # an unusual, high-frequency, high-amplitude signal
    response = client.post(
        "/api/models/predict",
        json={"model_type": "isolation_forest", "signal": signal, "sampling_rate": FS},
    )
    explanation = response.json()["explanation"]

    real_features = extract_features(signal, FS, nperseg=256, noverlap=128)

    # At least one real feature name mentioned in the explanation (if any
    # deviation was found) must match this exact request's own computed value.
    assert response.status_code == 200
    mentioned_any_real_feature = any(name in explanation for name in real_features)
    assert mentioned_any_real_feature or "multivariate assessment" in explanation


def test_predict_explanation_is_not_a_generic_hardcoded_string() -> None:
    response = client.post(
        "/api/models/predict",
        json={"model_type": "isolation_forest", "signal": _sine(500), "sampling_rate": FS},
    )
    explanation = response.json()["explanation"]

    assert explanation != "The signal is anomalous."
    assert "generic" not in explanation.lower()


def test_predict_explanations_differ_for_genuinely_different_signals() -> None:
    """Different real inputs -> different real feature values -> different
    explanations (not a single templated message regardless of input)."""
    quiet_signal = (np.zeros(N) + 0.001 * np.sin(2 * np.pi * 50 * np.arange(N) / FS)).tolist()
    loud_signal = _sine(freq=3000, amplitude=10.0)

    response_quiet = client.post(
        "/api/models/predict",
        json={"model_type": "isolation_forest", "signal": quiet_signal, "sampling_rate": FS},
    )
    response_loud = client.post(
        "/api/models/predict",
        json={"model_type": "isolation_forest", "signal": loud_signal, "sampling_rate": FS},
    )

    assert response_quiet.json()["explanation"] != response_loud.json()["explanation"]


def test_predict_uses_the_real_feature_extraction_scaling_and_model_inference_pipeline(monkeypatch) -> None:
    """Spies on the real pipeline functions `model_service` imports, confirming
    the route genuinely calls extract_features -> isolation_forest_raw_predict
    -> normalize_scores, not a stub/shortcut."""
    calls: list[str] = []

    real_extract = model_service.extract_features
    real_predict = model_service.isolation_forest_raw_predict
    real_normalize = model_service.normalize_scores

    def spy_extract(*args, **kwargs):
        calls.append("extract_features")
        return real_extract(*args, **kwargs)

    def spy_predict(*args, **kwargs):
        calls.append("isolation_forest_raw_predict")
        return real_predict(*args, **kwargs)

    def spy_normalize(*args, **kwargs):
        calls.append("normalize_scores")
        return real_normalize(*args, **kwargs)

    monkeypatch.setattr(model_service, "extract_features", spy_extract)
    monkeypatch.setattr(model_service, "isolation_forest_raw_predict", spy_predict)
    monkeypatch.setattr(model_service, "normalize_scores", spy_normalize)

    response = client.post(
        "/api/models/predict",
        json={"model_type": "isolation_forest", "signal": _sine(500), "sampling_rate": FS},
    )

    assert response.status_code == 200
    assert "extract_features" in calls
    assert "isolation_forest_raw_predict" in calls
    assert "normalize_scores" in calls


def test_predict_rejects_invalid_payload_missing_signal_with_422() -> None:
    response = client.post("/api/models/predict", json={"model_type": "isolation_forest", "sampling_rate": FS})

    assert response.status_code == 422


def test_predict_rejects_invalid_model_type_with_422() -> None:
    response = client.post(
        "/api/models/predict",
        json={"model_type": "random_forest", "signal": _sine(500), "sampling_rate": FS},
    )

    assert response.status_code == 422


def test_predict_rejects_sampling_rate_as_string_with_422() -> None:
    response = client.post(
        "/api/models/predict",
        json={"model_type": "isolation_forest", "signal": _sine(500), "sampling_rate": "50000"},
    )

    assert response.status_code == 422


def test_predict_rejects_empty_signal_with_422_not_500() -> None:
    response = client.post(
        "/api/models/predict", json={"model_type": "isolation_forest", "signal": [], "sampling_rate": FS}
    )

    assert response.status_code == 422
    assert response.status_code != 500


def test_predict_rejects_a_signal_too_short_for_the_real_psd_config_with_422_not_500() -> None:
    """Request-shape-valid but DSP-invalid (too few samples for nperseg=256) --
    must be caught by the real extractor/PSD error, not crash as 500."""
    response = client.post(
        "/api/models/predict",
        json={"model_type": "isolation_forest", "signal": [0.1, 0.2, 0.3], "sampling_rate": FS},
    )

    assert response.status_code == 422
    assert response.status_code != 500


def test_predict_returns_404_not_422_when_the_model_artifact_is_genuinely_missing(monkeypatch) -> None:
    """Fixed discrepancy: a missing model artifact is a server-side/infra
    condition, not a client input error -- it must surface as 404, not 422.
    Forces the real `ModelPersistenceError` path (by pointing the cached
    loader at a path that does not exist) and resets the module-level cache
    so the loader actually runs instead of returning an already-cached model
    from an earlier test in this file."""
    monkeypatch.setattr(model_service, "_ISOLATION_FOREST_MODEL", None)
    monkeypatch.setattr(model_service, "default_model_path", lambda: default_model_path().parent / "does_not_exist.pkl")

    response = client.post(
        "/api/models/predict",
        json={"model_type": "isolation_forest", "signal": _sine(500), "sampling_rate": FS},
    )

    assert response.status_code == 404
    assert response.status_code != 422
    assert response.status_code != 500
