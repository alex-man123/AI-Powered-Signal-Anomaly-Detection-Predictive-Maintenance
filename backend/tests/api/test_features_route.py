"""TASK 10.4 -- tests for `POST /api/features/extract`.

Uses the real `app.main.app` through `TestClient` (HTTP -> Pydantic schema ->
`feature_service` -> Phase 5.3's registry-driven `extract_features` -> response
schema -> JSON) -- never a mock of the feature layer, and never a hard-coded
feature count/name list copied into this file: every assertion is checked
directly against the live `FEATURE_REGISTRY`/`FREQUENCY_FEATURE_REGISTRY`
(TASK 5.1/5.2), so this suite keeps passing automatically if a feature is ever
added to either registry.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.features.registry import FEATURE_REGISTRY, FREQUENCY_FEATURE_REGISTRY
from app.main import app
from app.services import feature_service

client = TestClient(app)

FS = 50000.0  # Hz -- app.datasets.validators.SAMPLING_RATE_HZ
N = 1024
NPERSEG = 256
NOVERLAP = 128

REAL_FEATURE_NAMES = list(FEATURE_REGISTRY.keys()) + list(FREQUENCY_FEATURE_REGISTRY.keys())
REAL_FEATURE_COUNT = len(FEATURE_REGISTRY) + len(FREQUENCY_FEATURE_REGISTRY)


def _sine(freq: float, n: int = N, fs: float = FS, amplitude: float = 1.0) -> list[float]:
    t = np.arange(n) / fs
    return (amplitude * np.sin(2 * np.pi * freq * t)).tolist()


def _valid_payload(**overrides) -> dict:
    payload = {"signal": _sine(500), "sampling_rate": FS, "nperseg": NPERSEG, "noverlap": NOVERLAP}
    payload.update(overrides)
    return payload


# --- 1. Valid request -> 200 ---


def test_extract_features_valid_request_returns_200() -> None:
    response = client.post("/api/features/extract", json=_valid_payload())

    assert response.status_code == 200


# --- 2. Exact number of features from the registry ---


def test_extract_features_response_has_exactly_the_registry_feature_count() -> None:
    response = client.post("/api/features/extract", json=_valid_payload())
    body = response.json()

    assert len(body["feature_names"]) == REAL_FEATURE_COUNT
    assert len(body["values"]) == REAL_FEATURE_COUNT


# --- 3. Feature names match the registry exactly -- no missing/extra, order too ---


def test_extract_features_response_names_match_the_live_registry_exactly() -> None:
    """Not order-independent: the extractor's own documented contract is a
    fixed order (FEATURE_REGISTRY then FREQUENCY_FEATURE_REGISTRY, each in
    registry insertion order) -- this is an explicit part of the existing
    contract (app.features.extractor's own docstring), not an accidental
    dict ordering this test happens to rely on."""
    response = client.post("/api/features/extract", json=_valid_payload())
    body = response.json()

    assert body["feature_names"] == REAL_FEATURE_NAMES


def test_extract_features_response_names_have_no_missing_or_extra_features() -> None:
    response = client.post("/api/features/extract", json=_valid_payload())
    body = response.json()

    assert set(body["feature_names"]) == set(REAL_FEATURE_NAMES)


# --- 4. Values are numeric and valid (no NaN/Inf) ---


def test_extract_features_response_values_are_all_finite_numbers() -> None:
    response = client.post("/api/features/extract", json=_valid_payload())
    body = response.json()

    for value in body["values"]:
        assert isinstance(value, (int, float))
        assert np.isfinite(value)


# --- 5. The endpoint genuinely uses the real registry-driven extractor, not a hard-coded list ---


def test_extract_features_route_calls_the_real_extract_features(monkeypatch) -> None:
    calls: list[str] = []
    real_fn = feature_service.extract_features

    def spy(*args, **kwargs):
        calls.append("extract_features")
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(feature_service, "extract_features", spy)

    response = client.post("/api/features/extract", json=_valid_payload())

    assert response.status_code == 200
    assert calls == ["extract_features"]


def test_feature_service_does_not_hardcode_a_feature_name_list() -> None:
    """Static confirmation: `feature_service.py` never names an individual
    feature (e.g. no `"rms"`/`"kurtosis"` string literal) -- it must derive
    `feature_names` purely from whatever `extract_features` returns."""
    import ast
    import inspect

    source = inspect.getsource(feature_service)
    tree = ast.parse(source)
    module_docstring = ast.get_docstring(tree) or ""
    code_without_docstring = source.replace(module_docstring, "")

    for feature_name in REAL_FEATURE_NAMES:
        assert f'"{feature_name}"' not in code_without_docstring
        assert f"'{feature_name}'" not in code_without_docstring


def test_extract_features_response_is_consistent_with_a_direct_extractor_call() -> None:
    """Cross-checks the HTTP response against a direct, independent call to the
    same real extractor -- proves the endpoint is not silently returning a
    different/stale/hard-coded feature set."""
    signal = _sine(500)
    response = client.post(
        "/api/features/extract",
        json={"signal": signal, "sampling_rate": FS, "nperseg": NPERSEG, "noverlap": NOVERLAP},
    )
    body = response.json()

    from app.features.extractor import extract_features

    direct = extract_features(signal, FS, nperseg=NPERSEG, noverlap=NOVERLAP)

    assert body["feature_names"] == list(direct.keys())
    assert body["values"] == pytest.approx(list(direct.values()))


# --- 6. Invalid payload rejected via FastAPI/Pydantic validation ---


def test_extract_features_rejects_missing_signal_with_422() -> None:
    payload = _valid_payload()
    del payload["signal"]

    response = client.post("/api/features/extract", json=payload)

    assert response.status_code == 422


def test_extract_features_rejects_empty_signal_with_422() -> None:
    response = client.post("/api/features/extract", json=_valid_payload(signal=[]))

    assert response.status_code == 422


def test_extract_features_rejects_sampling_rate_as_string_with_422() -> None:
    response = client.post("/api/features/extract", json=_valid_payload(sampling_rate="50000"))

    assert response.status_code == 422


def test_extract_features_rejects_nperseg_as_string_with_422() -> None:
    response = client.post("/api/features/extract", json=_valid_payload(nperseg="256"))

    assert response.status_code == 422


def test_extract_features_rejects_noverlap_greater_or_equal_to_nperseg_with_422() -> None:
    response = client.post("/api/features/extract", json=_valid_payload(nperseg=256, noverlap=256))

    assert response.status_code == 422


def test_extract_features_rejects_nperseg_larger_than_signal_length_with_422() -> None:
    """Request-shape-valid but DSP-invalid (Pydantic cannot check this without
    knowing len(signal)) -- must be caught by the real extractor/PSD error and
    surfaced as 422, never 500."""
    response = client.post(
        "/api/features/extract", json=_valid_payload(signal=_sine(500, n=10), nperseg=256, noverlap=128)
    )

    assert response.status_code == 422
    assert response.status_code != 500


def test_extract_features_rejects_a_constant_signal_with_422_not_500() -> None:
    """A constant/all-zero window makes some time-domain features (e.g.
    crest_factor) mathematically undefined -- app.features.time_domain.FeatureError
    (a ValueError subclass) must surface as 422, not crash as a 500."""
    response = client.post("/api/features/extract", json=_valid_payload(signal=[0.0] * N))

    assert response.status_code == 422
    assert response.status_code != 500
