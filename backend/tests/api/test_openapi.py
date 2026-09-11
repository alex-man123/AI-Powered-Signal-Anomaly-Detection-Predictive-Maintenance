"""TASK 10.8 -- OpenAPI documentation completeness/correctness for Phase 10.

Verifies more than "does /docs return 200": that `/openapi.json` genuinely
reflects the real, already-implemented API -- every real Phase 10 endpoint
present, request/response schemas matching TASK 10.1's real Pydantic
contracts (types, enums, required-ness, constraints), and error responses
(404/422/500) documented exactly where each route can genuinely produce them
(not invented, not omitted).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# The exact, real Phase 10 endpoints -- verified against app/api/routes/*.py,
# not copied from the task prompt without checking.
EXPECTED_PHASE_10_ENDPOINTS = {
    ("get", "/api/datasets"),
    ("get", "/api/datasets/{dataset_id}"),
    ("post", "/api/fft"),
    ("post", "/api/psd"),
    ("post", "/api/spectrogram"),
    ("post", "/api/dsp/filter"),
    ("post", "/api/features/extract"),
    ("get", "/api/models"),
    ("get", "/api/models/{model_id}/performance"),
    ("post", "/api/models/predict"),
    ("get", "/api/models/sample-signal"),
    ("get", "/api/experiments"),
    ("get", "/api/experiments/{experiment_id}"),
}


def _spec() -> dict:
    return client.get("/openapi.json").json()


# --- Test 1: OpenAPI available ---


def test_openapi_json_returns_200_and_is_valid_json() -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, dict)
    assert "paths" in body
    assert "openapi" in body


# --- Test 2: docs available ---


def test_docs_returns_200() -> None:
    response = client.get("/docs")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_redoc_returns_200() -> None:
    response = client.get("/redoc")

    assert response.status_code == 200


# --- Test 3: endpoint coverage (exact, real endpoints only) ---


def test_openapi_contains_every_real_phase_10_endpoint() -> None:
    spec = _spec()
    actual = {
        (method, path)
        for path, methods in spec["paths"].items()
        for method in methods
        if path.startswith("/api/") and path != "/api/health"
    }

    assert EXPECTED_PHASE_10_ENDPOINTS <= actual


def test_openapi_does_not_invent_extra_undocumented_phase_10_paths() -> None:
    """The reverse check: no path/method exists in the real app beyond what
    this test explicitly expects (plus /api/health, pre-Phase-10) -- catches
    an accidentally-duplicated or leftover probe route."""
    spec = _spec()
    actual = {
        (method, path)
        for path, methods in spec["paths"].items()
        for method in methods
        if path.startswith("/api/") and path != "/api/health"
    }

    assert actual == EXPECTED_PHASE_10_ENDPOINTS


def test_no_duplicate_operation_ids_or_routes() -> None:
    spec = _spec()
    operation_ids = [
        details["operationId"]
        for methods in spec["paths"].values()
        for details in methods.values()
        if "operationId" in details
    ]

    assert len(operation_ids) == len(set(operation_ids))


# --- Test 4: request schema correctness for representative endpoints ---


def test_filter_request_schema_reflects_the_real_pydantic_contract() -> None:
    spec = _spec()
    schema = spec["components"]["schemas"]["FilterRequest"]

    # sampling_rate: strict float, gt=0 -> numeric type, not string, exclusiveMinimum present.
    assert schema["properties"]["sampling_rate"]["type"] == "number"
    assert schema["properties"]["sampling_rate"]["exclusiveMinimum"] == 0.0

    # btype: real Literal["lowpass","highpass","bandpass"] -> string enum.
    assert schema["properties"]["btype"]["type"] == "string"
    assert set(schema["properties"]["btype"]["enum"]) == {"lowpass", "highpass", "bandpass"}

    # cutoff: float | tuple[float, float] -> anyOf[number, 2-tuple].
    cutoff_any_of = schema["properties"]["cutoff"]["anyOf"]
    assert {"type": "number"} in cutoff_any_of
    assert any("prefixItems" in option for option in cutoff_any_of)

    # required fields match the real schema exactly.
    assert set(schema["required"]) == {"signal", "sampling_rate", "cutoff", "order", "btype"}


def test_predict_request_schema_reflects_the_real_model_type_enum() -> None:
    spec = _spec()
    schema = spec["components"]["schemas"]["PredictRequest"]

    model_type_schema = schema["properties"]["model_type"]
    ref = model_type_schema.get("$ref") or model_type_schema.get("allOf", [{}])[0].get("$ref")
    assert ref == "#/components/schemas/ModelType"

    model_type_enum = spec["components"]["schemas"]["ModelType"]
    assert set(model_type_enum["enum"]) == {"isolation_forest", "autoencoder"}


def test_dataset_detail_response_channels_constraint_is_reflected() -> None:
    """AC1/section 3: a real Pydantic constraint (ge=0, le=7 on each channel)
    must show up in OpenAPI, not just plain `array of integer`."""
    spec = _spec()
    channels_schema = spec["components"]["schemas"]["DatasetDetailResponse"]["properties"]["channels"]

    assert channels_schema["type"] == "array"
    assert channels_schema["items"]["minimum"] == 0
    assert channels_schema["items"]["maximum"] == 7


# --- Test 5: response schema correctness ---


def test_get_datasets_response_model_is_dataset_summary_list() -> None:
    spec = _spec()
    op = spec["paths"]["/api/datasets"]["get"]
    schema = op["responses"]["200"]["content"]["application/json"]["schema"]

    assert schema["items"]["$ref"] == "#/components/schemas/DatasetSummaryResponse"


def test_predict_response_model_matches_the_real_schema_fields() -> None:
    spec = _spec()
    schema = spec["components"]["schemas"]["PredictResponse"]

    assert set(schema["properties"].keys()) == {"anomaly_score", "status", "explanation"}
    assert schema["properties"]["anomaly_score"]["minimum"] == 0.0
    assert schema["properties"]["anomaly_score"]["maximum"] == 1.0

    status_ref = schema["properties"]["status"].get("$ref") or schema["properties"]["status"].get(
        "allOf", [{}]
    )[0].get("$ref")
    assert status_ref == "#/components/schemas/PredictionStatus"
    assert set(spec["components"]["schemas"]["PredictionStatus"]["enum"]) == {"NORMAL", "WARNING", "ANOMALY"}


def test_experiment_detail_response_includes_every_real_field() -> None:
    spec = _spec()
    schema = spec["components"]["schemas"]["ExperimentDetailResponse"]

    assert set(schema["properties"].keys()) == {
        "experiment_id",
        "representation",
        "model",
        "dataset_hash",
        "split_manifest_hash",
        "feature_dimension",
        "preprocessing_config",
        "feature_set",
        "model_config",
        "random_seed",
        "threshold_method",
        "threshold_value",
        "metrics",
        "timestamp",
    }


def test_model_response_includes_artifact_and_metrics() -> None:
    spec = _spec()
    schema = spec["components"]["schemas"]["ModelResponse"]

    assert set(schema["properties"].keys()) == {"model_type", "artifact", "metrics"}


# --- Test 6: validation documented via the real Pydantic schema (not duplicated manually) ---


def test_every_body_accepting_endpoint_documents_422_with_the_real_validation_error_schema() -> None:
    spec = _spec()

    body_endpoints = [
        ("post", "/api/fft"),
        ("post", "/api/psd"),
        ("post", "/api/spectrogram"),
        ("post", "/api/dsp/filter"),
        ("post", "/api/features/extract"),
        ("post", "/api/models/predict"),
    ]
    for method, path in body_endpoints:
        responses = spec["paths"][path][method]["responses"]
        assert "422" in responses
        schema_ref = responses["422"]["content"]["application/json"]["schema"]["$ref"]
        assert schema_ref == "#/components/schemas/HTTPValidationError"


def test_validation_error_schema_has_per_field_structure() -> None:
    spec = _spec()
    schema = spec["components"]["schemas"]["HTTPValidationError"]
    detail_items_schema_ref = schema["properties"]["detail"]["items"]["$ref"]
    validation_error_schema = spec["components"]["schemas"][detail_items_schema_ref.rsplit("/", 1)[-1]]

    assert {"loc", "msg", "type"} <= set(validation_error_schema["properties"].keys())


# --- Test 7: error responses (404/500) documented exactly where reachable ---


def test_404_documented_on_every_real_id_lookup_endpoint() -> None:
    spec = _spec()

    for method, path in (
        ("get", "/api/datasets/{dataset_id}"),
        ("get", "/api/models/{model_id}/performance"),
        ("get", "/api/experiments/{experiment_id}"),
    ):
        assert "404" in spec["paths"][path][method]["responses"], f"{method.upper()} {path} should document 404"


def test_404_documented_on_predict_now_that_a_missing_model_artifact_is_reachable() -> None:
    """Previously a disclosed discrepancy (see models.py's own docstring
    history): `post_predict` used to catch `ModelNotFoundError` without any
    real path to it (a missing artifact surfaced as 422 instead). Fixed in
    `model_service.py`'s cached loaders (`ModelPersistenceError` ->
    `ModelNotFoundError`) -- 404 is now genuinely reachable and must be
    documented."""
    spec = _spec()

    assert "404" in spec["paths"]["/api/models/predict"]["post"]["responses"]


def test_500_documented_on_every_real_endpoint() -> None:
    """TASK 10.7's global handler applies to literally every route -- the
    OpenAPI spec must reflect that uniformly, not only on a hand-picked
    subset."""
    spec = _spec()

    for path, methods in spec["paths"].items():
        if not path.startswith("/api/"):
            continue
        for method, details in methods.items():
            assert "500" in details["responses"], f"{method.upper()} {path} is missing a documented 500"


def test_500_response_schema_matches_the_real_global_handler_contract() -> None:
    spec = _spec()
    responses = spec["paths"]["/api/fft"]["post"]["responses"]

    schema_ref = responses["500"]["content"]["application/json"]["schema"]["$ref"]
    assert schema_ref == "#/components/schemas/ErrorResponse"
    error_schema = spec["components"]["schemas"]["ErrorResponse"]
    assert set(error_schema["properties"].keys()) == {"detail"}


# --- Test 8 (tagging, section 6): logical grouping ---


def test_endpoints_are_grouped_with_the_expected_tags() -> None:
    spec = _spec()
    expected_tags = {
        "/api/datasets": "Datasets",
        "/api/datasets/{dataset_id}": "Datasets",
        "/api/fft": "Signal Processing",
        "/api/psd": "Signal Processing",
        "/api/spectrogram": "Signal Processing",
        "/api/dsp/filter": "Signal Processing",
        "/api/features/extract": "Features",
        "/api/models": "Models",
        "/api/models/{model_id}/performance": "Models",
        "/api/models/predict": "Models",
        "/api/experiments": "Experiments",
        "/api/experiments/{experiment_id}": "Experiments",
    }

    for path, expected_tag in expected_tags.items():
        for details in spec["paths"][path].values():
            assert expected_tag in details.get("tags", []), f"{path} missing tag {expected_tag!r}"
