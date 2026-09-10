"""TASK 10.7 -- tests for global error handling (`app/api/error_handlers.py`).

AC1 (unhandled exception -> structured HTTP 500 JSON, no traceback leaked,
logged server-side) and AC2 (Pydantic validation -> HTTP 422 with per-field
details) are both exercised against REAL app wiring -- `app.main.app`'s
already-registered routes for AC2/Test 4, and a throwaway probe app built with
the SAME real `register_error_handlers()` function for AC1/Test 1 (there is no
existing real route that can be made to raise an uncontrolled exception
without breaking real behavior, so a dedicated route is the correct way to
exercise this path, per the task's own suggestion -- the mechanism itself,
`app.api.error_handlers`, is never mocked or duplicated).

IMPORTANT, discovered while implementing this task (documented in
`error_handlers.py`'s own module docstring): a handler registered for the
base `Exception` class is dispatched by Starlette's `ServerErrorMiddleware`,
which -- by Starlette's own explicit design -- re-raises the original
exception to the ASGI caller AFTER already sending the handled response to
the real client, specifically so test clients can surface the traceback for
debugging. `TestClient`'s default `raise_server_exceptions=True` would
therefore make a normal test call raise a raw Python exception instead of
returning a response object. Every test in this file that exercises the 500
path constructs its own client with `raise_server_exceptions=False` to
observe the actual HTTP response a real client would receive; AC2's 422 tests
use the default client (that path goes through `ExceptionMiddleware`, which
never re-raises).
"""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.error_handlers import register_error_handlers
from app.api.routes import datasets as datasets_route
from app.main import app as real_app
from app.services import feature_service, model_service

real_client = TestClient(real_app)
real_client_no_raise = TestClient(real_app, raise_server_exceptions=False)


def _build_probe_app() -> FastAPI:
    """A throwaway app wired with the SAME real `register_error_handlers`
    this project's actual `app.main` uses -- proves the mechanism itself,
    not a stand-in, handles an uncontrolled exception."""
    probe_app = FastAPI()
    register_error_handlers(probe_app)

    @probe_app.get("/probe/boom")
    def boom() -> None:
        raise RuntimeError("sensitive internal detail: /etc/secret/path, table users")

    return probe_app


# ---------------------------------------------------------------------------
# Test 1 -- unexpected exception (AC1)
# ---------------------------------------------------------------------------


def test_unhandled_exception_returns_500() -> None:
    client = TestClient(_build_probe_app(), raise_server_exceptions=False)

    response = client.get("/probe/boom")

    assert response.status_code == 500


def test_unhandled_exception_response_is_json() -> None:
    client = TestClient(_build_probe_app(), raise_server_exceptions=False)

    response = client.get("/probe/boom")

    assert response.headers["content-type"].startswith("application/json")
    assert isinstance(response.json(), dict)


def test_unhandled_exception_response_has_a_structured_generic_body() -> None:
    client = TestClient(_build_probe_app(), raise_server_exceptions=False)

    response = client.get("/probe/boom")

    assert response.json() == {"detail": "Internal server error"}


def test_unhandled_exception_never_leaks_the_traceback_or_exception_details() -> None:
    client = TestClient(_build_probe_app(), raise_server_exceptions=False)

    response = client.get("/probe/boom")
    body_text = response.text

    assert "Traceback" not in body_text
    assert "RuntimeError" not in body_text
    assert "sensitive internal detail" not in body_text
    assert "/etc/secret/path" not in body_text
    assert "table users" not in body_text
    assert ".py" not in body_text  # no source file path leaked either


def test_unhandled_exception_is_logged_server_side(caplog) -> None:
    client = TestClient(_build_probe_app(), raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR, logger="app.api.error_handlers"):
        client.get("/probe/boom")

    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(error_records) >= 1
    assert error_records[0].exc_info is not None  # traceback captured server-side
    assert "boom" in error_records[0].getMessage() or "GET" in error_records[0].getMessage()


# ---------------------------------------------------------------------------
# Test 2 -- Pydantic validation, single invalid field (AC2)
# ---------------------------------------------------------------------------


def test_validation_error_returns_422() -> None:
    response = real_client.post("/api/fft", json={"signal": [1.0, 2.0], "sampling_rate": "not_a_number"})

    assert response.status_code == 422


def test_validation_error_response_identifies_the_invalid_field() -> None:
    response = real_client.post("/api/fft", json={"signal": [1.0, 2.0], "sampling_rate": "not_a_number"})
    body = response.json()

    assert "detail" in body
    assert isinstance(body["detail"], list)
    locations = [tuple(error["loc"]) for error in body["detail"]]
    assert ("body", "sampling_rate") in locations


def test_validation_error_never_becomes_a_500() -> None:
    response = real_client.post("/api/fft", json={"signal": [1.0, 2.0], "sampling_rate": "not_a_number"})

    assert response.status_code != 500


# ---------------------------------------------------------------------------
# Test 3 -- multiple invalid fields simultaneously (AC2)
# ---------------------------------------------------------------------------


def test_validation_error_reports_every_invalid_field_when_several_are_wrong() -> None:
    response = real_client.post(
        "/api/dsp/filter",
        json={"signal": [], "sampling_rate": "bad", "cutoff": 100.0, "order": -1, "btype": "lowpass"},
    )

    assert response.status_code == 422
    locations = {tuple(error["loc"]) for error in response.json()["detail"]}

    assert ("body", "signal") in locations
    assert ("body", "sampling_rate") in locations
    assert ("body", "order") in locations
    assert len(response.json()["detail"]) >= 3


# ---------------------------------------------------------------------------
# Test 4 -- error handling works through the app's GLOBAL mechanism, across
# multiple REAL existing endpoints (datasets / features / models) -- not a
# per-route try/except and not an isolated helper.
# ---------------------------------------------------------------------------


def test_datasets_endpoint_uses_the_global_handler_for_an_unexpected_exception(monkeypatch) -> None:
    def broken_list_datasets():
        raise KeyError("unexpected real bug, not a validation error")

    # datasets.py imports `list_datasets` by name (unlike features.py/models.py,
    # which import their service module and call it via attribute access) --
    # the route module's own local name must be patched, not the service
    # module's, since Python already bound the reference at import time.
    monkeypatch.setattr(datasets_route, "list_datasets", broken_list_datasets)

    response = real_client_no_raise.get("/api/datasets")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "unexpected real bug" not in response.text


def test_features_endpoint_uses_the_global_handler_for_an_unexpected_exception(monkeypatch) -> None:
    def broken_run_feature_extraction(request):
        raise ZeroDivisionError("unexpected real bug in feature extraction")

    monkeypatch.setattr(feature_service, "run_feature_extraction", broken_run_feature_extraction)

    response = real_client_no_raise.post(
        "/api/features/extract",
        json={"signal": [1.0, 2.0, 3.0, 4.0], "sampling_rate": 50000.0, "nperseg": 2, "noverlap": 1},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "ZeroDivisionError" not in response.text


def test_models_endpoint_uses_the_global_handler_for_an_unexpected_exception(monkeypatch) -> None:
    def broken_list_models():
        raise AttributeError("unexpected real bug in model listing")

    monkeypatch.setattr(model_service, "list_models", broken_list_models)

    response = real_client_no_raise.get("/api/models")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "AttributeError" not in response.text


def test_global_handler_is_not_a_per_route_try_except() -> None:
    """Static confirmation: none of the existing routers (datasets, signals,
    features, models, experiments) catch the bare `Exception`/500 case
    themselves -- only `ValueError`/domain-specific exceptions are caught in
    each route (converted to 404/422), meaning an unrelated bug (KeyError,
    AttributeError, etc., as exercised above) can only ever be caught by the
    one global handler registered in `app.main`, not a per-route mechanism."""
    import ast
    import inspect

    from app.api.routes import datasets, experiments, features, models, signals

    for module in (datasets, experiments, features, models, signals):
        source = inspect.getsource(module)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is not None:
                caught = ast.unparse(node.type)
                assert caught not in ("Exception", "BaseException"), (
                    f"{module.__name__} catches bare {caught} -- error handling for unexpected "
                    "bugs must stay in the one global handler, not per-route"
                )
