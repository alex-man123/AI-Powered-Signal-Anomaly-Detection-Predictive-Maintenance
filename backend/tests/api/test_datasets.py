"""TASK 10.2 -- tests for `GET /api/datasets`, `GET /api/datasets/{id}`.

Uses the REAL `data/processed/split_manifest.json` (the project's actual, already
-produced 880-file manifest) through `app.main.app`'s real `TestClient` -- not a
second, parallel test-only database/fixture infrastructure. AC1's "no
placeholders" requirement is demonstrated by cross-checking the endpoint's
response directly against the same real, independent sources
(`app.datasets.validators`'s audit-confirmed constants, and the manifest file
itself) rather than merely asserting fixed numbers copied into this file.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.datasets.validators import MINIMUM_SIGNAL_LENGTH, MISSING_VALUE_COUNT, NUM_CHANNELS, SAMPLING_RATE_HZ
from app.main import app
from app.services.dataset_service import DEFAULT_MANIFEST_PATH, MAFAULDA_DATASET_ID

client = TestClient(app)


def _real_manifest_signal_count() -> int:
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    return sum(len(paths) for paths in manifest["splits"].values())


# --- 1. List endpoint ---


def test_list_datasets_returns_200_and_the_real_mafaulda_dataset() -> None:
    response = client.get("/api/datasets")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 1

    dataset = body[0]
    assert dataset["id"] == MAFAULDA_DATASET_ID
    assert dataset["name"] == "mafaulda"
    assert set(dataset.keys()) == {"id", "name", "signal_count"}


def test_list_datasets_signal_count_matches_the_real_split_manifest_independently() -> None:
    """Cross-checks the endpoint's number against the SAME real manifest file,
    read independently here -- not against a number copied into this test."""
    response = client.get("/api/datasets")

    assert response.json()[0]["signal_count"] == _real_manifest_signal_count()


# --- 2. Existing dataset detail ---


def test_get_dataset_detail_returns_200_for_the_real_dataset() -> None:
    response = client.get(f"/api/datasets/{MAFAULDA_DATASET_ID}")

    assert response.status_code == 200


def test_get_dataset_detail_sampling_rate_matches_the_real_audit_constant() -> None:
    """AC1: sampling_rate must come from the real audit source, not a placeholder --
    cross-checked directly against `app.datasets.validators.SAMPLING_RATE_HZ`, the
    same constant TASK 2.3's loader/validators use."""
    response = client.get(f"/api/datasets/{MAFAULDA_DATASET_ID}")
    body = response.json()

    assert body["sampling_rate"] == SAMPLING_RATE_HZ


def test_get_dataset_detail_channels_matches_the_real_audit_constant() -> None:
    response = client.get(f"/api/datasets/{MAFAULDA_DATASET_ID}")
    body = response.json()

    assert body["channels"] == list(range(NUM_CHANNELS))
    assert len(body["channels"]) == 8  # docs/dataset_audit/AUDIT_REPORT.md Sec.5/9


def test_get_dataset_detail_samples_per_signal_matches_the_real_audit_constant() -> None:
    response = client.get(f"/api/datasets/{MAFAULDA_DATASET_ID}")
    body = response.json()

    assert body["samples_per_signal"] == MINIMUM_SIGNAL_LENGTH
    assert body["samples_per_signal"] == 250_000


def test_get_dataset_detail_signal_count_matches_the_real_split_manifest_independently() -> None:
    response = client.get(f"/api/datasets/{MAFAULDA_DATASET_ID}")

    assert response.json()["signal_count"] == _real_manifest_signal_count()


def test_get_dataset_detail_missing_values_matches_the_real_audit_constant() -> None:
    """AC1 (TASK 11.8): missing_values must come from the real, TASK 1.5.8
    full-dataset-audit-confirmed constant, not a placeholder."""
    response = client.get(f"/api/datasets/{MAFAULDA_DATASET_ID}")
    body = response.json()

    assert body["missing_values"] == MISSING_VALUE_COUNT
    assert body["missing_values"] == 0


def test_get_dataset_detail_labels_are_real_known_signal_labels() -> None:
    response = client.get(f"/api/datasets/{MAFAULDA_DATASET_ID}")
    body = response.json()

    assert set(body["labels"]) <= {"normal", "imbalance", "horizontal-misalignment", "vertical-misalignment"}
    assert len(body["labels"]) > 0


def test_get_dataset_detail_response_matches_the_full_pydantic_schema_shape() -> None:
    response = client.get(f"/api/datasets/{MAFAULDA_DATASET_ID}")
    body = response.json()

    assert set(body.keys()) == {
        "id",
        "name",
        "signal_count",
        "sampling_rate",
        "channels",
        "samples_per_signal",
        "labels",
        "missing_values",
    }


# --- 3. Non-existing dataset ---


def test_get_dataset_detail_returns_404_for_a_nonexistent_id() -> None:
    response = client.get("/api/datasets/999")

    assert response.status_code == 404
    body = response.json()
    assert "detail" in body
    assert "999" in body["detail"]


def test_get_dataset_detail_404_is_not_a_500_or_empty_body() -> None:
    response = client.get("/api/datasets/999")

    assert response.status_code != 500
    assert response.json() != {}


def test_get_dataset_detail_returns_422_for_a_non_integer_id() -> None:
    """FastAPI's own path-parameter validation rejects a non-integer id before
    the route body ever runs -- this is the "ID invalid" case the task asks to
    handle "dacă schema/router-ul îl poate valida"."""
    response = client.get("/api/datasets/not-a-number")

    assert response.status_code == 422


# --- 4. No placeholders: real values, not hardcoded constants unrelated to the audit ---


def test_dataset_detail_values_are_not_the_common_placeholder_example_from_the_task() -> None:
    """The task's own example of a forbidden placeholder: samples=100000,
    sampling_rate=1000, channels=3. None of these appear in the real response."""
    response = client.get(f"/api/datasets/{MAFAULDA_DATASET_ID}")
    body = response.json()

    assert body["sampling_rate"] != 1000
    assert len(body["channels"]) != 3
    assert body["samples_per_signal"] != 100_000
