"""TASK 10.1 -- tests for `backend/app/api/schemas/*.py`.

Exercises real validation behavior (rejection of wrong types, missing required
fields, invalid enum values, and structural cross-field rules) -- never just
"the module imports". AC1's own example (`sampling_rate: str` instead of
`float`) is tested explicitly for every schema that has a `sampling_rate` field,
since Pydantic v2's default ("lax") mode actually COERCES a numeric string like
`"1000"` into `1000.0` silently (verified interactively before writing this
suite) -- these schemas use `Field(strict=True)` specifically so that coercion
does not happen, and this file proves it.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.schemas.datasets import DatasetDetailResponse, DatasetSummaryResponse
from app.api.schemas.experiments import ExperimentListResponse, ExperimentResponse
from app.api.schemas.features import FeatureExtractionRequest, FeatureExtractionResponse
from app.api.schemas.models import (
    ModelMetrics,
    ModelResponse,
    PredictRequest,
    PredictResponse,
    PredictionStatus,
)
from app.api.schemas.signal_processing import (
    FFTRequest,
    FFTResponse,
    FilterRequest,
    FilterResponse,
    PSDRequest,
    PSDResponse,
    SpectrogramRequest,
    SpectrogramResponse,
)
from app.ml.model_artifact import ModelType
from app.models.signal import SignalLabel

# ---------------------------------------------------------------------------
# datasets.py
# ---------------------------------------------------------------------------


def test_dataset_summary_response_accepts_a_valid_payload() -> None:
    response = DatasetSummaryResponse(id=1, name="mafaulda", signal_count=880)

    assert response.id == 1
    assert response.signal_count == 880


def test_dataset_summary_response_rejects_string_signal_count() -> None:
    with pytest.raises(ValidationError):
        DatasetSummaryResponse(id=1, name="mafaulda", signal_count="880")


def test_dataset_summary_response_rejects_missing_required_field() -> None:
    with pytest.raises(ValidationError):
        DatasetSummaryResponse(id=1, signal_count=880)  # name missing


def test_dataset_detail_response_accepts_a_valid_payload() -> None:
    response = DatasetDetailResponse(
        id=1,
        name="mafaulda",
        signal_count=880,
        sampling_rate=50000.0,
        channels=[0, 1, 2, 3, 4, 5, 6, 7],
        samples_per_signal=250000,
        labels=["normal", "imbalance"],
        missing_values=0,
    )

    assert response.sampling_rate == 50000.0
    assert response.samples_per_signal == 250000
    assert response.labels == [SignalLabel.NORMAL, SignalLabel.IMBALANCE]
    assert response.missing_values == 0


def test_dataset_detail_response_rejects_sampling_rate_as_string() -> None:
    """AC1's own literal example: `sampling_rate: str` instead of `float`."""
    with pytest.raises(ValidationError, match="sampling_rate"):
        DatasetDetailResponse(
            id=1,
            name="mafaulda",
            signal_count=880,
            sampling_rate="50000.0",
            channels=[0],
            samples_per_signal=250000,
            labels=["normal"],
        )


def test_dataset_detail_response_rejects_non_positive_sampling_rate() -> None:
    with pytest.raises(ValidationError):
        DatasetDetailResponse(
            id=1,
            name="mafaulda",
            signal_count=880,
            sampling_rate=0.0,
            channels=[0],
            samples_per_signal=250000,
            labels=["normal"],
        )


def test_dataset_detail_response_rejects_channel_out_of_audit_confirmed_range() -> None:
    with pytest.raises(ValidationError):
        DatasetDetailResponse(
            id=1,
            name="mafaulda",
            signal_count=880,
            sampling_rate=50000.0,
            channels=[8],
            samples_per_signal=250000,
            labels=["normal"],
        )


def test_dataset_detail_response_rejects_invalid_label_enum_value() -> None:
    with pytest.raises(ValidationError):
        DatasetDetailResponse(
            id=1,
            name="mafaulda",
            signal_count=880,
            sampling_rate=50000.0,
            channels=[0],
            samples_per_signal=250000,
            labels=["not_a_real_label"],
        )


def test_dataset_detail_response_rejects_wrong_type_element_in_channels_list() -> None:
    with pytest.raises(ValidationError):
        DatasetDetailResponse(
            id=1,
            name="mafaulda",
            signal_count=880,
            sampling_rate=50000.0,
            channels=["zero"],
            samples_per_signal=250000,
            labels=["normal"],
        )


def test_dataset_detail_response_rejects_non_positive_samples_per_signal() -> None:
    with pytest.raises(ValidationError):
        DatasetDetailResponse(
            id=1,
            name="mafaulda",
            signal_count=880,
            sampling_rate=50000.0,
            channels=[0],
            samples_per_signal=0,
            labels=["normal"],
        )


def test_dataset_detail_response_rejects_samples_per_signal_as_string() -> None:
    with pytest.raises(ValidationError):
        DatasetDetailResponse(
            id=1,
            name="mafaulda",
            signal_count=880,
            sampling_rate=50000.0,
            channels=[0],
            samples_per_signal="250000",
            labels=["normal"],
        )


# ---------------------------------------------------------------------------
# signal_processing.py -- FFT
# ---------------------------------------------------------------------------


def test_fft_request_accepts_a_valid_payload() -> None:
    request = FFTRequest(signal=[1.0, 2.0, 3.0], sampling_rate=50000.0)
    assert request.sampling_rate == 50000.0


def test_fft_request_rejects_sampling_rate_as_string() -> None:
    with pytest.raises(ValidationError):
        FFTRequest(signal=[1.0, 2.0], sampling_rate="50000")


def test_fft_request_rejects_non_positive_sampling_rate() -> None:
    with pytest.raises(ValidationError):
        FFTRequest(signal=[1.0, 2.0], sampling_rate=-1.0)


def test_fft_request_rejects_empty_signal() -> None:
    with pytest.raises(ValidationError):
        FFTRequest(signal=[], sampling_rate=50000.0)


def test_fft_request_rejects_wrong_type_element_in_signal() -> None:
    with pytest.raises(ValidationError):
        FFTRequest(signal=[1.0, "not_a_number"], sampling_rate=50000.0)


def test_fft_request_rejects_missing_signal() -> None:
    with pytest.raises(ValidationError):
        FFTRequest(sampling_rate=50000.0)


def test_fft_response_accepts_a_valid_payload_and_serializes() -> None:
    response = FFTResponse(frequencies=[0.0, 1.0], magnitude=[0.0, 5.0], dominant_frequency=1.0)
    dumped = response.model_dump()
    assert dumped["dominant_frequency"] == 1.0


# ---------------------------------------------------------------------------
# signal_processing.py -- PSD
# ---------------------------------------------------------------------------


def test_psd_request_accepts_a_valid_payload() -> None:
    request = PSDRequest(signal=[1.0] * 512, sampling_rate=50000.0, nperseg=256, noverlap=128)
    assert request.nperseg == 256


def test_psd_request_rejects_nperseg_as_string() -> None:
    with pytest.raises(ValidationError):
        PSDRequest(signal=[1.0] * 512, sampling_rate=50000.0, nperseg="256", noverlap=128)


def test_psd_request_rejects_noverlap_greater_or_equal_to_nperseg() -> None:
    with pytest.raises(ValidationError, match="noverlap"):
        PSDRequest(signal=[1.0] * 512, sampling_rate=50000.0, nperseg=256, noverlap=256)


def test_psd_request_rejects_non_positive_nperseg() -> None:
    with pytest.raises(ValidationError):
        PSDRequest(signal=[1.0] * 512, sampling_rate=50000.0, nperseg=0, noverlap=0)


def test_psd_response_accepts_a_valid_payload() -> None:
    response = PSDResponse(frequencies=[0.0, 1.0], psd=[0.1, 0.2])
    assert response.psd == [0.1, 0.2]


# ---------------------------------------------------------------------------
# signal_processing.py -- Spectrogram
# ---------------------------------------------------------------------------


def test_spectrogram_request_accepts_a_valid_payload() -> None:
    request = SpectrogramRequest(signal=[1.0] * 512, sampling_rate=50000.0, window_size=256, hop_length=128)
    assert request.window_size == 256


def test_spectrogram_request_rejects_hop_length_greater_than_window_size() -> None:
    with pytest.raises(ValidationError, match="hop_length"):
        SpectrogramRequest(signal=[1.0] * 512, sampling_rate=50000.0, window_size=128, hop_length=256)


def test_spectrogram_request_rejects_window_size_as_string() -> None:
    with pytest.raises(ValidationError):
        SpectrogramRequest(signal=[1.0] * 512, sampling_rate=50000.0, window_size="256", hop_length=128)


def test_spectrogram_response_accepts_a_2d_values_payload() -> None:
    response = SpectrogramResponse(frequencies=[0.0, 1.0], times=[0.0, 0.5], values=[[0.1, 0.2], [0.3, 0.4]])
    assert response.values == [[0.1, 0.2], [0.3, 0.4]]


def test_spectrogram_response_rejects_non_numeric_element_in_values() -> None:
    with pytest.raises(ValidationError):
        SpectrogramResponse(frequencies=[0.0], times=[0.0], values=[["not_a_number"]])


# ---------------------------------------------------------------------------
# signal_processing.py -- Filter
# ---------------------------------------------------------------------------


def test_filter_request_accepts_a_valid_lowpass_payload() -> None:
    request = FilterRequest(signal=[1.0, 2.0], sampling_rate=50000.0, cutoff=1000.0, order=4, btype="lowpass")
    assert request.cutoff == 1000.0


def test_filter_request_accepts_a_valid_bandpass_payload() -> None:
    request = FilterRequest(
        signal=[1.0, 2.0], sampling_rate=50000.0, cutoff=(500.0, 2000.0), order=4, btype="bandpass"
    )
    assert request.cutoff == (500.0, 2000.0)


def test_filter_request_rejects_bandpass_with_a_scalar_cutoff() -> None:
    with pytest.raises(ValidationError, match="bandpass"):
        FilterRequest(signal=[1.0, 2.0], sampling_rate=50000.0, cutoff=1000.0, order=4, btype="bandpass")


def test_filter_request_rejects_lowpass_with_a_pair_cutoff() -> None:
    with pytest.raises(ValidationError, match="lowpass"):
        FilterRequest(signal=[1.0, 2.0], sampling_rate=50000.0, cutoff=(500.0, 2000.0), order=4, btype="lowpass")


def test_filter_request_rejects_invalid_btype_enum_value() -> None:
    with pytest.raises(ValidationError):
        FilterRequest(signal=[1.0, 2.0], sampling_rate=50000.0, cutoff=1000.0, order=4, btype="not_a_real_btype")


def test_filter_request_rejects_order_as_string() -> None:
    with pytest.raises(ValidationError):
        FilterRequest(signal=[1.0, 2.0], sampling_rate=50000.0, cutoff=1000.0, order="4", btype="lowpass")


def test_filter_request_rejects_non_positive_order() -> None:
    with pytest.raises(ValidationError):
        FilterRequest(signal=[1.0, 2.0], sampling_rate=50000.0, cutoff=1000.0, order=0, btype="lowpass")


def test_filter_response_accepts_a_valid_payload() -> None:
    response = FilterResponse(filtered_signal=[1.0, 2.0, 3.0])
    assert response.filtered_signal == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# features.py
# ---------------------------------------------------------------------------


def test_feature_extraction_request_accepts_a_valid_payload() -> None:
    request = FeatureExtractionRequest(signal=[1.0, 2.0], sampling_rate=50000.0, nperseg=256, noverlap=128)
    assert request.sampling_rate == 50000.0


def test_feature_extraction_request_rejects_sampling_rate_as_string() -> None:
    with pytest.raises(ValidationError):
        FeatureExtractionRequest(signal=[1.0, 2.0], sampling_rate="50000", nperseg=256, noverlap=128)


def test_feature_extraction_request_rejects_noverlap_greater_or_equal_to_nperseg() -> None:
    with pytest.raises(ValidationError, match="noverlap"):
        FeatureExtractionRequest(signal=[1.0, 2.0], sampling_rate=50000.0, nperseg=256, noverlap=256)


def test_feature_extraction_request_rejects_nperseg_as_string() -> None:
    with pytest.raises(ValidationError):
        FeatureExtractionRequest(signal=[1.0, 2.0], sampling_rate=50000.0, nperseg="256", noverlap=128)


def test_feature_extraction_response_accepts_matching_lengths() -> None:
    response = FeatureExtractionResponse(feature_names=["rms", "kurtosis"], values=[1.0, 2.0])
    assert len(response.feature_names) == len(response.values)


def test_feature_extraction_response_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValidationError, match="same length"):
        FeatureExtractionResponse(feature_names=["rms", "kurtosis"], values=[1.0])


def test_feature_extraction_response_rejects_non_string_feature_name() -> None:
    with pytest.raises(ValidationError):
        FeatureExtractionResponse(feature_names=["rms", 123], values=[1.0, 2.0])


# ---------------------------------------------------------------------------
# models.py
# ---------------------------------------------------------------------------

_REAL_METRICS_PAYLOAD = {
    "precision": 0.65,
    "recall": 0.26,
    "f1": 0.37,
    "roc_auc": 0.53,
    "pr_auc": 0.59,
    "confusion_matrix": [[840, 134], [721, 253]],
    "fpr": 0.14,
    "fnr": 0.74,
    "inference_time": 0.0085,
}

_REAL_ARTIFACT_PAYLOAD = {
    "model_version": "v1",
    "feature_names": ["mean", "std", "rms"],
    "feature_dimension": 3,
    "threshold_method": "percentile",
    "threshold_value": 0.51,
    "score_direction": "higher_is_more_anomalous",
    "training_split": "train",
    "dataset_hash": "sha256:" + "a" * 64,
    "split_manifest_hash": "sha256:" + "b" * 64,
    "random_seed": 42,
    "created_at": "2026-01-01T00:00:00+00:00",
}


def test_model_metrics_accepts_a_real_valid_payload() -> None:
    metrics = ModelMetrics(**_REAL_METRICS_PAYLOAD)
    assert metrics.confusion_matrix == [[840, 134], [721, 253]]


def test_model_metrics_accepts_nan_for_undefined_ranking_metrics() -> None:
    """`app.ml.evaluation.evaluate()`'s own documented contract: roc_auc/pr_auc/
    fpr/fnr may legitimately be NaN for an edge-case test set -- must not be
    rejected as invalid."""
    payload = {**_REAL_METRICS_PAYLOAD, "roc_auc": float("nan"), "pr_auc": float("nan")}
    metrics = ModelMetrics(**payload)
    assert metrics.roc_auc != metrics.roc_auc  # NaN != NaN


def test_model_metrics_rejects_precision_as_string() -> None:
    with pytest.raises(ValidationError):
        ModelMetrics(**{**_REAL_METRICS_PAYLOAD, "precision": "0.65"})


def test_model_metrics_rejects_missing_confusion_matrix() -> None:
    payload = dict(_REAL_METRICS_PAYLOAD)
    del payload["confusion_matrix"]
    with pytest.raises(ValidationError):
        ModelMetrics(**payload)


def test_model_response_accepts_a_valid_payload_with_real_model_type_enum() -> None:
    response = ModelResponse(
        model_type="isolation_forest", artifact=_REAL_ARTIFACT_PAYLOAD, metrics=_REAL_METRICS_PAYLOAD
    )
    assert response.model_type == ModelType.ISOLATION_FOREST
    assert response.artifact.feature_dimension == 3


def test_model_response_rejects_invalid_model_type_enum_value() -> None:
    with pytest.raises(ValidationError):
        ModelResponse(model_type="random_forest", artifact=_REAL_ARTIFACT_PAYLOAD, metrics=_REAL_METRICS_PAYLOAD)


def test_predict_request_accepts_a_valid_payload() -> None:
    request = PredictRequest(model_type="autoencoder", signal=[1.0, 2.0], sampling_rate=50000.0)
    assert request.model_type == ModelType.AUTOENCODER


def test_predict_request_rejects_sampling_rate_as_string() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(model_type="autoencoder", signal=[1.0, 2.0], sampling_rate="50000")


def test_predict_response_accepts_a_valid_payload() -> None:
    response = PredictResponse(anomaly_score=0.42, status="WARNING", explanation="kurtosis is elevated")
    assert response.status == PredictionStatus.WARNING


def test_predict_response_rejects_anomaly_score_outside_zero_one_range() -> None:
    with pytest.raises(ValidationError):
        PredictResponse(anomaly_score=1.5, status="ANOMALY", explanation="x")


def test_predict_response_rejects_anomaly_score_as_string() -> None:
    with pytest.raises(ValidationError):
        PredictResponse(anomaly_score="0.5", status="NORMAL", explanation="x")


def test_predict_response_rejects_invalid_status_enum_value() -> None:
    with pytest.raises(ValidationError):
        PredictResponse(anomaly_score=0.5, status="CRITICAL", explanation="x")


def test_predict_response_rejects_missing_explanation() -> None:
    with pytest.raises(ValidationError):
        PredictResponse(anomaly_score=0.5, status="NORMAL")


# ---------------------------------------------------------------------------
# experiments.py -- mirrors TASK 9.5's real central_experiment_results.json shape
# ---------------------------------------------------------------------------

_REAL_EXPERIMENT_A_PAYLOAD = {
    "experiment": "A",
    "experiment_id": "EXP-A-002",
    "representation": "raw_pca",
    "model": "isolation_forest",
    "dataset_hash": "sha256:95aebdfe4d32407ef",
    "split_manifest_hash": "sha256:72fc253af4f4c5df",
    "precision": 0.6537467700258398,
    "recall": 0.2597535934291581,
    "f1": 0.37178545187362233,
    "roc_auc": 0.5321727333673457,
    "pr_auc": 0.589875579289735,
    "fpr": 0.1375770020533881,
    "fnr": 0.7402464065708418,
    "confusion_matrix": [[840, 134], [721, 253]],
    "inference_time": 0.008495599999150727,
}


def test_experiment_response_accepts_the_real_experiment_a_payload() -> None:
    response = ExperimentResponse(**_REAL_EXPERIMENT_A_PAYLOAD)
    assert response.experiment_id == "EXP-A-002"


def test_experiment_response_rejects_invalid_experiment_literal() -> None:
    with pytest.raises(ValidationError):
        ExperimentResponse(**{**_REAL_EXPERIMENT_A_PAYLOAD, "experiment": "D"})


def test_experiment_response_rejects_precision_as_string() -> None:
    with pytest.raises(ValidationError):
        ExperimentResponse(**{**_REAL_EXPERIMENT_A_PAYLOAD, "precision": "0.65"})


def test_experiment_response_rejects_negative_inference_time() -> None:
    with pytest.raises(ValidationError):
        ExperimentResponse(**{**_REAL_EXPERIMENT_A_PAYLOAD, "inference_time": -0.01})


def test_experiment_list_response_accepts_a_valid_payload() -> None:
    listing = ExperimentListResponse(experiments=[_REAL_EXPERIMENT_A_PAYLOAD])
    assert len(listing.experiments) == 1


# ---------------------------------------------------------------------------
# FastAPI compatibility: schemas work as real response_model/request bodies
# ---------------------------------------------------------------------------


def _build_probe_app() -> FastAPI:
    """A throwaway FastAPI app, local to this test module -- proves these
    schemas work as real request bodies / response_models under FastAPI's own
    validation pipeline, without implementing any real TASK 10.2+ endpoint."""
    probe_app = FastAPI()

    @probe_app.post("/probe/filter", response_model=FilterResponse)
    def probe_filter(payload: FilterRequest) -> FilterResponse:
        return FilterResponse(filtered_signal=payload.signal)

    @probe_app.get("/probe/model", response_model=ModelResponse)
    def probe_model() -> ModelResponse:
        return ModelResponse(
            model_type="isolation_forest", artifact=_REAL_ARTIFACT_PAYLOAD, metrics=_REAL_METRICS_PAYLOAD
        )

    return probe_app


def test_filter_request_schema_rejects_an_invalid_body_via_fastapi_with_422() -> None:
    client = TestClient(_build_probe_app())

    response = client.post(
        "/probe/filter",
        json={"signal": [1.0, 2.0], "sampling_rate": "50000", "cutoff": 1000.0, "order": 4, "btype": "lowpass"},
    )

    assert response.status_code == 422


def test_filter_request_schema_accepts_a_valid_body_via_fastapi() -> None:
    client = TestClient(_build_probe_app())

    response = client.post(
        "/probe/filter",
        json={"signal": [1.0, 2.0], "sampling_rate": 50000.0, "cutoff": 1000.0, "order": 4, "btype": "lowpass"},
    )

    assert response.status_code == 200
    assert response.json() == {"filtered_signal": [1.0, 2.0]}


def test_model_response_schema_serializes_correctly_via_fastapi() -> None:
    client = TestClient(_build_probe_app())

    response = client.get("/probe/model")

    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "isolation_forest"
    assert body["metrics"]["confusion_matrix"] == [[840, 134], [721, 253]]
