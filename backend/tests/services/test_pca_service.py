"""TASK 12.2 -- tests for `app.services.pca_service`.

Runs against the real, committed `models/experiment_a_pca.pkl` (TASK 9.2's own
artifact) and the real MAFAULDA files it names -- there is no synthetic/mock
PCA representation to substitute, since the whole point of this service is to
reuse that real, already-fitted object (never re-fit it here).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models.signal import SignalLabel
from app.services import pca_service
from app.services.pca_service import (
    ADDITIONAL_REAL_FILES,
    PC_COUNT_FOR_VISUALIZATION,
    VISUALIZATION_FILES,
    PCAVisualizationError,
    _load_pca_representation,
    _verify_files_are_real,
    get_pca_visualization,
)


@pytest.fixture(scope="module")
def visualization():
    return get_pca_visualization()


# --- real files / real classes ---


def test_additional_real_files_are_genuinely_listed_in_the_split_manifest() -> None:
    _verify_files_are_real(ADDITIONAL_REAL_FILES)  # must not raise


def test_verify_files_are_real_rejects_a_file_not_in_the_manifest() -> None:
    with pytest.raises(PCAVisualizationError):
        _verify_files_are_real(["not-a-real-file.csv"])


def test_visualization_files_cover_all_four_real_dataset_classes() -> None:
    real_class_names = {label.value for label in SignalLabel}
    covered_class_names = {path.split("/")[0] for path in VISUALIZATION_FILES}

    assert covered_class_names == real_class_names


# --- missing artifact: refuse, never fabricate ---


def test_missing_pca_artifact_raises_instead_of_fabricating_points(tmp_path: Path) -> None:
    with pytest.raises(PCAVisualizationError):
        _load_pca_representation(tmp_path / "does-not-exist.pkl")


# --- real output shape/content ---


def test_get_pca_visualization_returns_one_point_per_real_window(visualization) -> None:
    points, explained_variance_ratio, total_components = visualization

    assert len(points) > 0
    assert len(explained_variance_ratio) == PC_COUNT_FOR_VISUALIZATION
    assert total_components >= PC_COUNT_FOR_VISUALIZATION


def test_every_point_has_a_real_label_and_real_coordinates(visualization) -> None:
    points, _explained_variance_ratio, _total_components = visualization
    real_labels = {label.value for label in SignalLabel}

    for point in points:
        assert point["label"] in real_labels
        assert isinstance(point["pc1"], float)
        assert isinstance(point["pc2"], float)
        assert isinstance(point["pc3"], float)
        assert point["recording_id"] in VISUALIZATION_FILES


def test_all_four_real_classes_are_present_among_the_returned_points(visualization) -> None:
    points, _explained_variance_ratio, _total_components = visualization

    assert {point["label"] for point in points} == {label.value for label in SignalLabel}


def test_result_is_cached_across_calls(visualization) -> None:
    assert get_pca_visualization() is visualization
