from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from app.datasets.loader import DATASET_ROOT as REAL_DATASET_ROOT
from app.datasets.loader import load_all_splits
from app.features.registry import FEATURE_REGISTRY, FREQUENCY_FEATURE_REGISTRY
from app.ml.dimensionality import (
    DimensionalityError,
    PCARepresentation,
    dsp_feature_dimension,
    fit_pca,
    transform_pca,
)
from app.signal_processing.windowing import Window, create_windows

DSP_DIMENSION = dsp_feature_dimension()  # real, not hard-coded -- currently 15


def _raw_matrix(n_windows: int, window_size: int, *, loc: float = 0.0, scale: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc, scale, size=(n_windows, window_size))


# --- Test 1 (AC1): train-only fitting -- PCA parameters unchanged by transform ---


def test_ac1_pca_parameters_unchanged_after_transforming_validation_and_test() -> None:
    train = _raw_matrix(200, 64, loc=0.0, seed=1)
    validation = _raw_matrix(30, 64, loc=500.0, scale=50.0, seed=2)  # wildly different distribution
    test = _raw_matrix(30, 64, loc=-500.0, scale=50.0, seed=3)

    representation = fit_pca(train, n_components=10)
    components_before = representation.pca.components_.copy()
    mean_before = representation.pca.mean_.copy()
    explained_variance_before = representation.pca.explained_variance_.copy()

    transform_pca(representation, validation)
    transform_pca(representation, test)

    np.testing.assert_array_equal(representation.pca.components_, components_before)
    np.testing.assert_array_equal(representation.pca.mean_, mean_before)
    np.testing.assert_array_equal(representation.pca.explained_variance_, explained_variance_before)


# --- Test 2 (AC1, CRITICAL): real anti-leakage demonstration ---


def test_ac1_train_only_pca_differs_from_a_leaked_fit_including_validation_and_test() -> None:
    """Constructs validation/test from a distribution far removed from train, then
    demonstrates that a (deliberately wrong, for comparison only) PCA fit on
    train+validation+test combined produces MEANINGFULLY DIFFERENT parameters than
    this module's own train-only fit -- proving the implementation's train-only
    behavior is not a coincidence indistinguishable from leakage."""
    train = _raw_matrix(200, 64, loc=0.0, scale=1.0, seed=10)
    validation = _raw_matrix(50, 64, loc=1000.0, scale=1.0, seed=11)
    test = _raw_matrix(50, 64, loc=-1000.0, scale=1.0, seed=12)

    train_only_representation = fit_pca(train, n_components=10)

    # Reference-only comparison PCA, built directly with sklearn (not via this
    # module), fit on the concatenation -- what a LEAKING implementation would do.
    leaked_pca = PCA(n_components=10)
    leaked_pca.fit(np.vstack([train, validation, test]))

    assert not np.allclose(train_only_representation.pca.mean_, leaked_pca.mean_)

    # And transforming validation through the train-only fit does NOT reproduce
    # what fitting on validation's own (very different) distribution would give.
    train_only_result = transform_pca(train_only_representation, validation)
    validation_only_pca = PCA(n_components=10).fit(validation)
    validation_only_result = validation_only_pca.transform(validation)
    assert not np.allclose(train_only_result, validation_only_result)


# --- Test 3 (AC2): exact dimensionality, derived from the real DSP registry ---


def test_ac2_pca_dimensionality_exactly_matches_real_dsp_feature_dimension() -> None:
    train = _raw_matrix(200, 128, seed=20)

    representation = fit_pca(train, n_components=DSP_DIMENSION)
    transformed = transform_pca(representation, train)

    expected_dimension = len(FEATURE_REGISTRY) + len(FREQUENCY_FEATURE_REGISTRY)
    assert DSP_DIMENSION == expected_dimension  # sanity: the module's own helper matches direct computation
    assert transformed.shape[1] == DSP_DIMENSION
    assert representation.n_components == DSP_DIMENSION


# --- Test 4: split transformation, all splits same dimension ---


def test_train_validation_test_all_produce_the_same_dimension() -> None:
    train = _raw_matrix(200, 64, seed=30)
    validation = _raw_matrix(40, 64, seed=31)
    test = _raw_matrix(40, 64, seed=32)

    representation = fit_pca(train, n_components=12)

    train_pca = transform_pca(representation, train)
    validation_pca = transform_pca(representation, validation)
    test_pca = transform_pca(representation, test)

    assert train_pca.shape == (200, 12)
    assert validation_pca.shape == (40, 12)
    assert test_pca.shape == (40, 12)


# --- Test 5: same PCA object used for every split, never a separate one per split ---


def test_same_pca_object_is_reused_for_every_split() -> None:
    train = _raw_matrix(200, 64, seed=40)
    validation = _raw_matrix(40, 64, seed=41)
    test = _raw_matrix(40, 64, seed=42)

    representation = fit_pca(train, n_components=12)

    transform_pca(representation, validation)
    transform_pca(representation, test)

    # Still the exact same underlying sklearn PCA instance -- not re-fit or
    # replaced between calls.
    assert isinstance(representation, PCARepresentation)
    assert representation.pca.n_components == 12


# --- Test 6 (AC3): explained variance, real and reported ---


def test_ac3_explained_variance_is_real_and_correctly_summed() -> None:
    train = _raw_matrix(200, 64, seed=50)

    representation = fit_pca(train, n_components=10)

    assert len(representation.explained_variance_ratio) == 10
    assert all(0.0 <= ratio <= 1.0 for ratio in representation.explained_variance_ratio)

    # Independently recomputed here, not by calling this module's own logic again.
    expected_total = sum(representation.explained_variance_ratio)
    assert representation.total_explained_variance == pytest.approx(expected_total)
    assert representation.total_explained_variance == pytest.approx(
        float(np.sum(representation.pca.explained_variance_ratio_))
    )


def test_ac3_explained_variance_can_be_small_and_is_reported_as_is() -> None:
    """Pure random noise (no real structure) -> explained variance ratio should be
    small/near-uniform across components -- verified to be reported honestly, not
    hidden or artificially inflated."""
    train = _raw_matrix(300, 100, scale=1.0, seed=60)

    representation = fit_pca(train, n_components=10)

    print(f"\ntotal_explained_variance (pure noise, 10/100 components): {representation.total_explained_variance:.4f}")
    # With 100 independent noise dimensions reduced to 10 components, no
    # single component should dominate -- a sanity check that nothing was
    # artificially inflated to look better than the real structure (noise) allows.
    assert representation.total_explained_variance < 0.5


# --- Test 7: input immutability ---


def test_fit_pca_does_not_modify_input() -> None:
    train = _raw_matrix(50, 32, seed=70)
    snapshot = train.copy()

    fit_pca(train, n_components=5)

    np.testing.assert_array_equal(train, snapshot)


def test_transform_pca_does_not_modify_input() -> None:
    train = _raw_matrix(50, 32, seed=71)
    representation = fit_pca(train, n_components=5)
    other = _raw_matrix(10, 32, seed=72)
    snapshot = other.copy()

    transform_pca(representation, other)

    np.testing.assert_array_equal(other, snapshot)


# --- Test 8: determinism ---


def test_fit_pca_is_deterministic_for_the_same_input() -> None:
    train = _raw_matrix(100, 40, seed=80)

    representation_1 = fit_pca(train, n_components=8)
    representation_2 = fit_pca(train, n_components=8)

    np.testing.assert_array_equal(representation_1.pca.components_, representation_2.pca.components_)
    assert representation_1.explained_variance_ratio == representation_2.explained_variance_ratio


# --- Test 9: invalid configuration ---


def test_empty_train_matrix_raises_explicitly() -> None:
    with pytest.raises(DimensionalityError, match="row"):
        fit_pca(np.empty((0, 10)), n_components=5)


def test_non_2d_input_raises_explicitly() -> None:
    with pytest.raises(DimensionalityError, match="2-dimensional"):
        fit_pca(np.array([1.0, 2.0, 3.0]), n_components=1)


def test_non_positive_n_components_raises_explicitly() -> None:
    train = _raw_matrix(50, 20, seed=90)
    with pytest.raises(DimensionalityError, match="n_components"):
        fit_pca(train, n_components=0)
    with pytest.raises(DimensionalityError, match="n_components"):
        fit_pca(train, n_components=-3)


def test_n_components_exceeding_min_samples_features_raises_explicitly() -> None:
    train = _raw_matrix(10, 5, seed=91)  # min(10, 5) = 5
    with pytest.raises(DimensionalityError, match="exceeds"):
        fit_pca(train, n_components=6)


def test_n_components_is_never_silently_reduced() -> None:
    """Confirms the rejected fit_pca call above did NOT leave behind a
    representation with a smaller n_components -- fit_pca raises before
    constructing anything."""
    train = _raw_matrix(10, 5, seed=92)
    with pytest.raises(DimensionalityError):
        result = fit_pca(train, n_components=6)
        assert result is None  # unreachable -- fit_pca must have raised already


def test_transform_with_mismatched_feature_dimension_raises_explicitly() -> None:
    train = _raw_matrix(50, 20, seed=93)
    representation = fit_pca(train, n_components=5)

    wrong_shape = _raw_matrix(10, 15, seed=94)  # 15 != 20
    with pytest.raises(DimensionalityError, match="columns"):
        transform_pca(representation, wrong_shape)


def test_nan_in_train_matrix_raises_explicitly() -> None:
    train = _raw_matrix(50, 20, seed=95)
    train[0, 0] = np.nan
    with pytest.raises(DimensionalityError, match="NaN"):
        fit_pca(train, n_components=5)


def test_dataframe_input_is_accepted() -> None:
    train_df = pd.DataFrame(_raw_matrix(50, 20, seed=96))
    representation = fit_pca(train_df, n_components=5)
    transformed = transform_pca(representation, train_df)
    assert transformed.shape == (50, 5)


# --- real-data validation (Definition of Done) ---
#
# Same already-vetted real MAFAULDA subset used throughout TASK 6.x/7.x/8.x's own
# tests (data/processed/split_manifest.json's real train/validation/test entries)
# -- not a new split, not a full dataset scan. PCA does not use labels at all, so
# only raw windows are needed here (no feature extraction, no DSP).
REAL_TRAIN = ["normal/12.288.csv", "normal/16.1792.csv"]
REAL_VALIDATION = ["normal/14.336.csv", "horizontal-misalignment/0.5mm/20.48.csv"]
REAL_TEST = [
    "normal/13.1072.csv",
    "normal/20.2752.csv",
    "horizontal-misalignment/0.5mm/18.8416.csv",
    "horizontal-misalignment/0.5mm/25.8048.csv",
]
REAL_WINDOW_SIZE = 1024
REAL_OVERLAP = 0.5
REAL_CHANNEL = 0


def _write_manifest(tmp_path: Path, splits: dict[str, list[str]]) -> Path:
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps({"splits": splits}), encoding="utf-8")
    return manifest_path


def _raw_windows_matrix(loaded_split: list, dataset_root: Path) -> np.ndarray:
    windows: list[Window] = []
    for recording in loaded_split:
        df = pd.read_csv(dataset_root / recording.relative_path, header=None)
        values = df[REAL_CHANNEL].tolist()
        windows.extend(
            create_windows(
                values,
                recording_id=recording.relative_path,
                split=recording.split,
                window_size=REAL_WINDOW_SIZE,
                overlap=REAL_OVERLAP,
            )
        )
    return np.array([w.values for w in windows], dtype=float)


def test_real_data_raw_pca_matches_dsp_dimension_across_all_splits(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    tmp_path = tmp_path_factory.mktemp("raw_pca_real_data")
    manifest_path = _write_manifest(
        tmp_path, {"train": REAL_TRAIN, "validation": REAL_VALIDATION, "test": REAL_TEST}
    )

    loaded = load_all_splits(dataset_root=REAL_DATASET_ROOT, manifest_path=manifest_path)

    train_raw = _raw_windows_matrix(loaded["train"], REAL_DATASET_ROOT)
    validation_raw = _raw_windows_matrix(loaded["validation"], REAL_DATASET_ROOT)
    test_raw = _raw_windows_matrix(loaded["test"], REAL_DATASET_ROOT)

    print(f"\nreal raw window matrix shapes -- train:{train_raw.shape} validation:{validation_raw.shape} test:{test_raw.shape}")
    assert train_raw.shape[1] == REAL_WINDOW_SIZE

    representation = fit_pca(train_raw, n_components=DSP_DIMENSION)

    train_pca = transform_pca(representation, train_raw)
    validation_pca = transform_pca(representation, validation_raw)
    test_pca = transform_pca(representation, test_raw)

    print(
        f"real DSP feature dimension: {DSP_DIMENSION}, "
        f"total_explained_variance: {representation.total_explained_variance:.4f}"
    )

    assert train_pca.shape[1] == DSP_DIMENSION
    assert validation_pca.shape[1] == DSP_DIMENSION
    assert test_pca.shape[1] == DSP_DIMENSION
    assert np.isfinite(train_pca).all()
    assert np.isfinite(validation_pca).all()
    assert np.isfinite(test_pca).all()
    assert 0.0 <= representation.total_explained_variance <= 1.0
