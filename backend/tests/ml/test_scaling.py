from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from app.datasets.loader import DATASET_ROOT as REAL_DATASET_ROOT
from app.datasets.loader import load_all_splits
from app.datasets.validators import SAMPLING_RATE_HZ
from app.features.extractor import extract_feature_matrix
from app.ml import scaling
from app.ml.scaling import (
    DEFAULT_SCALER_FILENAME,
    ScalingError,
    apply_scaler,
    fit_scaler,
    load_scaler,
    save_scaler,
)
from app.signal_processing.windowing import Window, create_windows


# --- Test 1 / CRITICAL anti-leakage test (section 7/29) ---


def test_fit_scaler_uses_train_statistics_only_not_combined_with_other_distributions() -> None:
    """If someone changed the implementation to
    `StandardScaler().fit(np.vstack([train, val]))`, this test would fail: train's
    mean is 3.0, but a train+val combined mean would be dragged far toward 2000."""
    train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    validation = np.array([[1000.0], [2000.0], [3000.0]])

    scaler = fit_scaler(train)

    assert scaler.mean_[0] == pytest.approx(3.0)
    assert scaler.mean_[0] != pytest.approx(np.mean(np.vstack([train, validation])))

    # Applying it to validation must use TRAIN's mean/std, not re-derive from
    # validation's own (very different) distribution.
    expected_scaled_validation = (validation - np.mean(train)) / np.std(train, ddof=0)
    scaled_validation = apply_scaler(scaler, validation)
    np.testing.assert_allclose(scaled_validation, expected_scaled_validation)


# --- Test 2 — train transform: mean~=0, std~=1 ---


def test_train_transform_has_mean_zero_and_std_one_per_feature() -> None:
    rng = np.random.default_rng(42)
    train = rng.normal(loc=[10.0, -5.0, 100.0], scale=[2.0, 0.5, 20.0], size=(500, 3))

    scaler = fit_scaler(train)
    scaled_train = apply_scaler(scaler, train)

    np.testing.assert_allclose(np.mean(scaled_train, axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.std(scaled_train, axis=0, ddof=0), 1.0, atol=1e-10)


# --- Test 3 — validation uses train parameters (not centered at 0 in general) ---


def test_validation_transform_is_not_centered_at_zero_when_distributions_differ() -> None:
    train = np.array([[0.0], [1.0], [2.0]])
    validation = np.array([[100.0], [101.0], [102.0]])

    scaler = fit_scaler(train)
    scaled_validation = apply_scaler(scaler, validation)

    assert np.mean(scaled_validation) != pytest.approx(0.0)
    # Explicit, independently-derived expected value: train mean=1, train std=
    # sqrt(2/3); validation transformed using exactly those train parameters.
    expected = (validation - 1.0) / np.std([0.0, 1.0, 2.0], ddof=0)
    np.testing.assert_allclose(scaled_validation, expected)


# --- Test 4 — apply_scaler does not refit / mutate the scaler ---


def test_apply_scaler_does_not_modify_scaler_parameters() -> None:
    train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    scaler = fit_scaler(train)

    mean_before = scaler.mean_.copy()
    scale_before = scaler.scale_.copy()
    var_before = scaler.var_.copy()

    wildly_different = np.array([[9999.0, -9999.0], [5000.0, 1.0]])
    apply_scaler(scaler, wildly_different)
    apply_scaler(scaler, train)

    np.testing.assert_array_equal(scaler.mean_, mean_before)
    np.testing.assert_array_equal(scaler.scale_, scale_before)
    np.testing.assert_array_equal(scaler.var_, var_before)


def test_apply_scaler_never_calls_fit_or_fit_transform() -> None:
    """Static check that `apply_scaler`'s own source never calls `.fit(` or
    `.fit_transform(` -- semantically apply = transform only."""
    source = inspect.getsource(apply_scaler)
    assert ".fit(" not in source
    assert ".fit_transform(" not in source
    assert ".partial_fit(" not in source
    assert ".transform(" in source


# --- Test 5 — test split uses train parameters (same principle as validation) ---


def test_test_split_transform_uses_train_parameters() -> None:
    train = np.array([[5.0], [10.0], [15.0]])
    test = np.array([[500.0], [1000.0]])

    scaler = fit_scaler(train)
    scaled_test = apply_scaler(scaler, test)

    expected = (test - 10.0) / np.std([5.0, 10.0, 15.0], ddof=0)
    np.testing.assert_allclose(scaled_test, expected)


# --- Test 6 — shape preservation (ndarray and DataFrame) ---


def test_shape_is_preserved_for_ndarray() -> None:
    train = np.random.default_rng(0).normal(size=(20, 4))
    scaler = fit_scaler(train)

    scaled = apply_scaler(scaler, train)

    assert scaled.shape == train.shape


def test_shape_and_columns_are_preserved_for_dataframe() -> None:
    train = pd.DataFrame(
        np.random.default_rng(1).normal(size=(15, 3)), columns=["mean", "std", "rms"]
    )
    scaler = fit_scaler(train)

    scaled = apply_scaler(scaler, train)

    assert isinstance(scaled, pd.DataFrame)
    assert scaled.shape == train.shape
    assert list(scaled.columns) == list(train.columns)
    assert list(scaled.index) == list(train.index)


# --- Test 7 — input immutability ---


def test_fit_scaler_does_not_modify_input() -> None:
    train = np.array([[1.0, 2.0], [3.0, 4.0]])
    snapshot = train.copy()

    fit_scaler(train)

    np.testing.assert_array_equal(train, snapshot)


def test_apply_scaler_does_not_modify_input() -> None:
    train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = fit_scaler(train)
    snapshot = train.copy()

    apply_scaler(scaler, train)

    np.testing.assert_array_equal(train, snapshot)


def test_apply_scaler_does_not_modify_input_dataframe() -> None:
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    scaler = fit_scaler(train)
    snapshot = train.copy()

    apply_scaler(scaler, train)

    pd.testing.assert_frame_equal(train, snapshot)


# --- Test 8 — invalid feature dimension ---


def test_apply_scaler_raises_explicitly_on_feature_count_mismatch() -> None:
    train = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    scaler = fit_scaler(train)

    mismatched = np.array([[1.0, 2.0, 3.0, 4.0]])
    with pytest.raises(ScalingError, match="columns"):
        apply_scaler(scaler, mismatched)


def test_apply_scaler_raises_explicitly_on_dataframe_column_name_mismatch() -> None:
    train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    scaler = fit_scaler(train)

    reordered = pd.DataFrame({"b": [3.0, 4.0], "a": [1.0, 2.0]})
    with pytest.raises(ScalingError, match="columns"):
        apply_scaler(scaler, reordered)


def test_fit_scaler_raises_explicitly_on_empty_matrix() -> None:
    with pytest.raises(ScalingError, match="row"):
        fit_scaler(np.empty((0, 3)))


def test_fit_scaler_raises_explicitly_on_1d_input() -> None:
    with pytest.raises(ScalingError, match="2-dimensional"):
        fit_scaler(np.array([1.0, 2.0, 3.0]))


def test_fit_scaler_raises_explicitly_on_nan() -> None:
    train = np.array([[1.0, np.nan], [2.0, 3.0]])
    with pytest.raises(ScalingError, match="NaN"):
        fit_scaler(train)


def test_fit_scaler_raises_explicitly_on_inf() -> None:
    train = np.array([[1.0, np.inf], [2.0, 3.0]])
    with pytest.raises(ScalingError, match="NaN"):
        fit_scaler(train)


def test_apply_scaler_raises_explicitly_on_nan() -> None:
    train = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler = fit_scaler(train)
    with pytest.raises(ScalingError, match="NaN"):
        apply_scaler(scaler, np.array([[np.nan, 1.0]]))


# --- Test 9 — serialization / reload ---


def test_save_and_load_scaler_produces_identical_transform(tmp_path: Path) -> None:
    train = np.random.default_rng(7).normal(size=(50, 5))
    other = np.random.default_rng(8).normal(size=(10, 5))

    scaler = fit_scaler(train)
    path = save_scaler(scaler, tmp_path / DEFAULT_SCALER_FILENAME)

    assert path.exists()
    assert path.name == "scaler_v1.pkl"

    reloaded = load_scaler(path)

    original_output = apply_scaler(scaler, other)
    reloaded_output = apply_scaler(reloaded, other)

    np.testing.assert_allclose(original_output, reloaded_output)
    np.testing.assert_array_equal(reloaded.mean_, scaler.mean_)
    np.testing.assert_array_equal(reloaded.scale_, scaler.scale_)


def test_save_scaler_creates_parent_directories(tmp_path: Path) -> None:
    train = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler = fit_scaler(train)

    nested_path = tmp_path / "artifacts" / "v1" / DEFAULT_SCALER_FILENAME
    result_path = save_scaler(scaler, nested_path)

    assert result_path == nested_path
    assert nested_path.exists()


# --- zero-variance / constant feature (section 31) ---


def test_constant_feature_does_not_raise_and_scales_to_zero() -> None:
    """StandardScaler's own well-defined zero-variance behavior (scale_=1.0, not a
    divide-by-zero) is relied on as-is -- no custom workaround introduced."""
    train = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])

    scaler = fit_scaler(train)

    assert scaler.scale_[1] == pytest.approx(1.0)
    scaled = apply_scaler(scaler, train)
    np.testing.assert_allclose(scaled[:, 1], 0.0)


# --- architecture: scaling.py depends on nothing model-specific ---


def test_scaling_module_does_not_import_any_model_module() -> None:
    """Static, source-level dependency-direction check: `scaling.py` must not
    import `isolation_forest`, `autoencoder`, or `training` -- the dependency
    direction is strictly models -> scaling, never the reverse. Stays valid
    regardless of whether those model modules exist yet."""
    source = inspect.getsource(scaling)
    tree = ast.parse(source)

    imported_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.append(node.module)

    forbidden_substrings = ("isolation_forest", "autoencoder", "training")
    for name in imported_names:
        for forbidden in forbidden_substrings:
            assert forbidden not in name, f"scaling.py must not import {name!r}"


# --- real-data validation (Definition of Done) ---
#
# Same already-vetted real subset used by TASK 2.5/5.3's own tests (data/raw/
# mafaulda/horizontal-misalignment/0.5mm/*.csv via the real split_manifest.json) --
# not a new split, not a full dataset scan.
REAL_SUBSET = {
    "train": [
        "horizontal-misalignment/0.5mm/12.288.csv",
        "horizontal-misalignment/0.5mm/13.5168.csv",
    ],
    "validation": [
        "horizontal-misalignment/0.5mm/20.48.csv",
    ],
    "test": [
        "horizontal-misalignment/0.5mm/18.8416.csv",
    ],
}
REAL_WINDOW_SIZE = 1024
REAL_OVERLAP = 0.5
REAL_CHANNEL = 0
REAL_NPERSEG = 256
REAL_NOVERLAP = 128


def _write_manifest(tmp_path: Path, splits: dict[str, list[str]]) -> Path:
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps({"splits": splits}), encoding="utf-8")
    return manifest_path


@pytest.fixture(scope="module")
def real_feature_matrices(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    """Real MAFAULDA subset (2 train / 1 validation / 1 test recordings) -> real
    windowing -> real TASK 5.3 feature extraction, exactly as TASK 5.3's own test
    does. No new split, no full-dataset scan, no fitting/scaling happens here yet."""
    tmp_path = tmp_path_factory.mktemp("scaling_real_data")
    manifest_path = _write_manifest(tmp_path, REAL_SUBSET)

    loaded = load_all_splits(dataset_root=REAL_DATASET_ROOT, manifest_path=manifest_path)

    matrices: dict[str, pd.DataFrame] = {}
    for split_name, recordings in loaded.items():
        windows: list[Window] = []
        for recording in recordings:
            df = pd.read_csv(REAL_DATASET_ROOT / recording.relative_path, header=None)
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
        matrices[split_name] = extract_feature_matrix(
            windows, SAMPLING_RATE_HZ, nperseg=REAL_NPERSEG, noverlap=REAL_NOVERLAP
        )

    return matrices


def test_real_data_scaler_fit_on_train_only_and_applied_to_all_splits(
    real_feature_matrices: dict[str, pd.DataFrame],
) -> None:
    train_matrix = real_feature_matrices["train"]
    validation_matrix = real_feature_matrices["validation"]
    test_matrix = real_feature_matrices["test"]

    assert len(train_matrix) > 0
    assert len(validation_matrix) > 0
    assert len(test_matrix) > 0

    scaler = fit_scaler(train_matrix)

    scaled_train = apply_scaler(scaler, train_matrix)
    scaled_validation = apply_scaler(scaler, validation_matrix)
    scaled_test = apply_scaler(scaler, test_matrix)

    # Shape preserved for every split.
    assert scaled_train.shape == train_matrix.shape
    assert scaled_validation.shape == validation_matrix.shape
    assert scaled_test.shape == test_matrix.shape

    # No NaN/Inf introduced by scaling.
    assert np.isfinite(scaled_train.to_numpy()).all()
    assert np.isfinite(scaled_validation.to_numpy()).all()
    assert np.isfinite(scaled_test.to_numpy()).all()

    # Train (and only train) is guaranteed mean~=0 per feature, always -- even a
    # zero-variance feature centers to exactly 0.
    np.testing.assert_allclose(scaled_train.mean(axis=0).to_numpy(), 0.0, atol=1e-8)

    # std~=1 per feature -- EXCEPT for a real feature that happens to have zero
    # variance across this train subset (e.g. `band_energy_10_100hz`, whose
    # documented "< 2 bins -> 0.0" behavior triggers here: at nperseg=256/fs=50,000
    # the ~195 Hz bin spacing leaves zero full bins inside [10, 100] Hz, so this
    # feature is legitimately constant (0.0) for every real window at this
    # configuration). StandardScaler's own well-defined zero-variance behavior
    # (scale_=1.0, scaled output identically 0.0, std of a constant column is 0, not
    # 1) is asserted explicitly here rather than assumed away.
    nonzero_variance = scaler.var_ > 0
    zero_variance_columns = [
        name for name, has_variance in zip(train_matrix.columns, nonzero_variance) if not has_variance
    ]
    print(f"\nzero-variance train columns (excluded from the std~=1 check): {zero_variance_columns}")

    std_scaled_train = scaled_train.std(axis=0, ddof=0).to_numpy()
    np.testing.assert_allclose(std_scaled_train[nonzero_variance], 1.0, atol=1e-8)
    np.testing.assert_allclose(std_scaled_train[~nonzero_variance], 0.0, atol=1e-8)

    # Column identity/order preserved.
    assert list(scaled_train.columns) == list(train_matrix.columns)
    assert list(scaled_validation.columns) == list(validation_matrix.columns)
    assert list(scaled_test.columns) == list(test_matrix.columns)
