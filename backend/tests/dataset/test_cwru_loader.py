"""TASK 13.1 -- tests for `app.datasets.cwru_loader`.

The real CWRU Bearing Dataset is now present at `data/external/cwru/`
(uploaded by the user) -- this loader's directory/label/sampling-rate
contract was written and adjusted against that real, actual directory
listing and real `.mat` file contents (see the module's own docstring).
Tests here fall into two honest categories:

  1. Pure parsing-logic checks against tiny SYNTHETIC `.mat` fixtures built
     in-test via `scipy.io.savemat`, using the REAL directory-naming
     convention (`12k_..._Fault_Data/{B,IR,OR}/...`, bare `Normal/`) -- these
     verify the loader's parsing mechanics in isolation, never presented as
     real CWRU signal content.
  2. Checks against the REAL, present dataset at `data/external/cwru/`
     (`discover_recordings`/`load_all_recordings` with no root override) --
     genuine end-to-end verification, not a simulation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from app.datasets.cwru_loader import (
    DEFAULT_CWRU_ROOT,
    CWRULabel,
    CWRULoaderError,
    discover_recordings,
    load_all_recordings,
    load_recording,
)

_REAL_CWRU_DATASET_PRESENT = DEFAULT_CWRU_ROOT.exists()


def _write_synthetic_mat(path: Path, channel_keys: dict[str, np.ndarray]) -> None:
    """Writes a tiny, clearly-synthetic `.mat` file with the given variable
    names/arrays -- a test fixture shaped like the real CWRU `.mat` naming
    convention, never real CWRU signal content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(path), channel_keys)


@pytest.fixture()
def synthetic_signal() -> np.ndarray:
    # Deterministic, clearly-synthetic sine + noise, long enough for at least
    # one real MAFAULDA-configuration window (WINDOW_SIZE=1024).
    rng = np.random.default_rng(seed=42)
    t = np.arange(3000)
    return (np.sin(2 * np.pi * 50 * t / 12000) + 0.01 * rng.standard_normal(3000)).astype(np.float64)


# --- real, current absence of the dataset (against an isolated tmp root) ---


def test_discover_recordings_raises_when_dataset_root_missing(tmp_path: Path) -> None:
    with pytest.raises(CWRULoaderError, match="not found"):
        discover_recordings(dataset_root=tmp_path / "does-not-exist")


def test_load_all_recordings_raises_when_root_has_no_mat_files(tmp_path: Path) -> None:
    (tmp_path / "12k_Drive_End_Bearing_Fault_Data" / "B").mkdir(parents=True)
    with pytest.raises(CWRULoaderError, match="no \\.mat files"):
        load_all_recordings(dataset_root=tmp_path)


# --- synthetic-fixture parsing checks (real directory-naming convention) ---


def test_label_and_sampling_rate_derived_from_directory_name(tmp_path: Path, synthetic_signal) -> None:
    relative_path = "12k_Drive_End_Bearing_Fault_Data/IR/007/105_0.mat"
    _write_synthetic_mat(tmp_path / relative_path, {"X105_DE_time": synthetic_signal})

    recordings = load_recording(relative_path, dataset_root=tmp_path)

    assert len(recordings) == 1
    recording = recordings[0]
    assert recording.label == CWRULabel.INNER_RACE
    assert recording.sampling_rate == 12000.0
    assert recording.channel_name == "X105_DE_time"
    assert len(recording.values) == 3000


def test_48k_directory_prefix_resolves_to_48000_hz(tmp_path: Path, synthetic_signal) -> None:
    relative_path = "48k_Drive_End_Bearing_Fault_Data/OR/014/201@6_0.mat"
    _write_synthetic_mat(
        tmp_path / relative_path,
        {"X201_DE_time": synthetic_signal, "X201_FE_time": synthetic_signal * 2},
    )

    recordings = load_recording(relative_path, dataset_root=tmp_path)

    assert {r.channel_name for r in recordings} == {"X201_DE_time", "X201_FE_time"}
    assert all(r.label == CWRULabel.OUTER_RACE and r.sampling_rate == 48000.0 for r in recordings)


def test_explicit_sampling_rate_override_wins_over_directory(tmp_path: Path, synthetic_signal) -> None:
    relative_path = "12k_Drive_End_Bearing_Fault_Data/B/007/118_0.mat"
    _write_synthetic_mat(tmp_path / relative_path, {"X118_DE_time": synthetic_signal})

    recordings = load_recording(relative_path, dataset_root=tmp_path, sampling_rate_hz=48000.0)

    assert recordings[0].sampling_rate == 48000.0


def test_unrecognized_label_directory_raises(tmp_path: Path, synthetic_signal) -> None:
    relative_path = "12k_Drive_End_Bearing_Fault_Data/mystery-fault/1.mat"
    _write_synthetic_mat(tmp_path / relative_path, {"X001_DE_time": synthetic_signal})

    with pytest.raises(CWRULoaderError, match="label directories"):
        load_recording(relative_path, dataset_root=tmp_path)


def test_normal_file_without_explicit_override_raises(tmp_path: Path, synthetic_signal) -> None:
    """`Normal/` carries no `<N>k_...` directory segment in the real dataset
    -- see module docstring's disclosed sampling-rate ambiguity."""
    relative_path = "Normal/97_Normal_0.mat"
    _write_synthetic_mat(tmp_path / relative_path, {"X097_DE_time": synthetic_signal})

    with pytest.raises(CWRULoaderError, match="sampling rate"):
        load_recording(relative_path, dataset_root=tmp_path)


def test_normal_file_with_explicit_override_succeeds(tmp_path: Path, synthetic_signal) -> None:
    relative_path = "Normal/97_Normal_0.mat"
    _write_synthetic_mat(tmp_path / relative_path, {"X097_DE_time": synthetic_signal})

    recordings = load_recording(relative_path, dataset_root=tmp_path, sampling_rate_hz=12000.0)

    assert recordings[0].label == CWRULabel.NORMAL
    assert recordings[0].sampling_rate == 12000.0


def test_file_with_no_recognizable_time_series_variable_raises(tmp_path: Path, synthetic_signal) -> None:
    relative_path = "12k_Drive_End_Bearing_Fault_Data/B/007/1_0.mat"
    _write_synthetic_mat(tmp_path / relative_path, {"unrelated_variable": synthetic_signal})

    with pytest.raises(CWRULoaderError, match="naming convention"):
        load_recording(relative_path, dataset_root=tmp_path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(CWRULoaderError, match="not found"):
        load_recording("12k_Drive_End_Bearing_Fault_Data/B/007/missing.mat", dataset_root=tmp_path)


def test_discover_and_load_all_recordings_across_a_small_synthetic_tree(tmp_path: Path, synthetic_signal) -> None:
    _write_synthetic_mat(tmp_path / "Normal/1_Normal_0.mat", {"X001_DE_time": synthetic_signal})
    _write_synthetic_mat(
        tmp_path / "12k_Drive_End_Bearing_Fault_Data/B/007/2_0.mat", {"X002_DE_time": synthetic_signal}
    )

    relative_paths = discover_recordings(dataset_root=tmp_path)
    assert relative_paths == [
        "12k_Drive_End_Bearing_Fault_Data/B/007/2_0.mat",
        "Normal/1_Normal_0.mat",
    ]

    recordings = load_all_recordings(dataset_root=tmp_path, normal_sampling_rate_hz=12000.0)
    assert {r.label for r in recordings} == {CWRULabel.NORMAL, CWRULabel.BALL}


def test_load_all_recordings_raises_for_normal_files_without_override(tmp_path: Path, synthetic_signal) -> None:
    _write_synthetic_mat(tmp_path / "Normal/1_Normal_0.mat", {"X001_DE_time": synthetic_signal})

    with pytest.raises(CWRULoaderError, match="sampling rate"):
        load_all_recordings(dataset_root=tmp_path)


# --- against the real, present dataset ---


@pytest.mark.skipif(not _REAL_CWRU_DATASET_PRESENT, reason="Real CWRU dataset not present at data/external/cwru/")
def test_real_dataset_discovers_the_expected_number_of_real_mat_files() -> None:
    relative_paths = discover_recordings()
    assert len(relative_paths) == 161


@pytest.mark.skipif(not _REAL_CWRU_DATASET_PRESENT, reason="Real CWRU dataset not present at data/external/cwru/")
def test_real_dataset_loads_all_four_real_labels_with_a_normal_override() -> None:
    recordings = load_all_recordings(normal_sampling_rate_hz=12000.0)
    assert {r.label for r in recordings} == {
        CWRULabel.NORMAL,
        CWRULabel.BALL,
        CWRULabel.INNER_RACE,
        CWRULabel.OUTER_RACE,
    }
    assert {r.sampling_rate for r in recordings} == {12000.0, 48000.0}


@pytest.mark.skipif(not _REAL_CWRU_DATASET_PRESENT, reason="Real CWRU dataset not present at data/external/cwru/")
def test_real_dataset_raises_without_a_normal_override() -> None:
    with pytest.raises(CWRULoaderError, match="sampling rate"):
        load_all_recordings()
