import json
from pathlib import Path

import pytest

from app.datasets.loader import (
    DatasetLoaderError,
    NUM_CHANNELS,
    SAMPLING_RATE_HZ,
    load_all_splits,
    load_recording,
    load_split,
)
from app.models.signal import SignalLabel

REAL_DATASET_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "mafaulda"

# Real files, taken directly from data/processed/split_manifest.json (TASK 1.5.7) --
# not invented, not chosen at random. 3 per split, per this task's own AC.
REAL_SUBSET = {
    "train": [
        "horizontal-misalignment/0.5mm/12.288.csv",
        "horizontal-misalignment/0.5mm/13.5168.csv",
        "horizontal-misalignment/0.5mm/14.5408.csv",
    ],
    "validation": [
        "horizontal-misalignment/0.5mm/20.48.csv",
        "horizontal-misalignment/0.5mm/21.504.csv",
        "horizontal-misalignment/0.5mm/26.8288.csv",
    ],
    "test": [
        "horizontal-misalignment/0.5mm/18.8416.csv",
        "horizontal-misalignment/0.5mm/25.8048.csv",
        "horizontal-misalignment/0.5mm/33.1776.csv",
    ],
}


def _write_manifest(tmp_path: Path, splits: dict[str, list[str]]) -> Path:
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps({"splits": splits}), encoding="utf-8")
    return manifest_path


def _write_numeric_csv(path: Path, columns: int, rows: int = 3, header: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if header:
            f.write(",".join(f"col{i}" for i in range(columns)) + "\n")
        for r in range(rows):
            f.write(",".join(str(float(r + i)) for i in range(columns)) + "\n")


# --- Test 1 & 2: valid real data + split correctness (small real subset, 3 per split) ---


def test_real_subset_loads_with_correct_split_and_label(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, REAL_SUBSET)

    loaded = load_all_splits(dataset_root=REAL_DATASET_ROOT, manifest_path=manifest_path)

    assert set(loaded.keys()) == {"train", "validation", "test"}
    for split_name, relative_paths in REAL_SUBSET.items():
        recordings = loaded[split_name]
        assert len(recordings) == 3
        assert {r.relative_path for r in recordings} == set(relative_paths)
        for recording in recordings:
            # AC1: every recording is labeled with exactly the split assigned in the manifest.
            assert recording.split == split_name
            assert recording.label == SignalLabel.HORIZONTAL_MISALIGNMENT
            assert len(recording.signal_records) == NUM_CHANNELS
            channels = {sr.channel for sr in recording.signal_records}
            assert channels == set(range(NUM_CHANNELS))
            for sr in recording.signal_records:
                assert sr.sampling_rate == SAMPLING_RATE_HZ
                assert sr.file_path == recording.relative_path
                assert sr.operating_condition == "0.5mm"
                assert sr.machine_id is None


def test_load_split_matches_manifest_assignment_exactly(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, REAL_SUBSET)

    train = load_split("train", dataset_root=REAL_DATASET_ROOT, manifest_path=manifest_path)
    validation = load_split("validation", dataset_root=REAL_DATASET_ROOT, manifest_path=manifest_path)
    test = load_split("test", dataset_root=REAL_DATASET_ROOT, manifest_path=manifest_path)

    assert {r.relative_path for r in train} == set(REAL_SUBSET["train"])
    assert {r.relative_path for r in validation} == set(REAL_SUBSET["validation"])
    assert {r.relative_path for r in test} == set(REAL_SUBSET["test"])
    assert all(r.split == "train" for r in train)
    assert all(r.split == "validation" for r in validation)
    assert all(r.split == "test" for r in test)


def test_single_real_recording_via_load_recording() -> None:
    recording = load_recording("normal/12.288.csv", "train", dataset_root=REAL_DATASET_ROOT)

    assert recording.relative_path == "normal/12.288.csv"
    assert recording.split == "train"
    assert recording.label == SignalLabel.NORMAL
    assert len(recording.signal_records) == NUM_CHANNELS
    assert all(sr.operating_condition is None for sr in recording.signal_records)  # normal/ has no severity sub-folder


# --- Test 3: invalid structure ---


def test_wrong_column_count_is_rejected_explicitly(tmp_path: Path) -> None:
    _write_numeric_csv(tmp_path / "normal" / "bad.csv", columns=3)
    manifest_path = _write_manifest(tmp_path, {"train": ["normal/bad.csv"]})

    with pytest.raises(DatasetLoaderError, match="Channel count mismatch"):
        load_split("train", dataset_root=tmp_path, manifest_path=manifest_path)


def test_non_numeric_data_is_rejected_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "normal" / "text.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("a,b,c,d,e,f,g,h\nx,y,z,w,v,u,t,s\n", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path, {"train": ["normal/text.csv"]})

    with pytest.raises(DatasetLoaderError):
        load_split("train", dataset_root=tmp_path, manifest_path=manifest_path)


def test_all_channels_constant_is_rejected_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "normal" / "flat.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = ",".join(["1.0"] * NUM_CHANNELS)
    path.write_text("\n".join([row] * 5) + "\n", encoding="utf-8")
    manifest_path = _write_manifest(tmp_path, {"train": ["normal/flat.csv"]})

    with pytest.raises(DatasetLoaderError, match="constant or zero"):
        load_split("train", dataset_root=tmp_path, manifest_path=manifest_path)


# --- Test 4: missing file ---


def test_missing_file_is_rejected_explicitly(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, {"train": ["normal/does_not_exist.csv"]})

    with pytest.raises(DatasetLoaderError, match="does not exist"):
        load_split("train", dataset_root=tmp_path, manifest_path=manifest_path)


# --- Test 5: invalid label / structure ---


def test_unrecognized_state_folder_is_rejected_explicitly(tmp_path: Path) -> None:
    _write_numeric_csv(tmp_path / "not_a_real_state" / "12.288.csv", columns=NUM_CHANNELS, rows=5)
    manifest_path = _write_manifest(tmp_path, {"train": ["not_a_real_state/12.288.csv"]})

    with pytest.raises(DatasetLoaderError, match="UNKNOWN"):
        load_split("train", dataset_root=tmp_path, manifest_path=manifest_path)


# --- Path safety / manifest integrity ---


def test_path_escaping_dataset_root_is_rejected(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, {"train": ["../../etc/passwd"]})

    with pytest.raises(DatasetLoaderError, match="escapes dataset root"):
        load_split("train", dataset_root=tmp_path, manifest_path=manifest_path)


def test_unknown_split_name_is_rejected(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, REAL_SUBSET)

    with pytest.raises(DatasetLoaderError, match="not found in manifest"):
        load_split("nonexistent_split", dataset_root=REAL_DATASET_ROOT, manifest_path=manifest_path)


def test_missing_manifest_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetLoaderError, match="not found"):
        load_split("train", dataset_root=REAL_DATASET_ROOT, manifest_path=tmp_path / "no_such_manifest.json")
