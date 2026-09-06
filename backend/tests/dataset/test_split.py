import pytest

from app.datasets.class_distribution_audit import (
    analyze_imbalance,
    build_class_distribution,
    load_inventory_rows,
)
from app.datasets.mafaulda_parser import parse_recording_state
from app.datasets.split import (
    SplitRecord,
    create_split,
    validate_complete_coverage,
    validate_major_class_representation,
    validate_no_overlap,
)


def _synthetic_records(counts: dict[str, int]) -> list[SplitRecord]:
    records = []
    for state, n in counts.items():
        for i in range(n):
            records.append(SplitRecord(relative_path=f"{state}/file_{i:04d}.csv", state=state))
    return records


# --- Test 1: determinism ---


def test_same_seed_produces_identical_split() -> None:
    records = _synthetic_records({"normal": 50, "imbalance": 100})

    result_1 = create_split(records, seed=42)
    result_2 = create_split(records, seed=42)

    assert result_1 == result_2


def test_different_seed_can_produce_a_different_split() -> None:
    records = _synthetic_records({"normal": 50, "imbalance": 100})

    result_1 = create_split(records, seed=42)
    result_2 = create_split(records, seed=7)

    assert result_1.train != result_2.train or result_1.test != result_2.test


# --- Test 2: no overlap ---


def test_no_overlap_between_splits() -> None:
    records = _synthetic_records({"normal": 50, "imbalance": 100, "vertical-misalignment": 80})

    result = create_split(records, seed=42)

    assert validate_no_overlap(result) == []


# --- Test 3: complete coverage ---


def test_complete_coverage_of_input() -> None:
    records = _synthetic_records({"normal": 50, "imbalance": 100})
    all_paths = {r.relative_path for r in records}

    result = create_split(records, seed=42)

    assert validate_complete_coverage(result, all_paths) == []


# --- Test 4: class representation ---


def test_every_major_class_appears_in_every_split() -> None:
    records = _synthetic_records(
        {"normal": 49, "imbalance": 333, "horizontal-misalignment": 197, "vertical-misalignment": 301}
    )

    result = create_split(records, seed=42)

    major_classes = {"normal", "imbalance", "horizontal-misalignment", "vertical-misalignment"}
    assert validate_major_class_representation(result, major_classes) == []


# --- Test 5: invalid ratios ---


@pytest.mark.parametrize(
    "train_ratio,val_ratio,test_ratio",
    [
        (0.0, 0.5, 0.5),
        (0.7, 0.0, 0.3),
        (0.7, 0.15, 0.0),
        (0.7, 0.15, 0.20),  # sums to 1.05, not 1.0
        (-0.1, 0.6, 0.5),
    ],
)
def test_invalid_ratios_raise(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    records = _synthetic_records({"normal": 10})

    with pytest.raises(ValueError):
        create_split(records, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)


def test_empty_records_raise() -> None:
    with pytest.raises(ValueError):
        create_split([])


def test_record_with_missing_state_raises() -> None:
    records = [SplitRecord(relative_path="normal/a.csv", state="normal"), SplitRecord(relative_path="b.csv", state="")]

    with pytest.raises(ValueError):
        create_split(records)


# --- Test 6: insufficient samples ---


def test_class_with_fewer_than_three_files_is_flagged_insufficient() -> None:
    records = _synthetic_records({"normal": 50, "rare-class": 2})

    result = create_split(records, seed=42)

    assert "rare-class" in result.insufficient_classes
    assert "normal" not in result.insufficient_classes
    # still no duplicates/overlap and still fully covers the input, even for the
    # under-provisioned class -- it just can't have all 3 splits non-empty.
    all_paths = {r.relative_path for r in records}
    assert validate_no_overlap(result) == []
    assert validate_complete_coverage(result, all_paths) == []
    rare_counts = result.class_distribution["rare-class"]
    assert sum(rare_counts.values()) == 2


def test_class_with_exactly_zero_files_never_occurs_but_three_is_the_minimum_safe_count() -> None:
    records = _synthetic_records({"normal": 50, "tiny": 3})

    result = create_split(records, seed=42)

    assert "tiny" not in result.insufficient_classes
    tiny_counts = result.class_distribution["tiny"]
    assert all(count >= 1 for count in tiny_counts.values())
    assert sum(tiny_counts.values()) == 3


# --- Real dataset validation ---


def test_real_dataset_split_covers_every_inventoried_file_with_no_overlap() -> None:
    rows = load_inventory_rows()
    records = [
        SplitRecord(relative_path=row["relative_path"], state=parse_recording_state(row["relative_path"]).state)
        for row in rows
    ]
    all_paths = {r.relative_path for r in records}

    result = create_split(records, seed=42)

    assert validate_no_overlap(result) == []
    assert validate_complete_coverage(result, all_paths) == []
    assert len(result.train) + len(result.validation) + len(result.test) == len(rows)


def test_real_dataset_every_major_class_represented_in_every_split() -> None:
    rows = load_inventory_rows()
    records = [
        SplitRecord(relative_path=row["relative_path"], state=parse_recording_state(row["relative_path"]).state)
        for row in rows
    ]

    class_counts = build_class_distribution(rows)
    imbalance = analyze_imbalance(class_counts)
    major_classes = {
        state for state in class_counts if state not in imbalance.severely_underrepresented and state != "UNKNOWN"
    }
    assert major_classes == {"normal", "imbalance", "horizontal-misalignment", "vertical-misalignment"}

    result = create_split(records, seed=42)

    assert validate_major_class_representation(result, major_classes) == []
    assert result.insufficient_classes == ()


def test_real_dataset_split_is_deterministic_across_two_full_runs() -> None:
    rows = load_inventory_rows()
    records = [
        SplitRecord(relative_path=row["relative_path"], state=parse_recording_state(row["relative_path"]).state)
        for row in rows
    ]

    result_1 = create_split(records, seed=42)
    result_2 = create_split(records, seed=42)

    assert result_1 == result_2
