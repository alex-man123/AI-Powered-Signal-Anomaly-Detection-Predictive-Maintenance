import csv
from pathlib import Path

from app.datasets.mafaulda_parser import UNKNOWN, parse_recording_state

REPO_ROOT = Path(__file__).resolve().parents[3]
INVENTORY_CSV = REPO_ROOT / "docs" / "dataset_audit" / "file_inventory.csv"


def test_known_real_pattern_without_condition_subfolder() -> None:
    # Real path observed in data/raw/mafaulda/normal/ (no severity sub-folder).
    mapping = parse_recording_state("normal/12.288.csv")

    assert mapping.state == "normal"
    assert mapping.condition is None


def test_known_real_pattern_with_condition_subfolder() -> None:
    # Real path observed in data/raw/mafaulda/imbalance/10g/.
    mapping = parse_recording_state("imbalance/10g/13.9264.csv")

    assert mapping.state == "imbalance"
    assert mapping.condition == "10g"


def test_unknown_path_is_marked_unknown_not_ignored() -> None:
    mapping = parse_recording_state("some_unknown_folder/random_file.csv")

    assert mapping.state == UNKNOWN
    assert mapping.condition is None


def test_path_variation_across_all_four_observed_states() -> None:
    # Covers the other two real state folders (misalignment variants), each with a
    # different severity sub-folder naming unit (mm vs g).
    horizontal = parse_recording_state("horizontal-misalignment/0.5mm/12.288.csv")
    vertical = parse_recording_state("vertical-misalignment/0.51mm/12.4928.csv")

    assert horizontal.state == "horizontal-misalignment"
    assert horizontal.condition == "0.5mm"
    assert vertical.state == "vertical-misalignment"
    assert vertical.condition == "0.51mm"


def test_flat_filename_with_no_folder_is_unknown() -> None:
    mapping = parse_recording_state("loose_file.csv")

    assert mapping.state == UNKNOWN


def test_unexpected_deeper_nesting_is_unknown() -> None:
    mapping = parse_recording_state("normal/extra/nested/file.csv")

    assert mapping.state == UNKNOWN


def test_every_row_in_real_inventory_maps_to_a_known_state() -> None:
    """AC1/AC2, demonstrated against the real TASK 1.5.1 inventory: every one of the
    880 audited files must resolve to a known state (zero UNKNOWN expected, since the
    parser's KNOWN_STATES was derived from this exact dataset)."""
    with INVENTORY_CSV.open(newline="", encoding="utf-8") as f:
        inventory_rows = list(csv.DictReader(f))

    assert len(inventory_rows) > 0

    mappings = [parse_recording_state(row["relative_path"]) for row in inventory_rows]

    assert len(mappings) == len(inventory_rows)

    unknown = [m.relative_path for m in mappings if m.state == UNKNOWN]
    assert unknown == []

    observed_states = {m.state for m in mappings}
    assert observed_states == {
        "normal",
        "imbalance",
        "horizontal-misalignment",
        "vertical-misalignment",
    }
