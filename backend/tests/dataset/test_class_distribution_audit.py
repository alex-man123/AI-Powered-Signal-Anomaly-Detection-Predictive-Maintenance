from app.datasets.class_distribution_audit import (
    ROTATION_SOURCES,
    analyze_imbalance,
    build_class_distribution,
    build_rotation_audit,
    extract_rotation_info,
    load_inventory_rows,
)
from app.datasets.mafaulda_parser import UNKNOWN


def test_class_distribution_sums_to_inventory_total() -> None:
    rows = load_inventory_rows()
    class_counts = build_class_distribution(rows)

    assert sum(class_counts.values()) == len(rows)


def test_class_distribution_omits_no_class_and_has_zero_unknown() -> None:
    """AC1, against the real inventory: every one of the 4 real states from TASK 1.5.2
    is present, and (per TASK 1.5.2's own audit) UNKNOWN never occurs for this dataset."""
    rows = load_inventory_rows()
    class_counts = build_class_distribution(rows)

    assert set(class_counts) == {
        "normal",
        "imbalance",
        "horizontal-misalignment",
        "vertical-misalignment",
    }
    assert UNKNOWN not in class_counts


def test_analyze_imbalance_flags_a_synthetic_severely_underrepresented_class() -> None:
    """Proves the <10%-of-mean rule actually triggers — the real dataset doesn't
    exercise this branch, so it must be proven with a synthetic, clearly-imbalanced
    input instead of asserted against real (not severely imbalanced) data."""
    class_counts = {"a": 100, "b": 100, "c": 100, "d": 5}  # mean=76.25, 10%=7.625

    analysis = analyze_imbalance(class_counts)

    assert analysis.severely_underrepresented == ["d"]
    assert analysis.mean_files_per_class == 76.25


def test_analyze_imbalance_flags_nothing_when_balanced() -> None:
    class_counts = {"a": 100, "b": 95, "c": 105, "d": 98}

    analysis = analyze_imbalance(class_counts)

    assert analysis.severely_underrepresented == []


def test_analyze_imbalance_excludes_unknown_from_the_mean() -> None:
    class_counts = {"a": 100, "b": 100, UNKNOWN: 1}

    analysis = analyze_imbalance(class_counts)

    assert UNKNOWN not in analysis.percentage_of_mean
    assert analysis.mean_files_per_class == 100.0


def test_extract_rotation_info_known_real_filename_pattern() -> None:
    # Real filename observed in data/raw/mafaulda/normal/.
    info = extract_rotation_info("normal/12.288.csv", state="normal")

    assert info.rotation_source == "filename"
    assert info.rotation_frequency_hz == 12.288
    assert "12.288.csv" in info.evidence


def test_extract_rotation_info_unparseable_filename_is_unavailable_not_guessed() -> None:
    info = extract_rotation_info("some_state/not_a_number.csv", state="some_state")

    assert info.rotation_source == "unavailable"
    assert info.rotation_frequency_hz is None


def test_rotation_source_is_always_one_of_the_defined_enum_values() -> None:
    rows = load_inventory_rows()
    rotation_infos = build_rotation_audit(rows)

    assert len(rotation_infos) == len(rows)
    assert all(info.rotation_source in ROTATION_SOURCES for info in rotation_infos)


def test_unavailable_rotation_source_always_implies_null_frequency() -> None:
    rows = load_inventory_rows()
    rotation_infos = build_rotation_audit(rows)

    for info in rotation_infos:
        if info.rotation_source == "unavailable":
            assert info.rotation_frequency_hz is None
        else:
            assert info.rotation_frequency_hz is not None


def test_every_real_recording_has_a_filename_sourced_rotation_value() -> None:
    """Empirically confirmed by this audit: all 880 real filenames parse cleanly as a
    decimal number, so rotation_source == "filename" for every one of them (never
    "unavailable" for this dataset as currently inventoried)."""
    rows = load_inventory_rows()
    rotation_infos = build_rotation_audit(rows)

    assert all(info.rotation_source == "filename" for info in rotation_infos)
