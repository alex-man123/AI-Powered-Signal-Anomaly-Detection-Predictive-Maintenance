from app.datasets.class_distribution_audit import (
    build_rotation_audit,
    load_inventory_rows,
)
from app.datasets.replication_analysis import (
    UNKNOWN_ROTATION_KEY,
    ConditionGroup,
    build_condition_groups,
    build_condition_groups_with_severity,
    find_same_size_duplicates,
    group_key_for,
    summarize,
)
from app.datasets.class_distribution_audit import RotationInfo


def _info(relative_path: str, state: str, freq: float | None, source: str) -> RotationInfo:
    return RotationInfo(
        relative_path=relative_path,
        state=state,
        rotation_frequency_hz=freq,
        rotation_source=source,
        evidence="synthetic test fixture",
    )


def test_same_state_and_frequency_produce_the_same_group() -> None:
    infos = [
        _info("normal/a.csv", "normal", 25.0, "filename"),
        _info("normal/b.csv", "normal", 25.0, "filename"),
    ]

    groups = build_condition_groups(infos)

    assert len(groups) == 1
    assert groups[0].file_count == 2
    assert groups[0].is_replicated is True


def test_different_state_same_frequency_are_different_groups() -> None:
    infos = [
        _info("normal/a.csv", "normal", 25.0, "filename"),
        _info("imbalance/6g/b.csv", "imbalance", 25.0, "filename"),
    ]

    groups = build_condition_groups(infos)

    assert len(groups) == 2
    assert all(not g.is_replicated for g in groups)


def test_same_state_different_frequency_are_different_groups() -> None:
    infos = [
        _info("normal/a.csv", "normal", 25.0, "filename"),
        _info("normal/b.csv", "normal", 50.0, "filename"),
    ]

    groups = build_condition_groups(infos)

    assert len(groups) == 2
    assert all(not g.is_replicated for g in groups)


def test_no_tolerance_is_applied_to_near_but_distinct_frequencies() -> None:
    """Design decision (justified empirically against the real dataset, see
    docs/dataset_audit/replication_analysis.md): exact match, zero tolerance. 25.00 and
    25.03 are therefore treated as distinct groups, not merged."""
    infos = [
        _info("normal/a.csv", "normal", 25.00, "filename"),
        _info("normal/b.csv", "normal", 25.03, "filename"),
    ]

    groups = build_condition_groups(infos)

    assert len(groups) == 2


def test_unavailable_rotation_recordings_are_grouped_not_dropped() -> None:
    infos = [
        _info("normal/a.csv", "normal", None, "unavailable"),
        _info("normal/b.csv", "normal", None, "unavailable"),
    ]

    groups = build_condition_groups(infos)

    assert len(groups) == 1
    assert groups[0].rotation_frequency_hz is None
    assert groups[0].file_count == 2
    assert group_key_for(infos[0]) == ("normal", UNKNOWN_ROTATION_KEY)


def test_group_records_multiple_distinct_rotation_sources_when_present() -> None:
    infos = [
        _info("normal/a.csv", "normal", 25.0, "filename"),
        _info("normal/b.csv", "normal", 25.0, "metadata"),
    ]

    groups = build_condition_groups(infos)

    assert len(groups) == 1
    assert groups[0].rotation_sources == ("filename", "metadata")


def test_summarize_counts_replicated_and_singleton_groups() -> None:
    groups = [
        ConditionGroup("normal", 25.0, ("a", "b"), ("filename",)),  # replicated
        ConditionGroup("normal", 50.0, ("c",), ("filename",)),  # singleton
    ]

    summary = summarize(groups)

    assert summary == {
        "total_groups": 2,
        "replicated_groups": 1,
        "singleton_groups": 1,
        "total_recordings": 3,
    }


def test_real_inventory_groups_sum_to_inventory_total() -> None:
    rows = load_inventory_rows()
    rotation_infos = build_rotation_audit(rows)

    groups = build_condition_groups(rotation_infos)
    summary = summarize(groups)

    assert summary["total_recordings"] == len(rows)
    assert summary["total_groups"] == summary["replicated_groups"] + summary["singleton_groups"]


def test_real_inventory_has_no_true_same_severity_replication() -> None:
    """Empirically confirmed by this audit: within a single (state, condition)
    folder, every rotation_frequency_hz is unique — no file repeats an identical
    target speed under the identical severity. So the coarser (state, frequency)
    "replication" found above is entirely explained by different severities sharing a
    target speed, never by the same physical setup being recorded more than once."""
    rows = load_inventory_rows()

    severity_groups = build_condition_groups_with_severity(rows)

    assert all(len(paths) == 1 for paths in severity_groups.values())


def test_real_inventory_has_no_byte_size_duplicates() -> None:
    rows = load_inventory_rows()

    duplicates = find_same_size_duplicates(rows)

    assert duplicates == {}
