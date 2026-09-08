from pathlib import Path

from app.datasets.data_quality import (
    CLASS_CONSTANT_NONZERO,
    CLASS_OK,
    CLASS_ZERO_SIGNAL,
    STATUS_INVALID_STRUCTURE,
    STATUS_NON_NUMERIC_DATA,
    STATUS_OK,
    STATUS_READ_ERROR,
    analyze_file_quality,
)


def _write_csv(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


def test_nan_injection_is_detected(tmp_path: Path) -> None:
    """The task-mandated test: a synthetic file with valid numeric data plus at least
    one injected NaN must be detected, with the correct missing-value count."""
    path = tmp_path / "with_nan.csv"
    _write_csv(path, [[0.1, 1.0], [0.2, ""], [0.3, 3.0], [0.4, 4.0]])  # 1 NaN via empty field

    report = analyze_file_quality(path, "with_nan.csv")

    assert report.status == STATUS_OK
    assert report.has_missing_values is True
    assert report.missing_values == 1
    assert report.missing_percentage == (1 / (4 * 2)) * 100


def test_multiple_nan_values_are_all_counted(tmp_path: Path) -> None:
    path = tmp_path / "many_nan.csv"
    _write_csv(path, [[1.0, 2.0], ["", ""], [3.0, ""], [4.0, 5.0]])  # 3 NaN total

    report = analyze_file_quality(path, "many_nan.csv")

    assert report.missing_values == 3
    assert report.has_missing_values is True


def test_file_with_zero_nan_has_no_missing_values(tmp_path: Path) -> None:
    path = tmp_path / "clean.csv"
    _write_csv(path, [[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]])

    report = analyze_file_quality(path, "clean.csv")

    assert report.missing_values == 0
    assert report.has_missing_values is False


def test_constant_nonzero_signal_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "constant.csv"
    _write_csv(path, [[5.0], [5.0], [5.0], [5.0]])

    report = analyze_file_quality(path, "constant.csv")

    assert len(report.constant_channels) == 1
    assert report.constant_channels[0].classification == CLASS_CONSTANT_NONZERO
    assert report.zero_channels == ()


def test_zero_signal_is_detected_and_distinguished_from_constant_nonzero(tmp_path: Path) -> None:
    path = tmp_path / "zero.csv"
    _write_csv(path, [[0.0], [0.0], [0.0], [0.0]])

    report = analyze_file_quality(path, "zero.csv")

    assert len(report.zero_channels) == 1
    assert report.zero_channels[0].classification == CLASS_ZERO_SIGNAL
    assert report.constant_channels == ()


def test_varied_valid_signal_is_not_flagged_as_constant_or_zero(tmp_path: Path) -> None:
    path = tmp_path / "valid.csv"
    _write_csv(path, [[0.0], [1.0], [2.0], [1.0], [0.0]])

    report = analyze_file_quality(path, "valid.csv")

    assert report.constant_channels == ()
    assert report.zero_channels == ()
    assert report.channels[0].classification == CLASS_OK


def test_mixed_channels_only_flags_the_actually_constant_one(tmp_path: Path) -> None:
    # column 0 varies, column 1 is constant non-zero, column 2 is zero.
    path = tmp_path / "mixed.csv"
    _write_csv(path, [[0.0, 7.0, 0.0], [1.0, 7.0, 0.0], [2.0, 7.0, 0.0]])

    report = analyze_file_quality(path, "mixed.csv")

    assert report.channels[0].classification == CLASS_OK
    assert report.channels[1].classification == CLASS_CONSTANT_NONZERO
    assert report.channels[2].classification == CLASS_ZERO_SIGNAL
    assert report.all_channels_constant_or_zero is False  # column 0 is still OK


def test_all_channels_constant_or_zero_flags_the_worse_case(tmp_path: Path) -> None:
    path = tmp_path / "all_bad.csv"
    _write_csv(path, [[5.0, 0.0], [5.0, 0.0], [5.0, 0.0]])

    report = analyze_file_quality(path, "all_bad.csv")

    assert report.all_channels_constant_or_zero is True


def test_low_variance_but_non_constant_signal_is_not_flagged() -> None:
    """Explicit guard against the anti-pattern the task warns about: low variance is
    not the same as a constant signal."""
    import pandas as pd

    from app.datasets.data_quality import classify_channel

    series = pd.Series([1.0000, 1.0001, 1.0000, 1.0002, 1.0000])
    assert classify_channel(series) == CLASS_OK


def test_unreadable_file_produces_read_error_not_a_crash(tmp_path: Path) -> None:
    path = tmp_path / "not_really_a_csv.csv"
    path.write_bytes(b"\x00\x01\x02\xff\xfe\xfd" * 10)  # binary garbage

    report = analyze_file_quality(path, "not_really_a_csv.csv")

    assert report.status in (STATUS_READ_ERROR, STATUS_NON_NUMERIC_DATA, STATUS_INVALID_STRUCTURE)
    assert report.error_message is not None


def test_missing_file_produces_read_error(tmp_path: Path) -> None:
    path = tmp_path / "does_not_exist.csv"

    report = analyze_file_quality(path, "does_not_exist.csv")

    assert report.status == STATUS_READ_ERROR
    assert report.error_message is not None


def test_non_numeric_data_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "text.csv"
    _write_csv(path, [["abc", "def"], ["ghi", "jkl"]])

    report = analyze_file_quality(path, "text.csv")

    assert report.status == STATUS_NON_NUMERIC_DATA


def test_empty_file_is_invalid_structure(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("")

    report = analyze_file_quality(path, "empty.csv")

    assert report.status in (STATUS_INVALID_STRUCTURE, STATUS_READ_ERROR)
