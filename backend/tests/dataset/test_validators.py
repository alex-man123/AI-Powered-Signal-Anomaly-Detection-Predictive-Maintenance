import pytest

from app.datasets.validators import (
    EXPECTED_CHANNEL_COUNT,
    EXPECTED_SAMPLING_RATE_HZ,
    MINIMUM_SIGNAL_LENGTH,
    DatasetValidationError,
    validate_channel_count,
    validate_sampling_rate,
    validate_signal,
    validate_signal_length,
    validate_signal_shape,
)

# --- sampling rate (AC1's critical case) ---


def test_valid_sampling_rate_passes() -> None:
    validate_sampling_rate(EXPECTED_SAMPLING_RATE_HZ)  # must not raise


def test_mismatched_sampling_rate_raises_with_expected_and_actual_in_message() -> None:
    with pytest.raises(DatasetValidationError) as exc_info:
        validate_sampling_rate(44100.0, expected=EXPECTED_SAMPLING_RATE_HZ)

    message = str(exc_info.value)
    assert "expected=50000.0" in message
    assert "actual=44100.0" in message


def test_zero_sampling_rate_raises() -> None:
    with pytest.raises(DatasetValidationError):
        validate_sampling_rate(0, expected=EXPECTED_SAMPLING_RATE_HZ)


def test_negative_sampling_rate_raises() -> None:
    with pytest.raises(DatasetValidationError):
        validate_sampling_rate(-50000.0, expected=EXPECTED_SAMPLING_RATE_HZ)


# --- channel count ---


def test_valid_channel_count_passes() -> None:
    validate_channel_count(EXPECTED_CHANNEL_COUNT)  # must not raise


def test_mismatched_channel_count_raises_with_expected_and_actual() -> None:
    with pytest.raises(DatasetValidationError) as exc_info:
        validate_channel_count(3, expected=EXPECTED_CHANNEL_COUNT)

    message = str(exc_info.value)
    assert "expected=8" in message
    assert "actual=3" in message


def test_zero_channel_count_raises() -> None:
    with pytest.raises(DatasetValidationError):
        validate_channel_count(0, expected=EXPECTED_CHANNEL_COUNT)


# --- signal length ---


def test_valid_signal_length_passes() -> None:
    validate_signal_length(MINIMUM_SIGNAL_LENGTH)  # exactly at the floor: must not raise
    validate_signal_length(MINIMUM_SIGNAL_LENGTH + 1)  # above the floor: must not raise


def test_signal_length_below_minimum_raises() -> None:
    with pytest.raises(DatasetValidationError) as exc_info:
        validate_signal_length(100, minimum_length=MINIMUM_SIGNAL_LENGTH)

    message = str(exc_info.value)
    assert f"expected>={MINIMUM_SIGNAL_LENGTH}" in message
    assert "actual=100" in message


def test_zero_signal_length_raises() -> None:
    with pytest.raises(DatasetValidationError):
        validate_signal_length(0, minimum_length=MINIMUM_SIGNAL_LENGTH)


def test_empty_signal_length_raises_same_as_zero() -> None:
    with pytest.raises(DatasetValidationError):
        validate_signal_length(0)


# --- shape (rows, columns) = (samples, channels) ---


def test_valid_shape_passes() -> None:
    validate_signal_shape((MINIMUM_SIGNAL_LENGTH, EXPECTED_CHANNEL_COUNT))  # must not raise


def test_shape_with_wrong_channel_count_raises() -> None:
    with pytest.raises(DatasetValidationError, match="Channel count mismatch"):
        validate_signal_shape((MINIMUM_SIGNAL_LENGTH, 3))


def test_shape_with_insufficient_length_raises() -> None:
    with pytest.raises(DatasetValidationError, match="Signal length below minimum"):
        validate_signal_shape((100, EXPECTED_CHANNEL_COUNT))


def test_shape_not_two_dimensional_raises() -> None:
    with pytest.raises(DatasetValidationError, match="2-dimensional"):
        validate_signal_shape((MINIMUM_SIGNAL_LENGTH, EXPECTED_CHANNEL_COUNT, 1))  # type: ignore[arg-type]


def test_unexpected_shape_ordering_is_still_caught_as_a_mismatch() -> None:
    """(channels, samples) instead of (samples, channels) must be rejected, not
    silently accepted -- e.g. (8, 250_000) would pass a naive channel check only if
    someone mistakenly compared the wrong axis; validate_signal_shape always reads
    shape[1] as channels, so a transposed shape correctly fails the channel check."""
    with pytest.raises(DatasetValidationError, match="Channel count mismatch"):
        validate_signal_shape((EXPECTED_CHANNEL_COUNT, MINIMUM_SIGNAL_LENGTH))


# --- combined validate_signal ---


def test_validate_signal_passes_for_fully_valid_input() -> None:
    validate_signal(
        sampling_rate=EXPECTED_SAMPLING_RATE_HZ,
        shape=(MINIMUM_SIGNAL_LENGTH, EXPECTED_CHANNEL_COUNT),
    )  # must not raise


def test_validate_signal_raises_on_sampling_rate_mismatch_before_shape_is_checked() -> None:
    with pytest.raises(DatasetValidationError, match="Sampling rate mismatch"):
        validate_signal(sampling_rate=1234.0, shape=(1, 1))


def test_validate_signal_raises_on_shape_mismatch_when_sampling_rate_is_valid() -> None:
    with pytest.raises(DatasetValidationError, match="Channel count mismatch"):
        validate_signal(sampling_rate=EXPECTED_SAMPLING_RATE_HZ, shape=(MINIMUM_SIGNAL_LENGTH, 2))
