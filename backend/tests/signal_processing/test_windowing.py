import pytest

from app.signal_processing.windowing import (
    Window,
    WindowingError,
    compute_window_count,
    create_windows,
)

# --- AC1 (CRITICAL): no window may contain samples from two different recordings ---


def test_ac1_windows_never_mix_two_recordings_with_different_splits() -> None:
    """The single most important leakage test in the project. Recording A is all 1s
    (split=train), Recording B is all 2s (split=test), with DIFFERENT lengths so
    there's no accidental symmetry. create_windows() is called once per recording
    (the API structurally cannot accept more than one recording per call), and every
    resulting window is checked for exact value provenance -- a window containing
    e.g. [1,1,1,1,2,2,2,2] would fail this test immediately."""
    recording_a_values = [1.0] * 55  # split=train
    recording_b_values = [2.0] * 37  # split=test

    windows_a = create_windows(
        recording_a_values, recording_id="recording-A", split="train", window_size=20, overlap=0.5
    )
    windows_b = create_windows(
        recording_b_values, recording_id="recording-B", split="test", window_size=20, overlap=0.5
    )

    assert len(windows_a) > 0
    assert len(windows_b) > 0

    for window in windows_a:
        assert window.recording_id == "recording-A"
        assert window.split == "train"
        assert set(window.values) == {1.0}  # exclusively recording A's value
        assert 2.0 not in window.values

    for window in windows_b:
        assert window.recording_id == "recording-B"
        assert window.split == "test"
        assert set(window.values) == {2.0}  # exclusively recording B's value
        assert 1.0 not in window.values


def test_ac1_a_naive_concatenation_would_be_caught_by_boundary_values() -> None:
    """A stronger variant: place the two recordings' distinctive values so that if
    someone incorrectly concatenated A+B before windowing, a window straddling the
    boundary would contain both 1.0 and 2.0. Calling create_windows() correctly (once
    per recording) must never produce such a window."""
    recording_a_values = list(range(100))  # 0..99, split=train
    recording_b_values = list(range(1000, 1100))  # 1000..1099, split=validation

    windows_a = create_windows(
        recording_a_values, recording_id="A", split="train", window_size=10, overlap=0.0
    )
    windows_b = create_windows(
        recording_b_values, recording_id="B", split="validation", window_size=10, overlap=0.0
    )

    for window in windows_a:
        assert all(v < 100 for v in window.values)
        assert window.split == "train"
    for window in windows_b:
        assert all(v >= 1000 for v in window.values)
        assert window.split == "validation"


# --- split inheritance ---


def test_split_inheritance_across_three_recordings() -> None:
    recordings = [
        ("rec-A", "train", [0.0] * 40),
        ("rec-B", "validation", [1.0] * 40),
        ("rec-C", "test", [2.0] * 40),
    ]

    all_windows: dict[str, list[Window]] = {}
    for recording_id, split, values in recordings:
        all_windows[recording_id] = create_windows(
            values, recording_id=recording_id, split=split, window_size=10, overlap=0.0
        )

    assert all(w.split == "train" for w in all_windows["rec-A"])
    assert all(w.split == "validation" for w in all_windows["rec-B"])
    assert all(w.split == "test" for w in all_windows["rec-C"])
    # no window has any split other than its own recording's
    for recording_id, split, _ in recordings:
        assert all(w.split == split for w in all_windows[recording_id])


# --- window count (AC2): the mandatory worked example ---


def test_window_count_matches_the_worked_example_exactly() -> None:
    # N=100, W=20, overlap=50% -> step=10, N_windows = floor((100-20)/10)+1 = 9
    assert compute_window_count(signal_length=100, window_size=20, overlap=0.5) == 9

    windows = create_windows(
        list(range(100)), recording_id="rec", split="train", window_size=20, overlap=0.5
    )

    assert len(windows) == 9
    expected_boundaries = [
        (0, 20), (10, 30), (20, 40), (30, 50), (40, 60), (50, 70), (60, 80), (70, 90), (80, 100),
    ]
    actual_boundaries = [(w.start_sample, w.end_sample) for w in windows]
    assert actual_boundaries == expected_boundaries


@pytest.mark.parametrize(
    "signal_length,window_size,overlap,expected_count",
    [
        (100, 20, 0.0, 5),  # no overlap: step=20, floor(80/20)+1=5
        (100, 20, 0.5, 9),  # the worked example
        (100, 10, 0.9, 91),  # heavy overlap: step=1, floor(90/1)+1=91
        (50, 50, 0.0, 1),  # exactly one window fits, no remainder
        (49, 50, 0.0, 0),  # signal shorter than window -> zero windows
    ],
)
def test_window_count_formula_for_various_configurations(
    signal_length: int, window_size: int, overlap: float, expected_count: int
) -> None:
    assert compute_window_count(signal_length, window_size, overlap) == expected_count


# --- boundaries ---


def test_window_boundaries_and_values_match_exact_signal_slice() -> None:
    signal = [float(i) for i in range(100)]

    windows = create_windows(signal, recording_id="rec", split="train", window_size=20, overlap=0.5)

    for window in windows:
        assert 0 <= window.start_sample
        assert window.start_sample < window.end_sample
        assert window.end_sample <= len(signal)
        assert window.length == 20
        assert list(window.values) == signal[window.start_sample : window.end_sample]


def test_signal_shorter_than_window_size_produces_zero_windows_not_padding() -> None:
    signal = [1.0, 2.0, 3.0]

    windows = create_windows(signal, recording_id="rec", split="train", window_size=10, overlap=0.0)

    assert windows == []


def test_signal_exactly_window_size_produces_exactly_one_window() -> None:
    signal = list(range(50))

    windows = create_windows(signal, recording_id="rec", split="train", window_size=50, overlap=0.0)

    assert len(windows) == 1
    assert windows[0].start_sample == 0
    assert windows[0].end_sample == 50
    assert list(windows[0].values) == signal


# --- invalid configuration ---


def test_zero_window_size_is_rejected() -> None:
    with pytest.raises(WindowingError, match="window_size"):
        create_windows([1.0, 2.0, 3.0], recording_id="rec", split="train", window_size=0, overlap=0.0)


def test_negative_window_size_is_rejected() -> None:
    with pytest.raises(WindowingError, match="window_size"):
        create_windows([1.0, 2.0, 3.0], recording_id="rec", split="train", window_size=-5, overlap=0.0)


def test_negative_overlap_is_rejected() -> None:
    with pytest.raises(WindowingError, match="overlap"):
        create_windows([1.0] * 100, recording_id="rec", split="train", window_size=10, overlap=-0.1)


def test_overlap_of_one_or_more_is_rejected() -> None:
    with pytest.raises(WindowingError, match="overlap"):
        create_windows([1.0] * 100, recording_id="rec", split="train", window_size=10, overlap=1.0)

    with pytest.raises(WindowingError, match="overlap"):
        create_windows([1.0] * 100, recording_id="rec", split="train", window_size=10, overlap=1.5)


def test_compute_window_count_rejects_same_invalid_configs() -> None:
    with pytest.raises(WindowingError):
        compute_window_count(100, window_size=0, overlap=0.0)
    with pytest.raises(WindowingError):
        compute_window_count(100, window_size=10, overlap=1.0)


# --- determinism ---


def test_windowing_is_deterministic_across_repeated_calls() -> None:
    signal = [float(i) for i in range(137)]

    result_1 = create_windows(signal, recording_id="rec", split="train", window_size=17, overlap=0.3)
    result_2 = create_windows(signal, recording_id="rec", split="train", window_size=17, overlap=0.3)

    assert result_1 == result_2


# --- data is never modified ---


def test_windowing_does_not_transform_values() -> None:
    signal = [5.0, -3.2, 0.0, 100.0, -0.001]

    windows = create_windows(signal, recording_id="rec", split="train", window_size=5, overlap=0.0)

    assert len(windows) == 1
    assert list(windows[0].values) == signal  # exact values, not normalized/filtered/rounded
