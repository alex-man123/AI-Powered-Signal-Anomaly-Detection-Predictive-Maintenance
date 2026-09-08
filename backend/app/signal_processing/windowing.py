"""TASK 2.4 — windowing (signal segmentation), respecting split assignment.

Strictly per-recording: `create_windows()` takes exactly one recording's values, one
`recording_id`, and one `split` at a time — there is no API surface here that could
accept multiple recordings and concatenate them, which is itself a structural guard
against the project's single most important leakage rule (blueprint.md section 8:
"Overlap-ul... e permis in interiorul unei rulari alocate unui singur split, nu intre
ferestre care ar ajunge in split-uri diferite"). A caller with multiple recordings
must call this once per recording and keep the results separate — never concatenate
signals before windowing.

Pure segmentation only: no preprocessing, filtering, normalization, resampling,
FFT/PSD/STFT, or feature extraction happens here. Window values are an exact slice of
the input, never transformed.

Does not persist to the `signal_windows` table (TASK 2.1) or touch
`data/processed/split_manifest.json` — this module returns in-memory `Window` objects;
wiring them into persistence is a future task's responsibility, kept separate here.

Split is never derived, recomputed, or re-stratified here — it is exactly whatever the
caller passes in for `split` (e.g. a `LoadedRecording.split` from TASK 2.2's loader),
copied onto every window generated from that recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


class WindowingError(ValueError):
    """Raised for invalid windowing configuration — never silently corrected."""


@dataclass(frozen=True)
class Window:
    recording_id: str
    split: str
    start_sample: int
    end_sample: int
    values: tuple[float, ...]

    @property
    def length(self) -> int:
        return self.end_sample - self.start_sample


def _validate_config(window_size: int, overlap: float) -> float:
    if window_size <= 0:
        raise WindowingError(f"window_size must be > 0, got {window_size}")
    if not (0 <= overlap < 1):
        raise WindowingError(f"overlap must be in [0, 1), got {overlap}")

    step = window_size * (1 - overlap)
    if step <= 0:
        raise WindowingError(
            f"computed step must be > 0, got {step} (window_size={window_size}, overlap={overlap})"
        )
    return step


def compute_window_count(signal_length: int, window_size: int, overlap: float = 0.0) -> int:
    """N_windows = floor((N - W) / S) + 1 for N >= W, else 0. Pure formula, no
    padding/truncation of the underlying signal is implied or performed anywhere."""
    step = _validate_config(window_size, overlap)

    if signal_length < window_size:
        return 0

    return int((signal_length - window_size) // step) + 1


def create_windows(
    values: Sequence[float],
    *,
    recording_id: str,
    split: str,
    window_size: int,
    overlap: float = 0.0,
) -> list[Window]:
    """Segments exactly one recording's values into windows using the
    `[start, end)` convention (`window.values == values[start:end]`,
    `end - start == window_size` for every window returned).

    Window start positions are computed as `round(i * step)` for `i` in
    `range(window_count)` — not by repeatedly accumulating a possibly-fractional
    `step` (which would drift under floating-point error over many windows). This
    keeps every `start_sample`/`end_sample` an exact integer while still matching
    `compute_window_count`'s closed-form count exactly.
    """
    step = _validate_config(window_size, overlap)
    signal_length = len(values)
    window_count = compute_window_count(signal_length, window_size, overlap)

    windows = []
    for i in range(window_count):
        start = int(round(i * step))
        end = start + window_size
        windows.append(
            Window(
                recording_id=recording_id,
                split=split,
                start_sample=start,
                end_sample=end,
                values=tuple(values[start:end]),
            )
        )

    return windows
