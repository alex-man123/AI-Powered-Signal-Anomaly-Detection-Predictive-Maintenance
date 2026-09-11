"""TASK 13.1 -- loader for the CWRU (Case Western Reserve University) Bearing
Dataset, this project's secondary, cross-dataset-validation-only dataset
(docs/blueprint.md section 4/5/7: "CWRU. Folosit exclusiv pentru validare
cross-dataset in faza finala ... nu pentru dezvoltare MVP").

VERIFIED AGAINST THE REAL, DOWNLOADED DATASET (not merely public
documentation): the user uploaded a real CWRU mirror at
`data/external/cwru/`, and this module's directory/label/sampling-rate/
`.mat`-variable parsing below was written and adjusted against that actual
directory listing and actual `.mat` file contents (via `scipy.io.loadmat`
inspection), superseding an earlier version of this module that was written
before any real file existed and had to guess a layout from public
documentation alone. See git history for that earlier, disclosed-as-unverified
version.

REAL, OBSERVED DIRECTORY LAYOUT (`data/external/cwru/`):

    12k_Drive_End_Bearing_Fault_Data/{B,IR,OR}/<diameter>/[<orientation>/]<id>[@<orientation>]_<load>.mat
    12k_Fan_End_Bearing_Fault_Data/{B,IR,OR}/<diameter>/...
    48k_Drive_End_Bearing_Fault_Data/{B,IR,OR}/<diameter>/...
    Normal/<id>_Normal_<load>.mat

`B`/`IR`/`OR` = Ball / Inner Race / Outer Race fault location (standard
bearing-fault terminology; unambiguous from the folder names themselves).
`<diameter>` (e.g. `007`, `014`, `021`, `028`) and, for some Outer Race
subsets, an extra `<orientation>` level (`@3`/`@6`/`@12`, the fault's clock
position relative to the load zone) vary in nesting depth across the real
tree (e.g. 0.014" OR faults have no `@N` subdirectory at all -- only one
orientation was collected for that diameter in this real mirror). This
module's directory-scanning approach (checks every parent directory segment
for a recognized label/rate marker, never assumes one fixed path SHAPE) is
robust to this real inconsistency without special-casing it.

Sampling rate is embedded in the top-level directory name as a `<N>k_...`
prefix (`12k_...`, `48k_...`) -- extracted via regex, not an exact-string
lookup, so it does not depend on the exact `_Drive_End_Bearing_Fault_Data`/
`_Fan_End_Bearing_Fault_Data` suffix. **`Normal/` has no such prefix and its
real sampling rate is NOT invented here**: the CWRU Bearing Data Center's own
"Apparatus and Procedures" page (fetched and checked while writing this
module) states "[data] was collected at 12,000 samples per second, and data
was also collected at 48,000 samples per second for drive end bearing
faults" -- i.e. 48 kHz is documented as an ADDITIONAL rate specifically for
drive-end fault recordings, while the page does not explicitly state which
rate the Normal baseline files use. Given this genuine, unresolved ambiguity
in the authoritative source itself, `load_recording`/`load_all_recordings`
REQUIRE an explicit `sampling_rate_hz` for any file under `Normal/` and raise
`CWRULoaderError` if one is not given -- never silently assuming 12 kHz or
48 kHz.

`.mat` variable naming: confirmed directly (not merely documented) against
real files in this mirror -- e.g. `Normal/97_Normal_0.mat` contains
`X097_DE_time`, `X097_FE_time`, `X097RPM`; `.../IR/007/105_0.mat` contains
`X105_DE_time`, `X105_FE_time`, `X105_BA_time`, `X105RPM`. This module reads
every `*_DE_time`/`*_FE_time`/`*_BA_time` variable generically (a regex, not
one hardcoded key) as one channel each; `*RPM` is not currently read (out of
scope for TASK 13.1's own AC1 -- feature extraction here uses the same 15
DSP features as MAFAULDA, none of which need RPM).

NOT duplicated from `app.datasets.loader` (MAFAULDA): this module is
independent (CWRU's file format, label set, and sampling-rate variability are
all structurally different from MAFAULDA's single-format/single-rate CSVs --
see that module's own docstring), but produces windows via the exact same
`app.signal_processing.windowing.create_windows` and feeds them into the
exact same `app.features.extractor.extract_feature_matrix` /
`app.features.registry` used for MAFAULDA -- no second windowing or feature
extraction implementation exists here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.io import loadmat

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CWRU_ROOT = _REPO_ROOT / "data" / "external" / "cwru"


class CWRULabel(str, Enum):
    NORMAL = "normal"
    BALL = "ball"
    INNER_RACE = "inner-race"
    OUTER_RACE = "outer-race"


# Directory-name -> label, matching the REAL, verified folder names in this
# mirror exactly (`B`/`IR`/`OR`/`Normal`, case-insensitively).
LABEL_DIRECTORY_NAMES: dict[str, CWRULabel] = {
    "normal": CWRULabel.NORMAL,
    "b": CWRULabel.BALL,
    "ir": CWRULabel.INNER_RACE,
    "or": CWRULabel.OUTER_RACE,
}

# Matches the real top-level directory prefix ("12k_Drive_End_Bearing_Fault_Data",
# "12k_Fan_End_Bearing_Fault_Data", "48k_Drive_End_Bearing_Fault_Data") -- the
# leading "<N>k" segment, not the full (variable) suffix.
_SAMPLING_RATE_DIRECTORY_PATTERN = re.compile(r"^(\d+)k(?:_|$)", re.IGNORECASE)

# Confirmed directly against real files in this mirror (see module docstring):
# "<fileid>_<DE|FE|BA>_time". Matched generically -- never one hardcoded key.
_MAT_TIME_SERIES_KEY_PATTERN = re.compile(r".*_(DE|FE|BA)_time$", re.IGNORECASE)


class CWRULoaderError(ValueError):
    """Raised for anything this loader cannot resolve from real, actually
    present data: a missing dataset root, a file outside the required
    label directory contract, an unresolvable sampling rate (including the
    real, disclosed `Normal/`-rate ambiguity -- see module docstring), a
    `.mat` file that fails to parse, or a `.mat` file with no recognizable
    time-series variable. Never silently falls back to a guessed label,
    sampling rate, or channel."""


@dataclass(frozen=True)
class CWRURecording:
    """One real channel from one real CWRU `.mat` file. Mirrors the shape of
    `app.datasets.loader.LoadedRecording` (relative_path + label), extended
    with `sampling_rate` (which MAFAULDA does not need, since every MAFAULDA
    recording shares one fixed rate) and `channel_name` (the real `.mat`
    variable name this channel's values came from)."""

    relative_path: str
    label: CWRULabel
    sampling_rate: float
    channel_name: str
    values: tuple[float, ...]


def _label_from_path(relative_path: str) -> CWRULabel | None:
    parts = [part.lower() for part in PurePosixPath(relative_path).parts[:-1]]
    for part in parts:
        if part in LABEL_DIRECTORY_NAMES:
            return LABEL_DIRECTORY_NAMES[part]
    return None


def _sampling_rate_from_path(relative_path: str) -> float | None:
    parts = PurePosixPath(relative_path).parts[:-1]
    for part in parts:
        match = _SAMPLING_RATE_DIRECTORY_PATTERN.match(part)
        if match:
            return float(match.group(1)) * 1000.0
    return None


def _extract_time_series_channels(mat_contents: dict) -> dict[str, np.ndarray]:
    """Every real, non-metadata (`scipy.io.loadmat` prefixes its own metadata
    keys with `__`) variable matching the documented `*_time` convention --
    never assumes a single fixed variable name."""
    channels: dict[str, np.ndarray] = {}
    for key, value in mat_contents.items():
        if key.startswith("__"):
            continue
        if _MAT_TIME_SERIES_KEY_PATTERN.match(key):
            channels[key] = np.asarray(value, dtype=float).reshape(-1)
    return channels


def load_recording(
    relative_path: str,
    *,
    dataset_root: Path = DEFAULT_CWRU_ROOT,
    sampling_rate_hz: float | None = None,
) -> list[CWRURecording]:
    """Loads every real time-series channel out of one real CWRU `.mat` file.

    Args:
        relative_path: POSIX-style path relative to `dataset_root`, e.g.
            `"12k_Drive_End_Bearing_Fault_Data/IR/007/105_0.mat"` or
            `"Normal/97_Normal_0.mat"`.
        dataset_root: root of the real, user-uploaded CWRU tree.
        sampling_rate_hz: overrides the directory-derived sampling rate.
            REQUIRED for any file under `Normal/` (see module docstring for
            the real, disclosed ambiguity in CWRU's own documentation about
            that subset's rate) since no `<N>k_...` directory segment exists
            there to derive it from. If both a directory-derived rate AND an
            explicit override are present, the explicit override wins,
            deliberately (the caller asked for a specific rate).

    Returns:
        One `CWRURecording` per real time-series channel found in the file
        (a real CWRU file commonly has 2-3: DE/FE, sometimes also BA).

    Raises:
        CWRULoaderError: file missing; not under a recognized label
            directory; sampling rate not resolvable; `.mat` parse failure; or
            no recognizable time-series variable found inside it.
    """
    full_path = dataset_root / relative_path
    if not full_path.exists():
        raise CWRULoaderError(f"CWRU file not found: {full_path}")

    label = _label_from_path(relative_path)
    if label is None:
        raise CWRULoaderError(
            f"{relative_path!r} is not under one of the required label directories "
            f"{sorted(name for name in LABEL_DIRECTORY_NAMES)} -- see cwru_loader module "
            "docstring for the real, verified directory layout."
        )

    resolved_sampling_rate = sampling_rate_hz or _sampling_rate_from_path(relative_path)
    if resolved_sampling_rate is None:
        raise CWRULoaderError(
            f"Could not determine the real sampling rate for {relative_path!r}: no "
            "'<N>k_...' directory segment found, and no sampling_rate_hz override was given. "
            "This is expected for files under 'Normal/' -- CWRU's own Apparatus/Procedures page "
            "documents 48 kHz as an ADDITIONAL rate specifically for drive-end fault recordings, "
            "without stating the Normal baseline set's rate explicitly; pass sampling_rate_hz "
            "explicitly rather than have this loader guess (see module docstring)."
        )

    try:
        mat_contents = loadmat(full_path)
    except Exception as exc:
        raise CWRULoaderError(f"Failed to read {full_path} as a MATLAB .mat file: {exc}") from exc

    channels = _extract_time_series_channels(mat_contents)
    if not channels:
        present_keys = sorted(key for key in mat_contents if not key.startswith("__"))
        raise CWRULoaderError(
            f"{full_path} contains no variable matching the documented CWRU time-series "
            f"naming convention ('<fileid>_DE_time' / '_FE_time' / '_BA_time'); real "
            f"variable keys found in this file: {present_keys}"
        )

    return [
        CWRURecording(
            relative_path=relative_path,
            label=label,
            sampling_rate=resolved_sampling_rate,
            channel_name=channel_name,
            values=tuple(float(v) for v in values),
        )
        for channel_name, values in sorted(channels.items())
    ]


def discover_recordings(*, dataset_root: Path = DEFAULT_CWRU_ROOT) -> list[str]:
    """Every real `.mat` file under `dataset_root`, as POSIX-style relative
    paths, sorted deterministically. Purely a filesystem listing -- reads no
    file content.

    Raises:
        CWRULoaderError: if `dataset_root` does not exist.
    """
    if not dataset_root.exists():
        raise CWRULoaderError(
            f"CWRU dataset root not found: {dataset_root}. Upload the real CWRU Bearing Dataset "
            "under this exact path before cross-dataset validation (TASK 13.1) can be run -- "
            "see docs/results/cross_dataset_validation.md."
        )

    return sorted(str(path.relative_to(dataset_root).as_posix()) for path in dataset_root.rglob("*.mat"))


def load_all_recordings(
    *, dataset_root: Path = DEFAULT_CWRU_ROOT, normal_sampling_rate_hz: float | None = None
) -> list[CWRURecording]:
    """Loads every real channel from every real `.mat` file under
    `dataset_root`.

    Args:
        dataset_root: root of the real, user-uploaded CWRU tree.
        normal_sampling_rate_hz: forwarded as the `sampling_rate_hz` override
            for every file under `Normal/` specifically (files elsewhere
            derive their rate from their own `<N>k_...` directory and ignore
            this argument) -- see `load_recording`/module docstring for why
            `Normal/`'s real rate cannot be derived automatically.

    Raises:
        CWRULoaderError: root missing; root exists but contains zero `.mat`
            files; or (via `load_recording`) any individual file fails to
            resolve its label/sampling-rate/channels -- including every
            `Normal/` file, if `normal_sampling_rate_hz` is not given.
    """
    relative_paths = discover_recordings(dataset_root=dataset_root)
    if not relative_paths:
        raise CWRULoaderError(f"CWRU dataset root {dataset_root} exists but contains no .mat files.")

    recordings: list[CWRURecording] = []
    for relative_path in relative_paths:
        is_normal_file = _label_from_path(relative_path) == CWRULabel.NORMAL
        recordings.extend(
            load_recording(
                relative_path,
                dataset_root=dataset_root,
                sampling_rate_hz=normal_sampling_rate_hz if is_normal_file else None,
            )
        )
    return recordings
