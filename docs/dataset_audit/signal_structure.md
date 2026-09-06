# MAFAULDA Signal Structure Audit — TASK 1.5.3

> Source of truth: the real local files under `data/raw/mafaulda/`, inspected directly
> with pandas, for everything except sampling rate. The sampling rate (§5) is a
> deliberate exception: it is adopted from official MAFAULDA documentation (project
> decision) because the local files contain no way to derive or verify it independently
> — every place that value is used is labeled `sampling_rate_source:
> official_documentation`, never presented as a local measurement.

## 1. Scope

This audit inspects the real, local MAFAULDA CSV files to determine their actual
column/channel structure, row count, and — if determinable from the files themselves —
sampling rate and signal duration. No parsing of signal semantics, no filtering/FFT/PSD,
no feature extraction, and no train/val/test split happen here.

## 2. Files inspected

Selected deterministically: the first file (by `relative_path`, i.e. TASK 1.5.1's own
sort order) for **every distinct state** found in `docs/dataset_audit/file_inventory.csv`
— all 4 states discovered in TASK 1.5.2, not a manually hand-picked subset.

| # | relative_path | state | size_bytes (from file_inventory.csv) |
|---|---|---|---|
| 1 | `normal/12.288.csv` | `normal` | 17,642,507 |
| 2 | `imbalance/10g/13.9264.csv` | `imbalance` | 17,582,498 |
| 3 | `horizontal-misalignment/0.5mm/12.288.csv` | `horizontal-misalignment` | 17,122,444 |
| 4 | `vertical-misalignment/0.51mm/12.4928.csv` | `vertical-misalignment` | 17,149,830 |

Reproducible via `backend/app/datasets/signal_structure_audit.py::select_audit_files()`.

## 3. Observed CSV structure

Each file was read with `pandas.read_csv(path, header=None)` (no header row is present
— see below).

| File | Rows | Columns | dtypes | Missing values |
|---|---:|---:|---|---|
| `normal/12.288.csv` | 250,000 | 8 | all `float64` | none |
| `imbalance/10g/13.9264.csv` | 250,000 | 8 | all `float64` | none |
| `horizontal-misalignment/0.5mm/12.288.csv` | 250,000 | 8 | all `float64` | none |
| `vertical-misalignment/0.51mm/12.4928.csv` | 250,000 | 8 | all `float64` | none |

No header row: the first line of every inspected file is plain numeric data (verified
with `head -1` on the raw file, e.g. `normal/12.288.csv` starts with
`4.5595,0.1752,0.28721,-0.017751,-0.41565,0.032459,-0.11218,-0.12814`), not column
names. Columns are therefore positional (`0`–`7`), never renamed.

No metadata files of any kind exist anywhere under `data/raw/mafaulda/` — every file in
the tree has a `.csv` extension (re-confirmed for this audit; also established in TASK
1.5.1's inventory). There is no separate header/config/README file describing sampling
rate, channel names, or duration.

## 4. Channel structure

**Observed: 8 columns per file, consistently, across all 4 inspected states.** This
matches what the blueprint's dataset research section describes (triaxial accelerometer +
3 industrial accelerometers + tachometer + microphone = 8 channels), but that
correspondence is **not proven by this audit** — the files carry no column headers or
per-column labels, so **no semantic sensor name is assigned to any column here**. Assigning
"channel 0 = tachometer", etc. would be an assumption not supported by file content alone,
and is explicitly out of scope for this task (would belong to a task that cross-references
the external MAFAULDA sensor documentation with the local data, if ever undertaken).

## 5. Sampling rate

**Sampling rate could not be independently derived from the file contents alone** — see
the checks below. A value is nonetheless adopted for the project, sourced explicitly and
non-ambiguously from external documentation (project decision, recorded here):

```
sampling_rate_hz: 50000
sampling_rate_source: official_documentation (MAFAULDA page, UFRJ)
```

This mirrors the `rotation_source` transparency field introduced in TASK 1.5.4 — the
value is never presented as if it were measured from the local files, only as what it
actually is: an externally-sourced, explicitly labeled figure.

What was actually checked locally, in the 4 inspected files (this part did **not**
change — the files themselves still contain no such information):
- No header row, so no field labeled `sample_rate`, `fs`, `Hz`, or similar.
- No metadata file anywhere in `data/raw/mafaulda/` (confirmed empty search for any
  non-`.csv` file in the whole tree).
- No column is a timestamp or elapsed-time series: every one of the 8 columns in every
  inspected file was checked for monotonic increase (`pandas.Series.is_monotonic_increasing`)
  as a candidate time axis — **none of the 32 checked columns (8 columns × 4 files) is
  monotonic**. All 8 columns are raw oscillating sensor-amplitude values.
- Without a time/duration reference anywhere in the file, the sampling-rate formula
  `fs = num_samples / duration` cannot be *independently evaluated* from local data —
  `duration` is not known from local data, only `num_samples` (250,000, from §3) is.

**Indirect corroboration (not independent derivation):** with the externally-sourced
`fs = 50,000 Hz` and MAFAULDA's documented 5-second recording window, the expected
sample count is `50,000 × 5 = 250,000` — exactly what §3 measured in all 4 inspected
files. This is corroborating evidence that the adopted rate is plausible for this
dataset; it is **not** proof derived independently from the files, since the files carry
no time axis and the same row count could in principle correspond to a different
(rate, duration) pair. The distinction matters and is kept explicit throughout this
document, per project decision.

## 6. Signal duration

Using the externally-sourced sampling rate (§5) and the locally-measured
`num_samples = 250,000` (identical across all 4 inspected files):

```
duration_s = num_samples / sampling_rate_hz = 250,000 / 50,000 = 5.0 s
```

This duration is **derived**, not independently measured locally — it depends entirely
on accepting the externally-sourced `sampling_rate_hz` from §5. No file contains a time
or duration field that would allow computing this without that external input.

## 7. Cross-file consistency

| Property | Result |
|---|---|
| Channel consistency (column count + dtype) | **CONSISTENT** — 8 `float64` columns in all 4 files |
| Sampling-rate consistency | **CONSISTENT** — single adopted value (50,000 Hz, `official_documentation`) applies uniformly; not independently re-derived per file, but corroborated by identical row counts across all 4 |
| Row-count consistency | **CONSISTENT** — 250,000 rows in all 4 files |
| Duration consistency | **CONSISTENT** — 5.0 s for all 4 files, derived from the same adopted rate + measured row count |
| Missing values | **CONSISTENT** — none found in any of the 4 files |

## 8. External references

- Blueprint (`docs/blueprint.md`, section 4): "Sampling rate: 50 kHz, ferestre de 5
  secunde per fișier (250k eșantioane/canal)."
- MAFAULDA's official documentation (UFRJ page) states the same 50 kHz / 5 s convention.

**This is the adopted source for `sampling_rate_hz` (§5)** — a deliberate project
decision, not a local measurement. It is recorded as `sampling_rate_source:
official_documentation` everywhere this value is used, so no downstream reader mistakes
it for something measured from `data/raw/mafaulda/` directly.

## 9. Audit conclusion

- **AC1 — confirmed.** All 4 inspected files (covering all 4 states discovered in TASK
  1.5.2) have exactly 8 columns, all `float64`, 250,000 rows, no missing values, no
  header row. This is measured directly from the files, not cited from external sources.
- **AC2 — confirmed, via explicitly documented external source.** The local files
  contain no header, no metadata, and no monotonic time/duration column — a sampling
  rate cannot be independently derived from them alone. Per project decision, the value
  `sampling_rate_hz = 50,000` is adopted from MAFAULDA's official documentation (UFRJ),
  labeled `sampling_rate_source: official_documentation` rather than
  `measured_locally`. The locally-measured 250,000-row count (§3) is documented as
  indirect corroboration (consistent with 50 kHz × 5 s) — explicitly *not* an
  independent derivation, since no file contains a time axis. Signal duration (5.0 s,
  §6) is derived from this adopted rate, not measured directly.
