# MAFAULDA Recording Mapping — TASK 1.5.2

> Source of truth: the real folder/file listing under `data/raw/mafaulda/` and
> `docs/dataset_audit/file_inventory.csv` (TASK 1.5.1). Nothing here is copied from
> MAFAULDA documentation, tutorials, or the blueprint — every claim below was verified
> directly against the filesystem during this audit.

## Dataset structure observed

```
data/raw/mafaulda/
├── normal/
│   ├── 12.288.csv
│   ├── 13.1072.csv
│   └── ... (49 files, no sub-folder)
├── imbalance/
│   ├── 6g/...    (one of 7 severity sub-folders)
│   ├── 10g/...
│   ├── 15g/...
│   ├── 20g/...
│   ├── 25g/...
│   ├── 30g/...
│   └── 35g/...   (333 files total)
├── horizontal-misalignment/
│   ├── 0.5mm/...  (one of 4 severity sub-folders)
│   ├── 1.0mm/...
│   ├── 1.5mm/...
│   └── 2.0mm/...  (197 files total)
└── vertical-misalignment/
    ├── 0.51mm/... (one of 6 severity sub-folders)
    ├── 0.63mm/...
    ├── 1.27mm/...
    ├── 1.40mm/...
    ├── 1.78mm/...
    └── 1.90mm/... (301 files total)
```

Exactly **4 top-level folders** exist under `data/raw/mafaulda/` — no more, no fewer.
This is a strict subset of the full published MAFAULDA dataset (no bearing-fault
underhang/overhang folders are present in this repository's copy); the audit reports
what is actually here, not what MAFAULDA documentation describes elsewhere.

`normal/` is the only state with **no** severity sub-folder — files sit directly inside
it. Every other state has exactly one level of severity sub-folder between the state
folder and the file.

## Naming convention

Two — and only two — path shapes were found (verified: every one of the 880 inventoried
files matches one of them, see [Audit conclusion](#audit-conclusion)):

```
<state>/<filename>.csv               → only "normal/" uses this shape
<state>/<condition>/<filename>.csv    → every other state uses this shape
```

The **filename** itself (e.g. `12.288.csv`, `59.8016.csv`) is a plain decimal number in
every observed case. It encodes no state/condition information — the state and condition
are encoded entirely in the **folder path**, never in the filename. What that number
itself represents (it is plausibly a rotation-speed-related value, per the blueprint's
dataset notes) is **not determined here** — interpreting filename semantics is out of
scope for TASK 1.5.2 and belongs to a later audit task.

## Mapping precedence

Not applicable in the "conflicting sources" sense — the filename carries **no** state
information at all in this dataset, so there is nothing to arbitrate between folder and
filename. State and condition are read exclusively from the folder path:

```
folder (state + condition) > filename (carries no state information)
```

## State mapping

| Observed pattern | Derived state | Derived condition | Source |
|---|---|---|---|
| `normal/12.288.csv` | `normal` | *(none — no sub-folder)* | folder |
| `imbalance/10g/13.9264.csv` | `imbalance` | `10g` | folder |
| `imbalance/6g/59.392.csv` | `imbalance` | `6g` | folder |
| `horizontal-misalignment/0.5mm/12.288.csv` | `horizontal-misalignment` | `0.5mm` | folder |
| `horizontal-misalignment/2.0mm/60.8256.csv` | `horizontal-misalignment` | `2.0mm` | folder |
| `vertical-misalignment/0.51mm/12.4928.csv` | `vertical-misalignment` | `0.51mm` | folder |
| `vertical-misalignment/1.90mm/61.44.csv` | `vertical-misalignment` | `1.90mm` | folder |

All examples above are real rows taken from `docs/dataset_audit/file_inventory.csv`.

### States and their condition values (all values observed on disk)

| State (top-level folder, verbatim) | Condition sub-folders observed | File count |
|---|---|---|
| `normal` | *(none)* | 49 |
| `imbalance` | `6g`, `10g`, `15g`, `20g`, `25g`, `30g`, `35g` | 333 |
| `horizontal-misalignment` | `0.5mm`, `1.0mm`, `1.5mm`, `2.0mm` | 197 |
| `vertical-misalignment` | `0.51mm`, `0.63mm`, `1.27mm`, `1.40mm`, `1.78mm`, `1.90mm` | 301 |

State names are used **exactly as they appear on disk** (hyphenated:
`horizontal-misalignment`, `vertical-misalignment`), not rewritten to an
underscore-separated form — the real dataset takes priority over any illustrative
naming used elsewhere.

## UNKNOWN cases

**No `UNKNOWN` recordings were found during this audit.** All 880 files inventoried in
`docs/dataset_audit/file_inventory.csv` matched one of the two path shapes above with a
recognized top-level state — verified programmatically (see
`backend/tests/dataset/test_mafaulda_parser.py::test_every_row_in_real_inventory_maps_to_a_known_state`,
which runs the parser against every real inventoried row and asserts zero `UNKNOWN`).

`UNKNOWN` is nonetheless a fully supported, tested outcome of
`backend/app/datasets/mafaulda_parser.py::parse_recording_state` — it is returned
(never guessed, never silently dropped) for any path that doesn't match a known
top-level state or doesn't have exactly 2 or 3 path segments (e.g. a stray file with no
folder at all, or unexpected deeper nesting).

## Audit conclusion

The state/condition mapping is **complete** for the dataset currently present in this
repository: every one of the 880 files inventoried in TASK 1.5.1 resolves to one of the
four states above (`normal`, `imbalance`, `horizontal-misalignment`,
`vertical-misalignment`), each with its condition correctly extracted where applicable.
Zero files required the `UNKNOWN` fallback.

This mapping says nothing yet about sampling rate, channel meaning, rotation frequency,
replication counts, class balance, data quality, or train/val/test split — those are
TASK 1.5.3–1.5.8.
