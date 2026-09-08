# MAFAULDA Dataset Audit Report — TASK 1.5.9 (Phase 1.5 Consolidation)

> **This document is the single source of truth for dataset decisions in Phase 2 and
> later.** It consolidates TASK 1.5.1–1.5.8, re-verified against the real local
> dataset, real audit artifacts, and real code at the time this report was written — not
> reconstructed from memory or from external MAFAULDA documentation. Every claim below
> carries an explicit source. Where something cannot be verified locally, this document
> says so explicitly (`UNKNOWN`) rather than filling the gap with an assumption.

## A note on this report's own instructions, read before anything else

The TASK 1.5.9 prompt that generated this document asks it to answer, one by one,
"all questions from blueprint.md section 6." **Section 6 of `docs/blueprint.md`, as it
actually exists, is "Recommended Primary Dataset" — one paragraph recommending
MAFAULDA, containing zero audit questions.** The 7-question chain the task's own
acceptance criteria describe (file organization, channels, recording ID, rotation,
replication, class distribution, split) is verbatim in **blueprint.md section 4**
("Dataset Research" → "Dataset Audit Protocol"), not section 6. This was verified by
directly reading `docs/blueprint.md` in full before writing this report — not assumed.
Given the exact topical match between the task's own description and section 4's
actual content, this is treated as a numbering slip in the task instructions, not a
real ambiguity, and §6 of this document below answers the real section-4 protocol.
This discrepancy is reported explicitly rather than silently resolved, per this task's
own rule for handling cross-document contradictions.

---

## 1. Executive Summary

```
Dataset:                MAFAULDA (local subset)
Local dataset root:     data/raw/mafaulda/
Total files:             880
Classes:                 normal, imbalance, horizontal-misalignment, vertical-misalignment
Channels:                8 numeric columns per file (no header; physical sensor identity
                         not independently verified locally)
Sampling rate:           50,000 Hz — adopted from official documentation, NOT measured
                         locally (see §8)
Signal duration:         5.0 s — derived from the adopted rate, NOT measured locally
Rotation:                241 distinct target-frequency values, range 12.0832–62.2592 Hz,
                         source = "filename" for all 880 recordings (never tachometer)
Replication structure:   567 (state, frequency) groups; 230 have >=2 files, but 0 groups
                         remain once severity is also included in the key
Leakage risk:            LOW
Split strategy:          FILE
Train:                   611
Validation:              135
Test:                    134
Quality issues:          0 (0 missing values, 0 constant/zero channels, 0 read errors,
                         across all 880 files)
Final audit status:      PASS
```

Source: every value above is expanded with its exact source in §3–§18.

---

## 2. Audit scope and evidence

This report consolidates, and re-verifies against the live filesystem/artifacts at
report-writing time (not from memory), the following:

| Prior task | Artifact | Re-verified in this report |
|---|---|---|
| 1.5.1 | `docs/dataset_audit/file_inventory.csv`, `backend/scripts/audit_inventory.py` | Row count re-counted: 880, matches a fresh `find data/raw/mafaulda -type f \| wc -l` (880) |
| 1.5.2 | `docs/dataset_audit/recording_mapping.md`, `backend/app/datasets/mafaulda_parser.py` | Re-read; 0 `UNKNOWN` confirmed |
| 1.5.3 | `docs/dataset_audit/signal_structure.md` | Re-read; figures cross-checked against §5 |
| 1.5.4 | `docs/dataset_audit/class_distribution.md`, `class_distribution.png` | Class counts re-grepped from the file, match §9 |
| 1.5.5 | `docs/dataset_audit/replication_analysis.md` | Group counts re-grepped, match §11 |
| 1.5.6 | `docs/dataset_audit/leakage_analysis.md` | Final decision re-grepped, matches §12 |
| 1.5.7 | `data/processed/split_manifest.json`, `backend/app/datasets/split.py`, `backend/tests/dataset/test_split.py` | Re-loaded and re-checked for overlap/coverage at report-writing time (§13/§14) |
| 1.5.8 | `docs/dataset_audit/data_quality_report.md` | Re-read; 0 problematic files confirmed |

No prior audit artifact was modified while writing this report. No contradiction was
found between any two documents, or between any document and the live dataset — see
§23 equivalent check inline in §13/§18.

---

## 3. Dataset inventory

- **Dataset root:** `data/raw/mafaulda/`
- **Total files:** 880 — re-verified with `find data/raw/mafaulda -type f | wc -l` at
  report-writing time, matches `file_inventory.csv` row count exactly.
- **Total size:** 15,300,671,008 bytes (~14.25 GB) — Source:
  `docs/dataset_audit/file_inventory.csv` (sum of `size_bytes`), TASK 1.5.1.
- **Extension distribution:** 100% `.csv` — no other extension exists anywhere in the
  tree (TASK 1.5.1).
- **Folder structure:** 4 top-level state folders; 3 of them (`imbalance`,
  `horizontal-misalignment`, `vertical-misalignment`) each have one level of severity
  sub-folder; `normal/` has none. Source: `docs/dataset_audit/recording_mapping.md`
  (TASK 1.5.2), re-confirmed with a fresh `find` scan.
- **UNKNOWN/MISSING cases:** none. Every file maps to a known state (TASK 1.5.2,
  `mafaulda_parser.py::parse_recording_state`); re-verified: 0 `UNKNOWN` across all 880
  rows.

Source: `docs/dataset_audit/file_inventory.csv`.

---

## 4. File organization and recording mapping

- **How is the dataset organized?** `<state>/<filename>.csv` (for `normal`) or
  `<state>/<condition>/<filename>.csv` (for the other 3 states) — exactly two path
  shapes exist, verified for all 880 files.
- **How is class identified?** The top-level folder name, taken verbatim (e.g.
  `horizontal-misalignment`, not rewritten to underscore form).
- **How is condition identified?** The severity sub-folder name (e.g. `0.5mm`, `10g`),
  present for every state except `normal`.
- **Recording ID:** Not present. No field or path segment is a recording identifier —
  see §7.
- **Session ID:** Not present in path or filename — see §7.
- **Run ID:** Not present in path or filename — see §7.
- **What's in the path:** state + (optionally) condition.
- **What's in the filename:** a single decimal number (e.g. `12.288.csv`) — see §10 for
  what this number is understood to represent, and how confidently.

Source: `docs/dataset_audit/recording_mapping.md`, `backend/app/datasets/mafaulda_parser.py`.

---

## 5. Signal structure

Per TASK 1.5.3, verified on 4 real files (one per state — `normal/12.288.csv`,
`imbalance/10g/13.9264.csv`, `horizontal-misalignment/0.5mm/12.288.csv`,
`vertical-misalignment/0.51mm/12.4928.csv`), and re-confirmed structurally for **all
880 files** in TASK 1.5.8's full quality pass:

- **Columns:** 8, consistently — for all 4 sampled files (1.5.3) and independently
  re-confirmed for all 880 files (1.5.8: "distinct column counts observed: `{8}`").
- **Header:** none — first line of every sampled file is plain numeric data.
- **dtype:** `float64` for every column, in every file checked.
- **Rows:** 250,000, identical across all 4 sampled files (1.5.3) and — via row-count
  used to compute `missing_percentage` — implicitly consistent with all 880 files in
  1.5.8's full scan (no file produced a shape anomaly).
- **Missing values (structural sample, 1.5.3):** none in the 4 sampled files.
  **Full-dataset check (1.5.8):** 0 missing values in all 880 files.
- **Sampling rate / duration:** see §8 — not locally measurable.

Source: `docs/dataset_audit/signal_structure.md`, `docs/dataset_audit/data_quality_report.md`.

---

## 6. Blueprint Dataset Audit Protocol — explicit answers

*(This answers the 7-question chain in `docs/blueprint.md` section 4, "Dataset Audit
Protocol" — see the note at the top of this document for why section 4, not the
literal section 6.)*

### Question — Câte fișiere / recordings există, și cum sunt denumite?

**Answer:** 880 files. Naming: `<state>/[<condition>/]<frequency>.csv` — state and
condition (where present) are folder names; the filename is a decimal number (§4, §10).

**Source:** `docs/dataset_audit/file_inventory.csv`, `docs/dataset_audit/recording_mapping.md`.

**Status:** CONFIRMED.

### Question — Ce reprezintă fiecare coloană/canal, exact, per fișier?

**Answer:** 8 numeric (`float64`) columns per file, positionally identified
(`column_0`…`column_7`). **No physical sensor mapping (e.g. "column 0 = tachometer") is
independently verified from local data** — the files carry no header or channel labels.

**Source:** `docs/dataset_audit/signal_structure.md`.

**Status:** CONFIRMED for column count/dtype; sensor-identity mapping is UNKNOWN /
NOT INDEPENDENTLY VERIFIED.

### Question — Cum se identifică unic un "recording" (nume fișier → stare + condiție de operare)?

**Answer:** A recording is uniquely identified by its `relative_path`, which decomposes
into `state` (folder) + `condition` (sub-folder, when present) + a numeric filename.
There is no separate recording ID field — the full relative path is the only unique
identifier available.

**Source:** `docs/dataset_audit/recording_mapping.md`.

**Status:** CONFIRMED.

### Question — Ce turații/rotation frequencies apar, și cum variază între fișiere din aceeași clasă?

**Answer:** 241 distinct filename-derived values across the dataset, range
12.0832–62.2592 Hz. Per-state ranges: `normal` 12.288–61.44 Hz, `imbalance`
12.0832–62.0544 Hz, `horizontal-misalignment` 12.288–62.0544 Hz,
`vertical-misalignment` 12.0832–62.2592 Hz — ranges overlap closely across states,
consistent with a similar target-speed sweep repeated per condition.

**Source:** `docs/dataset_audit/class_distribution.md` §5–§6.

**Status:** CONFIRMED (value + provenance = `filename`; see §10 for confidence caveats).

### Question — Există replicări ale aceleiași condiții (mai multe fișiere pentru aceeași stare+turație)?

**Answer:** Yes, at the `(state, frequency)` level: 230 of 567 groups have ≥2 files.
**But** once severity/condition is also included in the key, **0** groups have ≥2
files — i.e. no two files share class **and** severity **and** target speed. The 230
"replicated" groups are fully explained by different severities coincidentally sharing
a target speed, not by repeated identical trials.

**Source:** `docs/dataset_audit/replication_analysis.md`.

**Status:** CONFIRMED.

### Question — Distribuția claselor: e echilibrată? Câte fișiere per clasă?

**Answer:** `normal` 49 (5.57%), `imbalance` 333 (37.84%), `horizontal-misalignment` 197
(22.39%), `vertical-misalignment` 301 (34.20%). No class falls below the 10%-of-mean
severe-underrepresentation threshold (mean=220, threshold=22) — but `normal` (the only
class the blueprint's training methodology actually uses) is still markedly smaller
than the fault classes, flagged as a Phase 6–9 risk, not silently accepted.

**Source:** `docs/dataset_audit/class_distribution.md`.

**Status:** CONFIRMED.

### Question — Ce poate ajunge legitim în train vs. test fără a încălca split-ul per fișier?

**Answer:** Any file can be allocated independently — TASK 1.5.6 found no session/group
unit larger than one file that needs to stay intact across splits (zero true
same-severity replicates, no session identifiers, weak content correlation between the
most plausible "related" pairs). File-level split (TASK 1.5.7) is the result: 611
train / 135 validation / 134 test, stratified per class, zero overlap.

**Source:** `docs/dataset_audit/leakage_analysis.md`, `data/processed/split_manifest.json`.

**Status:** CONFIRMED.

---

## 7. Recording / session identity

**No independently verified session/recording ID was found.** Checked explicitly in
TASK 1.5.6: a keyword search across all 880 `relative_path` values for
session/run/experiment/trial/recording/sample/measurement/test returned **zero
matches**. The only non-textual candidate is filesystem `mtime`: every file within one
`(state, condition)` folder shares exactly one calendar date (`2014-10-03` or
`2014-10-06`), which is **filesystem/packaging metadata, not a field inside the
data**, and is explicitly documented as a low-confidence, unresolved, uncorroborated
signal — not treated as a session ID.

"Same class + same rotation" (§6, replication question) is explicitly **not** treated
as "same session" — TASK 1.5.6/1.5.5 verified these are physically distinct recordings
(different severities), not repeated captures of one setup.

Source: `docs/dataset_audit/leakage_analysis.md` §3.2–3.3, `docs/dataset_audit/replication_analysis.md` §6.

---

## 8. Sampling rate and signal duration

```
Sampling rate:  50,000 Hz
Evidence:       ADOPTED from official MAFAULDA documentation (UFRJ page) — NOT measured
                or derived independently from local file content. Project decision,
                recorded explicitly (see docs/dataset_audit/signal_structure.md §5).
sampling_rate_source: official_documentation (not measured_locally)
```

**What was actually checked locally and found insufficient to derive this
independently:** no header, no metadata file anywhere in `data/raw/mafaulda/`, and none
of the 8 columns in any of the 4 sampled files is a monotonic time/duration series
(checked: 32 of 32 column-checks non-monotonic). Local evidence provides only
**indirect corroboration**: the adopted rate × MAFAULDA's documented 5 s window
predicts exactly 250,000 samples — which matches the locally-measured row count in all
880 files (1.5.8) — but this is consistency with an assumption, not independent
derivation, since no file contains a time axis.

```
Signal duration: 5.0 s
```
Derived as `250,000 / 50,000`, i.e. **dependent on the adopted (not locally measured)
sampling rate** — not an independent local measurement.

Source: `docs/dataset_audit/signal_structure.md` §5–§6, §9.

---

## 9. Classes / states

| Class / State | File count | Percentage | Major class? | Source |
|---|---:|---:|---|---|
| `normal` | 49 | 5.57% | YES | `class_distribution.md` §3 |
| `imbalance` | 333 | 37.84% | YES | `class_distribution.md` §3 |
| `horizontal-misalignment` | 197 | 22.39% | YES | `class_distribution.md` §3 |
| `vertical-misalignment` | 301 | 34.20% | YES | `class_distribution.md` §3 |
| **TOTAL** | **880** | **100%** | — | — |

No `UNKNOWN` class exists (re-verified: `mafaulda_parser.py` applied to all 880 rows
yields 0 `UNKNOWN`, TASK 1.5.2/1.5.7 tests).

**"Major class" definition used throughout (1.5.7/1.5.9):** a state not flagged as
severely underrepresented in TASK 1.5.4's imbalance analysis (count ≥ 10% of the mean
files/class — mean=220, threshold=22 files). All 4 real states clear this bar; TASK
1.5.4 did not itself coin a "major class" term, so this reuses its existing documented
threshold rather than inventing a new one.

Source: `docs/dataset_audit/class_distribution.md`, `backend/app/datasets/class_distribution_audit.py::analyze_imbalance`.

---

## 10. Rotation frequency

| Class / Group | Rotation range (Hz) | Source | Confidence |
|---|---:|---|---|
| `normal` | 12.288–61.44 | filename | See caveat below |
| `imbalance` | 12.0832–62.0544 | filename | See caveat below |
| `horizontal-misalignment` | 12.288–62.0544 | filename | See caveat below |
| `vertical-misalignment` | 12.0832–62.2592 | filename | See caveat below |

**Provenance for all 880 recordings: `rotation_source = "filename"`.** Never
`"tachometer"` — there is no locally-processed tachometer channel; the value is read
out of a filename, not measured by this project. `"metadata"` and `"unavailable"` are
supported, tested code paths (`extract_rotation_info`) but do not occur in this dataset
(0 of 880).

**Confidence caveat (Hz vs RPM):** no file states a unit. The Hz interpretation is an
explicit inference from local evidence — the observed 12.0832–62.2592 range is
implausible as RPM (near-stationary shaft) and physically standard as Hz for this rig
(~725–3,735 RPM equivalent); separately, 201 of 241 distinct values recur across
different state folders, consistent with a repeated target-speed sweep rather than
arbitrary per-file IDs. This is documented as an inference, not asserted as certain.

Source: `docs/dataset_audit/class_distribution.md` §5–§6.

---

## 11. Replications

```
Total (state, frequency) condition groups:  567
Groups with >=2 files:                       230
Groups with exactly 1 file:                  337
```

Per state: `normal` 49 groups / 0 replicated; `horizontal-misalignment` 146 groups / 47
replicated; `imbalance` 183 groups / 102 replicated; `vertical-misalignment` 189 groups
/ 81 replicated.

**Experimental replication vs. digital duplication vs. session — all distinguished:**
- **Digital duplicates:** 0 — every file has a unique `size_bytes` (880 distinct sizes
  for 880 files), and content-level spot checks (1.5.6) found no identical files.
- **True same-severity replication (identical state + condition + frequency):** 0 —
  once severity is included in the group key, every group has exactly 1 file.
- **"Same experimental condition" (state + frequency only) ≠ "same session":** the 230
  `(state, frequency)` groups are explained entirely by different severities sharing a
  target speed (verified via the finer grouping above and via signal-content checks
  showing weak correlation, comparable to unrelated file pairs).

Source: `docs/dataset_audit/replication_analysis.md`.

---

## 12. Leakage risk

```
Overall leakage risk:              LOW
Is file-level split safe?          YES
Is group/session-level split required?  NO
```

**Evidence:** zero exact duplicates; zero session/run/trial identifiers in any of 880
real paths; zero true same-`(state,condition,frequency)` replicates; weak
signal-content correlation (max |r|=0.44) between the most plausible "related" file
pairs (same frequency, different severity) — comparable to an unrelated baseline pair;
a measured content discontinuity (3 of 8 channels jumping 3.6×–23× beyond typical
within-file variation) in the one continuous-recording-split candidate tested. The one
open signal (filesystem mtime-date clustering) is real but unresolved and not
corroborated by any content-level dependency — documented, not acted upon.

This decision is authoritative for TASK 1.5.7 and this report — not re-derived or
second-guessed here.

Source: `docs/dataset_audit/leakage_analysis.md`.

---

## 13. Final split strategy

```
Split unit:        FILE
Train ratio:       0.70
Validation ratio:  0.15
Test ratio:        0.15
Seed:              42
Manifest:          data/processed/split_manifest.json
```

Ratio and seed are the documented defaults specified by TASK 1.5.7 itself — neither
`docs/blueprint.md` nor `docs/backlog.md` specifies a project-level ratio or seed.

Source: `data/processed/split_manifest.json`, `backend/app/datasets/split.py`.

---

## 14. Split manifest

Re-verified directly against the live `data/processed/split_manifest.json` and the live
`docs/dataset_audit/file_inventory.csv` at report-writing time (not assumed from TASK
1.5.7's own report):

```
Train count:       611
Validation count:  135
Test count:        134
Total:             880
```

| Check | Result |
|---|---|
| No overlap (train∩val, train∩test, val∩test) | **PASS** — 0, 0, 0 |
| Complete coverage (train∪val∪test == inventory) | **PASS** — union has 880 elements, exactly equal to the 880-element inventory set |
| Deterministic (same seed → same manifest) | **PASS** — re-verified by `backend/tests/dataset/test_split.py` (runs `create_split` twice with the real 880-record inventory and seed=42, asserts identical results) |
| Class representation (all 4 major classes, all 3 splits > 0) | **PASS** — `horizontal-misalignment` 137/30/30, `imbalance` 232/51/50, `normal` 33/8/8, `vertical-misalignment` 209/46/46 |

Source: `data/processed/split_manifest.json`, `backend/tests/dataset/test_split.py`, direct re-check at report-writing time.

---

## 15. Data quality

```
Files analyzed:              880 (full dataset, not a sample)
Files with missing values:   0
Files with constant signals: 0
Files with zero signals:     0
Unreadable files:            0
Other issues:                0 (column-count consistency also re-confirmed: all 880
                              files have exactly 8 columns)
```

No problematic-file table is included because none exists — 0 of 880 files triggered
any quality category. The general KEEP/EXCLUDE/REPAIR/MANUAL-REVIEW decision policy per
category (for if a future re-audit does find something) is documented in
`docs/dataset_audit/data_quality_report.md` §9 and is not repeated here in full.

Source: `docs/dataset_audit/data_quality_report.md`.

---

## 16. Final dataset decisions

Each rule below is derived directly from the audit sections referenced, not asserted
independently of them:

1. Dataset root is `data/raw/mafaulda/` — 880 files, 4 states, verified read-only
   throughout Phase 1.5 (§3).
2. Split unit is **FILE** — justified by TASK 1.5.6's leakage analysis (§12), not
   re-decided here.
3. Files are never split by window — TASK 1.5.7 explicitly allocates whole files, not
   windows (§13/§14).
4. Rotation provenance (`filename`, for all 880) must be preserved and never presented
   as a `tachometer` measurement (§10).
5. Sampling rate (50,000 Hz) is an **adopted external value**, not a local measurement
   — any code that depends on it (FFT, PSD, windowing) must treat it as such, not as an
   independently-verified fact (§8).
6. No file is currently marked `EXCLUDE`/`REPAIR REQUIRED` — data quality found zero
   issues (§15); this may change if the dataset is ever updated, at which point TASK
   1.5.8's audit should be re-run.
7. The `(state, frequency)` replication groups (§11) are **not** a split constraint —
   TASK 1.5.6 found no session/session-like dependency requiring them to stay together.
8. `split_manifest.json`'s existing train/validation/test allocation (§13/§14) is the
   authoritative split for Phase 2 — it must not be regenerated with a different seed
   or ratio without re-running the AC1–AC3 validations from TASK 1.5.7.

---

## 17. Phase 2 rules

Developers implementing Phase 2 (dataset loader, TASK 2.x) must respect:

- **Valid file paths:** only paths listed in `data/processed/split_manifest.json`
  (`splits.train`/`splits.validation`/`splits.test`) are valid inputs; they are
  relative to `data/raw/mafaulda/` (§13/§14).
- **Class mapping:** use `backend/app/datasets/mafaulda_parser.py::parse_recording_state`
  — do not re-derive state from paths independently, to avoid drift from the audited
  mapping (§4/§6).
- **Split manifest is authoritative:** do not regenerate it ad hoc; if it must change,
  re-run TASK 1.5.7's full validation suite (§14).
- **Quality exclusions:** none currently exist (§15) — no filtering logic is required
  for data quality reasons at this time.
- **Rotation provenance:** always carry `rotation_source` alongside any
  `rotation_frequency_hz` value consumed from `class_distribution.md`/
  `replication_analysis.md` data structures — never treat a `filename`-sourced value as
  a direct measurement (§10).
- **Group/session constraints:** none — file-level allocation is sufficient (§12); no
  additional grouping logic is required when loading train/val/test.
- **Sampling rate:** treat 50,000 Hz as an adopted external constant
  (`sampling_rate_source: official_documentation`), not a locally-verified fact —
  document this provenance wherever it's used (e.g. FFT/PSD calls) (§8).
- **Channel interpretation:** 8 positional columns; no physical sensor identity is
  locally verified — do not label columns as specific sensors without new evidence (§5/§6).
- **No leakage:** never split by window; never move a file between train/val/test after
  the fact (§12/§14).
- **Reproducibility:** `create_split(seed=42, ratios={0.70,0.15,0.15})` regenerates the
  exact same manifest — verified deterministic (§14); Phase 2 code should not hardcode
  the file lists, but read them from `split_manifest.json`.

---

## 18. Evidence / source matrix

| Claim / Decision | Value | Source | Verified? |
|---|---|---|---|
| Dataset root | `data/raw/mafaulda/` | `file_inventory.csv` | YES |
| Total files | 880 | `file_inventory.csv`, re-counted via `find` | YES |
| File organization | 2 path shapes (state[/condition]/freq.csv) | `recording_mapping.md` | YES |
| Classes | 4 (normal, imbalance, horizontal-misalignment, vertical-misalignment) | `class_distribution.md` | YES |
| Channels | 8 numeric columns, no header | `signal_structure.md`, `data_quality_report.md` | YES |
| Physical sensor mapping | Not independently verified | `signal_structure.md` §6 | UNKNOWN (explicitly) |
| Sampling rate | 50,000 Hz | `signal_structure.md` §5 | ADOPTED, not locally measured (explicitly marked) |
| Signal duration | 5.0 s | `signal_structure.md` §6 | DERIVED from adopted rate, not measured |
| Recording ID | Not present | `recording_mapping.md`, `leakage_analysis.md` §3.2 | YES (absence confirmed) |
| Session ID | Not present | `leakage_analysis.md` §3.2/3.3 | YES (absence confirmed); mtime-date clustering noted as unresolved, LOW confidence |
| Rotation values | 241 distinct, 12.0832–62.2592 Hz | `class_distribution.md` | YES (value); Hz-vs-RPM unit is a documented inference, not certain |
| Rotation source | `filename` for all 880 | `class_distribution.md` | YES |
| Replications | 567 groups, 230 with ≥2 files; 0 true same-severity replicates | `replication_analysis.md` | YES |
| Leakage risk | LOW | `leakage_analysis.md` | YES |
| Split strategy | FILE | `leakage_analysis.md`, `split.py` | YES |
| Train count | 611 | `split_manifest.json`, re-verified | YES |
| Validation count | 135 | `split_manifest.json`, re-verified | YES |
| Test count | 134 | `split_manifest.json`, re-verified | YES |
| Split overlap | None | `split_manifest.json`, re-verified at report time | YES |
| Data quality | 0 problematic files of 880 | `data_quality_report.md`, full-dataset run | YES |

No claim in this matrix is left without a source; the two `UNKNOWN`/adopted-value rows
(sensor mapping, sampling rate) are marked as such deliberately, per this task's own
instruction that this is preferable to an unearned assumption.

---

## 19. Audit Gate

```
[x] All Section 6 (blueprint section 4) questions answered
[x] All major claims have sources
[x] No unresolved contradictions
[x] Real dataset inventory consistent (880 == 880, re-verified at report time)
[x] Split manifest consistent (overlap=0, coverage=880/880, re-verified at report time)
[x] Leakage decision consistent with split strategy (LOW risk -> FILE split, both agree)
[x] Data quality report consistent (0 issues, re-confirmed against the full 880-file run)
[x] Phase 2 rules documented
```

Status: **PASS**

One process note, not a blocking condition: the task instructions that generated this
report cited "blueprint.md section 6" for the audit-protocol questions; the actual
questions live in section 4 (see the note after the title). This is a documentation
labeling discrepancy in the task prompt itself, not a dataset or audit inconsistency,
and does not affect the Gate status — flagged for visibility, per this task's own rule
against hiding contradictions.
