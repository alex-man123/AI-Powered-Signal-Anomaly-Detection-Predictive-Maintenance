# MAFAULDA Data Quality Report — TASK 1.5.8

> Source of truth: every one of the 880 real files under `data/raw/mafaulda/`, read
> directly with pandas, one file at a time (never all loaded simultaneously).
> Reproducible via `backend/app/datasets/data_quality.py` and
> `backend/scripts/run_data_quality_audit.py`. This is a **full** audit, not a sample —
> all 880 inventoried files were analyzed (§2).

## 1. Scope

This audit checks every real MAFAULDA file for: missing values (NaN), constant
channels, zero-signal channels, and read/parse/structure failures. No preprocessing, no
filtering, no repair, no exclusion is applied — detection and documented decision only.
`data/raw/mafaulda/` is read-only throughout; verified untouched after the audit (file
modification times unchanged).

Thresholds are objective, not arbitrary: a channel is "constant" only if
`max == min` among its present (non-missing) values; "zero" only if that constant value
is exactly `0`. No variance/std threshold is used — low variance is explicitly not
treated as evidence of a constant or corrupted signal, per the task's own instruction.

## 2. Dataset coverage

- Inventory files (`docs/dataset_audit/file_inventory.csv`): **880**
- Files successfully analyzed: **880**
- Files with read errors: **0**
- Coverage: **100%** — every inventoried file was read and classified; this is not a
  sampled check.

## 3. Missing values

**Zero files have any missing value.** Checked for all 880 files (250,000 rows × 8
columns = 2,000,000 values each, 1,760,000,000 values total across the dataset): every
file has `missing_values = 0`, `missing_percentage = 0.0000%`, `status = OK`.

No table of individual files is given here because there is nothing to list — the
programmatic check (`backend/tests/dataset/test_quality.py`, plus the full-dataset run
of `run_data_quality_audit.py`) confirms `len(missing_value_files) == 0`.

## 4. Constant signals

**Zero channels, in zero files, are constant-non-zero.** Every one of the 880 files ×
8 channels = 7,040 channel-checks found `max != min` for every channel in every file —
no flat/stuck sensor channel exists anywhere in this dataset.

## 5. Zero signals

**Zero channels, in zero files, are exactly zero-valued.** Same full check as §4,
specifically distinguishing "constant-non-zero" from "identically zero" — none of either
kind were found.

## 6. Read / parse errors

**None.** All 880 files parsed successfully as pure numeric CSV data with pandas
(`header=None`), matching TASK 1.5.3's structural finding (no header, 8 columns,
`float64`, 250,000 rows) for every single file, not just the 4 files TASK 1.5.3 sampled.

## 7. Other observed quality issues

- **Column-count consistency:** all 880 files have exactly **8** columns — the single
  distinct value observed across the full dataset (`{8}`). TASK 1.5.3 established this
  from a 4-file sample; this audit confirms it holds for every file, not just the
  sample.
- No other structural anomaly was found.

## 8. Summary

```
Total files:            880
Clean files:            880
Problematic files:        0
Missing-value files:      0
Constant-signal files:    0
Zero-signal files:        0
Unreadable files:         0
```

## 9. Decision

No problematic files exist for any category, so there is nothing to decide
per-file. The general policy, recorded here for completeness in case a future re-audit
(e.g. after a dataset update) does find an issue:

| Category | Decision if found | Reason |
|---|---|---|
| Missing values (any NaN) | MANUAL REVIEW | Amount and location matter (a few NaN in one channel of a multi-channel file is very different from a mostly-missing file) — a blanket KEEP/EXCLUDE rule would be a guess, not a decision grounded in the specific case. |
| Constant non-zero channel (single channel, multi-channel file) | KEEP, channel-level note | A single stuck channel does not necessarily invalidate the other 7 channels' data — this is a channel-level problem, not automatically a file-level one (per the task's explicit instruction not to over-exclude). |
| Zero-signal channel (single channel, multi-channel file) | KEEP, channel-level note | Could reflect an unused/disconnected sensor rather than a corrupted recording — same reasoning as above. |
| All channels constant or zero in one file | EXCLUDE | A file with no varying signal in any channel carries no information for anomaly detection — this is a file-level, not channel-level, problem. |
| Read/structure/non-numeric error | EXCLUDE (until repaired) | An unparseable file cannot be used as-is; whether it's worth repairing is a separate decision outside this audit's scope. |

**For the dataset as it currently exists, no category above is triggered, so no
exclusion, repair, or manual review is actually required.**

## 10. Impact on downstream pipeline

Phase 2 (dataset loader, TASK 2.x) can proceed on the assumption that every file listed
in `split_manifest.json` (TASK 1.5.7) is structurally clean: 8 numeric columns, 250,000
rows, zero missing values, zero constant/zero channels. No filtering, imputation, or
per-file exclusion logic needs to be built to handle a data-quality problem, because
none was found. This audit does not change `split_manifest.json` and does not
re-evaluate leakage (TASK 1.5.6) or the split itself (TASK 1.5.7) — if a future dataset
version does introduce quality issues, this audit should be re-run before trusting the
existing split.
