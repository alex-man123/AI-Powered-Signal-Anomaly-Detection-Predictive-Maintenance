# MAFAULDA Class & Rotation Distribution Audit — TASK 1.5.4

> Source of truth: `docs/dataset_audit/file_inventory.csv` (TASK 1.5.1), the state
> mapping established in `docs/dataset_audit/recording_mapping.md` (TASK 1.5.2), and
> direct inspection of the real filenames under `data/raw/mafaulda/`. No class, count,
> RPM/Hz value, or filename convention below is assumed from MAFAULDA documentation —
> everything is computed from the local inventory. Reproducible via
> `backend/app/datasets/class_distribution_audit.py` and
> `backend/scripts/generate_class_distribution_report.py`.

## 1. Scope

This audit counts recordings per class/state and determines, per recording, a rotation
frequency value together with its exact provenance. It is based entirely on the local
dataset inventory (`file_inventory.csv`) and the state mapping already established in
TASK 1.5.2 — no MAFAULDA CSV content is read, no signal processing (FFT/PSD/filtering)
is performed, and no train/val/test split, windowing, or ML happens here.

## 2. Dataset totals

- Total files: **880**
- Number of classes/states: **4** (`normal`, `imbalance`, `horizontal-misalignment`,
  `vertical-misalignment`) — zero `UNKNOWN`, consistent with TASK 1.5.2's finding.
- Number of files with rotation information: **880** (100%)
- Number without rotation information: **0**

## 3. Class distribution

| Class / State | File count | Percentage |
|---|---:|---:|
| `normal` | 49 | 5.57% |
| `imbalance` | 333 | 37.84% |
| `horizontal-misalignment` | 197 | 22.39% |
| `vertical-misalignment` | 301 | 34.20% |
| **TOTAL** | **880** | **100%** |

Validation: `sum(file_count_per_class) = 49 + 333 + 197 + 301 = 880 =
total_files_in_inventory`. **Confirmed equal** (see §8).

## 4. Class imbalance analysis

- Mean files/class: **220.0** (880 ÷ 4 real classes; `UNKNOWN` — none present — would be
  excluded from this mean if it existed, since it is a data-quality bucket, not a state)
- Severe-underrepresentation threshold (10% of mean, per AC2): **22.0 files**
- Percentage of mean, per class:

  | Class | Count | % of mean (220.0) | Severely underrepresented (<10% of mean)? |
  |---|---:|---:|---|
  | `normal` | 49 | 22.27% | No |
  | `imbalance` | 333 | 151.36% | No |
  | `horizontal-misalignment` | 197 | 89.55% | No |
  | `vertical-misalignment` | 301 | 136.82% | No |

- **Severely underrepresented classes (literal AC2 threshold): none.** No class falls
  below 22.0 files.

- **Risk assessment beyond the literal threshold (still worth flagging, per instruction
  not to treat "all classes present" as automatically "fine"):** `normal` (49 files) is
  still markedly smaller than every fault class (197–333 files) — less than a quarter of
  `imbalance`'s count. It does not cross the 10%-of-mean severity line, but the relative
  gap is real and matters specifically because the blueprint's training methodology
  (`docs/blueprint.md`, section 3) trains Isolation Forest/Autoencoder **exclusively on
  "normal"-labeled data**. Having the fewest recordings for the one class that is the
  sole source of training data — rather than for a fault class used only in
  evaluation — is a different, arguably more consequential, kind of imbalance than a
  straightforward "rare fault class" scenario. See §9 for the affected phases.

## 5. Rotation frequency provenance

All 880 filenames parse cleanly as a single decimal number (verified programmatically —
zero unparseable filenames). Two representative rows per state (full per-file data is
reproducible via `build_rotation_audit()`, not duplicated here for all 880 rows):

| Recording | State | Frequency (Hz) | Source | Evidence |
|---|---|---:|---|---|
| `normal/12.288.csv` | `normal` | 12.288 | filename | `12.288.csv` |
| `normal/13.1072.csv` | `normal` | 13.1072 | filename | `13.1072.csv` |
| `imbalance/10g/13.9264.csv` | `imbalance` | 13.9264 | filename | `13.9264.csv` |
| `imbalance/10g/14.5408.csv` | `imbalance` | 14.5408 | filename | `14.5408.csv` |
| `horizontal-misalignment/0.5mm/12.288.csv` | `horizontal-misalignment` | 12.288 | filename | `12.288.csv` |
| `vertical-misalignment/0.51mm/12.4928.csv` | `vertical-misalignment` | 12.4928 | filename | `12.4928.csv` |

**No `tachometer` or `metadata` values exist for this dataset.** Per TASK 1.5.3's
signal-structure audit, there is no header, no metadata file, and no locally-processed
tachometer channel anywhere in `data/raw/mafaulda/` — only raw sensor-amplitude columns.
A value read out of a filename is therefore always `rotation_source = "filename"`,
**never** `"tachometer"`, even though the number itself plausibly originates from a
tachometer measurement taken by MAFAULDA's original creators — we did not measure it
ourselves, we read it out of a filename, so it is labeled accordingly. `rotation_source
= "unavailable"` is a fully supported, tested code path (see
`test_extract_rotation_info_unparseable_filename_is_unavailable_not_guessed`); it is
simply never reached for this dataset, since every filename parses cleanly.

### Why the filename number is interpreted as Hz, not RPM

No file states a unit explicitly, so this is an inference — made transparent here, not
silently assumed:

- The 880 filename values range from **12.0832 to 62.2592** (see §6). As **RPM**, that
  would mean a shaft completing 12–62 full turns *per minute* — i.e. one rotation every
  ~1–5 seconds, an implausibly near-stationary rig for a vibration/unbalance/misalignment
  test setup. As **Hz** (12–62 full turns per *second* = ~725–3,735 RPM), it lands
  squarely in a standard motor/rotor operating range for this kind of test rig — physically
  coherent with the deliberately induced vibration faults (imbalance, misalignment) this
  dataset exists to capture.
- Structural evidence from the inventory itself: of the 241 distinct numeric values
  found across all 880 filenames, 201 recur across *multiple* different states/classes
  (e.g. the same value appears under `normal/` and under one or more fault folders).
  This is consistent with a controlled experiment that re-runs the same target rotation
  *speed* sweep for every condition — not with an arbitrary per-file ID or timestamp,
  which would not be expected to repeat across unrelated folders this often.
- This interpretation also happens to match MAFAULDA's external documentation
  (kept here strictly as secondary corroboration, not as the basis for the conclusion —
  the reasoning above stands on the locally-observed value range and repetition pattern
  alone).

No RPM→Hz conversion is applied to the stored `rotation_frequency_hz` value — the
filename number is used as-is, on the basis above that it already represents Hz.

## 6. Rotation frequency distribution

- All 880 recordings: source = `filename`, range **12.0832–62.2592 Hz**.
- Per state (min–max Hz, all source=`filename`):

  | State | n | Min (Hz) | Max (Hz) |
  |---|---:|---:|---:|
  | `normal` | 49 | 12.288 | 61.44 |
  | `imbalance` | 333 | 12.0832 | 62.0544 |
  | `horizontal-misalignment` | 197 | 12.288 | 62.0544 |
  | `vertical-misalignment` | 301 | 12.0832 | 62.2592 |

  Ranges overlap closely across all four states — consistent with the same target speed
  sweep being repeated per condition (§5).

## 7. Visualization

See `docs/dataset_audit/class_distribution.png` (generated by
`backend/scripts/generate_class_distribution_report.py`):

- **Left panel:** file count per class/state (bar chart, exact counts labeled).
- **Right panel:** rotation-frequency histogram, labeled `filename (n=880)` in the
  legend and titled to state explicitly that 0 recordings are `unavailable`. Only the
  `filename` category is drawn — no empty `tachometer`/`metadata` category is invented,
  since none exist for this dataset (per instruction not to fabricate categories with no
  real entries). The x-axis label states explicitly that the values come from filenames,
  not a direct measurement.

## 8. Audit validation

- Inventory total: **880**
- Class distribution total: **880** (49 + 333 + 197 + 301)
- Difference: **0**
- Coverage: **100%** of inventoried files have both a class/state (§3, all non-`UNKNOWN`)
  and a rotation-frequency provenance record (§5, all `filename`, none `unavailable`)

Also verified (see `backend/tests/dataset/test_class_distribution_audit.py`):
- Every `rotation_source` value is one of the 4 defined enum values.
- `rotation_source == "unavailable"` always implies `rotation_frequency_hz is None`,
  and every other source always has a non-null numeric value (tested against all 880
  real rows, not sampled).
- The severe-underrepresentation rule (§4) is proven with a synthetic, deliberately
  imbalanced dataset (`{a:100, b:100, c:100, d:5}` → flags `d`) — the real dataset
  doesn't happen to exercise that branch, so the logic is verified independently of it.

## 9. Risks for Phase 6–9

**Risk:** `normal` (49 files) is the class with the fewest recordings, and it is also
the *only* class the blueprint's methodology trains on (Isolation Forest/Autoencoder are
trained exclusively on "normal"-labeled windows per `docs/blueprint.md` section 3;
fault classes are used only for evaluation).

**Potential impact:**
- Fewer source recordings for `normal` means fewer distinct *operating conditions*
  (even after windowing produces many windows per file, they remain correlated samples
  from the same 49 underlying recordings — see the leakage-prevention discussion in
  blueprint section 8) available to characterize what "normal" looks like.
- A training distribution built from a narrower base could under-represent some normal
  operating variability, inflating false-positive risk (flagging legitimate but
  under-seen normal conditions as anomalous) once real evaluation begins.
- This is *not* the same failure mode as "too few fault examples to evaluate against" —
  fault classes here are comparatively well-represented (197–333 files) — so the usual
  worry about unstable minority-class metrics applies less; the risk is specifically
  about training-data diversity for the one class the models actually learn from.

**Affected phases:** Phase 6 (Isolation Forest training), Phase 7 (Autoencoder
training) — both consume only `normal` data; Phase 8 (IF vs. AE comparison) and Phase 9
(central DSP-vs-raw experiment) inherit whatever limitation exists in the upstream
trained models.

No mitigation (oversampling, augmentation, rebalancing, etc.) is implemented here —
per TASK 1.5.4 scope, this section only documents the risk for later phases to address
if it turns out to matter empirically.

## 10. Conclusion

- **AC1 — confirmed.** Class → file-count table is complete (4/4 real classes, `880`
  total, sum verified equal to the inventory total, §3/§8).
- **AC2 — confirmed, with an honest finding.** No class falls below the literal
  10%-of-mean severity threshold (§4). A real, if sub-threshold, imbalance is still
  flagged explicitly for Phase 6–9 (§9) rather than silently accepted just because the
  threshold wasn't crossed.
- **AC3 — confirmed.** Every one of the 880 rotation-frequency values carries an
  explicit `rotation_source` (`filename` for all 880 — never `tachometer`, since no
  locally-processed tachometer channel exists for this dataset). The Hz-vs-RPM
  interpretation is documented as a transparent inference from the locally-observed
  value range and repetition pattern (§5), not presented with unearned certainty.
