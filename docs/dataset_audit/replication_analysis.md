# MAFAULDA Replication Analysis — TASK 1.5.5

> Source of truth: `docs/dataset_audit/file_inventory.csv` (TASK 1.5.1), the state
> mapping from TASK 1.5.2, and the rotation-frequency provenance established in TASK
> 1.5.4. Nothing here is assumed — every group and count is computed from the real 880
> recordings. Reproducible via `backend/app/datasets/replication_analysis.py`.

## 1. Scope

This audit groups the 880 real recordings by condition and reports which groups have
more than one recording ("replicated") versus exactly one ("singleton"). It also checks,
separately, whether any files are byte-for-byte duplicates. No train/val/test split is
implemented here — this report is input for TASK 1.5.7.

## 2. Grouping methodology

**Group definition (as specified by TASK 1.5.5):**
```
group = (state, approximate rotation frequency)
```

**Rotation-frequency tolerance: exact match, zero tolerance.** Not an arbitrary choice —
justified by the real distribution of `rotation_frequency_hz` values across all 880
recordings:
- 241 distinct values exist in total.
- The **minimum gap** between any two *different* distinct values is **0.2048 Hz**.
- **Zero** pairs of distinct values are closer than 0.01 Hz.

Values are therefore already cleanly discrete — there is no floating-point noise of the
kind the task's illustrative example warns about (e.g. `24.98` vs `25.01` vs `25.03`
being "the same" condition). Grouping by exact numeric equality is sufficient and
introduces no risk of either splitting a true match or merging two genuinely different
speeds.

**Rotation source is preserved per group** (per TASK 1.5.4's provenance field) — see §5.
For this dataset every value's source is `filename` (TASK 1.5.4 finding: no
`tachometer`/`metadata` source exists locally), so no group currently mixes sources of
different certainty — but the code (`ConditionGroup.rotation_sources`) records this
per-group and is tested against a synthetic mixed-source case, so it would surface
immediately if a future re-audit found otherwise.

**Critical caveat — read before interpreting §3 as "the same experiment repeated":**
the group key is `(state, frequency)` only, **not** `(state, condition/severity,
frequency)`. `horizontal-misalignment`, `imbalance`, and `vertical-misalignment` each
have a severity sub-folder (`0.5mm`, `10g`, etc. — TASK 1.5.2) that this grouping
intentionally ignores, per the task's own definition. This means a "replicated" group
here can be **multiple different severities that happen to share a target speed** —
not the same physical setup recorded more than once. §6 verifies this distinction
directly against the real data.

## 3. Replication groups

**Total condition groups (state, frequency): 567.** Full per-group data is reproducible
via `build_condition_groups()`; a representative sample (including the two largest
groups) is shown below — not all 567 rows, to keep this document readable.

| Group | State | Approx. rotation Hz | Rotation sources | File count | Replicated? |
|---|---|---:|---|---:|---|
| G1 | `imbalance` | 45.4656 | filename | 5 | YES |
| G2 | `horizontal-misalignment` | 12.288 | filename | 4 | YES |
| G3 | `horizontal-misalignment` | 13.5168 | filename | 2 | YES |
| G4 | `vertical-misalignment` | 12.0832 | filename | 2 | YES |
| G5 | `normal` | 12.288 | filename | 1 | NO |
| G6 | `imbalance` | 12.288 | filename | 1 | NO |

**Manual verification (largest group, G1):** `imbalance` @ 45.4656 Hz resolves to
exactly these 5 real files, one per severity condition:
```
imbalance/6g/45.4656.csv
imbalance/10g/45.4656.csv
imbalance/20g/45.4656.csv
imbalance/30g/45.4656.csv
imbalance/35g/45.4656.csv
```
This is the direct illustration of the caveat in §2: these 5 files are **not** 5 takes
of the identical setup — they are the *same target speed* measured at 5 *different*
imbalance severities. Traced by hand against `data/raw/mafaulda/imbalance/`; all 5 exist.

**Manual verification (G2):** `horizontal-misalignment` @ 12.288 Hz resolves to
```
horizontal-misalignment/0.5mm/12.288.csv
horizontal-misalignment/1.0mm/12.288.csv
horizontal-misalignment/1.5mm/12.288.csv
horizontal-misalignment/2.0mm/12.288.csv
```
— one file per each of the 4 horizontal-misalignment severities, same pattern as G1.

## 4. Replication summary

```
Total condition groups (state, frequency): 567
Groups with >=2 files (replicated):        230
Groups with exactly 1 file (singleton):     337
Total recordings:                           880
```
(230 + 337 = 567 groups; group sizes {1: 337, 2: 160, 3: 58, 4: 11, 5: 1} files sum to
337·1 + 160·2 + 58·3 + 11·4 + 1·5 = 880 — verified equal to the inventory total, §8.)

Per state:

| State | Condition groups | Replicated (≥2) | Singleton (1) |
|---|---:|---:|---:|
| `normal` | 49 | 0 | 49 |
| `horizontal-misalignment` | 146 | 47 | 99 |
| `imbalance` | 183 | 102 | 81 |
| `vertical-misalignment` | 189 | 81 | 108 |
| **TOTAL** | **567** | **230** | **337** |

`normal` has zero replicated groups — expected, since it has no severity sub-folder to
generate cross-condition speed coincidences (§6 confirms it also has zero internal
repeats).

## 5. UNKNOWN rotation groups

**None exist for this dataset.** TASK 1.5.4 found `rotation_source = "unavailable"` for
0 of the 880 real recordings (every filename parses as a decimal number). The
`UNKNOWN_ROTATION` grouping bucket (`group_key_for` mapping any `unavailable` recording
to `(state, "UNKNOWN_ROTATION")`) is implemented and unit-tested
(`test_unavailable_rotation_recordings_are_grouped_not_dropped`), but is not exercised
by the real data as currently inventoried — it exists so that if a future dataset
version (or a re-run of this audit) does contain unavailable rotation values, they are
grouped and counted, never silently dropped.

## 6. Digital duplicates

**Zero byte-for-byte duplicates.** Checked without reading or hashing any file content:
every one of the 880 files has a **unique `size_bytes`** value in `file_inventory.csv`
(880 distinct sizes for 880 files). Since byte-identical files necessarily have
identical sizes, a unique size for every file rules out any byte-for-byte duplicate
directly — no need to hash ~14 GB of data (see performance constraints, §14 of the task
prompt).

**Separately, and more importantly for §2's caveat:** within any single `(state,
condition)` folder — i.e. the same class *and* the same severity — every
`rotation_frequency_hz` value is also unique. Checked directly:
`build_condition_groups_with_severity()` groups by `(state, condition, frequency)` and,
across all 18 real `(state, condition)` folders, **every resulting group has exactly 1
file** — zero groups with ≥2. In other words: **there is no true experimental
replication anywhere in this dataset** — no two files share the identical class,
severity, *and* target speed. Every recording, including within a single severity
folder, targets a distinct speed. This is the finding that resolves §2's caveat: the 230
"replicated" `(state, frequency)` groups from §3–4 are entirely explained by different
severities coincidentally sharing a target speed, never by repeated identical trials.

## 7. Implications for dataset split

## Split-design input for TASK 1.5.7

- **230 `(state, frequency)` groups span multiple severity conditions** (§3, §6) — these
  are **not** a leakage concern in the sense of "the same recording split across
  train/test": each file within such a group is still a physically distinct recording
  (different severity), so splitting them across train/val/test does **not** leak
  information about "the same signal" the way splitting windows *from the same file*
  would (blueprint section 8's actual leakage concern). TASK 1.5.7 does **not** need to
  treat `(state, frequency)` groups as an atomic split unit.
- **Zero true same-setup replicates exist** (§6): no file is a repeated trial of an
  identical (state, severity, speed) combination. This means TASK 1.5.7's per-file split
  (blueprint section 8: allocate whole recordings, not windows, to train/val/test) has
  no additional "which of these N identical-condition files go together" decision to
  make — every file already represents a distinct experimental point.
- **Rotation-frequency provenance for all 880 recordings is `filename`** (TASK 1.5.4) —
  uniformly, with zero `tachometer`/`metadata`/`unavailable` values. TASK 1.5.7 does not
  need to handle mixed-confidence rotation data or missing-rotation recordings for this
  dataset as currently inventoried.
- **No leakage risk was identified** beyond what blueprint section 8 already establishes
  (split by whole recording/file, never by window, to avoid within-file correlation
  leaking across splits) — this task's analysis surfaces no *additional* structural risk
  from condition-group overlap.
- **UNKNOWN_ROTATION groups: none** (§5) — no special handling required for missing
  rotation data in the split design.

No split logic is implemented here — this section is purely documented input for TASK
1.5.7, as required.

## 8. Audit validation

- Sum of all group file counts: **880**
- Total files in inventory: **880**
- Difference: **0**
- Consistency with TASK 1.5.4 class totals: `sum(class counts) = 49 + 333 + 197 + 301 =
  880 = total recordings` — unchanged from TASK 1.5.4, re-confirmed here since group
  totals are computed independently and must agree (verified:
  `test_real_inventory_groups_sum_to_inventory_total`).

## 9. Conclusion

- **AC1 — confirmed.** **230** condition groups `(state, frequency)` have ≥2 files,
  computed from the real 880-recording inventory (§3–4), not estimated.
- **AC2 — confirmed.** §7 documents exactly what TASK 1.5.7 needs: which groups are
  replicated, why that replication is a severity-sharing artifact rather than true
  duplication (§6), that no true same-setup replicate exists anywhere, that rotation
  provenance is uniform, and that no additional leakage risk beyond blueprint section
  8's existing per-file-split rule was found.
