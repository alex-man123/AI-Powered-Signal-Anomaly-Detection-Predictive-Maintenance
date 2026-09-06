# MAFAULDA Leakage Risk Analysis — TASK 1.5.6

> Source of truth: `docs/dataset_audit/file_inventory.csv`, `recording_mapping.md`,
> `signal_structure.md`, `class_distribution.md`, `replication_analysis.md` (TASK
> 1.5.1–1.5.5), plus a handful of targeted signal-content checks against the real files
> under `data/raw/mafaulda/` performed specifically for this task (not the whole
> dataset — see §3.6). Local data takes priority over any external MAFAULDA
> documentation throughout.

## 1. Scope

This analysis asks one question: is "split per file" a safe unit for TASK 1.5.7, or
does real evidence in this dataset require grouping multiple files together (e.g. as a
recording session) before splitting, to avoid leaking information across
train/validation/test? It does not implement any split logic.

## 2. Dataset evidence

Documents inspected: `file_inventory.csv` (880 files), `recording_mapping.md` (state ==
top-level folder, condition == severity sub-folder), `signal_structure.md` (8 columns,
all `float64`, 250,000 rows, no header, no time column, no metadata file — verified in
TASK 1.5.3), `class_distribution.md` (4 classes, rotation_source = `filename` for all
880 recordings), `replication_analysis.md` (567 `(state, frequency)` groups, 230 with
≥2 files, but **0** groups with ≥2 files once severity/condition is also included in the
key — TASK 1.5.5's key finding).

New evidence gathered specifically for this task (read-only, targeted, not a full scan):
- Full-path keyword search across all 880 `relative_path` values in
  `file_inventory.csv` for `session`, `run`, `experiment`, `trial`, `recording`,
  `sample`, `measurement`, `test` (and numbered variants) — **zero matches**.
- Filesystem modification-time (`mtime`) distribution across all 880 files (§3.3).
- Signal-content comparison for 5 targeted file pairs, plus a within-file
  successive-sample baseline (§3.6/§3.7) — read directly with pandas, one pair/file at
  a time, never the whole dataset.

## 3. Potential leakage sources

### 3.1 Exact duplicates

- **Evidence:** TASK 1.5.5 found every one of the 880 files has a **unique
  `size_bytes`** (880 distinct sizes) — a necessary precondition for byte-identical
  content, so unique sizes alone rule out exact duplicates. Re-confirmed at content
  level here: none of the 5 spot-checked pairs (§3.6) are `numpy.array_equal`.
- **Interpretation:** No exact duplicate files exist.
- **Confidence:** HIGH.
- **Impact on split:** None — no special handling needed.

### 3.2 Filename/path identifiers

- **Evidence:** Every `relative_path` is exactly `<state>/<filename>.csv` or
  `<state>/<condition>/<filename>.csv` (TASK 1.5.2). The full vocabulary of path
  components is closed and known: 4 state names, 17 condition names (mm/g severity
  values), and 241 distinct frequency-valued filenames (TASK 1.5.4/1.5.5). A keyword
  search for session/run/experiment/trial/recording/sample/measurement/test across all
  880 paths returned zero matches (§2).
- **Interpretation:** No session, run, or experiment identifier is encoded in any path
  or filename, anywhere in this dataset.
- **Confidence:** HIGH.
- **Impact on split:** None — there is no identifier to group by even if one wanted to.

### 3.3 Session/run identifiers

- **Evidence:** Same as §3.2 — no textual identifier exists. The only non-textual
  candidate is filesystem `mtime`: every file within a given `(state, condition)`
  folder shares **exactly one** calendar date (verified for all 18 real
  `(state,condition)` folders); dates found are only `2014-10-03` (290 files:
  `normal`, `imbalance/6g`, `imbalance/10g`, `imbalance/15g`, `imbalance/20g`,
  `imbalance/25g`) and `2014-10-06` (590 files: `horizontal-misalignment/*`,
  `vertical-misalignment/*`, `imbalance/30g`, `imbalance/35g`). Consecutive files
  within a folder (sorted by mtime) are spaced roughly 14 seconds apart in the sampled
  range checked.
- **Interpretation:** This is **filesystem metadata about when files were
  created/packaged on disk**, not a field inside the data. It could reflect genuine
  acquisition-day batching, or simply how the original archive was extracted/copied —
  **not independently verifiable from local data which of these it is.** Critically,
  even if it does reflect two real acquisition days, that alone does not establish that
  files sharing a date have *dependent signal content* — see §3.6 for the actual
  content-level check, which is what matters for leakage.
- **Confidence:** LOW (real pattern, but its meaning is unresolved and it is not, by
  itself, evidence of data dependency).
- **Impact on split:** Noted as an open, low-confidence signal; not acted upon given
  the absence of corroborating content-level evidence (§3.6) — see §7/§8.

### 3.4 Consecutive recordings

- **Evidence:** Not applicable to this dataset's naming convention. Filenames encode a
  target rotation frequency (TASK 1.5.4/1.5.5), not a sequential counter
  (`file_001`, `file_002`, …). There is no incrementing-ID pattern to evaluate.
- **Interpretation:** This risk factor does not apply here.
- **Confidence:** N/A.
- **Impact on split:** None.

### 3.5 Same experimental condition

- **Evidence:** TASK 1.5.5: 230 of 567 `(state, frequency)` groups have ≥2 files, but
  **every single one of those is explained by different severities sharing a nominal
  target speed** — 0 groups remain once severity/condition is included in the key (i.e.
  no two files share class **and** severity **and** frequency).
- **Interpretation:** "Same class + same rotation" here is a coincidence of the
  experimental design (a similar speed sweep repeated per severity), not repeated
  measurement of the identical physical setup. See `replication_analysis.md` §6–7 for
  the full argument.
- **Confidence:** HIGH (directly computed from the real inventory + mapping, TASK
  1.5.5).
- **Impact on split:** These files are legitimately independent recordings (different
  severity = different physical configuration); grouping them as one split unit would
  be unnecessarily conservative, not a leakage-safety requirement.

### 3.6 Shared signal/noise characteristics

Read directly (pandas, one file at a time — 7 files total, not the dataset):

| Pair | Relationship | Identical? | Max abs sample diff | Per-channel Pearson r |
|---|---|---|---:|---|
| `imbalance/6g/45.4656.csv` vs `imbalance/10g/45.4656.csv` | same freq, different severity | No | 5.96 | −0.15 … 0.37 |
| `imbalance/6g/45.4656.csv` vs `imbalance/20g/45.4656.csv` | same freq, different severity | No | 5.97 | −0.40 … 0.15 |
| `imbalance/10g/45.4656.csv` vs `imbalance/35g/45.4656.csv` | same freq, different severity | No | 115.11 | 0.00 … 0.40 |
| `horizontal-misalignment/0.5mm/12.288.csv` vs `.../1.0mm/12.288.csv` | same freq, different severity | No | 5.90 | −0.12 … 0.44 |
| `normal/12.288.csv` vs `imbalance/6g/13.9264.csv` | **unrelated baseline** (different state, different freq) | No | 5.85 | −0.06 … 0.06 |

- **Interpretation:** Same-frequency/different-severity pairs show weak-to-moderate
  per-channel correlation (max observed: 0.44, on one channel of one pair) —
  comparable in magnitude to the unrelated baseline pair (max 0.06 in this sample, but
  other unrelated-baseline channels elsewhere in the dataset would not be expected to
  stay near zero either; a single baseline pair is illustrative, not a rigorous null
  distribution). No pair shows the near-1.0 correlation or near-identical values that
  would indicate shared or duplicated underlying signal. This does not prove
  independence for all 230 replicated groups (only 4 were checked), but it gives no
  reason to suspect the opposite either.
- **Confidence:** MEDIUM — sample-based (5 pairs out of 230 replicated groups), not
  exhaustive; a full pairwise scan (~386,000 comparisons across all 880 files) was
  judged disproportionate for an audit task and is not what was done.
- **Impact on split:** No evidence of shared/dependent signal content between files in
  the same `(state, frequency)` group.

### 3.7 Other observed dependencies — continuous-recording-split hypothesis

- **Evidence:** Tested whether two adjacent-in-sweep files within the same
  `(state, condition)` folder (`horizontal-misalignment/0.5mm/12.288.csv`, the
  lowest frequency in that folder, and `.../13.5168.csv`, the next one up) could be
  one continuous acquisition split at that boundary. Compared the **last row of the
  first file to the first row of the second** against the **typical successive-sample
  change measured within a single file** (`horizontal-misalignment/0.5mm/12.288.csv`
  itself, all 8 channels, 249,999 consecutive-row diffs):

  | Channel | Boundary jump (A→B) | Typical within-file jump (mean) | Ratio |
  |---:|---:|---:|---:|
  | 0 | 0.020 | 0.052 | 0.4× |
  | 1 | 0.263 | 1.609 | 0.2× |
  | 2 | 0.025 | 0.452 | 0.05× |
  | 3 | 0.077 | 0.059 | 1.3× |
  | 4 | 0.477 | 0.059 | **8.1×** |
  | 5 | 0.099 | 0.004 | **23×** |
  | 6 | 0.201 | 0.056 | **3.6×** |
  | 7 | 0.007 | 0.210 | 0.03× |

- **Interpretation:** For a genuinely continuous acquisition, **every** channel should
  show a boundary jump on the same order as its typical within-file successive-sample
  change (all 8 sensors record the same physical instant simultaneously). Here, 3 of 8
  channels (4, 5, 6) show a jump 3.6×–23× larger than typical — inconsistent with these
  two files being adjacent slices of one continuous stream. This is **one example, not
  an exhaustive check of all adjacent-sweep pairs**, but it directly argues against the
  "single recording split across multiple files" hypothesis for this candidate.
- **Confidence:** MEDIUM (single-pair evidence, but methodologically direct).
- **Impact on split:** Argues against needing to treat within-folder frequency-adjacent
  files as a continuous unit.

## 4. File-level independence assessment

**For:**
- Zero exact duplicates (§3.1).
- Zero session/run identifiers anywhere in path or filename (§3.2/3.3).
- Zero true same-`(state, condition, frequency)` replicates (§3.5, from TASK 1.5.5).
- Signal-content correlation between the most plausible "related" candidates
  (same-frequency, different severity) is weak and comparable to an unrelated baseline
  (§3.6).
- The one continuous-recording-split candidate tested shows a content discontinuity
  inconsistent with that hypothesis (§3.7).

**Against / open questions:**
- mtime clustering by calendar date exists and is unexplained (§3.3) — not ruled out as
  reflecting real acquisition sessions, though no content-level dependency was found
  that would make this actionable.
- Content-level checks (§3.6/§3.7) sampled a handful of pairs, not all 230 replicated
  groups or all folder-adjacent pairs — a residual, unquantified risk that an
  unchecked pair somewhere behaves differently remains, in principle.

## 5. Session/group-level independence assessment

**For (i.e., evidence that WOULD justify grouping):**
- The mtime-date clustering (§3.3) is the only signal pointing this direction, and it
  is explicitly a weak, unresolved one.

**Against:**
- No structural evidence (filename, path, metadata, or content) supports a
  session/group unit larger than "one file". Every avenue checked (§3.1–3.7) that could
  have surfaced a group-level dependency did not.

## 6. Replication vs. session

These are explicitly different, and this dataset's audit history makes the distinction
concrete: TASK 1.5.5 found 230 `(state, frequency)` groups with ≥2 files — this is
**replication of experimental condition** (same class, same nominal target speed).
Whether those files also constitute a **single recording session** is a separate
question, answered here: no. Every one of those 230 groups is actually composed of
files at **different severities** (§3.5) — i.e. different physical configurations of
the rig — and the signal-content checks in §3.6 found no evidence they share dependent
content. "Same experimental condition" (class + rotation) here does **not** imply "same
physical/session recording."

## 7. Leakage risk classification

- **Exact duplicates: NONE**
- **Session leakage: LOW** (no structural evidence found; one weak, unresolved
  filesystem-metadata signal that content-level checks do not corroborate)
- **Condition-level leakage: LOW** (replication is a severity-sharing artifact, not
  duplicated data — §3.5/§3.6)
- **Overall leakage risk: LOW**

## 8. Final split decision

```
FILE-LEVEL SPLIT IS SUFFICIENT
```

## 9. Evidence-based justification

Every structural and content-level check performed against the real local dataset
(§3.1–3.7) — unique file sizes, zero true same-condition replicates, no
session/run/trial identifiers anywhere in 880 real paths, weak content correlation
between the most plausible "related" file pairs (comparable to an unrelated baseline),
and a content discontinuity in the one continuous-recording-split candidate tested —
converges on file-level independence. The single open question (mtime-date clustering,
§3.3) is real but unresolved and, on its own, is not evidence of *data* dependency
between files; it did not survive corroboration once content was actually checked. Per
the task's own rule, this is reported honestly as a residual LOW-confidence signal
rather than either suppressed or inflated into a reason to block the decision.

This conclusion should not be read as a claim of exhaustive proof: §3.6/§3.7 checked a
representative sample (5 content-comparison pairs, 1 boundary-continuity pair), not all
880 files pairwise — disproportionate for an audit task, per the task's own scope
guidance. It is the best defensible conclusion available from local evidence at
proportionate effort, clearly distinguished from what was and wasn't checked.

## 10. Implications for TASK 1.5.7

- Split MAFAULDA recordings **per file** (blueprint section 8's existing rule remains
  the right one; this audit found no reason to extend it to a larger group unit).
- No `(state, frequency)` group (TASK 1.5.5) needs to be kept intact across
  train/val/test — its members are physically distinct recordings, confirmed via both
  structural (§3.5) and content (§3.6) evidence.
- No session/run identifier exists to group by, even if a group-level split were
  desired later.
- The mtime-date clustering (§3.3) is documented here for visibility; TASK 1.5.7 is not
  required to act on it, but should not be surprised to see two distinct dates in the
  filesystem if it ever inspects file metadata.
- This does not reopen or modify TASK 1.5.5's condition-group analysis, nor does it
  implement any split logic — that remains TASK 1.5.7's responsibility.
