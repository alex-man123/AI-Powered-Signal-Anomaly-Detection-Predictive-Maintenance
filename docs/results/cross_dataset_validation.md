# Cross-Dataset Validation — MAFAULDA → CWRU (TASK 13.1)

**Status: MEASURED.** The real CWRU Bearing Dataset (161 real `.mat` files, uploaded by the user under `data/external/cwru/`) was evaluated end-to-end against the real, standing MAFAULDA-trained production models, with no retraining, refitting, or recalibration on CWRU anywhere in the pipeline. Every number below comes from one real, executed run of `python -m scripts.run_cross_dataset_validation` (backend/), reproduced by `backend/tests/scripts/test_run_cross_dataset_validation.py`'s own real-dataset tests. Nothing here is a placeholder, an estimate, or a projection.

**Headline, stated plainly and without spin:** at MAFAULDA's own calibrated threshold, both models flag essentially **every** CWRU window — normal and faulty alike — as anomalous. This means the calibrated *decision* (NORMAL vs. ANOMALY) does **not** transfer to CWRU. The underlying **continuous** anomaly scores, however, are not random relative to the true label (ROC-AUC 0.61–0.68, above the 0.50 chance level) — see Section 5 for what this does and does not support.

---

## 1. Dataset

| | MAFAULDA (training) | CWRU (external evaluation) |
|---|---|---|
| Role | Primary — all models/artifacts in this project are trained on it | Secondary, cross-dataset validation only (docs/blueprint.md §4/§5/§7) |
| Present in this environment? | Yes (`data/raw/mafaulda/`, 880 real files) | **Yes** — `data/external/cwru/`, 161 real `.mat` files (uploaded 2026-09-11) |
| Format | CSV | `.mat` (MATLAB), read via `scipy.io.loadmat` |
| Sampling rate | 50,000 Hz (fixed, every file) | **Two real rates present**: 12,000 Hz (`12k_Drive_End_Bearing_Fault_Data/`, `12k_Fan_End_Bearing_Fault_Data/`, `Normal/`) and 48,000 Hz (`48k_Drive_End_Bearing_Fault_Data/`) |
| Real classes | 4: `normal`, `imbalance`, `horizontal-misalignment`, `vertical-misalignment` | 4: `normal`, `ball`, `inner-race`, `outer-race` |
| Channels per file | 8 | 2–3 (`DE`/`FE`, sometimes also `BA`) |

**Number of windows evaluated (real, measured):**

| Sampling-rate group | Total windows | Normal | Anomaly (ball+inner-race+outer-race) |
|---|---|---|---|
| 12,000 Hz | 79,112 | 8,512 | 70,600 |
| 48,000 Hz | 86,368 | **0** | 86,368 |

Windowing: `window_size=1024` samples, `overlap=0.5` — reused unchanged from MAFAULDA's own configuration (see Section 6 for what this means physically at each rate). Feature extraction: the same 15 DSP features as MAFAULDA (`app.features.registry`), Welch PSD `nperseg=256`/`noverlap=128`, computed at CWRU's own real sampling rate for each group (never MAFAULDA's 50 kHz).

**A real, disclosed property of this particular download, not a defect:** the 48 kHz subset contains **only** fault recordings (`48k_Drive_End_Bearing_Fault_Data/`) — no `Normal/` file in this mirror was collected at 48 kHz. The 48 kHz evaluation below is therefore a single-class (100% anomaly) test set.

**`Normal/`'s real sampling rate — the one inferential step in this report, made explicit:** the real, downloaded `Normal/` files carry no `<N>k_...` directory segment (unlike every fault file). CWRU's own "Apparatus and Procedures" page (fetched directly while implementing this task) states: *"Digital data was collected at 12,000 samples per second, and data was also collected at 48,000 samples per second for drive end bearing faults."* Read literally, 48 kHz is documented as an **additional** rate specific to drive-end **fault** recordings — the Normal baseline set is not a drive-end fault, so by this wording it uses the general, base 12,000 Hz rate. This reading was confirmed with the user directly (2026-09-11) before running the evaluation below; `app.datasets.cwru_loader` never assumes it silently — it raises `CWRULoaderError` for any `Normal/` file unless `sampling_rate_hz`/`normal_sampling_rate_hz=12000.0` is passed explicitly.

**Structurally important caveat for reading every result below:** MAFAULDA's fault classes (imbalance, horizontal/vertical misalignment) and CWRU's (ball, inner-race, outer-race bearing defects) **do not overlap**. A model trained only on MAFAULDA's fault signatures is being asked, here, to generalize to an entirely different family of mechanical faults on different hardware/sensors — not merely the same fault type recorded differently.

---

## 2. What was implemented

### 2.1 `backend/app/datasets/cwru_loader.py`

- Reads `.mat` files via `scipy.io.loadmat`. Directory contract **verified directly against the real, downloaded files** (not merely public documentation): `<12k|48k>_<Drive|Fan>_End_Bearing_Fault_Data/<B|IR|OR>/<diameter>/[<orientation>/]<file>.mat` for faults, bare `Normal/<file>.mat` for the normal baseline set. Label directories (`B`/`IR`/`OR`/`Normal`) and the sampling-rate prefix (`<N>k_...`) are parsed generically (regex-based), robust to the real tree's inconsistent nesting depth (e.g. 0.014" outer-race faults have no `@N` orientation subdirectory in this download, while 0.007"/0.021" do).
- `.mat` variable naming (`X<fileid>_DE_time`/`_FE_time`/`_BA_time`) confirmed directly against real files while writing this module (e.g. `Normal/97_Normal_0.mat` → `X097_DE_time`, `X097_FE_time`, `X097RPM`).
- Never guesses: an unrecognized label directory, an unresolvable sampling rate (including every `Normal/` file without an explicit override — see Section 1), or a `.mat` file with no matching time-series variable all raise `CWRULoaderError` with a specific message.

### 2.2 `backend/scripts/run_cross_dataset_validation.py`

1. Loads the real, standing `models/isolation_forest_v1.pkl` **or** `models/autoencoder_v1.pt` plus its TASK 6.5 metadata sidecar (`load_model_artifact`) — never retrained.
2. Loads the real, standing `models/scaler_v1.pkl` — used only via `.transform()`, never refit.
3. Recomputes score normalization (`app.ml.scoring.calibrate`) from the **real MAFAULDA validation split only** and asserts the result reproduces the model's already-persisted `threshold_value` — verified to hold exactly (see Section 3) before any CWRU file is scored.
4. Loads every real CWRU recording, **groups them by their own real sampling rate** (12 kHz vs. 48 kHz — this download spans both under one root), and evaluates each group separately so no window computed at one frequency resolution is ever mixed with another's in the same evaluation.
5. Extracts the same 15 DSP features, scores with the model/scaler/calibration from steps 1–3, calls `app.ml.evaluation.evaluate` (TASK 8.1, unmodified). CWRU's 4 real classes are reduced to `0=normal`/`1=anomaly` for the metric computation; the real fault-type labels are preserved separately (`cwru_labels_present`).

---

## 3. Methodology — verified guarantees

- **Model trained on MAFAULDA, never retrained:** `models/isolation_forest_v1.pkl`/`models/autoencoder_v1.pt`, loaded unchanged via `load_model_artifact`. Verified structurally (this script never imports `app.ml.training`/`train_isolation_forest`/`train_autoencoder`) and by spy-based tests (`IsolationForest.fit`/`StandardScaler.fit`/`fit_transform` raise if called during a full pipeline run).
- **Scaler never refit:** `models/scaler_v1.pkl`, fit exclusively on MAFAULDA train data, used only via `apply_scaler`/`predict` (transform only).
- **Threshold never recalibrated on CWRU:** the value used is exactly the one already persisted in each model's `ModelArtifactMetadata` — **Isolation Forest: 0.510328048907322**, **Autoencoder: 0.016910125709661**. The intermediate min/max normalization (not persisted by TASK 6.5's schema) is recomputed from the **real MAFAULDA validation split only**, and the resulting threshold is asserted to reproduce the persisted value before any CWRU data is touched — **verified to hold exactly (within `1e-9`) for both models in the run these results come from.** CWRU never participates in this step.
- **Sampling rate:** CWRU's own real rate (12 kHz or 48 kHz, per group) is passed to `extract_feature_matrix`, never MAFAULDA's 50 kHz.
- **Labels:** CWRU's 4 real classes reduced to the same binary target MAFAULDA's own experiments use; fault-type detail preserved (`cwru_labels_present`).

---

## 4. Results (real, measured)

### 12,000 Hz group (79,112 windows: 8,512 normal / 70,600 anomaly)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion Matrix `[[TN,FP],[FN,TP]]` | FPR | FNR | Inference time (s) |
|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest | 0.8924 | 1.0000 | 0.9431 | **0.6124** | 0.9201 | `[[0, 8512], [0, 70600]]` | **1.0000** | 0.0000 | 0.1094 |
| Autoencoder | 0.8924 | 1.0000 | 0.9431 | **0.6752** | 0.9273 | `[[0, 8512], [0, 70600]]` | **1.0000** | 0.0000 | 0.0121 |

### 48,000 Hz group (86,368 windows: 0 normal / 86,368 anomaly — single-class, see Section 1)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion Matrix | FPR | FNR | Inference time (s) |
|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest | 1.0000 | 1.0000 | 1.0000 | `NaN` (single class — sklearn's own convention, TASK 8.1) | 1.0000 (real value for an all-positive set — TASK 8.1's documented convention, not fabricated) | `[[0, 0], [0, 86368]]` | `NaN` (0 actual negatives) | 0.0000 | 0.1284 |
| Autoencoder | 1.0000 | 1.0000 | 1.0000 | `NaN` | 1.0000 | `[[0, 0], [0, 86368]]` | `NaN` | 0.0000 | 0.0147 |

These `NaN`/trivial-1.0 values are `app.ml.evaluation.evaluate`'s own documented, deliberate behavior for a single-class test set (TASK 8.1) — not an error, and not evidence of "perfect" performance: with zero real normal examples in this rate group, there is nothing to compute a false-positive rate or a meaningful precision against.

---

## 5. Interpretation

**At the calibrated decision threshold, the models do not discriminate CWRU normal from CWRU faulty at all — everything is classified anomalous.** FPR = 1.0 at 12 kHz means every one of the 8,512 real CWRU normal windows was flagged as anomalous by both models, exactly like every real fault window. This is not "the model is slightly miscalibrated" — it is a complete failure of the calibrated *decision* to transfer.

**The continuous scores are not uninformative, however.** ROC-AUC (which does not depend on the threshold) is 0.6124 (Isolation Forest) and 0.6752 (Autoencoder) — above the 0.50 chance level, meaning the raw anomaly scores do rank at least some CWRU-normal windows below at least some CWRU-fault windows more often than not. This is modest, not strong, discriminative signal — nowhere near the levels this project's MAFAULDA-only results report (e.g. `docs/results/central_experiment_report.md`).

**A plausible (not proven) explanation this report can offer, without overclaiming:** MAFAULDA's validation-derived min/max normalization (`fit_score_normalizer`) defines a `[0,1]` scale calibrated to MAFAULDA's own score range. If CWRU's raw scores — computed from a structurally different machine, sensors, and (for imbalance/misalignment vs. bearing faults) a different fault physics — fall mostly or entirely *outside* that MAFAULDA-calibrated range, `normalize_scores`' clipping would push most/all CWRU scores to `1.0`, which is exactly consistent with FPR=1.0 at 12 kHz. **This report does not have the additional ablation (e.g. inspecting the raw, pre-normalization score distributions for CWRU vs. MAFAULDA-validation) needed to confirm this is the actual mechanism**, so it is stated as a plausible reading of the pattern, not a demonstrated cause.

**What this result does and does not say:**
- It does **not** show that Isolation Forest or Autoencoder is "better" — both saturate to the same degenerate all-anomalous decision at 12 kHz, and both produce a trivial 100%-positive result at 48 kHz (no normal examples exist there to distinguish them). The Autoencoder's modestly higher ROC-AUC (0.675 vs. 0.612) is the only real, measured difference between them here.
- It does **not** show that the underlying DSP feature set or either model is fundamentally incapable of separating normal from anomalous vibration signals in general — MAFAULDA-only results (elsewhere in `docs/results/`) show strong in-distribution performance from the same features/models.
- It **does** show that this project's current pipeline — specifically, the MAFAULDA-validation-calibrated threshold and score normalization — does not generalize its *decision* to an unrelated dataset/machine/fault-family without re-calibration. Whether re-calibrating normalization (still never fitting a new model) on CWRU's own distribution would recover useful discrimination is a natural next question this task's scope (evaluate the existing artifacts as-is, no recalibration on CWRU) does not answer.

---

## 6. Cross-dataset analysis

- **No shared fault type.** MAFAULDA's imbalance/misalignment faults and CWRU's ball/inner-race/outer-race bearing faults do not overlap mechanically — see Section 1. The near-chance-to-modest ROC-AUC is consistent with the models having learned MAFAULDA-specific fault signatures that only partially transfer to a different fault family, but this report cannot isolate that from the normalization-range effect discussed in Section 5 without further work.
- **Sampling rate / window duration.** `window_size=1024` samples is reused unchanged from MAFAULDA. At CWRU's 12 kHz this is ~85.3 ms per window (vs. ~20.5 ms for MAFAULDA at 50 kHz, and ~21.3 ms for CWRU's own 48 kHz group) — a real, disclosed ~4× difference in physical time span between the 12 kHz CWRU windows and MAFAULDA's own, a consequence of reusing a sample-count-based window size rather than a time-based one (out of this task's scope to redesign).
- **Channel/sensor differences.** MAFAULDA uses a fixed channel (0) of 8; CWRU's real files expose 2–3 differently-positioned accelerometers (DE/FE/BA). This evaluation used every real channel present in each CWRU file (each becomes its own window set) — no attempt was made to select "the most comparable" CWRU channel to MAFAULDA's channel 0, since no such correspondence is established by this task.
- **Class imbalance differs sharply between groups.** The 12 kHz group is heavily anomaly-weighted (89.2% of windows); the 48 kHz group is 100% anomaly. Precision/F1 in Section 4 must be read with this in mind — a trivial "always predict anomalous" classifier would already score precision=0.892/recall=1.0/F1=0.943 at 12 kHz by construction, which is exactly what both models' *decisions* reduce to here (FPR=1.0). The metric that actually reflects discrimination in this report is ROC-AUC, not precision/recall/F1.

---

## 7. Testing

- `backend/tests/dataset/test_cwru_loader.py` (15 tests): directory/label/sampling-rate parsing against synthetic `.mat` fixtures using the real, verified directory-naming convention; `Normal/`'s real override requirement; plus 3 tests against the real, present dataset (161 real files discovered; all 4 real labels and both real sampling rates loaded with the confirmed `normal_sampling_rate_hz=12000.0`; `Normal/` still raises without an explicit override).
- `backend/tests/scripts/test_run_cross_dataset_validation.py`: the module never references a training entry point; `IsolationForest.fit`/`StandardScaler.fit`/`fit_transform` spied and confirmed never called during a full pipeline run; calibration confirmed computed exactly once, from MAFAULDA-validation-sized data; a tampered-threshold scenario confirmed to raise rather than silently proceed; CWRU's real fault-type labels confirmed preserved; plus tests against the real dataset confirming both real sampling-rate groups are produced and neither model is retrained/recalibrated on CWRU in the real run.
- Full backend suite: all pre-existing tests still pass after this task's changes.

**Definition of Done: complete.** The CWRU loader is implemented and verified against the real, uploaded dataset; the pipeline processes real CWRU data end-to-end; both real MAFAULDA models were evaluated on it without any retraining, refitting, or recalibration on CWRU; the real, measured results are reported above, honestly, including the fact that the calibrated decision does not generalize — this is reported as the actual finding, not treated as a failure to hide.
