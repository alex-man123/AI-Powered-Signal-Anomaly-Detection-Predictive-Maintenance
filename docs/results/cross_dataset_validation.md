# Cross-Dataset Validation — MAFAULDA → CWRU (TASK 13.1)

**Status: BLOCKED — the real CWRU Bearing Dataset is not present in this repository/environment.** No cross-dataset metric in this report is a placeholder or an estimate; every "NOT YET MEASURED" below means exactly that: the real evaluation has not been run, because the real input data it needs does not exist here yet. Nothing in this document should be read as evidence of good, moderate, or poor generalization — that question is open until the dataset is uploaded and `scripts/run_cross_dataset_validation.py` is actually run against it.

Source of what IS real and already verified in this report: `backend/app/datasets/cwru_loader.py` (the loader) and `backend/scripts/run_cross_dataset_validation.py` (the evaluation pipeline), both written and tested (`backend/tests/dataset/test_cwru_loader.py`, `backend/tests/scripts/test_run_cross_dataset_validation.py`) as part of this task, plus one real, executed consistency check (Section 3) against the real, standing MAFAULDA production models.

---

## 1. Dataset

| | MAFAULDA (training) | CWRU (external evaluation) |
|---|---|---|
| Role | Primary — all 4 real models/artifacts in this project are trained on it | Secondary, cross-dataset validation only (docs/blueprint.md §4/§5/§7) |
| Present in this environment? | Yes (`data/raw/mafaulda/`, 880 real files) | **No** — `data/external/cwru/` does not exist; only the tracked `.gitkeep` placeholder is present |
| Format | CSV | `.mat` (MATLAB) |
| Sampling rate | 50,000 Hz (fixed, every file) | 12,000 Hz or 48,000 Hz depending on subset (per CWRU's own public documentation — docs/blueprint.md's CWRU section; not independently re-verified here since no real file is available to check) |
| Real classes | 4: `normal`, `imbalance`, `horizontal-misalignment`, `vertical-misalignment` | 4 (per public documentation): `normal`, `ball`, `inner-race`, `outer-race` |
| Channels per file | 8 | 1–3 (`DE`/`FE`/`BA` accelerometers) |

**Number of samples/windows evaluated: NOT YET MEASURED — 0.** No CWRU file has been loaded, windowed, or scored, because none exists in this environment.

**Sampling rate actually used for CWRU: NOT YET MEASURED.** `app.datasets.cwru_loader` resolves it per-file from the uploaded directory layout (`12khz/`/`48khz/` segment) — see Section 2 for the exact contract — but no real value has been read from a real file yet.

**Important, disclosed difference between the two datasets that any future result must be read against:** MAFAULDA is a multi-channel (8), single-sampling-rate (50 kHz), single-machine-type dataset with imbalance/misalignment faults; CWRU is a 1–3-channel, dual-sampling-rate (12/48 kHz), single-machine-type dataset with bearing (ball/race) faults. **The two datasets do not share a single real fault type** — MAFAULDA has no bearing faults, CWRU has no imbalance/misalignment faults. Any cross-dataset "anomaly" comparison is therefore a comparison of "does a model trained to recognize one family of mechanical faults flag an entirely different family of mechanical faults as anomalous", not the same fault type observed under a different sensor setup. This is a structurally important caveat for interpreting any future result (Section 5), not a defect in the implementation.

---

## 2. What was implemented

### 2.1 `backend/app/datasets/cwru_loader.py`

- Reads `.mat` files via `scipy.io.loadmat` (already a project dependency — `pyproject.toml`).
- Requires the uploaded CWRU tree to follow this layout: `data/external/cwru/<12khz|48khz>/<normal|ball|inner-race|outer-race>/.../<file>.mat`. **This layout is a contract this loader imposes, not a fact discovered by auditing real CWRU files** (unlike MAFAULDA's own directory convention, which TASK 1.5.1/1.5.2 derived from an actual audit of the real downloaded dataset) — no real CWRU files exist in this environment to audit. It is based on CWRU's publicly documented format (docs/blueprint.md's CWRU section) and is explicit and inspectable so it can be corrected the moment real files are available, if the real distribution turns out to differ.
- Within a `.mat` file, matches any variable named `<fileid>_<DE|FE|BA>_time` (a generic regex, not one hardcoded key) — CWRU's publicly documented time-series variable naming convention.
- Never guesses: a file outside a recognized label directory, a missing sampling-rate directory segment (with no explicit override), or a `.mat` file with no matching time-series variable all raise `CWRULoaderError` with a specific, actionable message — never a silently wrong label/rate/channel.
- `discover_recordings`/`load_all_recordings` raise `CWRULoaderError` immediately when `data/external/cwru/` does not exist — this is the actual, current, real behavior in this environment (verified by test, not simulated).

### 2.2 `backend/scripts/run_cross_dataset_validation.py`

End-to-end pipeline, structured to mirror `scripts/run_experiment_b.py`/`run_experiment_c.py`'s own "load the real standing artifact, never train/fit anything" pattern exactly (see Section 3 for what "reuse" means precisely here, and Section 4 for the guarantees this provides):

1. Loads the real, standing `models/isolation_forest_v1.pkl` **or** `models/autoencoder_v1.pt` plus its real TASK 6.5 metadata sidecar (`app.ml.model_artifact.load_model_artifact`) — never re-trained.
2. Loads the real, standing `models/scaler_v1.pkl` the metadata names (`app.ml.inference.load_scaler`) — used only via `.transform()` (`app.ml.scaling.apply_scaler` / `app.ml.inference.predict`), never refit.
3. Recomputes score normalization (`app.ml.scoring.calibrate`) from the **real MAFAULDA validation split only** (`scripts.run_experiment_a.VALIDATION_FILES`) — the same files/method/percentile Experiments A/B/C already use — and asserts the result reproduces the model's **already-persisted** `threshold_value` before proceeding. CWRU never participates in this step.
4. Loads real CWRU recordings (`app.datasets.cwru_loader.load_all_recordings`), windows them (same `window_size`/`overlap` as MAFAULDA, CWRU's own real sampling rate for the Welch PSD), extracts the same 15 DSP features (`app.features.registry`/`app.features.extractor`, unmodified), scores them with the model/scaler/calibration from steps 1–3, and calls `app.ml.evaluation.evaluate` (TASK 8.1, unmodified) — CWRU's 4 real classes reduced to the same `0=normal`/`1=anomaly` binary target MAFAULDA's own experiments already use, fault-type detail preserved separately (`cwru_labels_present`).

---

## 3. Methodology — what is real and already verified vs. what requires the dataset

**Already real and verified** (no CWRU data needed for this part):

- Running `_load_calibrated_model` against the real, standing `models/isolation_forest_v1.pkl` + `models/scaler_v1.pkl` and the real MAFAULDA validation files reproduces the exact threshold already persisted in `models/isolation_forest_v1.json` (**0.510328048907322**) to within `1e-9` — verified directly by executing the script's own code path in this environment (not assumed). The equivalent check for the Autoencoder (persisted threshold **0.016910125709661**, `models/autoencoder_v1.json`) is exercised by `backend/tests/scripts/test_run_cross_dataset_validation.py`.
- Model trained on MAFAULDA: **yes** — `models/isolation_forest_v1.pkl`/`models/autoencoder_v1.pt`, unchanged, loaded via `load_model_artifact`.
- Feature extraction used on CWRU (once real files exist): identical to MAFAULDA's own — same 15 features, same order, same `nperseg=256`/`noverlap=128`; only `fs` differs (CWRU's own real sampling rate, not MAFAULDA's 50 kHz).
- Scaler used: `models/scaler_v1.pkl`, fit exclusively on MAFAULDA train data — never refit on CWRU (enforced structurally: the script never imports `StandardScaler.fit`/`fit_transform`, and a spy-based test confirms it is never called during a full pipeline run).
- Threshold used: the value already persisted in the model's own `ModelArtifactMetadata` sidecar — never recalibrated on CWRU (enforced structurally: calibration is computed and asserted-consistent *before* any CWRU file is even loaded).
- **No training/refit/fine-tuning/recalibration on CWRU anywhere**: confirmed by (a) the module never importing `app.ml.training`/`train_isolation_forest`/`train_autoencoder`, (b) a spy on `IsolationForest.fit` and `StandardScaler.fit`/`fit_transform` raising if called during a full pipeline run.

**Requires the real dataset (NOT YET MEASURED)**:

- Any CWRU feature value, any CWRU score, any CWRU-based metric.
- Confirmation that the directory-layout contract in Section 2.1 actually matches the real uploaded files (it cannot be checked without them).

---

## 4. Results

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion Matrix | FPR | FNR | Inference time |
|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest (`isolation_forest_v1`, threshold=0.510328) | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED |
| Autoencoder (`autoencoder_v1`, threshold=0.016910) | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED |

**Why:** `scripts/run_cross_dataset_validation.py::run_cross_dataset_validation()` raises `CWRULoaderError` at the CWRU-loading step (verified directly, `backend/tests/scripts/test_run_cross_dataset_validation.py::test_run_cross_dataset_validation_raises_when_cwru_dataset_is_not_present`) because `data/external/cwru/` does not exist. No number in the table above has been computed, estimated, or approximated by any other means.

---

## 5. Interpretation

**Not applicable yet.** No real generalization result exists to interpret. In particular, this report does **not** claim:
- that the models generalize well, moderately, or poorly to CWRU;
- that Isolation Forest or Autoencoder is "better" for cross-dataset use;
- any expected direction for the result.

Once real CWRU data is uploaded under `data/external/cwru/` following the layout in Section 2.1 and `python -m scripts.run_cross_dataset_validation` is run from `backend/`, this section must be rewritten using the real, printed `metrics` dict for both models — replacing every `NOT YET MEASURED` cell in Section 4, not adding to them.

---

## 6. Cross-dataset analysis (methodological, not results-based)

Since no real CWRU measurement exists yet, this section only lays out real, disclosed structural differences a future result must be read against — it does not speculate about what the result will be.

- **No shared fault type.** As noted in Section 1, MAFAULDA's fault classes (imbalance, horizontal/vertical misalignment) and CWRU's (ball, inner-race, outer-race bearing defects) do not overlap. A model trained only to recognize MAFAULDA's fault signatures being evaluated on CWRU is being asked to generalize across mechanically different anomaly types, not merely across a different sensor/dataset for the *same* fault. A future low score on CWRU could indicate the model is overfit to MAFAULDA's specific fault signatures, that bearing faults produce a genuinely different statistical footprint in these 15 features than imbalance/misalignment faults, or both — the result alone will not distinguish between these without further analysis.
- **Sampling rate mismatch.** MAFAULDA is fixed at 50 kHz; CWRU is 12 kHz or 48 kHz depending on subset. The frequency-domain features (`dominant_frequency`, `spectral_centroid`, `spectral_bandwidth`, `spectral_energy`, `spectral_entropy`, `band_energy_10_100hz`) are computed via Welch PSD at each dataset's own real sampling rate — this keeps each feature physically meaningful in Hz for its own dataset, but the underlying frequency **resolution** (`fs`/`nperseg`) differs between a 50 kHz and a 12/48 kHz window of the same sample count. A result could indicate genuine model behavior, or could partly reflect this resolution difference — this report does not have grounds to attribute a future result to one cause over the other without further ablation.
- **Window duration mismatch.** `window_size=1024` samples is reused unchanged from MAFAULDA's own configuration. At 50 kHz this is ~20.5 ms; at 12 kHz it is ~85.3 ms; at 48 kHz it is ~21.3 ms — so a CWRU (12 kHz) window covers roughly 4× more physical time than a MAFAULDA window. This is a real, disclosed consequence of reusing the sample-count-based window size rather than a time-based one, not something this task's scope (evaluate the existing pipeline as-is, no pipeline redesign) permits silently equalizing.
- **Channel/sensor differences.** MAFAULDA windows come from a fixed channel (0) across 8 available channels; CWRU's real files carry 1–3 differently-named accelerometer channels (DE/FE/BA) at different physical mounting points. Which CWRU channel(s) are most comparable to MAFAULDA's channel 0 is not established by this task and is not assumed here.

These are the same kind of caveats already disclosed for MAFAULDA-only results elsewhere in this project (e.g. `docs/results/central_experiment_report.md`'s own limitations section) — real, structural, and not something a bigger/better model would automatically resolve.

---

## 7. Testing

- `backend/tests/dataset/test_cwru_loader.py` (10 tests, all passing): directory/label/sampling-rate parsing against synthetic `.mat` fixtures (never presented as real CWRU data); real-dataset-absence behavior (`discover_recordings`/`load_all_recordings` raising `CWRULoaderError` against the actual, currently-missing `data/external/cwru/`).
- `backend/tests/scripts/test_run_cross_dataset_validation.py` (7 tests, all passing): the real entry point raising against the real (missing) CWRU root; the module never referencing a training entry point; `IsolationForest.fit`/`StandardScaler.fit`/`fit_transform` spied and confirmed never called during a full pipeline run against a synthetic CWRU-shaped fixture tree; calibration confirmed computed exactly once, from MAFAULDA-validation-sized data, never from CWRU; a tampered-threshold scenario confirmed to raise rather than silently proceed; CWRU's real fault-type labels confirmed preserved alongside the binary metric.
- Full backend suite (`pytest`, run from `backend/`): all pre-existing tests still pass after this task's changes (see final report for the exact count).

**Definition of Done: NOT complete.** Per this task's own instructions, this is stated explicitly rather than silently implied: the CWRU loader is implemented, the pipeline can process CWRU once real files exist, and the "no retraining" guarantee is verified — but AC1 itself ("modelele antrenate pe MAFAULDA sunt evaluate pe CWRU ... rezultatele raportate onest") requires a real evaluation to have actually been run, which requires the real CWRU dataset. **The user must upload the real CWRU Bearing Dataset under `data/external/cwru/` (layout: Section 2.1) before this task's AC1 can be satisfied.**
