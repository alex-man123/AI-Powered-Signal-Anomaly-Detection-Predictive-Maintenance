# Isolation Forest vs. Autoencoder

Source of results: `backend/tests/ml/test_evaluation.py::test_real_data_same_evaluate_function_for_isolation_forest_and_autoencoder` (TASK 8.1's real end-to-end evaluation), re-run on 2026-09-10 to produce the numbers below. Both models are passed through the exact same `evaluate()` function (`backend/app/ml/evaluation.py`, TASK 8.1) — no separate metric-computation logic exists for either model.

---

## 1. Evaluation Setup

- **Isolation Forest:** `app.ml.isolation_forest.train_isolation_forest` (TASK 6.2), scored via `app.ml.inference.predict` + `app.ml.scoring.to_anomaly_score` (raw `decision_function` flipped to this project's `higher_is_more_anomalous` convention), threshold calibrated via `app.ml.scoring.calibrate`.
- **Autoencoder:** `app.ml.autoencoder.Autoencoder` (TASK 7.1) trained via `app.ml.training.train_autoencoder` (TASK 7.2), scored via `app.ml.inference.reconstruction_error` (already `higher_is_more_anomalous`, no sign flip needed), threshold calibrated via the same `app.ml.scoring.calibrate`.
- **Threshold method:** `percentile`, `percentile_value=95`, calibrated on the **validation** split only for both models — never on test.
- **Score direction:** `higher_is_more_anomalous` for both models (this project's established convention — see TASK 6.5's `score_direction` field).
- **Random seed:** 42 for both models' training.
- **Feature representation:** 15 DSP features per window — the 9 time-domain features (`app.ml.features.registry.FEATURE_REGISTRY`: mean, std, variance, rms, peak, peak_to_peak, skewness, kurtosis, crest_factor) plus the 6 frequency-domain features (`FREQUENCY_FEATURE_REGISTRY`: dominant_frequency, spectral_centroid, spectral_bandwidth, spectral_energy, spectral_entropy, band_energy_10_100hz), scaled with a single shared `StandardScaler` (TASK 5.4) fit on the train split.

## 2. Test Set

Both models were evaluated on the **exact same** test feature matrix, verified directly rather than assumed:

- Same `data/processed/split_manifest.json` **test** entries: `normal/13.1072.csv`, `normal/20.2752.csv`, `horizontal-misalignment/0.5mm/18.8416.csv`, `horizontal-misalignment/0.5mm/25.8048.csv` — 2 normal + 2 fault recordings.
- Same windowing: `window_size=1024` samples (~20.48 ms at the dataset's 50 kHz sampling rate), `overlap=0.5`, single channel (channel 0).
- Same feature extraction call (`app.features.extractor.extract_feature_matrix`, same `nperseg=256`/`noverlap=128`) and the same train-fitted scaler applied to both models' test inputs.
- Same label aggregation: `normal -> 0`, `horizontal-misalignment -> 1` (this project's only two classes represented in this particular test subset — see Limitations).
- **Confirmed identical, not assumed:** both models' confusion matrices sum to exactly **1948** test windows, split exactly **974 normal / 974 anomaly** for both models — the same test set, the same ground truth, not two different subsets that happened to be evaluated separately.

Note on scaling: Isolation Forest's own training function (`train_isolation_forest`) fits its scaler on the normal-labeled subset of train internally; the Autoencoder pipeline fits a scaler on train directly via `app.ml.scaling.fit_scaler`. In this run the train split (`normal/12.288.csv`, `normal/16.1792.csv`) contains **only** normal-labeled windows, so both scalers are fit on the identical underlying data and are numerically equivalent here — there is no scaling inconsistency between the two evaluations in this particular run.

## 3. Metrics

All metrics come from `app.ml.evaluation.evaluate(y_true, y_pred, anomaly_scores, inference_time)` (TASK 8.1), the same function call for both models:

- **Precision / Recall / F1** — from binary predictions (`y_pred`, at the calibrated threshold).
- **ROC-AUC / PR-AUC** — from the continuous `anomaly_scores`, not from `y_pred`.
- **Confusion matrix** — `[[TN, FP], [FN, TP]]`.
- **FPR** (`FP/(FP+TN)`) — proportion of normal windows misclassified as anomalies.
- **FNR** (`FN/(FN+TP)`) — proportion of real anomalies missed.
- **Inference time** — wall-clock seconds (`time.perf_counter()`) around the test-set scoring call only (model already trained/loaded, threshold already calibrated on validation — training, calibration, and dataset loading are excluded).

## 4. Comparative Results

| Metric | Isolation Forest | Autoencoder |
|---|---:|---:|
| Precision | 0.6502 | 0.6243 |
| Recall | 0.1355 | 0.1160 |
| F1 | 0.2243 | 0.1957 |
| ROC-AUC | 0.6383 | 0.5847 |
| PR-AUC | 0.6073 | 0.5583 |
| FPR | 0.0729 | 0.0698 |
| FNR | 0.8645 | 0.8840 |
| Inference time (s, 1948 windows) | 0.0093 | 0.0014 |

(Values as measured on 2026-09-10; re-running the same test reproduces the same metric values exactly — training/scoring is seeded — except `inference_time`, which is a real wall-clock measurement and varies slightly run-to-run, e.g. 0.0093–0.0140 s for Isolation Forest and 0.0014–0.0017 s for Autoencoder across two consecutive runs.)

## 5. Confusion Matrices

### Isolation Forest

| | Predicted Normal | Predicted Anomaly |
|---|---:|---:|
| **Actual Normal** | 903 | 71 |
| **Actual Anomaly** | 842 | 132 |

### Autoencoder

| | Predicted Normal | Predicted Anomaly |
|---|---:|---:|
| **Actual Normal** | 906 | 68 |
| **Actual Anomaly** | 861 | 113 |

## 6. Analysis

- **Precision** (0.650 vs. 0.624): when either model flags a window as anomalous, it is correct about 62-65% of the time — comparable between the two, Isolation Forest modestly ahead.
- **Recall** (0.136 vs. 0.116): both models detect only about 12-14% of the real anomalies in this test set — the dominant finding of this evaluation (see Limitations for why).
- **F1** (0.224 vs. 0.196): reflects the low recall dominating both scores; Isolation Forest modestly ahead.
- **ROC-AUC** (0.638 vs. 0.585) / **PR-AUC** (0.607 vs. 0.558): Isolation Forest's continuous anomaly score ranks normal-vs-anomaly windows somewhat better than the Autoencoder's reconstruction error does, on this specific test set. Both are well above the random-ranking baseline (0.5) but far from strong separation.
- **FPR** (0.073 vs. 0.070) / **FNR** (0.864 vs. 0.884): FPR is close between the two (Autoencoder marginally lower); FNR is high for both, with Isolation Forest marginally lower.
- **Inference time**: the Autoencoder scored the 1948-window test set roughly 5-10x faster than Isolation Forest in these measurements (0.0014 s vs. 0.0093 s in this run). This is a real, repeatedly observed gap in this setup (a single vectorized PyTorch forward pass vs. an ensemble of many decision trees), not a one-off measurement artifact.

Across five of the seven classification-quality metrics (precision, recall, F1, ROC-AUC, PR-AUC), Isolation Forest scores modestly higher than the Autoencoder; FPR is essentially tied; the Autoencoder is clearly faster at inference. None of the classification-metric gaps are large in absolute terms (the biggest is ROC-AUC, a 0.054 difference).

## 7. Limitations

- **Tiny training subset:** both models were trained on only 2 real normal recordings (`normal/12.288.csv`, `normal/16.1792.csv`) — the same minimal real-data subset already used throughout TASK 6.x/7.x's own smoke tests, reused here per this task's explicit instruction not to retrain bigger models just to produce a nicer-looking report. The very low recall (~12-14%) for both models is very likely a direct consequence of this: an Isolation Forest/Autoencoder trained on two recordings has seen only a narrow slice of what "normal" vibration actually looks like, and consequently fails to flag most real fault windows as sufficiently different. These results characterize that specific small-scale setup, not the models' ceiling performance with this project's full ~611-recording train split.
- **Single fault class represented:** the test subset's "anomaly" class is exclusively `horizontal-misalignment` — no `imbalance` or `vertical-misalignment` examples were included in this particular test run. These results do not demonstrate how either model performs against those other two fault types.
- **Small test set:** 1948 windows drawn from only 4 recordings (2 normal, 2 fault) — enough to compute every metric meaningfully, but not a large-sample estimate of population-level performance.
- **Threshold calibrated on a small validation set:** the `percentile=95` threshold was calibrated on 1 normal + 1 fault validation recording — a threshold calibrated on a larger, more representative validation set could shift these numbers (particularly recall/FPR/FNR, which are threshold-dependent; ROC-AUC/PR-AUC are threshold-independent and unaffected by this).
- **Inference time is hardware/implementation-dependent:** measured on this machine's CPU, single-process, no batching optimization beyond each library's own default — not a benchmark of theoretical best-case throughput for either approach.
- **No cross-dataset validation:** these results are specific to MAFAULDA and this project's current DSP/feature configuration; no CWRU or other dataset comparison exists yet.

## 8. Conclusion

On this specific evaluation (a small real MAFAULDA test subset, models trained on an equally small real training subset), Isolation Forest measured modestly higher than the Autoencoder on five of seven classification metrics (precision, recall, F1, ROC-AUC, PR-AUC), the two models were essentially tied on FPR, and the Autoencoder measured substantially faster at inference time. None of the classification-metric differences are large — the biggest, ROC-AUC, is a 0.054 gap — so this result should be read as "Isolation Forest performed somewhat better on this run's detection-quality metrics, while the Autoencoder was clearly faster," not as one architecture being decisively superior overall.

The more consequential finding in this evaluation is that **both models currently miss the large majority of real anomalies** (recall 11-14%, FNR 86-88%) — a limitation of the training data scale used here (2 real recordings), not necessarily of either model architecture. This report does not draw a conclusion about which architecture is better suited to this project's full-scale problem; that question requires re-running this same evaluation once both models are trained on this project's complete train split, which is outside this task's scope (no model was retrained beyond what TASK 6.x/7.x already established, per this task's explicit instruction).
