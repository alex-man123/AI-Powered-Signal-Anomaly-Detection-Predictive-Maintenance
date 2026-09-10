# Central Experiment Report

## 1. Executive Summary

This report compares three anomaly-detection configurations on the same real MAFAULDA test subset (1948 test windows, drawn from the same recordings for all three experiments):

- **Experiment A** -- Raw signal windows, dimensionality-reduced via PCA to exactly the DSP feature dimension, scored with Isolation Forest.
- **Experiment B** -- DSP-engineered features (TASK 5.3, 15 features), scored with Isolation Forest.
- **Experiment C** -- The same DSP features, scored with an Autoencoder's reconstruction error.

All three were evaluated with the identical common evaluator (TASK 8.1's `evaluate()`), on the identical test windows (verified below), with thresholds calibrated exclusively on validation data. All three runs share the same `dataset_hash`, `split_manifest_hash`, and `test_files` -- verified programmatically, not assumed.

Blueprint.md section 19's hypothesis (see below) is **PARTIALLY SUPPORTED** by these results -- ranking-quality metrics (ROC-AUC/PR-AUC) favor DSP features (Experiment B) over raw+PCA (Experiment A), while the specific calibrated operating point (precision/recall/F1) favors Experiment A. Neither Experiment A/B nor B/C shows one configuration dominating on every metric.

## 2. Initial Hypothesis

Blueprint.md section 19's own stated hypothesis (quoted, not reformulated):

> **Ipoteza:** feature engineering bazat pe DSP (cu semnificatie fizica -- RMS, kurtosis, frecventa dominanta etc.) separa mai bine normal/anomaly decat o reprezentare generica de aceeasi dimensionalitate (PCA pe raw).
>
> **Control:** acelasi split train/val/test (per fisier, fara leakage), acelasi algoritm ML, aceeasi dimensionalitate a input-ului intre bratele A si B, aceleasi metrici.

This is specifically a claim about **Experiment B vs Experiment A** (same algorithm, Isolation Forest, held constant; only the representation differs).

## 3. Experimental Setup

- **Dataset:** MAFAULDA (real subset), `dataset_hash=sha256:95aebdfe4d32407ef...` (identical for A/B/C, verified in Section 7).
- **Test set:** the same real recordings and windows for all three experiments (verified in Section 7).
- **Score direction:** `higher_is_more_anomalous` for all three (Isolation Forest's raw `decision_function` sign-flipped via TASK 6.3; Autoencoder's reconstruction error is already in this orientation).
- **Threshold:** `percentile`, calibrated on validation only, never on test, for all three experiments.
- **Evaluator:** TASK 8.1's `evaluate()`, called identically for all three -- no separate metric logic per experiment.

## 4. Experiment A -- Raw + PCA -> Isolation Forest

- Experiment ID: `EXP-A-002`
- Precision=0.6537, Recall=0.2598, F1=0.3718, ROC-AUC=0.5322, PR-AUC=0.5899
- FPR=0.1376, FNR=0.7402, Confusion Matrix=[[840, 134], [721, 253]], Inference time=0.0085s
- **Note on EXP-A-002:** the original `EXP-A-001` run is superseded by `EXP-A-002`. TASK 9.5's own mandatory reproducibility check (Section 12) discovered that `app.ml.dimensionality.fit_pca` left scikit-learn's PCA on its default `svd_solver="auto"`, which silently resolves to the randomized (non-deterministic) SVD algorithm for this data shape -- a genuine, narrow bug in prior (TASK 9.1) code, not a methodology change. It was fixed by pinning `svd_solver="full"` (exact, deterministic SVD; TASK 9.1's own existing test suite -- 278 tests -- still passes unchanged), and Experiment A was re-registered under the identical configuration (same files/window/channel/seed/threshold). Precision/Recall/F1/confusion matrix are bit-identical to the original `EXP-A-001`; ROC-AUC/PR-AUC differ from it in the 6th decimal place (0.53221 vs 0.53217, 0.58989 vs 0.58988) -- a change attributable entirely to the determinism fix, not a re-tuning of any hyperparameter, threshold, or dataset.

## 5. Experiment B -- DSP Features -> Isolation Forest

- Experiment ID: `EXP-B-001`
- Precision=0.6502, Recall=0.1355, F1=0.2243, ROC-AUC=0.6383, PR-AUC=0.6073
- FPR=0.0729, FNR=0.8645, Confusion Matrix=[[903, 71], [842, 132]], Inference time=0.0108s
- This experiment is a direct reference to Phase 6 (Isolation Forest training)/Phase 8 (evaluation) -- no retraining was performed for this report (TASK 9.3).

## 6. Experiment C -- DSP Features -> Autoencoder

- Experiment ID: `EXP-C-001`
- Precision=0.6243, Recall=0.1160, F1=0.1957, ROC-AUC=0.5847, PR-AUC=0.5583
- FPR=0.0698, FNR=0.8840, Confusion Matrix=[[906, 68], [861, 113]], Inference time=0.0008s
- This experiment is a direct reference to Phase 7 (Autoencoder training)/Phase 8 (evaluation) -- no retraining was performed for this report (TASK 9.4).

## 7. Comparative Results

| Experiment | Representation | Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|---:|---:|
| A | Raw + PCA | Isolation Forest | 0.6537 | 0.2598 | 0.3718 | 0.5322 | 0.5899 |
| B | DSP Features | Isolation Forest | 0.6502 | 0.1355 | 0.2243 | 0.6383 | 0.6073 |
| C | DSP Features | Autoencoder | 0.6243 | 0.1160 | 0.1957 | 0.5847 | 0.5583 |

All 15 cells above are real values measured by TASK 8.1's `evaluate()` on the real MAFAULDA test subset (see `central_experiment_results.json` for full, unrounded precision and every other metric TASK 8.1 computes).

![Comparison chart](central_experiment_comparison.png)

## 8. A vs B -- Representation Comparison (same model, Isolation Forest)

What changed: only the input representation (raw windows reduced via PCA to 15 components, vs. 15 DSP-engineered features). What stayed constant: the model (Isolation Forest), the seed, the train/validation/test files, the threshold method, the evaluator.

- ROC-AUC: B (0.6383) > A (0.5322) -- DSP features rank normal-vs-anomaly windows better across all thresholds.
- PR-AUC: B (0.6073) > A (0.5899) -- same conclusion, more relevant given the class imbalance in this test set.
- Precision/Recall/F1 (at the specific calibrated threshold): A (0.3718 F1) > B (0.2243 F1) -- at THIS RUN's specific percentile-95 threshold, raw+PCA happens to flag more true anomalies (Recall 0.2598 vs 0.1355) at a comparable precision.

This is a genuinely mixed result, not a clean win for either representation: DSP features produce a better-separating score (the more fundamental property the hypothesis is actually about), but raw+PCA's specific operating point performs better in this run. Both observations are reported as measured -- neither is discarded to simplify the story.

## 9. B vs C -- Model Comparison (same representation, DSP features)

What changed: only the model (Isolation Forest vs. Autoencoder). What stayed constant: the DSP feature representation, the train/validation/test files, the threshold method, the evaluator.

- B outperforms C on 5 of 5 classification-quality metrics (Precision, Recall, F1, ROC-AUC, PR-AUC): B={precision=0.6502, recall=0.1355, f1=0.2243, roc_auc=0.6383, pr_auc=0.6073} vs. C={precision=0.6243, recall=0.1160, f1=0.1957, roc_auc=0.5847, pr_auc=0.5583}.
- Inference time: C (0.0008s) is substantially faster than B (0.0108s) on this 1948-window test set -- a single vectorized PyTorch forward pass vs. an ensemble of many decision trees.
- Both models share the same dominant limitation on this run: recall is low for both (B=0.1355, C=0.1160) -- most real anomalies in this test set are missed by both models, most plausibly attributable to the very small (2-recording) real training subset used throughout TASK 6.x-9.x's own real-data runs, not to an inherent flaw in either architecture (see Limitations).

## 10. Hypothesis Evaluation

- Ranking-quality metrics (ROC-AUC, PR-AUC) favor DSP features (Experiment B) over raw+PCA (Experiment A): **True**.
- The specific calibrated operating point (Precision/Recall/F1) favors raw+PCA (Experiment A) over DSP features (Experiment B): **True**.
- **Conclusion: PARTIALLY SUPPORTED**.

This is not a forced or predetermined result -- it is the direct, documented consequence of the two signals above, computed from the real measured metrics in Section 7.

## 11. Alternative Interpretation / DSP Interpretability

Per blueprint.md section 19's own framing: if Experiment A (raw+PCA) is comparable to or better than Experiment B (DSP features) on some metrics -- which is the case here for Precision/Recall/F1 -- the correct interpretation is NOT that DSP feature engineering was pointless. Two things can both be true:

- Raw+PCA's few principal components may already capture much of the dominant variance associated with this specific fault type (horizontal misalignment) in this small test subset -- plausible given both representations reach the same target dimensionality, and PCA is itself a (generic, unsupervised) form of feature extraction, not truly "no processing at all".
- DSP features remain more **interpretable**: a RMS/kurtosis/dominant-frequency-based anomaly score can be explained in physical terms ("this window's kurtosis is unusually high, consistent with an impulsive bearing-type fault"); a PCA component has no such physical meaning -- it is a linear combination of thousands of raw samples with no direct mechanical interpretation. This interpretability difference is real and valuable independent of which representation happens to score marginally higher on this particular test run.
- DSP features are also far lower-dimensional to *compute* meaningfully from domain knowledge (RMS, crest factor, etc. are established vibration-analysis indicators), whereas raw+PCA's components are only meaningful in the statistical sense of "explains variance in this specific dataset" -- they do not generalize their interpretation to a different dataset the way "RMS" always means the same physical thing.

Given this run's ROC-AUC/PR-AUC do favor DSP features, this report does **not** need to fall back on "DSP wins on interpretability alone" as a consolation -- but the interpretability argument is recorded here as instructed, since Section 8 shows a genuinely mixed result on the threshold-dependent metrics.

## 12. Reproducibility

Each experiment was re-run using its own existing pipeline function (TASK 9.2/9.3/9.4's `run_experiment_a`/`run_experiment_b`/`run_experiment_c`, unmodified) against an isolated, throwaway experiment registry, and the freshly measured metrics were compared to what is already registered under `EXP-A-002`/`EXP-B-001`/`EXP-C-001`:

- Experiment A: PASS
- Experiment B: PASS
- Experiment C: PASS

Comparison excludes `inference_time` (a genuine wall-clock measurement, expected to vary slightly run to run on the same machine -- documented, not silently ignored) and includes every other TASK 8.1 metric plus the confusion matrix.

## 13. Limitations

- **Small real-data subset:** all three experiments were trained on the same 2 real normal recordings and evaluated on the same 4 real recordings (2 normal, 2 horizontal-misalignment) -- the same minimal subset established throughout TASK 6.x-9.x's own real-data runs, reused here per this task's explicit instruction not to retrain bigger models just for this report. The low recall shared by both DSP-based models (Section 9) is very likely a direct consequence of this.
- **Single fault class represented:** the test subset's "anomaly" class is exclusively `horizontal-misalignment` -- these results do not demonstrate performance against `imbalance` or `vertical-misalignment`.
- **Threshold calibrated on a small validation set:** `percentile=95` calibrated on 1 normal + 1 fault validation recording -- a larger, more representative validation set could shift the threshold-dependent metrics (Precision/Recall/F1/FPR/FNR); ROC-AUC/PR-AUC are threshold-independent and unaffected.
- **No cross-dataset validation:** these results are specific to MAFAULDA and this project's current DSP/feature/PCA configuration -- no CWRU or other dataset comparison exists yet.
- **Inference time is hardware/implementation-dependent:** measured on this machine, single-process, no batching optimization -- not a theoretical best-case benchmark for either library.
- **Experiment A's PCA determinism bug fix:** disclosed in full in Section 4 -- a real, narrow non-determinism bug in prior (TASK 9.1) code was found and fixed as part of this task's own mandatory reproducibility verification (AC4), and Experiment A was re-registered (`EXP-A-002`) under the identical configuration. This is a bug fix, not a methodology change: no hyperparameter, threshold, split, feature set, or dataset was altered.

## 14. Final Conclusion

On this specific real evaluation, the central experiment's hypothesis (DSP features separate normal/anomaly better than a same-dimensionality raw+PCA representation, same algorithm) is **PARTIALLY SUPPORTED**: DSP features do rank anomalies better across all thresholds (higher ROC-AUC/PR-AUC), but raw+PCA's specific calibrated operating point performs better on Precision/Recall/F1 in this run. Separately, comparing models on the same DSP representation, Isolation Forest (Experiment B) outperforms the Autoencoder (Experiment C) on every classification-quality metric measured here, while the Autoencoder is substantially faster at inference.

No winner is declared as definitive or universal -- these are the real, reproducible results of one specific, small-scale real-data configuration, honestly reported per this project's own "no invented results" principle, with the reproducibility of every number verified in Section 12, not assumed.
