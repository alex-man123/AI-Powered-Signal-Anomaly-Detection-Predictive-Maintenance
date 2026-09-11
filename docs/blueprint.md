# BLUEPRINT — AI-Powered Signal Anomaly Detection & Predictive Maintenance

> Technical decision document. Contains no implementation. All numerical results are marked `TO BE MEASURED` until the experiments are actually executed.

---

## 1. Executive Summary

We are building a platform that receives vibration signals from industrial equipment and answers the question: **"Is this signal consistent with normal operation, or does it indicate anomalous behavior?"**

The system combines classical DSP (filtering, FFT, PSD, spectrogram) with two anomaly detection approaches — Isolation Forest (classical baseline) and a PyTorch Autoencoder (deep learning) — trained predominantly on "normal" data and evaluated using real labels from the dataset. The central piece of the project, from an interview-value perspective, is the **raw signal → AI vs. raw signal → DSP → features → AI experiment**, which empirically demonstrates whether signal processing provides real value rather than assuming that it does.

The final deliverable is a web application (FastAPI + React/TypeScript), not a notebook and not a Streamlit dashboard.

---

## 2. Project Feasibility

Realistic for a single developer, working incrementally, under the following constraints:

* **We do not estimate workload in number of sessions.** There is no point tying a learning project (DSP + ML + DL + backend + frontend, learned thoroughly rather than merely copied) to an arbitrary deadline. The correct progress criterion is a phase gate, not a timer:

```text
Phase N → implementation → tests → explanation → user confirmation → Gate → Phase N+1
```

A phase is not considered complete because "enough time has passed", but because its success criterion (defined in Section 33) has been fulfilled and confirmed by you.

* The main risk of failure is not technical (each component is well established in the literature), but **scope creep** — the temptation to add 3D, experiment tracking, History, Settings, etc. before the foundation (DSP + baseline + central experiment) is solid.
* Decision: treat the project as a **strictly scoped MVP** + an explicit list of "later phases", rather than as one monolithic block of requirements.

---

## 3. Problem Definition

Correct formulation (according to Section 46 of your requirements):

> **AI-powered vibration anomaly detection for predictive maintenance** — not "failure prediction", not "Remaining Useful Life", until degradation tracking is actually implemented.

Methodological formulation (according to Section 5):

> Training is unsupervised/semi-supervised (the model learns only from data labeled "normal"), while evaluation is supervised (we use the dataset's real labels as ground truth during testing).

This distinction must be explicitly documented in the README, in a dedicated section, so that it does not appear contradictory to someone reading "you have labels, but you call it unsupervised."

---

## 4. Dataset Research

### MAFAULDA (Machinery Fault Database, UFRJ)

* **Creator:** Laboratório de Sinais, Multimídia e Telecomunicações, UFRJ (Rio de Janeiro).
* **Availability:** Public, official UFRJ page + Kaggle mirror ("Machinery Fault Dataset").
* **Content:** 1,951 multivariate time series acquired on a SpectraQuest Alignment-Balance-Vibration (ABVT) simulator. 6 states: normal, imbalance, horizontal misalignment, vertical misalignment, inner bearing fault (underhang), outer bearing fault (overhang).
* **Sensors:** Triaxial accelerometer on the underhang bearing + 3 industrial accelerometers (axial/radial/tangential) on the overhang bearing + tachometer + microphone → 8 columns per file.
* **Sampling rate:** 50 kHz, 5-second windows per file (250k samples/channel).
* **Format:** CSV — directly usable with pandas, without MATLAB conversion.
* **Problems:** Large volume per file (~13 GB for the entire dataset); rotation frequency varies between files and must be documented as a variable operating condition.
* **DSP suitability:** Very good — continuous signal, multi-channel, allows filtering/FFT/spectrogram on each channel independently.
* **ML/DL suitability:** Good — sufficient windows per class for Isolation Forest and Autoencoder training.
* **Data leakage risk:** High if splitting is performed naively on windows from the same file (see Section 8). The split must be performed at the file/run level.

**Dataset Audit Protocol — mandatory before any preprocessing (Phase 1.5, see Section 33):**

```text
MAFAULDA Dataset (raw files)

        ↓

How many files / recordings exist, and how are they named?

        ↓

What does each column/channel represent exactly, per file?

        ↓

How is a "recording" uniquely identified (filename → state + operating condition)?

        ↓

What rotation speeds / rotation frequencies exist, and how do they vary between files of the same class?

        ↓

Are there replications of the same condition (multiple files for the same state + rotation)?

        ↓

Class distribution: is it balanced? How many files per class?

        ↓

What can legitimately go into train vs. test without violating the file-level split?
```

The result of this audit (not assumptions) directly feeds the decisions in Section 8 (Data Leakage Strategy) and Phase 2. If the audit shows, for example, that certain classes have very few files, this must be known before designing the train/validation/test split, not discovered afterward.

### CWRU Bearing Dataset (Case Western Reserve University)

* **Creator:** Case Western Reserve University Bearing Data Center.
* **Availability:** Public, `.mat` files.
* **Content:** Artificially induced bearing faults (EDM) — ball, inner race, outer race — at fault diameters of 0.007"–0.021"/0.040", under 0–3 HP loads.
* **Sampling rate:** 12 kHz and 48 kHz.
* **Format:** MATLAB `.mat` — requires `scipy.io.loadmat`, less convenient than CSV.
* **Problems:** A single machine/bearing type and a strictly controlled laboratory environment; it is one of the most widely cited benchmarks in the literature, which is an advantage in terms of recognition, but also a signal that models may be over-optimized for it.
* **Suitability for the project:** Excellent as a **secondary validation dataset** — we test whether our pipeline (trained conceptually around the MAFAULDA logic) generalizes to a dataset different in origin, sensors, and fault types.

### Third Dataset Analyzed: NASA IMS Bearing Dataset

* **Content:** Bearings operated until actual failure (run-to-failure), useful for degradation tracking / RUL.
* **Reason for rejection from MVP:** There is no discrete normal/anomaly labeling — we would have to define it ourselves ("from what temporal point do we consider degradation to have started?"), introducing subjectivity that is difficult to justify in an MVP. The files are large and difficult to handle. **It remains a candidate for the "Remaining Useful Life" extension** (Section 45), not for the MVP.

---

## 5. Dataset Comparison Table

| Criterion       | MAFAULDA                   | CWRU                     | NASA IMS                               |
| --------------- | -------------------------- | ------------------------ | -------------------------------------- |
| Format          | CSV                        | `.mat`                   | Large ASCII files                      |
| Channels        | 8 (multi-sensor)           | 1–3 per file             | 4 channels                             |
| Sampling rate   | 50 kHz                     | 12/48 kHz                | 20 kHz                                 |
| Classes         | 6 (balanced by fault type) | 4 (normal + 3 faults)    | No discrete labels                     |
| Labeling        | Clear, per file            | Clear, per file          | Ambiguous (requires custom definition) |
| Leakage risk    | High if split naively      | High if split naively    | N/A for classification                 |
| DSP suitability | Very good                  | Very good                | Good, but different purpose (RUL)      |
| MVP suitability | **Primary**                | **Secondary validation** | Rejected for MVP                       |
| License/usage   | Public, research use       | Public, research use     | Public, research use                   |

---

## 6. Recommended Primary Dataset

**MAFAULDA.**

Reason: multi-channel + 6 varied classes (not only bearing faults, but also imbalance/misalignment) provides a meaningful anomaly score with real variation, native CSV format simplifies ingestion, and the richness of the sensors supports the Dataset/Signal Analysis UI pages.

---

## 7. Recommended Secondary Dataset

**CWRU.**

Used exclusively for cross-dataset validation in the final phase (Phase 13/later) — not for MVP development.

Purpose: demonstrate (or honestly disprove) the generalization capability of the pipeline.

---

## 8. Data Leakage Strategy

**Decision:** the split is performed at the **recording file level**, not at the individual window level.

* **Reason:** consecutive windows from the same MAFAULDA file are highly correlated (same run, same operating condition, identical background noise). A random window split would allow the model to "recognize" run-specific characteristics rather than general characteristics of the fault state.
* **Rejected alternative:** random split on windows (common in many online tutorials). Trade-off: it would produce artificially inflated metrics (false-optimistic F1), which cannot be used as credible evidence of technical competence.
* **Implementation:** `train/val/test` is assigned per file (`recording_id`), before windowing. Normalization (mean/std) is fitted **only** on the training data and then applied to validation/test — never the other way around.
* **Window overlap:** overlap (e.g. 50%) is allowed **within** a recording assigned to a single split, not between windows that would end up in different splits.
* **Documentation:** a dedicated "Data Leakage Prevention" section in the README, including the exact split diagram.

---

## 9. Product Vision

A web application that simulates a real condition-monitoring platform: the user selects a dataset/signal, explores it in time/frequency domains, runs anomaly detection, sees the score and explanation, and compares models.

It is not a notebook with static graphs — every interaction (zoom, interval selection, filter parameter changes) recalculates or re-renders live.

---

## 10. System Architecture

```text
React (Vite/TS) ──HTTP/JSON──> FastAPI ──> Services layer

                                      │

                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼

             signal_processing/    features/           ml/
             (preprocessing,       (time-domain,       (isolation_forest,
              filtering,           frequency-domain)   autoencoder,
              fft, psd,             features)          scoring)
              spectrogram)

                    │                 │                 │
                    └─────────────────┴─────────────────┘
                                      ▼
                               models/
                         (.pkl / .pt files saved to disk)
                                      ▼
                              SQLite
                    (metadata, experiment runs, predictions)
```

Each layer is independently testable (pytest), without dependency on FastAPI — DSP and feature engineering are pure NumPy/SciPy functions.

---

## 11. Backend Architecture

The structure proposed in Section 27 of the requirements is correct and is kept almost unchanged, with one addition: `app/api/schemas/` is kept separate from any domain model, so Pydantic request/response validation does not become mixed with business logic.

```text
backend/

└── app/

    ├── api/

    │   ├── routes/          (health, datasets, signals, models, experiments)

    │   └── schemas/         (Pydantic request/response models)

    ├── core/                (config.py, logging.py)

    ├── datasets/            (loader.py, validators.py)

    ├── signal_processing/   (preprocessing, filtering, fft, psd, spectrogram, windowing)

    ├── features/            (time_domain, frequency_domain, extractor)

    ├── ml/                  (isolation_forest, autoencoder, training, inference, scoring, evaluation)

    ├── services/            (orchestration — combines the layers above)

    └── main.py
```

**Decision → Reason → Alternative → Trade-off:**

* **Decision:** business logic lives in `services/`; route handlers remain thin (validation + service call).
* **Reason:** testability — `signal_analysis_service.py` can be tested without starting the HTTP server.
* **Alternative:** logic directly inside route handlers (common in simple FastAPI tutorials).
* **Trade-off:** one additional layer of indirection, but clearer separation of responsibilities and faster unit tests.

---

## 12. Frontend Architecture

The structure from Section 29 of the requirements is adopted in full.

One note: `components/charts/` will contain thin wrappers around Plotly (e.g. `TimeSeriesChart.tsx`, `FFTChart.tsx`) so that Plotly configuration is not duplicated across every page.

---

## 13. DSP Architecture

Pipeline:

```text
raw signal
    ↓
preprocessing (detrend, normalize)
    ↓
windowing
    ↓
filtering (optional, configurable)
    ↓
FFT / PSD / STFT
```

* **Filtering:** Butterworth low-pass/high-pass/band-pass, with order and cutoff frequencies configurable through the request, not hard-coded. We use `filtfilt` (zero-phase filtering) to avoid phase distortion that could artificially shift events in time — critical for vibration analysis where fault-event timing matters.

* **FFT:** simple and fast, suitable for identifying dominant frequencies on a static segment.

* **Welch PSD:** preferred over a simple FFT when we need a more robust estimate in the presence of noise (averages across multiple overlapping windows) — the frequency-resolution vs. reduced-variance trade-off is explained in the UI, not only in code.

* **STFT/Spectrogram:** used to visualize how frequencies evolve over time, with the window-size vs. hop-length trade-off (frequency resolution vs. time resolution) explicitly exposed and configurable from the DSP Lab UI.

---

## 14. Feature Engineering Architecture

Separate module, with each feature implemented as a pure function and covered by a dedicated test.

**Time-domain features:**

* mean
* standard deviation
* variance
* RMS
* peak
* peak-to-peak
* skewness
* kurtosis
* crest factor

**Frequency-domain features:**

* dominant frequency
* spectral centroid
* spectral bandwidth
* spectral energy
* spectral entropy
* band energy

For each feature, document (according to Section 15 of the requirements):

* formula
* physical interpretation
* relevance for fault detection

For example, kurtosis may increase with short-duration impacts typical of bearing faults, while crest factor detects peaks without necessarily increasing the mean.

Extensibility comes from the fact that `extractor.py` iterates over a registry of functions — adding a new feature does not require modifying the rest of the code.

---

## 15. ML Architecture — Isolation Forest

Pipeline:

```text
normal windows
    ↓
feature extraction
    ↓
feature matrix
    ↓
IsolationForest.fit()
    ↓
raw anomaly score
    ↓
normalization
    ↓
threshold
    ↓
NORMAL / WARNING / ANOMALY
```

**Decision → Reason → Alternative → Trade-off:**

* **Decision:** Isolation Forest as the baseline.
* **Reason:** it does not require distributional assumptions, trains quickly, works well with moderately sized feature vectors, and is easy to explain in an interview (it isolates observations through random partitioning — anomalies are isolated in fewer splits).
* **Alternative:** One-Class SVM, Local Outlier Factor.
* **Trade-off:** Isolation Forest is faster and scales better than One-Class SVM, but is less sensitive to "local" anomalies within a dense cluster than LOF — acceptable for a baseline, with the limitation explicitly discussed.

---

## 16. Autoencoder Architecture

**Decision A vs. B (Section 19 of the requirements):**

|                                 | Raw waveform segments                                        | Feature vectors                                                 |
| ------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------- |
| Interpretability                | Low (what exactly did the network "see"?)                    | High (reconstruction error can be linked to a specific feature) |
| Architecture complexity         | Higher (Conv1D layers)                                       | Lower (dense layers)                                            |
| Computational cost              | Higher                                                       | Lower                                                           |
| Relationship to DSP             | Weak — ignores much of what we built in the feature pipeline | Strong — directly validates the DSP pipeline                    |
| Relevance to central experiment | Does not contribute directly                                 | Contributes directly                                            |

**Recommendation: Autoencoder on feature vectors** for the MVP as the primary variant, because it directly connects to the central experiment (Section 21), where raw vs. DSP+features is compared.

**Autoencoder on raw waveform** remains an optional future experiment (Section 45), not mandatory for the MVP, specifically to avoid doubling the scope with a comparison that is not central to the project's story.

Architecture: encoder with 2–3 dense layers (input dimension = number of features → small bottleneck), symmetric decoder. Simple enough to draw the architecture on a sheet of paper during an interview.

---

## 17. Anomaly Scoring

* **Raw model score:** raw output (`decision_function` for Isolation Forest, reconstruction MSE for Autoencoder) — the scales are different and cannot be directly compared.

* **Normalized anomaly score:** min-max or percentile scaling based on the score distribution from the **validation set** (not the test set — otherwise leakage), mapped to `[0, 1]`.

* **Threshold:** calibrated on the validation set using the
