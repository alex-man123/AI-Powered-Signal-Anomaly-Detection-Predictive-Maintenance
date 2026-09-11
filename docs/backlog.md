# COMPLETE IMPLEMENTATION BACKLOG — AI Signal Anomaly Detection Platform

> **Source of truth:** `blueprint.md`. All numerical results marked as `NOT YET MEASURED` until actual execution.

> **Scope note:** For feature engineering tasks (Phase 5) and repetitive UI tasks (Phase 11), related sub-elements are grouped into a single task with Acceptance Criteria per element rather than being split into separate full tasks. Otherwise, the document would contain >150 tasks with identical structure. Each grouping remains individually verifiable.

---

# EPIC 1 — Project Foundation

## PHASE 1 — Project Setup

### TASK 1.1 — Initialize Repository and Directory Structure

**Priority:** P0 | **Dependencies:** None | **Blocks:** 1.2, 1.4

**User Story:** As a developer, I want a clear repository structure so that each component (backend/frontend/data/models/docs) has a predictable location.

**Description:** Create the folder structure defined in the blueprint (section 29): `backend/`, `frontend/`, `data/{raw,processed,external}`, `models/`, `notebooks/`, `scripts/`, `docs/`.

**Implementation Steps:**

1. Create the top-level directories.

2. Create `.gitignore` (Python, Node, `.env`, `data/raw/*`, `models/*.pt`, `models/*.pkl`, `__pycache__`, `.venv`, `node_modules`).

3. Create `.env.example` with anticipated variables (`DATABASE_URL`, `MODEL_DIR`, `DATA_DIR`, `LOG_LEVEL`).

4. Create `README.md` with a section skeleton to be progressively completed throughout the phases.

**Acceptance Criteria:**

* [ ] AC1: Given the repository is cloned, when I run `ls`, then I see all top-level directories defined in section 29 of the blueprint.

* [ ] AC2: `.gitignore` excludes `.venv`, `node_modules`, `data/raw/*`, `*.pkl`, `*.pt`, `.env`.

* [ ] AC3: `.env.example` contains no real secrets.

**Testing:** N/A (structural, verified manually + code review).

**Definition of Done:** Structure created, `.gitignore` functional (verified with `git status` after adding ignored files), README skeleton exists.

**Files/Modules Expected:** `/`, `.gitignore`, `.env.example`, `README.md`.

---

### TASK 1.2 — Configure Backend with uv + pyproject.toml

**Priority:** P0 | **Dependencies:** 1.1 | **Blocks:** 1.3, 1.5

**User Story:** As a developer, I want a reproducible Python environment managed with `uv` so that dependencies are versioned and installation is fast.

**Implementation Steps:**

1. Run `uv init` inside `backend/`.

2. Define `pyproject.toml` with dependencies: `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings` (for `BaseSettings` in modern Pydantic v2 — `BaseSettings` is no longer included in the base `pydantic` package), `sqlalchemy` (metadata persistence, Phase 2+), `numpy`, `scipy`, `pandas`, `scikit-learn`, `torch`, `pytest`, `httpx` (for TestClient).

3. Run `uv sync` to generate `uv.lock` and `.venv`.

4. Verify environment activation and successful `fastapi` import.

**Acceptance Criteria:**

* [ ] AC1: Given `backend/pyproject.toml` exists, when I run `uv sync`, then `.venv` is created without errors.

* [ ] AC2: `uv run python -c "import fastapi, numpy, scipy, sklearn, torch, sqlalchemy, pydantic_settings"` runs without `ImportError`.

* [ ] AC3: `uv.lock` is generated and can be committed.

**Testing:** Manual verification of the sync command + import smoke test.

**Definition of Done:** Functional environment, all critical packages import successfully.

**Files/Modules Expected:** `backend/pyproject.toml`, `backend/uv.lock`.

---

### TASK 1.3 — FastAPI Skeleton + Health Endpoint

**Priority:** P0 | **Dependencies:** 1.2 | **Blocks:** 1.6, 10.x

**User Story:** As a developer, I want a health endpoint so that I can quickly verify that the backend is running and accessible from the frontend.

**Implementation Steps:**

1. Create `app/main.py` with the FastAPI instance + CORS middleware (frontend origin allowed through `.env`).

2. Create `app/api/routes/health.py` with `GET /api/health`.

3. Create `app/core/config.py` (Pydantic `BaseSettings`, reads `.env`).

4. Create `app/core/logging.py` (structured logging configuration, log level from environment).

**Acceptance Criteria:**

* [ ] AC1: Given the backend is running (`uvicorn app.main:app`), when I call `GET /api/health`, then I receive HTTP 200.

* [ ] AC2: The response body contains `{"status": "ok"}`.

* [ ] AC3: A pytest integration test (using `TestClient`) verifies AC1 and AC2.

**Testing:** `tests/api/test_health.py` — integration test using `TestClient`.

**Definition of Done:** Endpoint functional, test passing, logging configured, no business logic inside the route handler.

**Files/Modules Expected:** `backend/app/main.py`, `backend/app/api/routes/health.py`, `backend/app/core/config.py`, `backend/app/core/logging.py`, `backend/tests/api/test_health.py`.

---

### TASK 1.4 — Initialize Frontend (Vite + React + TypeScript)

**Priority:** P0 | **Dependencies:** 1.1 | **Blocks:** 1.5, 11.x

**Implementation Steps:**

1. Run `npm create vite@latest frontend -- --template react-ts`.

2. Install and configure Tailwind CSS (`tailwind.config.ts`, `postcss.config.js`, `index.css` with Tailwind directives).

3. Install shadcn/ui (initialize + basic components: `button`, `card`, `badge`).

4. Install `plotly.js` + `react-plotly.js` + types.

**Acceptance Criteria:**

* [ ] AC1: Given the frontend is installed, when I run `npm run dev`, then the application starts on `localhost:5173` without console errors.

* [ ] AC2: A Tailwind class (e.g. `bg-slate-900`) applied to an element is visually reflected.

* [ ] AC3: A shadcn/ui component (`<Button>`) renders without TypeScript errors.

**Testing:** Manual visual verification + `tsc --noEmit` without errors.

**Definition of Done:** Dev server starts, Tailwind active, shadcn/ui functional, Plotly importable.

**Files/Modules Expected:** `frontend/` complete, `frontend/tailwind.config.ts`, `frontend/src/index.css`.

---

### TASK 1.5 — Initial Design Tokens (Design System)

**Priority:** P1 | **Dependencies:** 1.4 | **Blocks:** 11.x, 12.x

**Description:** Define the color palette (dark-first, single accent), typography (Inter + JetBrains Mono), and spacing scale before implementing any UI components (according to section 30 of the blueprint).

**Implementation Steps:**

1. Define CSS custom variables in `frontend/src/styles/globals.css` (`--color-bg`, `--color-surface`, `--color-accent`, `--color-normal`, `--color-warning`, `--color-anomaly`, `--font-sans`, `--font-mono`).

2. Configure Tailwind `theme.extend` to use these variables.

3. Document the tokens in `docs/design-system.md`.

**Acceptance Criteria:**

* [ ] AC1: All status colors (NORMAL/WARNING/ANOMALY) are defined exactly once as variables and are not hard-coded inside components.

* [ ] AC2: `docs/design-system.md` contains the complete palette and spacing rules.

**Testing:** Manual visual review.

**Definition of Done:** Tokens defined and documented, zero hard-coded colors in subsequent UI commits.

**Files/Modules Expected:** `frontend/src/styles/globals.css`, `docs/design-system.md`.

---

### TASK 1.6 — Frontend-Backend Connectivity (End-to-End Smoke Test)

**Priority:** P0 | **Dependencies:** 1.3, 1.4 | **Blocks:** Phase 2

**Implementation Steps:**

1. Configure `VITE_API_URL` in the frontend `.env`.

2. Create `frontend/src/services/api.ts` with a minimal fetch client.

3. Call `/api/health` from a placeholder page and display the status.

**Acceptance Criteria:**

* [ ] AC1: Given both servers are running, when I load the main page, then I see the status `"ok"` received from the backend (not hard-coded in the frontend).

* [ ] AC2: If the backend is stopped, the frontend displays a visible error state rather than a blank screen or crash.

**Testing:** Manual verification with backend running/stopped.

**Definition of Done:** End-to-end connectivity visually confirmed, error state handled.

**Files/Modules Expected:** `frontend/src/services/api.ts`, placeholder status component.

---

## PHASE 1.5 — Dataset Audit (Mandatory, Blocks Phase 2)

> **KNOWN BLOCKER:** These tasks require the real MAFAULDA files in `data/raw/mafaulda/`. They must not be executed against assumed or simulated data.

### TASK 1.5.1 — Locate and Inventory Dataset Files

**Priority:** P0 | **Dependencies:** 1.1 | **Blocks:** 1.5.2–1.5.9, Phase 2

**User Story:** As an ML engineer, I want a complete inventory of the available MAFAULDA files so that I can correctly design the train/validation/test split.

**Implementation Steps:**

1. Confirm the location of the uploaded files (`data/raw/mafaulda/`).

2. Recursively scan and list all files (name, extension, size).

3. Save the raw inventory as `docs/dataset_audit/file_inventory.csv`.

**Acceptance Criteria:**

* [ ] AC1: Given the MAFAULDA files have been uploaded, when I run the inventory script, then I obtain a CSV containing all discovered files without omissions (verified by comparing `find` count vs. CSV row count).

* [ ] AC2: If no files are found, the script explicitly reports an error rather than treating an empty inventory as a success.

**Testing:** Unit test using a small synthetic folder with dummy files, verifying that detected count = created file count.

**Definition of Done:** Complete inventory generated, or explicit error reported if the dataset is missing.

**Files/Modules Expected:** `backend/scripts/audit_inventory.py`, `docs/dataset_audit/file_inventory.csv`.

**Notes:** **Blocked until the real dataset is uploaded by the user.**

---

### TASK 1.5.2 — Identify Recording Structure (Filename → State + Condition)

**Priority:** P0 | **Dependencies:** 1.5.1 | **Blocks:** 1.5.5, 2.x

**Implementation Steps:**

1. Parse the actual file naming convention (not one assumed from online documentation).

2. Extract per file: class/state (normal/imbalance/misalignment/etc.), if encoded in the folder path or filename.

3. Document the exact mapping observed.

**Acceptance Criteria:**

* [ ] AC1: Every file in the inventory has an associated state derived from the actual structure (folder/name), not assumed from the blueprint.

* [ ] AC2: If a file cannot be mapped to a known state, it is explicitly marked as `UNKNOWN` rather than silently ignored.

**Testing:** Unit tests using known filename samples → verify correct mapping.

**Definition of Done:** Complete mapping documented in `docs/dataset_audit/recording_mapping.md`.

**Files/Modules Expected:** `backend/app/datasets/mafaulda_parser.py`, `docs/dataset_audit/recording_mapping.md`.

---

### TASK 1.5.3 — Identify Channels, Sampling Rate, and Signal Length

**Priority:** P0 | **Dependencies:** 1.5.1

**Implementation Steps:**

1. Read a sample of files using pandas.

2. Confirm the actual number of columns/channels per file and their names.

3. Confirm the actual sampling rate (verified, not merely cited from literature) and signal duration.

**Acceptance Criteria:**

* [ ] AC1: The observed number of columns is explicitly documented (it may differ from what external sources suggest).

* [ ] AC2: The sampling rate is confirmed either from file metadata or calculated from the number of samples / known duration, with the exact source of confirmation documented.

**Testing:** Verification on at least 3 files from different classes, compared for consistency.

**Definition of Done:** Written report containing the confirmed real data structure.

**Files/Modules Expected:** `docs/dataset_audit/signal_structure.md`.

---

### TASK 1.5.4 — Class and Rotation Distribution

**Priority:** P0 | **Dependencies:** 1.5.2

**Implementation Steps:**

1. Count files per class.

2. Extract/calculate rotation per file, with the **source explicitly documented for every value** — a value read from a filename must not be treated as a physical measurement equivalent to one obtained from a tachometer. Mandatory fields per recording:

```text
rotation_source: "tachometer" | "metadata" | "filename" | "unavailable"

rotation_frequency_hz: <value> (null if unavailable)
```

3. Generate a histogram of class and rotation distributions, annotated with the source.

**Acceptance Criteria:**

* [ ] AC1: A complete `class → file count` table exists, with no classes omitted.

* [ ] AC2: If the distribution is strongly imbalanced (e.g. one class has <10% of the average count), this is explicitly flagged as a risk for Phases 6–9 rather than ignored.

* [ ] AC3: Every reported rotation value is explicitly accompanied by `rotation_source` — the report must not present a filename-derived value with the same confidence as one directly measured by a tachometer.

**Testing:** Manual verification of total count (sum across classes = total files in inventory).

**Definition of Done:** Report + distribution plot saved.

**Files/Modules Expected:** `docs/dataset_audit/class_distribution.md`, `docs/dataset_audit/class_distribution.png`.

---

### TASK 1.5.5 — Detect Replications/Duplicates of the Same Conditions

**Priority:** P1 | **Dependencies:** 1.5.2, 1.5.4

**Implementation Steps:**

1. Group files by `(class, approximate rotation)`.

2. Identify groups containing multiple files (real replications) vs. unique conditions.

**Acceptance Criteria:**

* [ ] AC1: The report explicitly indicates how many `(class, rotation)` groups contain ≥2 replicated files.

* [ ] AC2: This information directly informs the split decision (Task 1.5.7).

**Testing:** Manual verification on a known subset.

**Definition of Done:** Replication analysis report generated.

**Files/Modules Expected:** `docs/dataset_audit/replication_analysis.md`.

---

### TASK 1.5.6 — Analyze Data Leakage Risk Specific to the Real Dataset

**Priority:** P0 | **Dependencies:** 1.5.1–1.5.5 | **Blocks:** 1.5.7

**Description:** Do not generically assume that "split per file is sufficient" — verify whether there are additional leakage factors (e.g. multiple files from the same recording session, with identical background noise, that should be treated as a single group during splitting).

**Acceptance Criteria:**

* [ ] AC1: The report explicitly documents whether the "per file" split unit is sufficient or whether it must be extended to "per session group".

* [ ] AC2: The final split decision is justified using evidence from the audit rather than assumed from the blueprint.

**Testing:** N/A (documented analysis, manually reviewed).

**Definition of Done:** Split decision written and justified with direct references to the real data.

**Files/Modules Expected:** `docs/dataset_audit/leakage_analysis.md`.

---

### TASK 1.5.7 — Define Final Train/Validation/Test Split Strategy

**Priority:** P0 | **Dependencies:** 1.5.6 | **Blocks:** Phase 2

**Implementation Steps:**

1. Assign files (not windows) to train/validation/test, stratified by class.

2. Document exactly which files go into each split (explicit list, reproducible with a fixed seed).

**Acceptance Criteria:**

* [ ] AC1: No file appears in more than one split.

* [ ] AC2: Each split contains representation from every major class (explicitly verified, not assumed).

* [ ] AC3: The split is deterministic (same seed → same allocation, verified with a test).

**Testing:** Unit test that runs the split function twice with the same seed and verifies identical results.

**Definition of Done:** Final split list saved as an artifact (`data/processed/split_manifest.json`).

**Files/Modules Expected:** `backend/app/datasets/split.py`, `data/processed/split_manifest.json`, `backend/tests/dataset/test_split.py`.

---

### TASK 1.5.8 — Data Quality Verification (Missing Values, Corrupted Signals)

**Priority:** P1 | **Dependencies:** 1.5.3

**Implementation Steps:**

1. Check NaN/missing values per file.

2. Check constant/zero signals (potential defective sensor recording).

**Acceptance Criteria:**

* [ ] AC1: The report explicitly lists every file with detected quality issues.

* [ ] AC2: If problematic files exist, the decision (exclude/repair) is documented with the reason.

**Testing:** Test using a synthetic file with injected NaN values → verify detection.

**Definition of Done:** Data quality report generated.

**Files/Modules Expected:** `docs/dataset_audit/data_quality_report.md`.

---

### TASK 1.5.9 — Final Audit Report (Consolidated Document)

**Priority:** P0 | **Dependencies:** 1.5.1–1.5.8 | **Blocks:** Phase 2 (Gate)

**Description:** Consolidate all 1.5.x tasks into a single document that serves as the source of truth for all dataset decisions in subsequent phases.

**Acceptance Criteria:**

* [ ] AC1: The document explicitly answers all questions from section 6 of the blueprint (file organization, channels, recording ID, rotation, replications, split).

* [ ] AC2: No statement in the document is assumed — each statement has a source (file/script that generated it).

**Definition of Done:** `docs/dataset_audit/AUDIT_REPORT.md` is complete and reviewed.

## **Files/Modules Expected:** `docs/dataset_audit/AUDIT_REPORT.md`.


# EPIC 2 — Data & Signal Processing Foundation

## PHASE 2 — Dataset Integration

### TASK 2.1 — Canonical Data Model (SignalRecord, Recording)

**Priority:** P0 | **Dependencies:** Phase 1.5 Gate | **Blocks:** 2.2–2.5

**Implementation Steps:**

1. Define `SignalRecord` / `Recording` (Pydantic + SQLAlchemy) with the fields from section 25 of the blueprint: id, sampling_rate, channel, values (file reference, not stored in the DB), machine_id/operating_condition, label.

2. Create the SQLite schema (`datasets`, `signals`, `signal_windows`).

**Acceptance Criteria:**

* [ ] AC1: The model rejects a `sampling_rate ≤ 0`.

* [ ] AC2: The model rejects an unknown label (strict enum, not a free-form string).

**Testing:** Pydantic unit tests for validation (valid/invalid values).

**Definition of Done:** Models defined, migrated, and tested.

**Files/Modules Expected:** `backend/app/models/signal.py`, `backend/app/core/database.py`.

---

### TASK 2.2 — Dataset Loader (Based on Audit Results)

**Priority:** P0 | **Dependencies:** 2.1, 1.5.9 | **Blocks:** 2.4

**Implementation Steps:**

1. Implement `loader.py` that reads files according to the actual structure confirmed during the audit (not an assumed structure).

2. Populate the `Recording` / `SignalRecord` models from each file.

**Acceptance Criteria:**

* [ ] AC1: Given the `split_manifest.json` from Task 1.5.7, when the loader is executed, then every recording is correctly assigned the split (`train`/`val`/`test`) allocated to it.

* [ ] AC2: The loader explicitly rejects a file that does not comply with the structure confirmed in the audit (it must not silently ignore it).

**Testing:** Integration test on a small real subset (2–3 files per split).

**Definition of Done:** Loader functional and tested on real data.

**Files/Modules Expected:** `backend/app/datasets/loader.py`, `backend/tests/dataset/test_loader.py`.

---

### TASK 2.3 — Dataset Validators (Sampling Rate, Channels, Shape)

**Priority:** P0 | **Dependencies:** 2.1

**Implementation Steps:**

1. Implement validation for: sampling rate consistent with the audit findings, expected number of channels, and minimum signal length.

**Acceptance Criteria:**

* [ ] AC1: Given a file with a sampling rate different from the one confirmed in the audit, when it is loaded, then the loader raises an explicit error and does not silently process it.

**Testing:** Unit tests with valid/invalid file fixtures.

**Definition of Done:** Validators implemented and tested.

**Files/Modules Expected:** `backend/app/datasets/validators.py`.

---

### TASK 2.4 — Windowing (Signal Segmentation) with Split Preservation

**Priority:** P0 | **Dependencies:** 2.2 | **Blocks:** Phase 3, 5, 6, 7

**Implementation Steps:**

1. Implement a windowing function (`window_size`, `overlap` configurable) that operates **within** a single recording and never crosses a recording boundary.

2. Windows inherit the split of their source recording.

**Acceptance Criteria:**

* [ ] AC1 (CRITICAL): Given two recordings assigned to different splits, when windows are generated, then no window contains data from both recordings.

* [ ] AC2: The configured overlap (e.g. 50%) produces the expected number of windows for a known signal length (mathematically verified).

**Testing:** Dedicated unit test specifically for AC1 (the most important leakage test in the entire project), plus a test for the number of generated windows.

**Definition of Done:** Windowing implemented, with the leakage test explicitly passing.

**Files/Modules Expected:** `backend/app/signal_processing/windowing.py`, `backend/tests/signal_processing/test_windowing.py`.

---

### TASK 2.5 — End-to-End Dataset Integration Tests

**Priority:** P1 | **Dependencies:** 2.2, 2.3, 2.4

**Acceptance Criteria:**

* [ ] AC1: The complete `file → loader → validation → windowing` pipeline runs on the real subset without errors.

* [ ] AC2: The total number of windows per split is reported and reasonable (not zero and not extremely imbalanced without explanation).

**Definition of Done:** Integration test passes on real data.

**Files/Modules Expected:** `backend/tests/dataset/test_integration.py`.

---

## PHASE 3 — Signal Processing (Preprocessing + Filtering)

### TASK 3.1 — Preprocessing: Detrending, Normalization, Standardization

**Priority:** P0 | **Dependencies:** 2.4 | **Blocks:** 3.2, 4.x, 5.x

**Architecture Note (avoids confusion with TASK 5.4):** This task normalizes/standardizes the **raw signal** (per window, before filtering/FFT), not feature vectors. It is a separate step with a different purpose from feature scaling in TASK 5.4 (which operates on extractor output rather than the signal). The two must not be confused or redundantly applied on top of each other.

**Implementation Steps:**

1. Implement `detrend()` (`scipy.signal.detrend`, linear).

2. Implement `normalize()` (min-max) and `standardize()` (z-score), with parameters fit **only on the training set** (according to the decision in section 8 of the blueprint).

**Acceptance Criteria:**

* [ ] AC1: Given a signal with a known synthetic linear trend added, when detrending is applied, then the residual trend remains below a defined tolerance (e.g. residual slope < 1e-3).

* [ ] AC2: Given standardization parameters calculated on the training set, when they are applied to the test set, then the resulting test mean is NOT necessarily 0 (explicit verification that the scaler was not refit on the test set — anti-leakage check).

**Testing:** `backend/tests/signal_processing/test_preprocessing.py`.

**Definition of Done:** Functions implemented and tested, with no fitting on test/validation data.

**Files/Modules Expected:** `backend/app/signal_processing/preprocessing.py`.

---

### TASK 3.2 — Butterworth Filters (Low-Pass, High-Pass, Band-Pass) + filtfilt

**Priority:** P0 | **Dependencies:** 3.1 | **Blocks:** 4.x

**Implementation Steps:**

1. Implement `butter_filter(signal, cutoff, fs, order, btype)` using `scipy.signal.butter` + `filtfilt` (zero-phase).

2. Parameters (`cutoff`, `order`) must be arguments, not hard-coded constants.

3. Validation: `cutoff < Nyquist (fs/2)`, otherwise raise an explicit error.

**Acceptance Criteria:**

* [ ] AC1: Given a synthetic signal = 10 Hz + 200 Hz components, when a low-pass filter with a 50 Hz cutoff is applied, then the 200 Hz component is attenuated by at least 20 dB (verified using FFT on the filtered signal).

* [ ] AC2: Given `cutoff ≥ Nyquist`, when the function is called, then an explicit error is raised rather than silently producing an incorrect result.

* [ ] AC3: Filtering with `filtfilt` does not introduce phase shift (verified by checking that the position of a known peak remains at the same index ±1 sample).

**Testing:** `backend/tests/signal_processing/test_filtering.py` with synthetic multi-component signals.

**Definition of Done:** All three filter types implemented and tested with measurable acceptance criteria.

**Files/Modules Expected:** `backend/app/signal_processing/filtering.py`.

---

## PHASE 4 — Spectral Analysis

### TASK 4.1 — FFT + Frequency Axis + Magnitude Spectrum

**Priority:** P0 | **Dependencies:** 3.1 | **Blocks:** 4.3, 5.2, 9.x

**Implementation Steps:**

1. Implement `compute_fft(signal, fs)` → returns frequencies and magnitude, using `numpy.fft.rfft` (real-valued signal) + `numpy.fft.rfftfreq`.

2. Implement `dominant_frequency(freqs, magnitude)`.

**Acceptance Criteria:**

* [ ] AC1: Given a synthetic 50 Hz sinusoidal signal sampled at 1000 Hz, when FFT is calculated, then the detected dominant frequency is within ±1 Hz of 50 Hz.

* [ ] AC2: Given `fs = 1000 Hz`, the maximum frequency in the returned frequency axis does not exceed the Nyquist frequency (500 Hz).

**Testing:** `backend/tests/signal_processing/test_fft.py`.

**Definition of Done:** FFT + dominant frequency implemented and tested on a known synthetic signal.

**Files/Modules Expected:** `backend/app/signal_processing/fft.py`.

---

### TASK 4.2 — Welch PSD

**Priority:** P0 | **Dependencies:** 4.1 | **Blocks:** 4.3, 5.2

**Implementation Steps:**

1. Implement a wrapper around `scipy.signal.welch`, with configurable `nperseg` / `noverlap`.

**Acceptance Criteria:**

* [ ] AC1: Given the same synthetic 50 Hz signal with added white noise, when Welch PSD is calculated, then the dominant peak remains at 50 Hz ±1 Hz, with visibly lower variance than a simple FFT on the same noisy signal (explicit comparison in the test).

**Testing:** `backend/tests/signal_processing/test_psd.py`.

**Definition of Done:** Welch PSD implemented and compared against a simple FFT in a test.

**Files/Modules Expected:** `backend/app/signal_processing/psd.py`.

---

### TASK 4.3 — STFT / Spectrogram

**Priority:** P0 | **Dependencies:** 4.1 | **Blocks:** 12.x (PCA/visualization later)

**Implementation Steps:**

1. Implement a wrapper around `scipy.signal.spectrogram`, with configurable window size and hop length.

**Acceptance Criteria:**

* [ ] AC1: Given a synthetic signal whose frequency changes halfway through the duration (50 Hz → 150 Hz), when the spectrogram is calculated, then the dominant energy visibly shifts from 50 Hz to 150 Hz between the first and second halves of the time axis (verified using the frequency index with maximum magnitude per time column).

* [ ] AC2: Increasing the window size increases frequency resolution and decreases time resolution (verified by comparing the dominant-band width for two different configurations).

**Testing:** `backend/tests/signal_processing/test_spectrogram.py`.

**Definition of Done:** STFT implemented, with the time/frequency resolution trade-off verified through an explicit test.

**Files/Modules Expected:** `backend/app/signal_processing/spectrogram.py`.

---

## PHASE 5 — Feature Engineering

### TASK 5.1 — Feature Registry + Time-Domain Features

**Priority:** P0 | **Dependencies:** 3.1 | **Blocks:** 5.3, 6.x, 7.x, 9.x

**Description:** Implement mean, standard deviation, variance, RMS, peak, peak-to-peak, skewness, kurtosis, and crest factor as pure functions registered in an extensible registry.

**Implementation Steps:**

1. `features/time_domain.py` — one function per feature, using the uniform signature `f(signal: np.ndarray) -> float`.

2. `features/registry.py` — dictionary `{name: function}`, extensible by adding an entry without modifying existing code.

**Acceptance Criteria (per feature, verified using synthetic signals with known properties):**

* [ ] AC1: RMS on a sinusoidal signal with amplitude A is approximately A/√2 (±1% tolerance).

* [ ] AC2: Kurtosis on pure Gaussian noise is approximately 3 (non-excess definition) or approximately 0 (excess kurtosis) — the convention used must be explicitly specified, with a documented tolerance.

* [ ] AC3: Crest factor on a signal with rare, sharp peaks is visibly higher than on a pure sinusoid with the same RMS power.

* [ ] AC4: Each feature has a dedicated test using known input and analytically calculated expected output.

**Testing:** `backend/tests/features/test_time_domain.py` — one test per feature.

**Definition of Done:** All 9 features implemented, each with an individual passing test, and documented (formula + physical interpretation + fault relevance) in the docstring.

**Files/Modules Expected:** `backend/app/features/time_domain.py`, `backend/app/features/registry.py`.

---

### TASK 5.2 — Frequency-Domain Features

**Priority:** P0 | **Dependencies:** 4.1, 4.2, 5.1 | **Blocks:** 5.3

**Description:** Implement dominant frequency, spectral centroid, spectral bandwidth, spectral energy, spectral entropy, and energy in frequency bands.

**Acceptance Criteria:**

* [ ] AC1: Given a pure sinusoidal signal, spectral entropy is close to its minimum (signal concentrated around one frequency), compared with white noise where entropy is close to its maximum — explicitly tested as a comparison.

* [ ] AC2: Spectral centroid on a signal with energy concentrated at a known frequency X is approximately X.

* [ ] AC3: Each feature has a dedicated test using a synthetic signal.

**Testing:** `backend/tests/features/test_frequency_domain.py`.

**Definition of Done:** All 6 features implemented and individually tested.

**Files/Modules Expected:** `backend/app/features/frequency_domain.py`.

---

### TASK 5.3 — Complete Feature Extractor (Feature Matrix)

**Priority:** P0 | **Dependencies:** 5.1, 5.2 | **Blocks:** Phase 6, 7, 9

**Implementation Steps:**

1. `extractor.py` — receives a signal window, runs all functions from the registry, and returns a feature vector/row with consistent column names.

2. Apply it to all windows → feature matrix (train/validation/test separately).

**Acceptance Criteria:**

* [ ] AC1: Given the same signal window, when processed twice, the output is identical (deterministic, with no hidden randomness).

* [ ] AC2: The feature matrix for the training split contains no NaN/Inf values (explicitly verified).

* [ ] AC3: The number of columns = the number of features in the registry (automatically verified, not hard-coded).

**Testing:** `backend/tests/features/test_extractor.py`.

**Definition of Done:** Complete extractor, tested on real data (subset).

**Files/Modules Expected:** `backend/app/features/extractor.py`.

---

### TASK 5.4 — Feature Scaling (Shared Component, NOT Model-Specific)

**Priority:** P0 | **Dependencies:** 5.3 | **Blocks:** 6.2, 7.2, 9.x

**Description:** Important architectural distinction to prevent double scaling or hidden dependencies between models:

```text
Signal preprocessing (Phase 3)        Feature scaling (HERE, Phase 5)

    detrend                              StandardScaler on FEATURE VECTORS
    normalize/standardize RAW SIGNAL    (output from TASK 5.3),
    (optional, per window)                fit only on train
             │                                  │
             ▼                                  ▼
    used for filtering/FFT/PSD          used as DIRECT INPUT
    (Phase 3-4), NOT the same step      for Isolation Forest (6.2)
    as the scaling below                 AND Autoencoder (7.2)
```

The scaler is a **shared** component consumed identically by both models — it must not live inside `app/ml/isolation_forest.py`, otherwise the Autoencoder could accidentally become dependent on IF-specific code.

**Implementation Steps:**

1. `app/ml/scaling.py` — `fit_scaler(train_features) -> scaler`, `apply_scaler(scaler, features) -> scaled_features`.

2. Fit the scaler once (on train), save it as a separate artifact (`scaler_v1.pkl`), and reference it through `scaler_artifact` in the Model Artifact Contract (TASK 6.5) for **both** models.

**Acceptance Criteria:**

* [ ] AC1: `StandardScaler` (or equivalent) is fit exclusively on the training set and applied (`transform`) to validation/test — verified through a test that would fail if the scaler were fit on the entire dataset.

* [ ] AC2: Code in `app/ml/isolation_forest.py` and `app/ml/autoencoder.py` / `training.py` imports the **same** `scaling.py` module rather than using duplicated implementations.

**Testing:** `backend/tests/ml/test_scaling.py`.

**Definition of Done:** Scaler implemented exactly once, saved as a reusable artifact, and consumed identically by Phase 6 and Phase 7.

**Files/Modules Expected:** `backend/app/ml/scaling.py`.

---

# EPIC 3 — Machine Learning & Deep Learning


## PHASE 6 — Isolation Forest Baseline

### TASK 6.2 — Train Isolation Forest (normal data only)

**Priority:** P0 | **Dependencies:** 5.4 | **Blocks:** 6.3, 8.x, 9.x

**Description:** The model is trained exclusively on windows labeled "normal" from the train split (according to the methodology defined in Section 5 of the blueprint — unsupervised/semi-supervised training).

**Acceptance Criteria:**

* [ ] AC1 (CRITICAL): Given the train set, when filtering the training data, then no example with a label other than "normal" reaches `.fit()` — verified through an explicit test that counts the labels in the actual training set used.

* [ ] AC2: The trained model produces scores (`decision_function`) for any new window without shape errors.

* [ ] AC3: Fixed seed → repeated training runs produce identical scores (reproducibility).

**Testing:** `backend/tests/ml/test_isolation_forest_training.py`.

**Definition of Done:** Model trained and saved, with the AC1 test (label anti-leakage) explicitly passing.

**Files/Modules Expected:** `backend/app/ml/isolation_forest.py`.

---

### TASK 6.3 — Scoring, Score Normalization, Threshold Calibration (Configurable Method)

**Priority:** P0 | **Dependencies:** 6.2 | **Blocks:** 8.x, 6.5

**Implementation Steps:**

1. Normalize the raw score to [0,1] using the score distribution from the **validation set**.

2. Implement threshold calibration as a **selectable strategy**, not a fixed value: `threshold_method` can be `"percentile"` (configurable `percentile_value` parameter, e.g. 95) or `"validation_f1_optimal"` (select the threshold that maximizes F1 on the validation set, if fault examples exist in validation). The default method and the reasoning behind its selection must be explicitly documented and must not be presented as a "universal truth".

**Acceptance Criteria:**

* [ ] AC1: The threshold is calculated from the validation set, not the test set (verified through code — the calibration function never receives test data as input).

* [ ] AC2: The normalized score is always in [0,1] for any input (verified using extreme synthetic data).

* [ ] AC3: The calibration function accepts the `threshold_method` parameter and produces different, verifiable results for the two implemented methods (explicitly tested, not assumed).

**Testing:** `backend/tests/ml/test_scoring.py`.

**Definition of Done:** Scoring function + threshold calibrated using the selected method, saved as part of the model artifact (see TASK 6.5).

**Files/Modules Expected:** `backend/app/ml/scoring.py`.

---

### TASK 6.4 — Model Persistence (Save/Load) + Inference

**Priority:** P0 | **Dependencies:** 6.2, 6.3 | **Blocks:** 10.x

**Acceptance Criteria:**

* [ ] AC1: Given a saved model, when it is reloaded, then predictions on the same input are identical to those produced before saving (bit-exact or within a documented numerical tolerance).

**Testing:** `backend/tests/ml/test_persistence.py`.

**Definition of Done:** Save/load functionality implemented and tested.

**Files/Modules Expected:** `backend/app/ml/inference.py`, `models/isolation_forest_v1.pkl`.

---

### TASK 6.5 — Model Artifact Contract (Standard Metadata, Shared by IF + AE)

**Priority:** P0 | **Dependencies:** 6.3, 6.4 | **Blocks:** 7.3 (reuses the same contract), 9.x, 10.5

**User Story:** As an ML engineer, I want every saved model to be accompanied by an explicit metadata contract so that any subsequent prediction (or audit) can be traced exactly to the configuration that produced the model, without assumptions.

**Description:** Each model artifact (`.pkl` for Isolation Forest, `.pt` for Autoencoder — see Phase 7) is accompanied by a JSON sidecar file with the same base name, containing:

```json
{
  "model_type": "isolation_forest",
  "model_version": "v1",
  "feature_names": ["rms", "kurtosis", "crest_factor", "..."],
  "feature_dimension": 15,
  "scaler_artifact": "scaler_v1.pkl",
  "threshold_method": "percentile",
  "threshold_value": 0.73,
  "score_direction": "higher_is_more_anomalous",
  "training_split": "train",
  "dataset_hash": "sha256:...",
  "split_manifest_hash": "sha256:...",
  "random_seed": 42,
  "created_at": "ISO8601 timestamp"
}
```

**Implementation Steps:**

1. Implement `app/ml/model_artifact.py` with a function `save_model_artifact(model, metadata: ModelArtifactMetadata, path)` and `load_model_artifact(path) -> (model, metadata)`.

2. `dataset_hash` and `split_manifest_hash` must be calculated (e.g. SHA-256 over the contents of `split_manifest.json` from TASK 1.5.7) — never invented and never omitted.

3. `score_direction` exists explicitly because Isolation Forest (`decision_function`, lower score = more anomalous) and Autoencoder (reconstruction error, higher score = more anomalous) have different native conventions — the contract normalizes this for consumers in Phase 10/11.

**Acceptance Criteria:**

* [ ] AC1: Given a saved model, when reading the JSON sidecar file, then all fields from the schema above are present and non-null.

* [ ] AC2: `dataset_hash` / `split_manifest_hash` are calculated from the actual source file (verified: changing a single character in `split_manifest.json` changes the hash).

* [ ] AC3: Given two models trained with different configurations (e.g. different `threshold_method`), the metadata correctly reflects the difference — it is not a static copied artifact.

**Testing:** `backend/tests/ml/test_model_artifact.py`.

**Definition of Done:** Contract implemented, applied to Isolation Forest (Phase 6), and reused identically for Autoencoder (Phase 7), with no duplicated schema.

**Files/Modules Expected:** `backend/app/ml/model_artifact.py`, `backend/tests/ml/test_model_artifact.py`.

---

## PHASE 7 — Autoencoder (PyTorch, on Feature Vectors)

### TASK 7.1 — Encoder/Decoder Architecture

**Priority:** P0 | **Dependencies:** 5.3 | **Blocks:** 7.2

**Implementation Steps:**

1. Define `Autoencoder(nn.Module)` — dense encoder (input_dim → hidden → bottleneck), symmetric decoder.

2. Dimensions must be configurable parameters, not hard-coded.

**Acceptance Criteria:**

* [ ] AC1: Given a batch with shape `(N, num_features)`, the forward pass returns an output with the identical shape `(N, num_features)`.

* [ ] AC2: The number of network parameters is explicitly documented (verify that the architecture remains "easy to explain" and is not excessively large, according to Section 16 of the blueprint).

**Testing:** `backend/tests/ml/test_autoencoder_architecture.py`.

**Definition of Done:** Architecture implemented and shape-tested.

**Files/Modules Expected:** `backend/app/ml/autoencoder.py`.

---

### TASK 7.2 — Training Loop + Validation + Early Stopping

**Priority:** P0 | **Dependencies:** 7.1, 5.4 | **Blocks:** 7.3

**Implementation Steps:**

1. DataLoader over normalized feature vectors (train = "normal" only, following the same rule as Isolation Forest).

2. Loss = reconstruction MSE. Optimizer = Adam.

3. Early stopping based on validation loss.

4. Fixed seed for reproducibility.

**Acceptance Criteria:**

* [ ] AC1: Final training loss is significantly lower than the loss in the first epoch (monotonic decrease is not required — loss may fluctuate from epoch to epoch and the model remains fully valid; the wrong criterion would reject perfectly normal training runs). Training converges without `NaN`/`Inf` at any step.

* [ ] AC2: **Validation** loss is tracked separately from training loss throughout the entire training process (not only training loss) — required so that early stopping can detect real overfitting rather than training noise.

* [ ] AC3: Training never sees non-normal examples (same verification as 6.2-AC1).

* [ ] AC4: Running with the same seed produces identical loss curves.

**Testing:** `backend/tests/ml/test_autoencoder_training.py`.

**Definition of Done:** Functional, reproducible training loop with checkpointing implemented.

**Files/Modules Expected:** `backend/app/ml/training.py`.

---

### TASK 7.3 — Reconstruction Error + Threshold + Persistence

**Priority:** P0 | **Dependencies:** 7.2, 6.5 | **Blocks:** 8.x

**Acceptance Criteria:**

* [ ] AC1: Given validation data (known normal + anomaly examples), the mean reconstruction error on anomalies is visibly higher than on normal examples (reported, not assumed — if this is not true, it must be reported honestly, according to the "do not invent results" rule).

* [ ] AC2: Threshold is calibrated using the same configurable strategy from TASK 6.3 (`percentile` or `validation_f1_optimal`), on validation data, not test data.

* [ ] AC3: A saved/reloaded model produces identical reconstructions.

* [ ] AC4: The saved artifact follows **exactly the same metadata contract from TASK 6.5** (`model_type: "autoencoder"`, `score_direction: "higher_is_more_anomalous"`, remaining fields identical in structure) — no separate/duplicated schema for Autoencoder.

**Testing:** `backend/tests/ml/test_autoencoder_inference.py`.

**Definition of Done:** Complete Autoencoder pipeline functional, tested, and saved.

**Files/Modules Expected:** `backend/app/ml/inference.py` (extended), `models/autoencoder_v1.pt`.

---

## PHASE 8 — Model Evaluation Framework

### TASK 8.1 — Common Evaluation Framework (IF + AE on the Same Test Set)

**Priority:** P0 | **Dependencies:** 6.4, 7.3 | **Blocks:** Phase 9

**Implementation Steps:**

1. Implement `evaluate(model, test_features, test_labels) -> metrics dict`, used identically for both models.

2. Calculate: precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, FPR, FNR, inference time.

**Acceptance Criteria:**

* [ ] AC1: Given the same test set for both models, when running evaluation, then both use exactly the same metric calculation function (verified through code, with no duplicated logic).

* [ ] AC2: All metrics are calculated from real predictions, not hard-coded — explicit status `NOT YET MEASURED` until the first real run.

* [ ] AC3: The confusion matrix is generated correctly for the binary case (normal vs. aggregated anomaly) — verified using a manually calculated example with known predictions/labels.

**Testing:** `backend/tests/ml/test_evaluation.py` with synthetic predictions/labels where the correct result is calculated manually.

**Definition of Done:** Common evaluation framework implemented and tested, applied to both models with real reported results (not invented).

**Files/Modules Expected:** `backend/app/ml/evaluation.py`.

---

### TASK 8.2 — Comparative Report: Isolation Forest vs. Autoencoder

**Priority:** P0 | **Dependencies:** 8.1

**Acceptance Criteria:**

* [ ] AC1: The comparison table contains real measured values for both models on the same test set.

* [ ] AC2: The report conclusion does not assume a winner before seeing the results.

**Definition of Done:** `docs/results/isolation_forest_vs_autoencoder.md` generated with real data.

**Files/Modules Expected:** `docs/results/isolation_forest_vs_autoencoder.md`.

---

### TASK 8.3 — Experiment Run ID Registry

**Priority:** P0 | **Dependencies:** 8.1, 6.5 | **Blocks:** Phase 9 (all tasks)

**User Story:** As an ML engineer, I want every experiment run to receive a unique identifier so that results can be explicitly referenced (in reports, API, frontend) without ambiguity about which configuration produced them.

**Description:** Formalize a run ID on top of the structure already established in TASK 6.5 (Model Artifact Contract) and TASK 9.5 (aggregated report) — do not introduce a new schema, only an identifier linking the two. Each experiment in the A/B/C matrix (TASK 9.2–9.4) receives an `experiment_id` (e.g. `EXP-A-001`) at runtime, stored together with the metadata:

```json
{
  "experiment_id": "EXP-A-001",
  "dataset_hash": "sha256:...",
  "split_manifest_hash": "sha256:...",
  "representation": "raw_pca",
  "model": "isolation_forest",
  "feature_dimension": 15,
  "random_seed": 42,
  "preprocessing_config": {},
  "model_config": {},
  "threshold": {
    "method": "percentile",
    "value": 0.73
  },
  "metrics": {},
  "created_at": "ISO8601 timestamp"
}
```

**Implementation Steps:**

1. `app/ml/experiment_registry.py` — function `register_experiment_run(config, metrics) -> experiment_id`, with a conventional ID format (`EXP-{A|B|C}-{NNN}`).

2. Each run from TASK 9.2/9.3/9.4 calls this function instead of manually writing ad-hoc result files.

**Acceptance Criteria:**

* [ ] AC1: Every experiment in the A/B/C matrix has a unique `experiment_id`, generated automatically, not manually selected/hard-coded.

* [ ] AC2: Given an `experiment_id`, the complete associated configuration + metrics can be retrieved (round-trip test: save → read → identity).

* [ ] AC3: The frontend (TASK 11.7, Experiments page) displays the `experiment_id` next to each result in the matrix as an explicit reference (e.g. "Experiment A — Raw + PCA + Isolation Forest — Run: EXP-A-001").

**Testing:** `backend/tests/ml/test_experiment_registry.py`.

**Definition of Done:** Registry implemented and used by all three Phase 9 experiments, with no schema duplication relative to TASK 6.5.

**Files/Modules Expected:** `backend/app/ml/experiment_registry.py`.

---

# EPIC 4 — Central Experiment

## PHASE 9 — Central DSP vs Raw Experiment

### TASK 9.1 — Raw + PCA Representation (Controlled Dimensionality)

**Priority:** P0 | **Dependencies:** 2.4, 5.4 | **Blocks:** 9.4

**Implementation Steps:**

1. Extract raw windows (without DSP) for train/validation/test.

2. Fit PCA **only on train**, reducing to exactly the same number of components as the DSP feature vector (from TASK 5.3).

3. Apply the PCA transformation to validation/test.

**Acceptance Criteria:**

* [ ] AC1 (CRITICAL): PCA is fit exclusively on windows from the train split (verified through code — no execution path allows fitting on validation/test).

* [ ] AC2: Resulting dimensionality (raw + PCA) == DSP feature vector dimensionality, verified automatically.

* [ ] AC3: Explained variance of the retained components is explicitly reported (not hidden), even if it is low.

**Testing:** `backend/tests/experiments/test_raw_pca.py`.

**Definition of Done:** Raw + PCA representation generated for all splits, with dimensionality confirmed equal to DSP.

**Files/Modules Expected:** `backend/app/ml/dimensionality.py`.

---

### TASK 9.2 — Experiment A: Raw + PCA → Isolation Forest

**Priority:** P0 | **Dependencies:** 9.1, 6.2 (reused), 8.3 | **Blocks:** 9.4

**Acceptance Criteria:**

* [ ] AC1: Training is methodologically identical to 6.2 (normal data only from train), but using the raw + PCA representation.

* [ ] AC2: Evaluated using exactly the same framework from 8.1, on the same test set (same windows) as Experiment B.

* [ ] AC3: The run receives a unique `experiment_id` through the registry from TASK 8.3 (e.g. `EXP-A-001`).

**Definition of Done:** Experiment A model + metrics generated, status `NOT YET MEASURED` → replaced with real values after execution.

**Files/Modules Expected:** `backend/scripts/run_experiment_a.py`, `models/experiment_a_isolation_forest.pkl`.

---

### TASK 9.3 — Experiment B: DSP Features → Isolation Forest

**Priority:** P0 | **Dependencies:** 5.3, 6.2, 8.3 | **Blocks:** 9.4

**Description:** Directly reuse the model from Phase 6 (this is already exactly this experiment) — the task here is only to explicitly place it in the comparison matrix, assign it an `experiment_id` (e.g. `EXP-B-001`), and generate the associated report.

**Acceptance Criteria:**

* [ ] AC1: The model and metrics from Phase 6/8 are referenced directly, without unnecessary duplicate retraining.

* [ ] AC2: The run has a unique `experiment_id` registered through TASK 8.3.

**Definition of Done:** Experiment B documented as a reference to Phase 6/8, with an assigned `experiment_id`.

---

### TASK 9.4 — Experiment C: DSP Features → Autoencoder

**Priority:** P0 | **Dependencies:** 7.3, 8.3 | **Blocks:** 9.5

**Description:** Directly reuse the model from Phase 7, with its own `experiment_id` (e.g. `EXP-C-001`).

**Acceptance Criteria:**

* [ ] AC1: Metrics from Phase 7/8 are directly referenced in the comparison matrix.

* [ ] AC2: The run has a unique `experiment_id` registered through TASK 8.3.

**Definition of Done:** Experiment C documented as a reference to Phase 7/8, with an assigned `experiment_id`.

---

### TASK 9.5 — Result Aggregation + Final Central Experiment Report

**Priority:** P0 | **Dependencies:** 9.2, 9.3, 9.4 | **Blocks:** Phase 12 (UI), Phase 13 (README)

**Implementation Steps:**

1. Build the final table: Experiment A/B/C × (Precision, Recall, F1, ROC-AUC, PR-AUC).

2. Write the interpretation: explain what the result means, regardless of which variant "wins".

3. Include a visualization (e.g. comparative bar chart) for the frontend (Phase 12).

4. For each of the three experiments, save a reproducibility artifact (JSON) containing: `dataset_hash`, `split_manifest_hash` (same as in TASK 6.5), `preprocessing_config` (detrend/normalization parameters used), `feature_set` (exact feature list for B and C), `model_config` (hyperparameters), `random_seed`, `threshold_method` + `threshold_value`, complete metrics from TASK 8.1, `timestamp`.

**Acceptance Criteria:**

* [ ] AC1: All three cells of the matrix contain real measured values, not placeholders, at the time the task is completed.

* [ ] AC2: The report explicitly discusses whether the result confirms or refutes the initial hypothesis (Section 19 of the blueprint), without forcing a predefined conclusion.

* [ ] AC3: If Experiment A (raw + PCA) is comparable to or better than B, the report explicitly discusses the alternative interpretation (DSP value = interpretability, not necessarily raw performance).

* [ ] AC4: Each of the three experiments has a complete, verifiable JSON reproducibility artifact — if someone reruns the same experiment using the saved artifact as configuration input, they obtain identical metrics (explicit reproducibility test, not an assumption).

**Definition of Done:** `docs/results/central_experiment_report.md` complete with real data.

**Files/Modules Expected:** `docs/results/central_experiment_report.md`, `backend/scripts/aggregate_experiment_results.py`.

---


# EPIC 5 — API & Backend Integration

## PHASE 10 — FastAPI

### TASK 10.1 — Pydantic Schemas (Request/Response) for All Endpoints

**Priority:** P0 | **Dependencies:** Phase 2-9 (existing models) | **Blocks:** 10.2–10.8

**Acceptance Criteria:**

* [ ] AC1: Every schema explicitly rejects a payload with an incorrect data type (e.g. `sampling_rate: str` instead of `float`), verified through tests.

**Testing:** `backend/tests/api/test_schemas.py`.

**Files/Modules Expected:** `backend/app/api/schemas/*.py`.

---

### TASK 10.2 — Dataset Endpoints (`GET /api/datasets`, `GET /api/datasets/{id}`)

**Priority:** P0 | **Dependencies:** 10.1, 2.x

**Acceptance Criteria:**

* [ ] AC1: Given an existing dataset, when `GET /api/datasets/{id}` is called, then the response contains real metadata (samples, sampling rate, channels) from the audit, not placeholder values.

* [ ] AC2: Given a non-existent ID, then HTTP 404 is returned with a clear error message.

**Testing:** `backend/tests/api/test_datasets.py`.

**Files/Modules Expected:** `backend/app/api/routes/datasets.py`, `backend/app/services/dataset_service.py`.

---

### TASK 10.3 — Signal Processing Endpoints (`/fft`, `/psd`, `/spectrogram`, `/dsp/filter`)

**Priority:** P0 | **Dependencies:** 10.1, Phase 3-4

**Acceptance Criteria:**

* [ ] AC1: Given a valid signal + filter parameters, when `POST /api/dsp/filter` is called, then the response contains the filtered signal with the same properties already verified in the Phase 3 tests (reuse existing logic, do not reimplement it).

* [ ] AC2: Given invalid parameters (cutoff ≥ Nyquist), then HTTP 422 is returned with an explicit message, not HTTP 500.

**Testing:** `backend/tests/api/test_signal_processing_routes.py`.

**Files/Modules Expected:** `backend/app/api/routes/signals.py`, `backend/app/services/dsp_service.py`.

---

### TASK 10.4 — Feature Extraction Endpoint (`POST /api/features/extract`)

**Priority:** P0 | **Dependencies:** 10.1, 5.3

**Acceptance Criteria:**

* [ ] AC1: The response contains exactly the number of features defined in the registry, with consistent column names.

**Testing:** `backend/tests/api/test_features_route.py`.

**Files/Modules Expected:** `backend/app/api/routes/features.py`.

---

### TASK 10.5 — Model Endpoints (`GET /api/models`, `/api/models/{id}/performance`, `POST /api/models/predict`)

**Priority:** P0 | **Dependencies:** 10.1, Phase 6-8

**Acceptance Criteria:**

* [ ] AC1: `/api/models` lists Isolation Forest and Autoencoder with real metrics from Phase 8 (not hard-coded in the route).

* [ ] AC2: `POST /api/models/predict` returns a normalized anomaly score [0,1] + status (NORMAL/WARNING/ANOMALY) + a text explanation based on real features calculated for the given signal (not generic text).

**Testing:** `backend/tests/api/test_models_route.py`.

**Files/Modules Expected:** `backend/app/api/routes/models.py`, `backend/app/services/model_service.py`.

---

### TASK 10.6 — Experiment Endpoints (`GET /api/experiments`, `/api/experiments/{id}`)

**Priority:** P1 | **Dependencies:** 10.1, 9.5

**Acceptance Criteria:**

* [ ] AC1: Returns the A/B/C experiment matrix with real values from `central_experiment_report.md` / the associated JSON artifact.

**Testing:** `backend/tests/api/test_experiments_route.py`.

**Files/Modules Expected:** `backend/app/api/routes/experiments.py`.

---

### TASK 10.7 — Global Error Handling + Centralized Validation

**Priority:** P0 | **Dependencies:** 10.2–10.6

**Acceptance Criteria:**

* [ ] AC1: Any unhandled exception produces HTTP 500 with a structured JSON body (no stack trace exposed in production), and is logged server-side.

* [ ] AC2: Pydantic validation errors produce HTTP 422 with field-level details.

**Testing:** `backend/tests/api/test_error_handling.py`.

**Files/Modules Expected:** `backend/app/api/error_handlers.py`.

---

### TASK 10.8 — OpenAPI Documentation + Full Verification

**Priority:** P1 | **Dependencies:** 10.2–10.7

**Acceptance Criteria:**

* [ ] AC1: `GET /docs` displays all endpoints with the correct request/response schemas.

**Definition of Done:** Complete OpenAPI documentation, manually verified.

---

# EPIC 6 — Frontend

## PHASE 11 — Frontend

### TASK 11.1 — Application Shell (Sidebar, Navigation, Layout)

**Priority:** P0 | **Dependencies:** 1.5, 1.6 | **Blocks:** 11.2–11.9

**Acceptance Criteria:**

* [ ] AC1: Navigation between all pages (Dashboard, Signals, Signal Analysis, DSP Lab, Anomaly Detection, Models, Experiments, Dataset) works without a full page reload (React Router, client-side routing).

* [ ] AC2: Responsive layout — the sidebar collapses correctly below a defined breakpoint (manually verified at 2 screen widths).

**Files/Modules Expected:** `frontend/src/layouts/AppShell.tsx`, `frontend/src/pages/*.tsx` (scaffolds).

---

### TASK 11.2 — Dashboard Page

**Priority:** P0 | **Dependencies:** 11.1, 10.5

**Acceptance Criteria:**

* [ ] AC1: Given a backend with real data, the Dashboard displays the current anomaly score, status (NORMAL/WARNING/ANOMALY), threshold used, and number of signals analyzed, all coming from the API rather than hard-coded. **A "model confidence %" must not be displayed** — neither Isolation Forest nor the Autoencoder in this project produces a calibrated probability; displaying an unexplained confidence percentage would amount to inventing a number.

* [ ] AC2: A loading state is displayed while the API request is in progress; an error state is displayed if the API request fails.

**Testing:** Manual verification + optional component test (React Testing Library) for loading/error states.

**Files/Modules Expected:** `frontend/src/pages/Dashboard.tsx`.

---

### TASK 11.3 — Signal Analysis Page (Tabs: Time/Frequency/Spectrogram/Features/AI Analysis)

**Priority:** P0 | **Dependencies:** 11.1, 10.3, 10.4, 10.5

**Acceptance Criteria:**

* [ ] AC1: The time-domain chart supports zoom and hover interactions (manually verified — Plotly built-in functionality).

* [ ] AC2: Switching between tabs preserves the selected signal (does not reset the user's selection).

**Files/Modules Expected:** `frontend/src/pages/SignalAnalysis.tsx`, `frontend/src/components/charts/*.tsx`.

---

### TASK 11.4 — DSP Lab Page (Configurable Filter Parameters, Raw vs. Filtered Comparison)

**Priority:** P0 | **Dependencies:** 11.1, 10.3

**Acceptance Criteria:**

* [ ] AC1: Changing the filter cutoff value in the UI triggers a new API request and updates the chart without reloading the page.

* [ ] AC2: The UI visibly explains Nyquist frequency, cutoff, and filter order (short explanatory text, not just a slider without context) — according to the requirement in Section 11 of the blueprint.

**Files/Modules Expected:** `frontend/src/pages/DSPLab.tsx`.

---

### TASK 11.5 — Models Page

**Priority:** P1 | **Dependencies:** 11.1, 10.5

**Acceptance Criteria:**

* [ ] AC1: Both models (Isolation Forest, Autoencoder) are displayed with real metrics from the API.

* [ ] AC2: Selecting the active model persists at least within the current session state and affects the Anomaly Detection page.

**Files/Modules Expected:** `frontend/src/pages/Models.tsx`.

---

### TASK 11.6 — Anomaly Detection / AI Analysis View

**Priority:** P0 | **Dependencies:** 11.1, 10.5

**Acceptance Criteria:**

* [ ] AC1: Given an analyzed signal, the UI displays the anomaly score, threshold used, status, and text indicators based on real feature values (not generic hard-coded text). **It must not display a "confidence %"** — this is mathematically undefined for the models used in this project (see the note in TASK 11.2).

* [ ] AC2: UI wording explicitly follows the clarification from Section 17 of the blueprint — the score is presented as "how unusual the signal is", not as a "% physical failure".

* [ ] AC3: The UI displays a short interpretation (e.g. "High anomaly score compared with the learned normal baseline"), not just the raw number.

**Files/Modules Expected:** `frontend/src/components/anomaly/AnomalyPanel.tsx`.

---

### TASK 11.7 — Experiments Page (A/B/C Matrix)

**Priority:** P1 | **Dependencies:** 11.1, 10.6

**Acceptance Criteria:**

* [ ] AC1: Displays the Representation × Model matrix from Section 19 of the blueprint, with real values from the API.

**Files/Modules Expected:** `frontend/src/pages/Experiments.tsx`.

---

### TASK 11.8 — Dataset Page

**Priority:** P1 | **Dependencies:** 11.1, 10.2

**Acceptance Criteria:**

* [ ] AC1: Displays real metadata from the audit (samples, sampling rate, duration, channels, missing values) — no placeholder values.

**Files/Modules Expected:** `frontend/src/pages/Dataset.tsx`.

---

### TASK 11.9 — Consistent Global Loading/Error/Empty States

**Priority:** P1 | **Dependencies:** 11.2–11.8

**Acceptance Criteria:**

* [ ] AC1: Every page that fetches data implements all three states (not only the happy path).

**Files/Modules Expected:** `frontend/src/components/ui/LoadingState.tsx`, `ErrorState.tsx`, `EmptyState.tsx`.

---

# EPIC 7 — Explainability & Polish

## PHASE 12 — Explainability + 3D PCA + Polish

### TASK 12.1 — Anomaly Explanation Based on Real Features

**Priority:** P0 | **Dependencies:** 11.6, 5.3

**Acceptance Criteria:**

* [ ] AC1: Given an anomalous signal, the explanation lists features with significant deviation from the normal baseline (calculated, not invented) — e.g. "RMS +32% vs. normal baseline", with the percentage calculated from real data.

* [ ] AC2: UI text must not describe the score as a "physical damage percentage" (according to the explicit blueprint constraint).

**Files/Modules Expected:** `backend/app/services/explanation_service.py`, `frontend/src/components/anomaly/ExplanationList.tsx`.

---

### TASK 12.2 — PCA 3D Feature Space Visualization

**Priority:** P1 | **Dependencies:** 9.1, 11.7

**Acceptance Criteria:**

* [ ] AC1: Points in the 3D visualization are colored according to the real class (normal/fault types) from the dataset, not randomly.

* [ ] AC2: The UI explicitly states that PCA is used for visualization/dimensionality reduction, not as proof of guaranteed separability (according to Section 21 of the blueprint).

**Files/Modules Expected:** `frontend/src/components/charts/PCA3DPlot.tsx`.

---

### TASK 12.3 — Final Visual Polish (Design Token Consistency, Subtle Animations)

**Priority:** P2 | **Dependencies:** all Phase 11 tasks

**Acceptance Criteria:**

* [ ] AC1: Zero hard-coded colors outside `globals.css` (verified manually/through grep).

---

# EPIC 8 — Final Validation

## PHASE 13 — Final Validation & Documentation

### TASK 13.1 — CWRU Loader + Cross-Dataset Validation

**Priority:** P1 | **Dependencies:** Phase 6-9 stable

**Acceptance Criteria:**

* [ ] AC1: Models trained on MAFAULDA are evaluated on CWRU (using equivalent feature extraction), with results reported honestly regardless of whether generalization is good or poor.

**Notes:** **This dataset must also be uploaded by the user** — same network constraint as MAFAULDA.

**Files/Modules Expected:** `backend/app/datasets/cwru_loader.py`, `docs/results/cross_dataset_validation.md`.

---

### TASK 13.2 — Complete Final README

**Priority:** P0 | **Dependencies:** all previous phases

**Acceptance Criteria:**

* [ ] AC1: README contains all sections from Section 24 of the blueprint (Overview, Architecture, Signal Processing, ML, DL, Evaluation, Results, Installation, Usage, Future Work), plus the dedicated "Data Leakage Prevention" section.

* [ ] AC2: No performance number in the README is invented — all performance figures come from `docs/results/*.md`.

**Files/Modules Expected:** `README.md`.

---

### TASK 13.3 — Cleanup, Screenshots, Interview Documentation

**Priority:** P2 | **Dependencies:** 13.2

**Acceptance Criteria:**

* [ ] AC1: `docs/interview_prep.md` contains real answers (not generic ones) to the questions from Section 36 of the blueprint, referencing concrete decisions made in the project.

**Files/Modules Expected:** `docs/interview_prep.md`, `docs/screenshots/`.


# DEPENDENCY GRAPH (summary by Phase, not by individual task)

```text
Phase 1 (Setup)

   │
   ▼
Phase 1.5 (Dataset Audit) ◄── BLOCKER: requires MAFAULDA upload

   │
   ▼
Phase 2 (Dataset Integration) ◄── depends on split_manifest.json (1.5.7)

   │
   ▼
Phase 3 (Preprocessing + Filtering)

   │
   ▼
Phase 4 (FFT / PSD / Spectrogram) ◄── depends on Phase 3 (preprocessing)

   │
   ▼
Phase 5 (Feature Engineering, including TASK 5.4 — COMMON Feature Scaling, not owned by IF) ◄── depends on Phase 3 + Phase 4

   │
   ├──────────────┐
   ▼              ▼
Phase 6 (IF)   Phase 7 (Autoencoder)   ◄── both consume the SAME scaler from 5.4, independent of each other

   │              │
   └──────┬───────┘
          ▼
    Phase 8 (Evaluation Framework + TASK 8.3 Experiment Run ID Registry) ◄── depends on BOTH Phase 6 and 7

          │
          ▼
    Phase 9 (Central Experiment) ◄── depends on Phase 2 (raw data), Phase 5 (features),
          │                        Phase 6 (reused IF), Phase 7 (reused AE), Phase 8 (framework)

          ▼
    Phase 10 (FastAPI) ◄── depends on Phase 2-9 (all business logic already exists)

          │
          ▼
    Phase 11 (Frontend) ◄── depends on Phase 10 (functional API)

          │
          ▼
    Phase 12 (Explainability + PCA 3D) ◄── depends on Phase 9 (features/PCA) + Phase 11 (UI)

          │
          ▼
    Phase 13 (Final Validation) ◄── depends on the entire project + CWRU upload
```

---

# CRITICAL PATH

The path that, if delayed, delays the entire project:

```text
1.1 → 1.2 → 1.3 → [Phase 1.5, ALL tasks, blocked by dataset upload]

→ 1.5.7 (split manifest) → 2.1 → 2.2 → 2.4 (anti-leakage windowing)

→ 3.1 → 3.2 → 4.1 → 4.2 → 5.1 → 5.2 → 5.3 → 5.4 (common scaling)

→ 6.2 (IF training) AND 7.2 (AE training, logically in parallel — both consume the scaler from 5.4)

→ 8.1 (evaluation framework) → 8.3 (experiment run ID registry)

→ 9.1 (raw+PCA) → 9.2 (Exp A) → 9.5 (final central experiment report)

→ 10.5 (predict endpoint) → 11.6 (Anomaly Detection UI)

→ 13.2 (final README)
```

**The longest real blocker on the critical path:** Phase 1.5 — not technical, but caused by data availability (user dataset upload). The rest of the chain (Phase 2 → 9) is a normal technical, sequential dependency.

---

# IMPLEMENTATION ORDER (actual execution order)

1. Phase 1 (tasks 1.1 → 1.6, fully automatable, no blocker)

2. **STOP at the Phase 1.5 gate** — explicitly request dataset upload from the user

3. Phase 1.5 (after upload) → Phase 2 → Phase 3 → Phase 4 → Phase 5 (strictly sequential, each depending on the previous phase)

4. Phase 6 and Phase 7 (can be reported as implemented in the same "round", but the code is written sequentially, not literally in parallel)

5. Phase 8 → Phase 9 (central experiment, the highest-value component)

6. Phase 10 (complete API on top of already validated logic)

7. Phase 11 (Frontend on top of functional API)

8. Phase 12 (explainability + polish)

9. Phase 13 (also requires CWRU upload for task 13.1; the remaining work — README, cleanup — has no blocker)

---

# PHASE GATE CRITERIA (conditions for moving to the next phase)

| Gate      | Passing condition                                                                                                                                                  |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1 → 1.5   | Both servers start successfully, `pytest` passes on Phase 1 tests (health endpoint)                                                                                |
| 1.5 → 2   | `AUDIT_REPORT.md` is complete, `split_manifest.json` is generated and validated (no file overlap between splits)                                                   |
| 2 → 3     | The windowing leakage test (2.4-AC1) explicitly passes                                                                                                             |
| 3 → 4     | All preprocessing/filtering tests pass, verified on a known synthetic signal                                                                                       |
| 4 → 5     | FFT/PSD/Spectrogram validated on synthetic signals with known frequencies                                                                                          |
| 5 → 6/7   | Feature matrix generated without NaN/Inf for all splits, AND the common scaler (TASK 5.4) is fit exactly once on train and consumed identically by both models     |
| 6/7 → 8   | Both models are trained exclusively on "normal" data (verified through dedicated AC1 label-leakage tests), each with a complete Model Artifact Contract (TASK 6.5) |
| 8 → 9     | Common evaluation framework is tested and applied identically to both models                                                                                       |
| 9 → 10    | A/B/C experimental matrix is complete with real values, not placeholders                                                                                           |
| 10 → 11   | All endpoints tested with `TestClient`, `GET /docs` functional                                                                                                     |
| 11 → 12   | Complete click-through workflow (dataset → analysis → prediction) works without errors                                                                             |
| 12 → 13   | Anomaly explanations verified to be based on real values, not generic text                                                                                         |
| 13 → DONE | README complete, all reported figures originate from `docs/results/*.md`                                                                                           |

---

**Backlog fully generated. No implementation has been executed outside this document yet.**
