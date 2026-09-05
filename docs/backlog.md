# BACKLOG COMPLET DE IMPLEMENTARE — AI Signal Anomaly Detection Platform

> Sursă de adevăr: `blueprint.md`. Toate rezultatele numerice marcate `NOT YET MEASURED` până la execuția reală.
> Notă de scop: pentru task-uri de feature engineering (Phase 5) și pentru task-uri de UI repetitive (Phase 11), grupez sub-elemente înrudite într-un singur task cu Acceptance Criteria per element, nu ca task-uri separate complete — altfel documentul ar avea >150 de task-uri identice ca structură. Fiecare grupare rămâne verificabilă individual.

---

# EPIC 1 — Project Foundation

## PHASE 1 — Project Setup

### TASK 1.1 — Inițializare repository și structură de directoare
**Priority:** P0 | **Dependencies:** None | **Blocks:** 1.2, 1.4

**User Story:** Ca developer, vreau o structură de repository clară, astfel încât fiecare componentă (backend/frontend/data/models/docs) să aibă un loc predictibil.

**Description:** Creează structura de foldere din blueprint (secțiunea 29): `backend/`, `frontend/`, `data/{raw,processed,external}`, `models/`, `notebooks/`, `scripts/`, `docs/`.

**Implementation Steps:**
1. Creează directoarele de top-level.
2. Creează `.gitignore` (Python, Node, `.env`, `data/raw/*`, `models/*.pt`, `models/*.pkl`, `__pycache__`, `.venv`, `node_modules`).
3. Creează `.env.example` cu variabilele anticipate (`DATABASE_URL`, `MODEL_DIR`, `DATA_DIR`, `LOG_LEVEL`).
4. Creează `README.md` cu schelet de secțiuni (completat progresiv pe parcursul fazelor).

**Acceptance Criteria:**
- [ ] AC1: Given repo-ul clonat, when rulez `ls`, then văd toate directoarele de top-level din secțiunea 29 a blueprint-ului.
- [ ] AC2: `.gitignore` exclude `.venv`, `node_modules`, `data/raw/*`, `*.pkl`, `*.pt`, `.env`.
- [ ] AC3: `.env.example` nu conține valori secrete reale.

**Testing:** N/A (structural, verificat manual + code review).

**Definition of Done:** structură creată, `.gitignore` funcțional (verificat cu `git status` după adăugare fișiere ignorate), README schelet există.

**Files/Modules Expected:** `/`, `.gitignore`, `.env.example`, `README.md`.

---

### TASK 1.2 — Configurare backend cu uv + pyproject.toml
**Priority:** P0 | **Dependencies:** 1.1 | **Blocks:** 1.3, 1.5

**User Story:** Ca developer, vreau un mediu Python reproductibil gestionat cu `uv`, astfel încât dependențele să fie versionate și instalarea să fie rapidă.

**Implementation Steps:**
1. `uv init` în `backend/`.
2. Definește `pyproject.toml` cu dependențe: `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings` (pentru `BaseSettings` în API-ul modern Pydantic v2 — `BaseSettings` nu mai e în `pydantic` de bază), `sqlalchemy` (persistență metadata, Phase 2+), `numpy`, `scipy`, `pandas`, `scikit-learn`, `torch`, `pytest`, `httpx` (pentru TestClient).
3. `uv sync` pentru a genera `uv.lock` și `.venv`.
4. Verifică activarea mediului și importul `fastapi`.

**Acceptance Criteria:**
- [ ] AC1: Given `backend/pyproject.toml` există, when rulez `uv sync`, then se creează `.venv` fără erori.
- [ ] AC2: `uv run python -c "import fastapi, numpy, scipy, sklearn, torch, sqlalchemy, pydantic_settings"` rulează fără `ImportError`.
- [ ] AC3: `uv.lock` e generat și committabil.

**Testing:** Verificare manuală a comenzii de sync + import smoke-test.

**Definition of Done:** mediu funcțional, toate pachetele critice importabile.

**Files/Modules Expected:** `backend/pyproject.toml`, `backend/uv.lock`.

---

### TASK 1.3 — Schelet FastAPI + health endpoint
**Priority:** P0 | **Dependencies:** 1.2 | **Blocks:** 1.6, 10.x

**User Story:** Ca developer, vreau un endpoint de health, astfel încât să pot verifica rapid că backend-ul rulează și e accesibil din frontend.

**Implementation Steps:**
1. Creează `app/main.py` cu instanța FastAPI + CORS middleware (origine frontend permisă din `.env`).
2. Creează `app/api/routes/health.py` cu `GET /api/health`.
3. Creează `app/core/config.py` (Pydantic `BaseSettings`, citește `.env`).
4. Creează `app/core/logging.py` (config logging structurat, nivel din env).

**Acceptance Criteria:**
- [ ] AC1: Given backend-ul pornit (`uvicorn app.main:app`), when fac `GET /api/health`, then primesc HTTP 200.
- [ ] AC2: Body-ul răspunsului conține `{"status": "ok"}`.
- [ ] AC3: Un test pytest de integrare (folosind `TestClient`) verifică AC1 și AC2.

**Testing:** `tests/api/test_health.py` — integration test cu `TestClient`.

**Definition of Done:** endpoint funcțional, test verde, logging configurat, fără logică de business în route handler.

**Files/Modules Expected:** `backend/app/main.py`, `backend/app/api/routes/health.py`, `backend/app/core/config.py`, `backend/app/core/logging.py`, `backend/tests/api/test_health.py`.

---

### TASK 1.4 — Inițializare frontend (Vite + React + TypeScript)
**Priority:** P0 | **Dependencies:** 1.1 | **Blocks:** 1.5, 11.x

**Implementation Steps:**
1. `npm create vite@latest frontend -- --template react-ts`.
2. Instalează Tailwind CSS + configurare (`tailwind.config.ts`, `postcss.config.js`, `index.css` cu directive Tailwind).
3. Instalează shadcn/ui (init + componente de bază: `button`, `card`, `badge`).
4. Instalează `plotly.js` + `react-plotly.js` + tipuri.

**Acceptance Criteria:**
- [ ] AC1: Given frontend-ul instalat, when rulez `npm run dev`, then aplicația pornește pe `localhost:5173` fără erori în consolă.
- [ ] AC2: O clasă Tailwind (ex. `bg-slate-900`) aplicată pe un element se reflectă vizual.
- [ ] AC3: Un component shadcn/ui (`<Button>`) se randează fără erori TypeScript.

**Testing:** verificare manuală vizuală + `tsc --noEmit` fără erori.

**Definition of Done:** dev server pornește, Tailwind activ, shadcn/ui funcțional, Plotly importabil.

**Files/Modules Expected:** `frontend/` complet, `frontend/tailwind.config.ts`, `frontend/src/index.css`.

---

### TASK 1.5 — Design tokens inițiale (design system)
**Priority:** P1 | **Dependencies:** 1.4 | **Blocks:** 11.x, 12.x

**Description:** Definim paleta de culori (dark-first, un singur accent), typography (Inter + JetBrains Mono), spacing scale, înainte de orice componentă UI (per secțiunea 30 blueprint).

**Implementation Steps:**
1. Definește variabile CSS custom în `frontend/src/styles/globals.css` (`--color-bg`, `--color-surface`, `--color-accent`, `--color-normal`, `--color-warning`, `--color-anomaly`, `--font-sans`, `--font-mono`).
2. Configurează Tailwind `theme.extend` să folosească aceste variabile.
3. Documentează tokens în `docs/design-system.md`.

**Acceptance Criteria:**
- [ ] AC1: Toate culorile de status (NORMAL/WARNING/ANOMALY) sunt definite o singură dată, ca variabile, nu hard-codate în componente.
- [ ] AC2: `docs/design-system.md` conține paleta completă și regulile de spacing.

**Testing:** review vizual manual.

**Definition of Done:** tokens definite și documentate, zero culori hard-codate în commit-urile ulterioare de UI.

**Files/Modules Expected:** `frontend/src/styles/globals.css`, `docs/design-system.md`.

---

### TASK 1.6 — Conectivitate frontend-backend (smoke test end-to-end)
**Priority:** P0 | **Dependencies:** 1.3, 1.4 | **Blocks:** Phase 2

**Implementation Steps:**
1. Configurează `VITE_API_URL` în `.env` frontend.
2. Creează `frontend/src/services/api.ts` cu un client fetch minimal.
3. Apelează `/api/health` dintr-o pagină placeholder și afișează statusul.

**Acceptance Criteria:**
- [ ] AC1: Given ambele servere pornite, when încarc pagina principală, then văd afișat statusul "ok" primit real de la backend (nu hard-codat în frontend).
- [ ] AC2: Dacă backend-ul e oprit, frontend-ul afișează o stare de eroare vizibilă, nu un ecran alb sau crash.

**Testing:** verificare manuală cu backend pornit/oprit.

**Definition of Done:** conectivitate end-to-end confirmată vizual, stare de eroare tratată.

**Files/Modules Expected:** `frontend/src/services/api.ts`, componentă placeholder de status.

---

## PHASE 1.5 — Dataset Audit (obligatorie, blochează Phase 2)

> **BLOCKER CUNOSCUT:** aceste task-uri necesită fișierele reale MAFAULDA în `/mnt/user-data/uploads`. Nu se execută pe date presupuse sau simulate.

### TASK 1.5.1 — Localizare și inventariere fișiere dataset
**Priority:** P0 | **Dependencies:** 1.1 | **Blocks:** 1.5.2–1.5.9, Phase 2

**User Story:** Ca ML engineer, vreau un inventar complet al fișierelor MAFAULDA disponibile, astfel încât să pot proiecta corect split-ul train/val/test.

**Implementation Steps:**
1. Confirmă locația fișierelor uploadate (`/mnt/user-data/uploads`).
2. Scanează recursiv, listează toate fișierele (nume, extensie, dimensiune).
3. Salvează inventarul brut ca `docs/dataset_audit/file_inventory.csv`.

**Acceptance Criteria:**
- [ ] AC1: Given fișierele MAFAULDA urcate, when rulez scriptul de inventariere, then obțin un CSV cu toate fișierele găsite, fără omisiuni (verificat prin numărare `find` vs. rânduri CSV).
- [ ] AC2: Dacă niciun fișier nu e găsit, scriptul raportează explicit eroare, nu un inventar gol tratat ca "succes".

**Testing:** test unitar pe un folder sintetic mic cu fișiere fictive, verificând că numărul detectat = numărul creat.

**Definition of Done:** inventar complet generat sau eroare explicită dacă lipsesc datele.

**Files/Modules Expected:** `backend/scripts/audit_inventory.py`, `docs/dataset_audit/file_inventory.csv`.

**Notes:** **Blocat până la upload real al dataset-ului de către utilizator.**

---

### TASK 1.5.2 — Identificare structură recording (nume fișier → stare + condiție)
**Priority:** P0 | **Dependencies:** 1.5.1 | **Blocks:** 1.5.5, 2.x

**Implementation Steps:**
1. Parsează convenția de denumire reală a fișierelor (nu presupusă din documentație online).
2. Extrage per fișier: clasă/stare (normal/imbalance/misalignment/etc.), dacă e codificată în calea folderului sau numele fișierului.
3. Documentează maparea exactă observată.

**Acceptance Criteria:**
- [ ] AC1: Fiecare fișier din inventar are o stare asociată, derivată din structura reală (folder/nume), nu presupusă din blueprint.
- [ ] AC2: Dacă un fișier nu poate fi mapat la o stare cunoscută, e marcat explicit `UNKNOWN`, nu ignorat silențios.

**Testing:** test unitar pe mostre de nume de fișier cunoscute → verifică maparea corectă.

**Definition of Done:** mapare completă documentată în `docs/dataset_audit/recording_mapping.md`.

**Files/Modules Expected:** `backend/app/datasets/mafaulda_parser.py`, `docs/dataset_audit/recording_mapping.md`.

---

### TASK 1.5.3 — Identificare canale, sampling rate, lungime semnal
**Priority:** P0 | **Dependencies:** 1.5.1

**Implementation Steps:**
1. Citește un eșantion de fișiere cu pandas.
2. Confirmă numărul real de coloane/canale per fișier și numele lor.
3. Confirmă sampling rate-ul real (verificat, nu doar citat din literatură) și durata semnalului.

**Acceptance Criteria:**
- [ ] AC1: Numărul de coloane observat e documentat explicit (poate diferi de ce sugerează sursele externe).
- [ ] AC2: Sampling rate-ul e confirmat fie din metadata fișierului, fie calculat din numărul de eșantioane / durata cunoscută, și documentat cu sursa exactă a confirmării.

**Testing:** verificare pe minim 3 fișiere din clase diferite, comparate pentru consistență.

**Definition of Done:** raport scris cu structura reală confirmată a datelor.

**Files/Modules Expected:** `docs/dataset_audit/signal_structure.md`.

---

### TASK 1.5.4 — Distribuția claselor și turațiilor (rotation frequency)
**Priority:** P0 | **Dependencies:** 1.5.2

**Implementation Steps:**
1. Numără fișiere per clasă.
2. Extrage/calculează turația per fișier, cu **sursa documentată explicit per valoare** — nu se tratează o valoare citită din numele fișierului ca măsurătoare fizică echivalentă cu una din tahometru. Câmp obligatoriu per recording:
   ```
   rotation_source: "tachometer" | "metadata" | "filename" | "unavailable"
   rotation_frequency_hz: <valoare> (null dacă unavailable)
   ```
3. Generează histogramă a distribuției claselor și a turațiilor, adnotată cu sursa.

**Acceptance Criteria:**
- [ ] AC1: Există un tabel `clasă → număr de fișiere` complet, fără clase omise.
- [ ] AC2: Dacă distribuția e puternic dezechilibrată (ex. o clasă are <10% din numărul mediu), asta e semnalat explicit ca risc pentru Phase 6-9, nu ignorat.
- [ ] AC3: Fiecare valoare de turație raportată e însoțită explicit de `rotation_source` — raportul nu prezintă o valoare din filename cu aceeași încredere ca una măsurată direct de tahometru.

**Testing:** verificare manuală a sumei totale (suma pe clase = total fișiere din inventar).

**Definition of Done:** raport + grafic distribuție salvate.

**Files/Modules Expected:** `docs/dataset_audit/class_distribution.md`, `docs/dataset_audit/class_distribution.png`.

---

### TASK 1.5.5 — Detectare replicări/duplicate ale acelorași condiții
**Priority:** P1 | **Dependencies:** 1.5.2, 1.5.4

**Implementation Steps:**
1. Grupează fișierele după (clasă, turație aproximativă).
2. Identifică grupuri cu multiple fișiere (replicări reale) vs. condiții unice.

**Acceptance Criteria:**
- [ ] AC1: Raportul indică explicit câte grupuri (clasă, turație) au ≥2 fișiere replicate.
- [ ] AC2: Aceste informații alimentează direct decizia de split (task 1.5.7).

**Testing:** verificare manuală pe un subset cunoscut.

**Definition of Done:** raport de replicări generat.

**Files/Modules Expected:** `docs/dataset_audit/replication_analysis.md`.

---

### TASK 1.5.6 — Analiză risc de leakage specifică datelor reale
**Priority:** P0 | **Dependencies:** 1.5.1–1.5.5 | **Blocks:** 1.5.7

**Description:** Nu presupunem generic "split per fișier e suficient" — verificăm dacă există factori suplimentari de leakage (ex. mai multe fișiere din aceeași sesiune de înregistrare, cu zgomot de fundal identic, care ar trebui tratate ca un singur grup la split).

**Acceptance Criteria:**
- [ ] AC1: Raportul documentează explicit dacă unitatea de split "per fișier" e suficientă sau dacă trebuie extinsă la "per grup de sesiune".
- [ ] AC2: Decizia finală de split e justificată cu date din audit, nu presupusă din blueprint.

**Testing:** N/A (analiză documentată, revizuită manual).

**Definition of Done:** decizie de split scrisă și justificată cu referință directă la datele reale.

**Files/Modules Expected:** `docs/dataset_audit/leakage_analysis.md`.

---

### TASK 1.5.7 — Definire strategie finală train/val/test split
**Priority:** P0 | **Dependencies:** 1.5.6 | **Blocks:** Phase 2

**Implementation Steps:**
1. Alocă fișiere (nu ferestre) la train/val/test, stratificat pe clasă.
2. Documentează exact ce fișiere merg în fiecare split (listă explicită, reproductibilă cu seed fixat).

**Acceptance Criteria:**
- [ ] AC1: Niciun fișier nu apare în mai mult de un split.
- [ ] AC2: Fiecare split conține reprezentare din fiecare clasă majoră (verificat explicit, nu presupus).
- [ ] AC3: Split-ul e determinist (același seed → aceeași alocare, verificat cu test).

**Testing:** test unitar care rulează funcția de split de două ori cu același seed și verifică identitatea rezultatelor.

**Definition of Done:** listă finală de split salvată ca artifact (`data/processed/split_manifest.json`).

**Files/Modules Expected:** `backend/app/datasets/split.py`, `data/processed/split_manifest.json`, `backend/tests/dataset/test_split.py`.

---

### TASK 1.5.8 — Verificare calitate date (missing values, semnal corupt)
**Priority:** P1 | **Dependencies:** 1.5.3

**Implementation Steps:**
1. Verifică NaN/valori lipsă per fișier.
2. Verifică semnale constante/zero (posibil senzor defect în înregistrare).

**Acceptance Criteria:**
- [ ] AC1: Raportul listează explicit orice fișier cu probleme de calitate detectate.
- [ ] AC2: Dacă există fișiere problematice, decizia (excludere/reparare) e documentată cu motiv.

**Testing:** test pe fișier sintetic cu NaN injectat → verifică detectarea.

**Definition of Done:** raport calitate date generat.

**Files/Modules Expected:** `docs/dataset_audit/data_quality_report.md`.

---

### TASK 1.5.9 — Raport final de audit (document consolidat)
**Priority:** P0 | **Dependencies:** 1.5.1–1.5.8 | **Blocks:** Phase 2 (Gate)

**Description:** Consolidează toate task-urile 1.5.x într-un singur document, sursă de adevăr pentru toate deciziile de dataset din fazele următoare.

**Acceptance Criteria:**
- [ ] AC1: Documentul răspunde explicit la toate întrebările din secțiunea 6 a blueprint-ului (organizare fișiere, canale, recording ID, turații, replicări, split).
- [ ] AC2: Nicio afirmație din document nu e presupusă — fiecare are sursă (fișier/script care a generat-o).

**Definition of Done:** `docs/dataset_audit/AUDIT_REPORT.md` complet, revizuit.

**Files/Modules Expected:** `docs/dataset_audit/AUDIT_REPORT.md`.

---

# EPIC 2 — Data & Signal Processing Foundation

## PHASE 2 — Dataset Integration

### TASK 2.1 — Model canonical de date (SignalRecord, Recording)
**Priority:** P0 | **Dependencies:** Phase 1.5 Gate | **Blocks:** 2.2–2.5

**Implementation Steps:**
1. Definește `SignalRecord`/`Recording` (Pydantic + SQLAlchemy) cu câmpurile din blueprint secțiunea 25: id, sampling_rate, channel, values (referință la fișier, nu în DB), machine_id/operating_condition, label.
2. Creează schema SQLite (`datasets`, `signals`, `signal_windows`).

**Acceptance Criteria:**
- [ ] AC1: Modelul validează respingerea unui sampling_rate ≤ 0.
- [ ] AC2: Modelul validează respingerea unui label necunoscut (enum strict, nu string liber).

**Testing:** teste unitare Pydantic pentru validare (valori valide/invalide).

**Definition of Done:** modele definite, migrate, testate.

**Files/Modules Expected:** `backend/app/models/signal.py`, `backend/app/core/database.py`.

---

### TASK 2.2 — Dataset loader (bazat pe rezultatul auditului)
**Priority:** P0 | **Dependencies:** 2.1, 1.5.9 | **Blocks:** 2.4

**Implementation Steps:**
1. Implementează `loader.py` care citește fișierele conform structurii reale confirmate în audit (nu presupusă).
2. Populează modelele `Recording`/`SignalRecord` din fiecare fișier.

**Acceptance Criteria:**
- [ ] AC1: Given `split_manifest.json` din task 1.5.7, when rulez loader-ul, then fiecare recording e etichetat corect cu split-ul (train/val/test) alocat.
- [ ] AC2: Loader-ul respinge explicit un fișier care nu respectă structura confirmată în audit (nu îl ignoră silențios).

**Testing:** test de integrare pe un subset mic real (2-3 fișiere per split).

**Definition of Done:** loader funcțional, testat pe date reale.

**Files/Modules Expected:** `backend/app/datasets/loader.py`, `backend/tests/dataset/test_loader.py`.

---

### TASK 2.3 — Validatori dataset (sampling rate, canale, shape)
**Priority:** P0 | **Dependencies:** 2.1

**Implementation Steps:**
1. Implementează validări: sampling rate consistent cu ce a confirmat auditul, număr de canale așteptat, lungime minimă semnal.

**Acceptance Criteria:**
- [ ] AC1: Given un fișier cu sampling rate diferit de cel confirmat în audit, when e încărcat, then loader-ul aruncă o eroare explicită, nu îl procesează silențios.

**Testing:** teste unitare cu fixture-uri de fișiere valide/invalide.

**Definition of Done:** validatori impl., testați.

**Files/Modules Expected:** `backend/app/datasets/validators.py`.

---

### TASK 2.4 — Windowing (segmentare semnal) cu respectarea split-ului
**Priority:** P0 | **Dependencies:** 2.2 | **Blocks:** Phase 3, 5, 6, 7

**Implementation Steps:**
1. Implementează funcție de windowing (window_size, overlap configurabile) care operează **în interiorul** unui singur recording, niciodată peste graniță de recording.
2. Ferestrele moștenesc split-ul recording-ului sursă.

**Acceptance Criteria:**
- [ ] AC1 (CRITIC): Given două recordinguri alocate la split-uri diferite, when generez ferestre, then nicio fereastră nu conține date din ambele recordinguri.
- [ ] AC2: Overlap-ul configurat (ex. 50%) produce numărul așteptat de ferestre pentru o lungime de semnal cunoscută (verificat matematic).

**Testing:** test unitar dedicat exact pentru AC1 (cel mai important test de leakage din tot proiectul), plus test pentru numărul de ferestre generate.

**Definition of Done:** windowing implementat, testul de leakage trece explicit.

**Files/Modules Expected:** `backend/app/signal_processing/windowing.py`, `backend/tests/signal_processing/test_windowing.py`.

---

### TASK 2.5 — Teste de integrare dataset end-to-end
**Priority:** P1 | **Dependencies:** 2.2, 2.3, 2.4

**Acceptance Criteria:**
- [ ] AC1: Pipeline complet `fișier → loader → validare → windowing` rulează pe subsetul real fără erori.
- [ ] AC2: Numărul total de ferestre per split e raportat și rezonabil (nu zero, nu dezechilibrat extrem fără explicație).

**Definition of Done:** test de integrare verde pe date reale.

**Files/Modules Expected:** `backend/tests/dataset/test_integration.py`.

---

## PHASE 3 — Signal Processing (Preprocessing + Filtering)

### TASK 3.1 — Preprocessing: detrending, normalization, standardization
**Priority:** P0 | **Dependencies:** 2.4 | **Blocks:** 3.2, 4.x, 5.x

**Notă de arhitectură (evită confuzia cu TASK 5.4):** acest task normalizează/standardizează **semnalul brut** (per fereastră, înainte de filtrare/FFT), nu feature vectors. E un pas separat, cu scop diferit, de scaling-ul de features din TASK 5.4 (care rulează pe ieșirea extractorului, nu pe semnal). Cele două nu trebuie confundate sau aplicate redundant una peste alta.

**Implementation Steps:**
1. Implementează `detrend()` (scipy.signal.detrend, liniar).
2. Implementează `normalize()` (min-max) și `standardize()` (z-score), cu parametri fit-uiți **doar pe train** (per decizia din blueprint secțiunea 8).

**Acceptance Criteria:**
- [ ] AC1: Given un semnal cu trend liniar cunoscut adăugat sintetic, when aplic detrend, then trendul rezidual e sub o toleranță definită (ex. panta reziduală < 1e-3).
- [ ] AC2: Given parametrii de standardizare calculați pe train, when aplicați pe test, then media rezultată pe test NU e neapărat 0 (semn că nu s-a refăcut fit pe test — verificare explicită anti-leakage).

**Testing:** `backend/tests/signal_processing/test_preprocessing.py`.

**Definition of Done:** funcții implementate, testate, fără fit pe test/val.

**Files/Modules Expected:** `backend/app/signal_processing/preprocessing.py`.

---

### TASK 3.2 — Filtre Butterworth (low-pass, high-pass, band-pass) + filtfilt
**Priority:** P0 | **Dependencies:** 3.1 | **Blocks:** 4.x

**Implementation Steps:**
1. Implementează `butter_filter(signal, cutoff, fs, order, btype)` folosind `scipy.signal.butter` + `filtfilt` (zero-phase).
2. Parametrii (cutoff, order) sunt argumente, nu constante hard-codate.
3. Validare: cutoff < Nyquist (fs/2), altfel eroare explicită.

**Acceptance Criteria:**
- [ ] AC1: Given un semnal sintetic = sumă de 10 Hz + 200 Hz, when aplic low-pass cu cutoff 50 Hz, then componenta de 200 Hz e atenuată cu minim 20 dB (verificat prin FFT pe semnalul filtrat).
- [ ] AC2: Given cutoff ≥ Nyquist, when apelez funcția, then se ridică o eroare explicită, nu un rezultat silențios greșit.
- [ ] AC3: Filtrarea cu `filtfilt` nu introduce shift de fază (verificat: poziția unui vârf cunoscut rămâne la același index ±1 eșantion).

**Testing:** `backend/tests/signal_processing/test_filtering.py` cu semnale sintetice multi-componentă.

**Definition of Done:** toate cele 3 tipuri de filtru implementate și testate cu AC măsurabile.

**Files/Modules Expected:** `backend/app/signal_processing/filtering.py`.

---

## PHASE 4 — Spectral Analysis

### TASK 4.1 — FFT + axă de frecvență + magnitude spectrum
**Priority:** P0 | **Dependencies:** 3.1 | **Blocks:** 4.3, 5.2, 9.x

**Implementation Steps:**
1. Implementează `compute_fft(signal, fs)` → returnează frequencies, magnitude, folosind `numpy.fft.rfft` (semnal real) + `numpy.fft.rfftfreq`.
2. Implementează `dominant_frequency(freqs, magnitude)`.

**Acceptance Criteria:**
- [ ] AC1: Given un semnal sinusoidal sintetic de 50 Hz eșantionat la 1000 Hz, when calculez FFT, then frecvența dominantă detectată e în ±1 Hz de 50 Hz.
- [ ] AC2: Given fs = 1000 Hz, frecvența maximă din axa returnată nu depășește Nyquist (500 Hz).

**Testing:** `backend/tests/signal_processing/test_fft.py`.

**Definition of Done:** FFT + dominant frequency implementate, testate pe semnal sintetic cunoscut.

**Files/Modules Expected:** `backend/app/signal_processing/fft.py`.

---

### TASK 4.2 — Welch PSD
**Priority:** P0 | **Dependencies:** 4.1 | **Blocks:** 4.3, 5.2

**Implementation Steps:**
1. Implementează wrapper peste `scipy.signal.welch`, cu `nperseg`/`noverlap` configurabile.

**Acceptance Criteria:**
- [ ] AC1: Given același semnal sintetic de 50 Hz cu zgomot alb adăugat, when calculez Welch PSD, then vârful dominant rămâne la 50 Hz ±1 Hz, cu varianță vizibil mai mică decât FFT simplu pe același semnal zgomotos (comparație explicită în test).

**Testing:** `backend/tests/signal_processing/test_psd.py`.

**Definition of Done:** Welch PSD implementat, comparat cu FFT simplu în test.

**Files/Modules Expected:** `backend/app/signal_processing/psd.py`.

---

### TASK 4.3 — STFT / Spectrogramă
**Priority:** P0 | **Dependencies:** 4.1 | **Blocks:** 12.x (PCA/vizualizare later)

**Implementation Steps:**
1. Implementează wrapper peste `scipy.signal.spectrogram`, cu window size și hop length configurabile.

**Acceptance Criteria:**
- [ ] AC1: Given un semnal sintetic cu frecvență care se schimbă la jumătatea duratei (50 Hz → 150 Hz), when calculez spectrograma, then energia dominantă se deplasează vizibil de la 50 la 150 Hz între prima și a doua jumătate a axei de timp (verificat prin index-ul frecvenței cu magnitudine maximă per coloană temporală).
- [ ] AC2: Mărirea window size-ului crește rezoluția în frecvență și scade rezoluția în timp (verificat comparând lățimea benzii dominante pentru 2 configurații diferite).

**Testing:** `backend/tests/signal_processing/test_spectrogram.py`.

**Definition of Done:** STFT implementat, trade-off-ul rezoluție timp/frecvență verificat printr-un test explicit.

**Files/Modules Expected:** `backend/app/signal_processing/spectrogram.py`.

---

## PHASE 5 — Feature Engineering

### TASK 5.1 — Feature registry + time-domain features
**Priority:** P0 | **Dependencies:** 3.1 | **Blocks:** 5.3, 6.x, 7.x, 9.x

**Description:** Implementează mean, std, variance, RMS, peak, peak-to-peak, skewness, kurtosis, crest factor ca funcții pure înregistrate într-un registry extensibil.

**Implementation Steps:**
1. `features/time_domain.py` — o funcție per feature, semnătură uniformă `f(signal: np.ndarray) -> float`.
2. `features/registry.py` — dicționar `{nume: funcție}`, extensibil prin adăugare de intrare, fără modificarea codului existent.

**Acceptance Criteria (per feature, verificate cu semnal sintetic cu proprietăți cunoscute):**
- [ ] AC1: RMS pe un semnal sinusoidal de amplitudine A e ≈ A/√2 (±1% toleranță).
- [ ] AC2: Kurtosis pe zgomot gaussian pur e ≈ 3 (definiție non-excess) sau ≈0 (excess kurtosis — se specifică explicit convenția folosită), ±toleranță documentată.
- [ ] AC3: Crest factor pe un semnal cu vârfuri rare și ascuțite e vizibil mai mare decât pe un sinusoidal pur de aceeași putere RMS.
- [ ] AC4: Fiecare feature are un test dedicat cu input cunoscut și output așteptat calculat analitic.

**Testing:** `backend/tests/features/test_time_domain.py` — un test per feature.

**Definition of Done:** toate cele 9 features implementate, fiecare cu test individual verde, documentate (formulă + interpretare fizică + relevanță pentru fault) în docstring.

**Files/Modules Expected:** `backend/app/features/time_domain.py`, `backend/app/features/registry.py`.

---

### TASK 5.2 — Frequency-domain features
**Priority:** P0 | **Dependencies:** 4.1, 4.2, 5.1 | **Blocks:** 5.3

**Description:** dominant frequency, spectral centroid, spectral bandwidth, spectral energy, spectral entropy, energie pe benzi de frecvență.

**Acceptance Criteria:**
- [ ] AC1: Given un semnal sinusoidal pur, spectral entropy e apropiată de minim (semnal concentrat pe o singură frecvență), comparativ cu zgomot alb unde entropy e apropiată de maxim — testat comparativ explicit.
- [ ] AC2: Spectral centroid pe un semnal cu energie concentrată la o frecvență cunoscută X e ≈ X.
- [ ] AC3: Fiecare feature are test dedicat cu semnal sintetic.

**Testing:** `backend/tests/features/test_frequency_domain.py`.

**Definition of Done:** toate cele 6 features implementate și testate individual.

**Files/Modules Expected:** `backend/app/features/frequency_domain.py`.

---

### TASK 5.3 — Feature extractor complet (matrice de features)
**Priority:** P0 | **Dependencies:** 5.1, 5.2 | **Blocks:** Phase 6, 7, 9

**Implementation Steps:**
1. `extractor.py` — primește o fereastră de semnal, rulează toate funcțiile din registry, returnează un vector/rând de features cu nume de coloane consistente.
2. Aplicat pe toate ferestrele → matrice de features (train/val/test separat).

**Acceptance Criteria:**
- [ ] AC1: Given aceeași fereastră de semnal, rulată de două ori, output-ul e identic (determinism, fără randomness ascuns).
- [ ] AC2: Matricea de features pentru split-ul train nu conține NaN/Inf (verificat explicit).
- [ ] AC3: Numărul de coloane = numărul de features din registry (verificat automat, nu hard-codat).

**Testing:** `backend/tests/features/test_extractor.py`.

**Definition of Done:** extractor complet, testat pe date reale (subset).

**Files/Modules Expected:** `backend/app/features/extractor.py`.

---

### TASK 5.4 — Feature Scaling (componentă comună, NU specifică unui model)
**Priority:** P0 | **Dependencies:** 5.3 | **Blocks:** 6.2, 7.2, 9.x

**Description:** Distincție importantă de arhitectură, ca să nu se aplice scaling de două ori sau să se creeze o dependență ascunsă între modele:

```
Signal preprocessing (Phase 3)         Feature scaling (AICI, Phase 5)
   detrend                                StandardScaler pe FEATURE VECTORS
   normalize/standardize SEMNALUL RAW      (rezultatul din TASK 5.3),
   (opțional, per fereastră)               fit doar pe train
        │                                        │
        ▼                                        ▼
   folosit pentru filtrare/FFT/PSD         folosit ca INPUT direct
   (Phase 3-4), NU e același pas cu        pentru Isolation Forest (6.2)
   scaling-ul de mai jos                   ȘI Autoencoder (7.2)
```

Scaler-ul e o componentă **comună**, consumată identic de ambele modele — nu trăiește în `app/ml/isolation_forest.py`, ca Autoencoder-ul să nu depindă accidental de codul specific IF.

**Implementation Steps:**
1. `app/ml/scaling.py` — `fit_scaler(train_features) -> scaler`, `apply_scaler(scaler, features) -> scaled_features`.
2. Scaler-ul fit-uit o singură dată (pe train), salvat ca artifact separat (`scaler_v1.pkl`), referit prin `scaler_artifact` în Model Artifact Contract (TASK 6.5) de către **ambele** modele.

**Acceptance Criteria:**
- [ ] AC1: `StandardScaler` (sau echivalent) e fit-uit exclusiv pe train, aplicat (transform) pe val/test — verificat printr-un test care ar eșua dacă s-ar face fit pe tot dataset-ul.
- [ ] AC2: Codul din `app/ml/isolation_forest.py` și `app/ml/autoencoder.py`/`training.py` importă **același** modul `scaling.py`, nu implementări duplicate.

**Testing:** `backend/tests/ml/test_scaling.py`.

**Definition of Done:** scaler implementat o singură dată, salvat ca artifact reutilizabil, consumat identic de Phase 6 și Phase 7.

**Files/Modules Expected:** `backend/app/ml/scaling.py`.

---

# EPIC 3 — Machine Learning & Deep Learning

## PHASE 6 — Isolation Forest Baseline

### TASK 6.2 — Antrenare Isolation Forest (doar pe date normale)
**Priority:** P0 | **Dependencies:** 5.4 | **Blocks:** 6.3, 8.x, 9.x

**Description:** Modelul se antrenează exclusiv pe ferestre etichetate "normal" din split-ul train (per metodologia secțiunii 5 blueprint — antrenare unsupervised/semi-supervised).

**Acceptance Criteria:**
- [ ] AC1 (CRITIC): Given setul de train, when filtrez datele de antrenare, then niciun exemplu cu label diferit de "normal" nu ajunge în `.fit()` — verificat printr-un test explicit care numără label-urile din setul de antrenare efectiv folosit.
- [ ] AC2: Modelul antrenat produce scoruri (`decision_function`) pentru orice fereastră nouă, fără erori de shape.
- [ ] AC3: Seed fixat → antrenări repetate produc scoruri identice (reproducibilitate).

**Testing:** `backend/tests/ml/test_isolation_forest_training.py`.

**Definition of Done:** model antrenat, salvat, testul AC1 (anti-leakage de labels) trece explicit.

**Files/Modules Expected:** `backend/app/ml/isolation_forest.py`.

---

### TASK 6.3 — Scoring, normalizare scor, threshold calibration (metodă configurabilă)
**Priority:** P0 | **Dependencies:** 6.2 | **Blocks:** 8.x, 6.5

**Implementation Steps:**
1. Normalizează scorul brut la [0,1] folosind distribuția scorurilor pe **validation set**.
2. Implementează calibrarea threshold-ului ca **strategie selectabilă**, nu valoare fixă: `threshold_method` poate fi `"percentile"` (parametru `percentile_value`, ex. 95, configurabil) sau `"validation_f1_optimal"` (alege pragul care maximizează F1 pe validation set, dacă există exemple de fault în validation). Metoda implicită și motivul alegerii ei se documentează explicit, nu se prezintă ca "adevăr universal".

**Acceptance Criteria:**
- [ ] AC1: Threshold-ul e calculat din validation set, nu din test set (verificat prin cod — funcția de calibrare nu primește niciodată date de test ca input).
- [ ] AC2: Scorul normalizat e mereu în [0,1] pentru orice input (verificat cu date extreme sintetice).
- [ ] AC3: Funcția de calibrare acceptă parametrul `threshold_method` și produce rezultate diferite, verificabile, pentru cele două metode implementate (testat explicit, nu presupus).

**Testing:** `backend/tests/ml/test_scoring.py`.

**Definition of Done:** funcție de scoring + threshold calibrat prin metoda aleasă, salvate ca parte a artifact-ului modelului (vezi TASK 6.5).

**Files/Modules Expected:** `backend/app/ml/scoring.py`.

---

### TASK 6.4 — Persistență model (save/load) + inference
**Priority:** P0 | **Dependencies:** 6.2, 6.3 | **Blocks:** 10.x

**Acceptance Criteria:**
- [ ] AC1: Given un model salvat, when e reîncărcat, then predicțiile pe același input sunt identice cu cele dinainte de salvare (bit-exact sau în toleranță numerică documentată).

**Testing:** `backend/tests/ml/test_persistence.py`.

**Definition of Done:** save/load funcțional, testat.

**Files/Modules Expected:** `backend/app/ml/inference.py`, `models/isolation_forest_v1.pkl`.

---

### TASK 6.5 — Model Artifact Contract (metadata standard, comun IF + AE)
**Priority:** P0 | **Dependencies:** 6.3, 6.4 | **Blocks:** 7.3 (reutilizează același contract), 9.x, 10.5

**User Story:** Ca ML engineer, vreau ca fiecare model salvat să fie însoțit de un contract de metadata explicit, astfel încât orice predicție ulterioară (sau audit) să poată fi trasată exact la configurația care a produs modelul, fără presupuneri.

**Description:** Fiecare artifact de model (`.pkl` pentru Isolation Forest, `.pt` pentru Autoencoder — vezi Phase 7) e însoțit de un fișier JSON sidecar cu aceeași bază de nume, conținând:

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
1. Implementează `app/ml/model_artifact.py` cu o funcție `save_model_artifact(model, metadata: ModelArtifactMetadata, path)` și `load_model_artifact(path) -> (model, metadata)`.
2. `dataset_hash` și `split_manifest_hash` se calculează (ex. SHA-256 pe conținutul `split_manifest.json` din TASK 1.5.7) — nu inventate, nu omise.
3. `score_direction` există explicit pentru că Isolation Forest (`decision_function`, scor mai mic = mai anormal) și Autoencoder (reconstruction error, scor mai mare = mai anormal) au convenții native diferite — contractul normalizează asta pentru consumatorii din Phase 10/11.

**Acceptance Criteria:**
- [ ] AC1: Given un model salvat, when citesc fișierul JSON sidecar, then toate câmpurile din schema de mai sus sunt prezente și nenule.
- [ ] AC2: `dataset_hash`/`split_manifest_hash` sunt calculate real din fișierul sursă (verificat: schimbarea unui singur caracter în `split_manifest.json` schimbă hash-ul).
- [ ] AC3: Given două modele antrenate cu configurații diferite (ex. threshold_method diferit), metadata reflectă corect diferența — nu e un artefact static copiat.

**Testing:** `backend/tests/ml/test_model_artifact.py`.

**Definition of Done:** contract implementat, aplicat la Isolation Forest (Phase 6) și reutilizat identic la Autoencoder (Phase 7), fără duplicare de schemă.

**Files/Modules Expected:** `backend/app/ml/model_artifact.py`, `backend/tests/ml/test_model_artifact.py`.

---

## PHASE 7 — Autoencoder (PyTorch, pe feature vectors)

### TASK 7.1 — Arhitectură Encoder/Decoder
**Priority:** P0 | **Dependencies:** 5.3 | **Blocks:** 7.2

**Implementation Steps:**
1. Definește `Autoencoder(nn.Module)` — encoder dense (input_dim → hidden → bottleneck), decoder simetric.
2. Dimensiunile sunt parametri configurabili, nu hard-codate.

**Acceptance Criteria:**
- [ ] AC1: Given un batch de shape `(N, num_features)`, forward pass returnează output de shape identică `(N, num_features)`.
- [ ] AC2: Numărul de parametri ai rețelei e documentat explicit (verificare că arhitectura rămâne "ușor de explicat", nu exagerat de mare, per blueprint secțiunea 16).

**Testing:** `backend/tests/ml/test_autoencoder_architecture.py`.

**Definition of Done:** arhitectură implementată, testată pentru shape.

**Files/Modules Expected:** `backend/app/ml/autoencoder.py`.

---

### TASK 7.2 — Training loop + validare + early stopping
**Priority:** P0 | **Dependencies:** 7.1, 5.4 | **Blocks:** 7.3

**Implementation Steps:**
1. DataLoader pe feature vectors normalizate (train = doar "normal", per aceeași regulă ca Isolation Forest).
2. Loss = MSE reconstrucție. Optimizer Adam.
3. Early stopping pe baza loss-ului de validare.
4. Seed fixat pentru reproducibilitate.

**Acceptance Criteria:**
- [ ] AC1: Loss-ul final de antrenare e semnificativ mai mic decât loss-ul din prima epocă (nu se cere scădere monotonă — loss-ul poate fluctua epocă cu epocă și modelul rămâne complet valid; criteriul greșit ar respinge antrenări perfect normale). Antrenarea converge fără `NaN`/`Inf` la niciun pas.
- [ ] AC2: Loss-ul de **validare** e urmărit separat de loss-ul de train pe tot parcursul antrenării (nu doar train loss) — condiție pentru ca early stopping (task 7.2) să poată detecta overfitting real, nu doar zgomot pe train.
- [ ] AC3: Antrenarea nu vede niciodată exemple non-normale (aceeași verificare ca 6.2-AC1).
- [ ] AC4: Rularea cu același seed produce loss curves identice.

**Testing:** `backend/tests/ml/test_autoencoder_training.py`.

**Definition of Done:** training loop funcțional, reproductibil, checkpointing implementat.

**Files/Modules Expected:** `backend/app/ml/training.py`.

---

### TASK 7.3 — Reconstruction error + threshold + persistență
**Priority:** P0 | **Dependencies:** 7.2, 6.5 | **Blocks:** 8.x

**Acceptance Criteria:**
- [ ] AC1: Given date de validare (normale + anomalii cunoscute), reconstruction error mediu pe anomalii e vizibil mai mare decât pe normale (raportat, nu presupus — dacă nu e adevărat, se raportează onest ca rezultat, per regula "nu inventa rezultate").
- [ ] AC2: Threshold calibrat prin aceeași strategie configurabilă din TASK 6.3 (`percentile` sau `validation_f1_optimal`), pe validation, nu test.
- [ ] AC3: Model salvat/reîncărcat produce reconstrucții identice.
- [ ] AC4: Artifact-ul salvat respectă **exact același contract de metadata din TASK 6.5** (`model_type: "autoencoder"`, `score_direction: "higher_is_more_anomalous"`, restul câmpurilor identice ca structură) — nicio schemă separată/duplicată pentru Autoencoder.

**Testing:** `backend/tests/ml/test_autoencoder_inference.py`.

**Definition of Done:** pipeline complet Autoencoder funcțional, testat, salvat.

**Files/Modules Expected:** `backend/app/ml/inference.py` (extins), `models/autoencoder_v1.pt`.

---

## PHASE 8 — Model Evaluation Framework

### TASK 8.1 — Evaluation framework comun (IF + AE pe același test set)
**Priority:** P0 | **Dependencies:** 6.4, 7.3 | **Blocks:** Phase 9

**Implementation Steps:**
1. Funcție `evaluate(model, test_features, test_labels) -> metrics dict`, folosită identic pentru ambele modele.
2. Calculează: precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, FPR, FNR, timp inference.

**Acceptance Criteria:**
- [ ] AC1: Given același test set pentru ambele modele, when rulez evaluarea, then ambele folosesc exact aceeași funcție de calcul metrici (verificat prin cod, nu duplicare de logică).
- [ ] AC2: Toate metricile sunt calculate din predicții reale, nu hard-codate — status explicit `NOT YET MEASURED` până la prima rulare reală.
- [ ] AC3: Confusion matrix e generată corect pentru cazul binar (normal vs. anomaly agregat) — verificat cu un exemplu de predicții/labels cunoscute manual.

**Testing:** `backend/tests/ml/test_evaluation.py` cu predicții/labels sintetice unde rezultatul corect e calculat manual.

**Definition of Done:** framework comun, testat, aplicat pe ambele modele cu rezultate reale raportate (nu inventate).

**Files/Modules Expected:** `backend/app/ml/evaluation.py`.

---

### TASK 8.2 — Raport comparativ Isolation Forest vs. Autoencoder
**Priority:** P0 | **Dependencies:** 8.1

**Acceptance Criteria:**
- [ ] AC1: Tabelul comparativ conține valori reale măsurate pentru ambele modele, pe același test set.
- [ ] AC2: Concluzia raportului nu presupune un câștigător înainte de a vedea rezultatele.

**Definition of Done:** `docs/results/isolation_forest_vs_autoencoder.md` generat cu date reale.

**Files/Modules Expected:** `docs/results/isolation_forest_vs_autoencoder.md`.

---

### TASK 8.3 — Experiment Run ID Registry
**Priority:** P0 | **Dependencies:** 8.1, 6.5 | **Blocks:** Phase 9 (toate task-urile)

**User Story:** Ca ML engineer, vreau ca fiecare rulare de experiment să primească un identificator unic, astfel încât rezultatele să poată fi referite explicit (în rapoarte, API, frontend) fără ambiguitate despre ce configurație le-a produs.

**Description:** Formalizează un ID de run peste structura deja existentă în TASK 6.5 (Model Artifact Contract) și TASK 9.5 (raport agregat) — nu introduce o schemă nouă, doar un identificator care leagă cele două. Fiecare experiment din matricea A/B/C (TASK 9.2-9.4) primește un `experiment_id` (ex. `EXP-A-001`) la momentul rulării, salvat împreună cu metadata:

```json
{
  "experiment_id": "EXP-A-001",
  "dataset_hash": "sha256:...",
  "split_manifest_hash": "sha256:...",
  "representation": "raw_pca",
  "model": "isolation_forest",
  "feature_dimension": 15,
  "random_seed": 42,
  "preprocessing_config": {...},
  "model_config": {...},
  "threshold": {"method": "percentile", "value": 0.73},
  "metrics": {...},
  "created_at": "ISO8601 timestamp"
}
```

**Implementation Steps:**
1. `app/ml/experiment_registry.py` — funcție `register_experiment_run(config, metrics) -> experiment_id`, format ID convențional (`EXP-{A|B|C}-{NNN}`).
2. Fiecare rulare din TASK 9.2/9.3/9.4 apelează această funcție, în loc să scrie manual fișiere de rezultate ad-hoc.

**Acceptance Criteria:**
- [ ] AC1: Fiecare experiment din matricea A/B/C are un `experiment_id` unic, generat automat, nu ales manual/hard-codat.
- [ ] AC2: Given un `experiment_id`, pot recupera întreaga configurație + metrici asociate (test de round-trip: salvare → citire → identitate).
- [ ] AC3: Frontend-ul (TASK 11.7, Experiments page) afișează `experiment_id` lângă fiecare rezultat din matrice, ca referință explicită (ex. "Experiment A — Raw + PCA + Isolation Forest — Run: EXP-A-001").

**Testing:** `backend/tests/ml/test_experiment_registry.py`.

**Definition of Done:** registry implementat, folosit de toate cele 3 experimente din Phase 9, fără duplicare de schemă față de TASK 6.5.

**Files/Modules Expected:** `backend/app/ml/experiment_registry.py`.

---

# EPIC 4 — Experimentul Central

## PHASE 9 — Central DSP vs Raw Experiment

### TASK 9.1 — Reprezentare Raw + PCA (dimensionalitate controlată)
**Priority:** P0 | **Dependencies:** 2.4, 5.4 | **Blocks:** 9.4

**Implementation Steps:**
1. Extrage ferestre brute (fără DSP) pentru train/val/test.
2. Fit PCA **doar pe train**, reducere la exact același număr de componente cât are feature vector-ul DSP (din task 5.3).
3. Aplică transformarea PCA pe val/test.

**Acceptance Criteria:**
- [ ] AC1 (CRITIC): PCA e fit-uit exclusiv pe ferestre din train (verificat prin cod — nicio cale de execuție nu permite fit pe val/test).
- [ ] AC2: Dimensionalitatea rezultată (raw+PCA) == dimensionalitatea feature vector-ului DSP, exact, verificat automat.
- [ ] AC3: Varianța explicată de componentele păstrate e raportată explicit (nu ascunsă), chiar dacă e mică.

**Testing:** `backend/tests/experiments/test_raw_pca.py`.

**Definition of Done:** reprezentare raw+PCA generată pentru toate split-urile, dimensionalitate confirmată egală cu DSP.

**Files/Modules Expected:** `backend/app/ml/dimensionality.py`.

---

### TASK 9.2 — Experiment A: Raw+PCA → Isolation Forest
**Priority:** P0 | **Dependencies:** 9.1, 6.2 (reutilizat), 8.3 | **Blocks:** 9.4

**Acceptance Criteria:**
- [ ] AC1: Antrenare identică metodologic cu 6.2 (doar pe "normal" din train), dar pe reprezentarea raw+PCA.
- [ ] AC2: Evaluat cu exact același framework din 8.1, pe același test set (aceleași ferestre) ca Experimentul B.
- [ ] AC3: Rularea primește un `experiment_id` unic prin registry-ul din TASK 8.3 (ex. `EXP-A-001`).

**Definition of Done:** model + metrici Experiment A generate, status `NOT YET MEASURED` → înlocuit cu valori reale după rulare.

**Files/Modules Expected:** `backend/scripts/run_experiment_a.py`, `models/experiment_a_isolation_forest.pkl`.

---

### TASK 9.3 — Experiment B: DSP Features → Isolation Forest
**Priority:** P0 | **Dependencies:** 5.3, 6.2, 8.3 | **Blocks:** 9.4

**Description:** Reutilizează direct modelul din Phase 6 (deja e exact acest experiment) — task-ul aici e doar de a-l încadra explicit în matricea comparativă, a-i atribui un `experiment_id` (ex. `EXP-B-001`) și a genera raportul asociat.

**Acceptance Criteria:**
- [ ] AC1: Modelul și metricile din Phase 6/8 sunt referite direct, fără reantrenare duplicată inutilă.
- [ ] AC2: Rularea are un `experiment_id` unic înregistrat prin TASK 8.3.

**Definition of Done:** Experiment B documentat ca referință la Phase 6/8, cu `experiment_id` atribuit.

---

### TASK 9.4 — Experiment C: DSP Features → Autoencoder
**Priority:** P0 | **Dependencies:** 7.3, 8.3 | **Blocks:** 9.5

**Description:** Reutilizează direct modelul din Phase 7, cu `experiment_id` propriu (ex. `EXP-C-001`).

**Acceptance Criteria:**
- [ ] AC1: Metricile din Phase 7/8 sunt referite direct în matricea comparativă.
- [ ] AC2: Rularea are un `experiment_id` unic înregistrat prin TASK 8.3.

**Definition of Done:** Experiment C documentat ca referință la Phase 7/8, cu `experiment_id` atribuit.

---

### TASK 9.5 — Agregare rezultate + raport final al experimentului central
**Priority:** P0 | **Dependencies:** 9.2, 9.3, 9.4 | **Blocks:** Phase 12 (UI), Phase 13 (README)

**Implementation Steps:**
1. Construiește tabelul final: Experiment A/B/C × (Precision, Recall, F1, ROC-AUC, PR-AUC).
2. Scrie interpretarea: ce înseamnă rezultatul, indiferent care variantă "câștigă".
3. Include vizualizare (ex. bar chart comparativ) pentru frontend (Phase 12).
4. Pentru fiecare din cele 3 experimente, salvează un artifact de reproducibilitate (JSON), conținând: `dataset_hash`, `split_manifest_hash` (aceleași ca în TASK 6.5), `preprocessing_config` (parametri detrend/normalize folosiți), `feature_set` (lista exactă de features, pentru B și C), `model_config` (hiperparametri), `random_seed`, `threshold_method` + `threshold_value`, metricile complete din task 8.1, `timestamp`.

**Acceptance Criteria:**
- [ ] AC1: Toate cele 3 celule ale matricei au valori reale măsurate, nu placeholder, la momentul finalizării task-ului.
- [ ] AC2: Raportul discută explicit dacă rezultatul confirmă sau infirmă ipoteza inițială (secțiunea 19 blueprint), fără a forța o concluzie predefinită.
- [ ] AC3: Dacă Experiment A (raw+PCA) e comparabil sau mai bun decât B, raportul discută explicit interpretarea alternativă (valoare DSP = interpretabilitate, nu neapărat performanță brută).
- [ ] AC4: Fiecare din cele 3 experimente are un artifact de reproducibilitate JSON complet (schema de mai sus), verificabil — dacă cineva rulează din nou același experiment cu artifact-ul salvat ca input de configurare, obține metrici identice (test explicit de reproducibilitate, nu presupunere).

**Definition of Done:** `docs/results/central_experiment_report.md` complet, cu date reale.

**Files/Modules Expected:** `docs/results/central_experiment_report.md`, `backend/scripts/aggregate_experiment_results.py`.

---

# EPIC 5 — API & Backend Integration

## PHASE 10 — FastAPI

### TASK 10.1 — Pydantic schemas (request/response) pentru toate endpoint-urile
**Priority:** P0 | **Dependencies:** Phase 2-9 (modele existente) | **Blocks:** 10.2–10.8

**Acceptance Criteria:**
- [ ] AC1: Fiecare schemă respinge explicit un payload cu tip de date greșit (ex. `sampling_rate: str` în loc de `float`), testat.

**Testing:** `backend/tests/api/test_schemas.py`.

**Files/Modules Expected:** `backend/app/api/schemas/*.py`.

---

### TASK 10.2 — Endpoint-uri Datasets (`GET /api/datasets`, `GET /api/datasets/{id}`)
**Priority:** P0 | **Dependencies:** 10.1, 2.x

**Acceptance Criteria:**
- [ ] AC1: Given un dataset existent, when `GET /api/datasets/{id}`, then răspunsul conține metadata reală (samples, sampling rate, channels) din audit, nu valori placeholder.
- [ ] AC2: Given un id inexistent, then HTTP 404 cu mesaj de eroare clar.

**Testing:** `backend/tests/api/test_datasets.py`.

**Files/Modules Expected:** `backend/app/api/routes/datasets.py`, `backend/app/services/dataset_service.py`.

---

### TASK 10.3 — Endpoint-uri Signal Processing (`/fft`, `/psd`, `/spectrogram`, `/dsp/filter`)
**Priority:** P0 | **Dependencies:** 10.1, Phase 3-4

**Acceptance Criteria:**
- [ ] AC1: Given un semnal valid + parametri de filtrare, when `POST /api/dsp/filter`, then răspunsul conține semnalul filtrat cu aceleași proprietăți verificate deja în testele Phase 3 (reutilizare logică, nu reimplementare).
- [ ] AC2: Given parametri invalizi (cutoff ≥ Nyquist), then HTTP 422 cu mesaj explicit, nu 500.

**Testing:** `backend/tests/api/test_signal_processing_routes.py`.

**Files/Modules Expected:** `backend/app/api/routes/signals.py`, `backend/app/services/dsp_service.py`.

---

### TASK 10.4 — Endpoint Feature Extraction (`POST /api/features/extract`)
**Priority:** P0 | **Dependencies:** 10.1, 5.3

**Acceptance Criteria:**
- [ ] AC1: Răspunsul conține exact numărul de features din registry, cu nume de coloane consistente.

**Testing:** `backend/tests/api/test_features_route.py`.

**Files/Modules Expected:** `backend/app/api/routes/features.py`.

---

### TASK 10.5 — Endpoint-uri Models (`GET /api/models`, `/api/models/{id}/performance`, `POST /api/models/predict`)
**Priority:** P0 | **Dependencies:** 10.1, Phase 6-8

**Acceptance Criteria:**
- [ ] AC1: `/api/models` listează Isolation Forest și Autoencoder cu metrici reale din Phase 8 (nu hard-codate în route).
- [ ] AC2: `POST /api/models/predict` returnează anomaly score normalizat [0,1] + status (NORMAL/WARNING/ANOMALY) + explicație text bazată pe features reale calculate pentru semnalul dat (nu text generic).

**Testing:** `backend/tests/api/test_models_route.py`.

**Files/Modules Expected:** `backend/app/api/routes/models.py`, `backend/app/services/model_service.py`.

---

### TASK 10.6 — Endpoint Experiments (`GET /api/experiments`, `/api/experiments/{id}`)
**Priority:** P1 | **Dependencies:** 10.1, 9.5

**Acceptance Criteria:**
- [ ] AC1: Returnează matricea experimentală (A/B/C) cu valorile reale din `central_experiment_report.md`/artifact JSON asociat.

**Testing:** `backend/tests/api/test_experiments_route.py`.

**Files/Modules Expected:** `backend/app/api/routes/experiments.py`.

---

### TASK 10.7 — Error handling global + validare centralizată
**Priority:** P0 | **Dependencies:** 10.2–10.6

**Acceptance Criteria:**
- [ ] AC1: Orice excepție necontrolată produce HTTP 500 cu body JSON structurat (nu stack trace expus în producție), logat server-side.
- [ ] AC2: Erorile de validare Pydantic produc HTTP 422 cu detalii per câmp.

**Testing:** `backend/tests/api/test_error_handling.py`.

**Files/Modules Expected:** `backend/app/api/error_handlers.py`.

---

### TASK 10.8 — OpenAPI docs + verificare completă
**Priority:** P1 | **Dependencies:** 10.2–10.7

**Acceptance Criteria:**
- [ ] AC1: `GET /docs` afișează toate endpoint-urile cu schema request/response corectă.

**Definition of Done:** documentație OpenAPI completă, verificată manual.

---

# EPIC 6 — Frontend

## PHASE 11 — Frontend

### TASK 11.1 — Application shell (sidebar, navigare, layout)
**Priority:** P0 | **Dependencies:** 1.5, 1.6 | **Blocks:** 11.2–11.9

**Acceptance Criteria:**
- [ ] AC1: Navigarea între toate paginile (Dashboard, Signals, Signal Analysis, DSP Lab, Anomaly Detection, Models, Experiments, Dataset) funcționează fără reload complet al paginii (React Router, client-side routing).
- [ ] AC2: Layout responsive — sidebar colapsează corect sub un breakpoint definit (verificat manual la 2 lățimi de ecran).

**Files/Modules Expected:** `frontend/src/layouts/AppShell.tsx`, `frontend/src/pages/*.tsx` (schelete).

---

### TASK 11.2 — Dashboard page
**Priority:** P0 | **Dependencies:** 11.1, 10.5

**Acceptance Criteria:**
- [ ] AC1: Given backend-ul cu date reale, Dashboard afișează anomaly score curent, status (NORMAL/WARNING/ANOMALY), threshold-ul folosit și numărul de semnale analizate, toate provenite din API, nu hard-codate. **Nu se afișează un "model confidence %"** — nici Isolation Forest, nici Autoencoder-ul din acest proiect nu produc o probabilitate calibrată; a afișa un procent de încredere neînsoțit de o metodă de calibrare reală ar fi echivalent cu a inventa o cifră.
- [ ] AC2: Loading state afișat cât timp cererea API e în curs; error state afișat dacă API-ul eșuează.

**Testing:** verificare manuală + eventual test de componentă (React Testing Library) pentru loading/error states.

**Files/Modules Expected:** `frontend/src/pages/Dashboard.tsx`.

---

### TASK 11.3 — Signal Analysis page (tabs: Time/Frequency/Spectrogram/Features/AI Analysis)
**Priority:** P0 | **Dependencies:** 11.1, 10.3, 10.4, 10.5

**Acceptance Criteria:**
- [ ] AC1: Graficul de time domain permite zoom și hover (verificat manual — Plotly built-in).
- [ ] AC2: Comutarea între taburi păstrează semnalul selectat (nu resetează selecția utilizatorului).

**Files/Modules Expected:** `frontend/src/pages/SignalAnalysis.tsx`, `frontend/src/components/charts/*.tsx`.

---

### TASK 11.4 — DSP Lab page (parametri filtru configurabili, comparație raw vs filtered)
**Priority:** P0 | **Dependencies:** 11.1, 10.3

**Acceptance Criteria:**
- [ ] AC1: Modificarea cutoff-ului filtrului din UI declanșează un nou apel API și actualizează graficul, fără reload de pagină.
- [ ] AC2: UI explică vizibil Nyquist frequency, cutoff, filter order (text scurt, nu doar slider fără context) — per cerința secțiunii 11 blueprint.

**Files/Modules Expected:** `frontend/src/pages/DSPLab.tsx`.

---

### TASK 11.5 — Models page
**Priority:** P1 | **Dependencies:** 11.1, 10.5

**Acceptance Criteria:**
- [ ] AC1: Ambele modele (Isolation Forest, Autoencoder) afișate cu metrici reale din API.
- [ ] AC2: Selectarea modelului activ persistă (măcar în state-ul sesiunii curente) și afectează pagina de Anomaly Detection.

**Files/Modules Expected:** `frontend/src/pages/Models.tsx`.

---

### TASK 11.6 — Anomaly Detection / AI Analysis view
**Priority:** P0 | **Dependencies:** 11.1, 10.5

**Acceptance Criteria:**
- [ ] AC1: Given un semnal analizat, afișează anomaly score, threshold-ul folosit, status, și indicatori text bazați pe valori reale de features (nu text generic hard-codat). **Nu afișează "confidence %"** — nedefinit matematic pentru modelele din acest proiect (vezi nota din TASK 11.2).
- [ ] AC2: Formularea din UI respectă explicit precizarea din blueprint secțiunea 17 — scorul e prezentat ca "cât de neobișnuit", nu ca "% defect fizic".
- [ ] AC3: UI afișează un text scurt de interpretare (ex. "Scor ridicat de anomalie față de baseline-ul normal învățat"), nu doar cifra goală.

**Files/Modules Expected:** `frontend/src/components/anomaly/AnomalyPanel.tsx`.

---

### TASK 11.7 — Experiments page (matricea A/B/C)
**Priority:** P1 | **Dependencies:** 11.1, 10.6

**Acceptance Criteria:**
- [ ] AC1: Afișează matricea Representation × Model din secțiunea 19 a blueprint-ului, cu valori reale din API.

**Files/Modules Expected:** `frontend/src/pages/Experiments.tsx`.

---

### TASK 11.8 — Dataset page
**Priority:** P1 | **Dependencies:** 11.1, 10.2

**Acceptance Criteria:**
- [ ] AC1: Afișează metadata reală din audit (samples, sampling rate, duration, channels, missing values) — nicio valoare placeholder.

**Files/Modules Expected:** `frontend/src/pages/Dataset.tsx`.

---

### TASK 11.9 — Loading/Error/Empty states consistente global
**Priority:** P1 | **Dependencies:** 11.2–11.8

**Acceptance Criteria:**
- [ ] AC1: Fiecare pagină care face fetch de date are cele 3 stări implementate (nu doar happy path).

**Files/Modules Expected:** `frontend/src/components/ui/LoadingState.tsx`, `ErrorState.tsx`, `EmptyState.tsx`.

---

# EPIC 7 — Explainability & Polish

## PHASE 12 — Explainability + 3D PCA + Polish

### TASK 12.1 — Anomaly explanation bazată pe features reale
**Priority:** P0 | **Dependencies:** 11.6, 5.3

**Acceptance Criteria:**
- [ ] AC1: Given un semnal anormal, explicația listează features cu deviație semnificativă față de baseline-ul normal (calculat, nu inventat) — ex. "RMS +32% vs. baseline normal", cu procentul calculat real.
- [ ] AC2: Textul UI nu descrie scorul ca "procent de deteriorare fizică" (per constrângerea explicită a blueprint-ului).

**Files/Modules Expected:** `backend/app/services/explanation_service.py`, `frontend/src/components/anomaly/ExplanationList.tsx`.

---

### TASK 12.2 — PCA 3D feature space visualization
**Priority:** P1 | **Dependencies:** 9.1, 11.7

**Acceptance Criteria:**
- [ ] AC1: Punctele din vizualizarea 3D sunt colorate după clasa reală (normal/fault types) din dataset, nu aleator.
- [ ] AC2: UI menționează explicit că PCA e folosit pentru vizualizare/reducere dimensională, nu ca dovadă de separabilitate garantată (per secțiunea 21 blueprint).

**Files/Modules Expected:** `frontend/src/components/charts/PCA3DPlot.tsx`.

---

### TASK 12.3 — Polish vizual final (consistență design tokens, animații subtile)
**Priority:** P2 | **Dependencies:** toate task-urile Phase 11

**Acceptance Criteria:**
- [ ] AC1: Zero culori hard-codate în afara `globals.css` (verificare manuală/grep).

---

# EPIC 8 — Final Validation

## PHASE 13 — Final Validation & Documentation

### TASK 13.1 — CWRU loader + validare cross-dataset
**Priority:** P1 | **Dependencies:** Phase 6-9 stabile

**Acceptance Criteria:**
- [ ] AC1: Modelele antrenate pe MAFAULDA sunt evaluate pe CWRU (după un feature extraction echivalent), rezultatele raportate onest, indiferent dacă generalizarea e bună sau slabă.

**Notes:** **Necesită și acest dataset urcat de utilizator** — aceeași constrângere de rețea ca MAFAULDA.

**Files/Modules Expected:** `backend/app/datasets/cwru_loader.py`, `docs/results/cross_dataset_validation.md`.

---

### TASK 13.2 — README final complet
**Priority:** P0 | **Dependencies:** toate fazele anterioare

**Acceptance Criteria:**
- [ ] AC1: README conține toate secțiunile din blueprint secțiunea 24 (Overview, Architecture, Signal Processing, ML, DL, Evaluation, Results, Installation, Usage, Future Work), plus secțiunea dedicată "Data Leakage Prevention".
- [ ] AC2: Nicio cifră de performanță din README nu e inventată — toate provin din `docs/results/*.md`.

**Files/Modules Expected:** `README.md`.

---

### TASK 13.3 — Cleanup, screenshots, documentație de interviu
**Priority:** P2 | **Dependencies:** 13.2

**Acceptance Criteria:**
- [ ] AC1: `docs/interview_prep.md` conține răspunsuri reale (nu generice) la întrebările din blueprint secțiunea 36, referind decizii concrete luate în proiect.

**Files/Modules Expected:** `docs/interview_prep.md`, `docs/screenshots/`.


---

# DEPENDENCY GRAPH (rezumat pe Phase, nu pe fiecare task individual)

```
Phase 1 (Setup)
   │
   ▼
Phase 1.5 (Dataset Audit) ◄── BLOCKER: necesită upload MAFAULDA
   │
   ▼
Phase 2 (Dataset Integration) ◄── depinde de split_manifest.json (1.5.7)
   │
   ▼
Phase 3 (Preprocessing + Filtering)
   │
   ▼
Phase 4 (FFT / PSD / Spectrogram) ◄── depinde de Phase 3 (preprocessing)
   │
   ▼
Phase 5 (Feature Engineering, inclusiv TASK 5.4 — Feature Scaling COMUN, nu deținut de IF) ◄── depinde de Phase 3 + Phase 4
   │
   ├──────────────┐
   ▼              ▼
Phase 6 (IF)   Phase 7 (Autoencoder)   ◄── ambele consumă ACELAȘI scaler din 5.4, independente una de alta
   │              │
   └──────┬───────┘
          ▼
     Phase 8 (Evaluation Framework + TASK 8.3 Experiment Run ID Registry) ◄── depinde de AMBELE Phase 6 și 7
          │
          ▼
     Phase 9 (Central Experiment) ◄── depinde de Phase 2 (raw data), Phase 5 (features), 
          │                            Phase 6 (IF reutilizat), Phase 7 (AE reutilizat), Phase 8 (framework)
          ▼
     Phase 10 (FastAPI) ◄── depinde de Phase 2-9 (toată logica de business există deja)
          │
          ▼
     Phase 11 (Frontend) ◄── depinde de Phase 10 (API funcțională)
          │
          ▼
     Phase 12 (Explainability + PCA 3D) ◄── depinde de Phase 9 (features/PCA) + Phase 11 (UI)
          │
          ▼
     Phase 13 (Final Validation) ◄── depinde de tot proiectul + upload CWRU
```

---

# CRITICAL PATH

Drumul care, dacă întârzie, întârzie tot proiectul:

```
1.1 → 1.2 → 1.3 → [Phase 1.5, TOATE task-urile, blocate de upload dataset]
→ 1.5.7 (split manifest) → 2.1 → 2.2 → 2.4 (windowing anti-leakage)
→ 3.1 → 3.2 → 4.1 → 4.2 → 5.1 → 5.2 → 5.3 → 5.4 (scaling comun)
→ 6.2 (antrenare IF) ȘI 7.2 (antrenare AE, în paralel logic — ambele consumă scaler-ul din 5.4)
→ 8.1 (evaluation framework) → 8.3 (experiment run ID registry)
→ 9.1 (raw+PCA) → 9.2 (Exp A) → 9.5 (raport final experiment central)
→ 10.5 (endpoint predict) → 11.6 (Anomaly Detection UI)
→ 13.2 (README final)
```

**Cel mai lung blocaj real de pe critical path:** Phase 1.5 — nu tehnic, ci de disponibilitate a datelor (upload utilizator). Restul lanțului (Phase 2 → 9) e blocaj tehnic normal, secvențial.

---

# IMPLEMENTATION ORDER (ordinea reală de execuție)

1. Phase 1 (task-urile 1.1 → 1.6, complet automatizabil, fără blocker)
2. **STOP la poarta Phase 1.5** — cerere explicită de upload dataset către utilizator
3. Phase 1.5 (după upload) → Phase 2 → Phase 3 → Phase 4 → Phase 5 (secvențial strict, fiecare depinde de precedenta)
4. Phase 6 și Phase 7 (pot fi raportate ca implementate în aceeași "rundă", dar codul se scrie secvențial, nu literal paralel)
5. Phase 8 → Phase 9 (experimentul central, piesa cu cea mai mare valoare)
6. Phase 10 (API completă peste logica deja validată)
7. Phase 11 (Frontend peste API funcțională)
8. Phase 12 (explainability + polish)
9. Phase 13 (necesită și upload CWRU pentru task 13.1; restul — README, cleanup — nu are blocker)

---

# PHASE GATE CRITERIA (condiție de trecere la faza următoare)

| Gate | Condiție de trecere |
|---|---|
| 1 → 1.5 | Ambele servere pornesc, `pytest` rulează verde pe testele Phase 1 (health endpoint) |
| 1.5 → 2 | `AUDIT_REPORT.md` complet, `split_manifest.json` generat și validat (fără overlap fișiere între split-uri) |
| 2 → 3 | Testul de leakage din windowing (2.4-AC1) trece explicit |
| 3 → 4 | Toate testele de filtrare/preprocessing verzi, verificate pe semnal sintetic cunoscut |
| 4 → 5 | FFT/PSD/Spectrogramă validate pe semnale sintetice cu frecvențe cunoscute |
| 5 → 6/7 | Matricea de features generată fără NaN/Inf pentru toate split-urile, ȘI scaler-ul comun (TASK 5.4) fit-uit o singură dată pe train, consumat identic de ambele modele |
| 6/7 → 8 | Ambele modele antrenate exclusiv pe date "normale" (verificat prin testele AC1 dedicate anti-leakage-de-labels), fiecare cu Model Artifact Contract (TASK 6.5) complet |
| 8 → 9 | Framework de evaluare comun, testat, aplicat identic pe ambele modele |
| 9 → 10 | Matricea experimentală A/B/C completă cu valori reale, nu placeholder |
| 10 → 11 | Toate endpoint-urile testate cu `TestClient`, `GET /docs` funcțional |
| 11 → 12 | Workflow complet click-through (dataset → analiză → predicție) fără erori |
| 12 → 13 | Explicațiile de anomalie verificate ca fiind bazate pe valori reale, nu text generic |
| 13 → DONE | README complet, toate cifrele raportate provin din `docs/results/*.md` |

---

**Backlog generat complet. Fără implementare încă executată în afara acestui document.**