# BLUEPRINT — AI-Powered Signal Anomaly Detection & Predictive Maintenance

> Document de decizii tehnice. Nu conține implementare. Toate rezultatele numerice sunt marcate `TO BE MEASURED` până la execuția reală a experimentelor.

---

## 1. Executive Summary

Construim o platformă care primește semnale de vibrații de la echipamente industriale și răspunde la întrebarea: *"Acest semnal este compatibil cu funcționarea normală, sau indică un comportament anormal?"*

Sistemul combină DSP clasic (filtrare, FFT, PSD, spectrogramă) cu două abordări de anomaly detection — Isolation Forest (baseline clasic) și un Autoencoder PyTorch (deep learning) — antrenate predominant pe date "normale" și evaluate cu label-uri reale din dataset. Piesa centrală a proiectului, din punct de vedere al valorii pentru interviu, este **experimentul comparativ raw signal → AI vs. raw signal → DSP → features → AI**, care demonstrează empiric dacă procesarea semnalului aduce valoare reală, nu presupusă.

Livrabilul final este o aplicație web (FastAPI + React/TypeScript), nu un notebook, nu un dashboard Streamlit.

---

## 2. Project Feasibility

Realist pentru un singur developer, lucrând incremental, cu următoarele constrângeri:

- **Nu estimăm volumul de muncă în număr de sesiuni.** Nu are sens să legăm un proiect de învățare (DSP + ML + DL + backend + frontend, învățate temeinic, nu doar copiate) de un deadline arbitrar. Criteriul de progres corect e phase-gate, nu cronometru:

  ```
  Phase N → implementare → teste → explicație → confirmare utilizator → Gate → Phase N+1
  ```

  O fază nu se consideră terminată pentru că "a trecut destul timp", ci pentru că criteriul ei de succes (definit în secțiunea 33) e îndeplinit și confirmat de tine.
- Riscul principal de eșec nu e tehnic (fiecare componentă e cunoscută în literatură), ci **de scope creep** — tentația de a adăuga 3D, Experiments tracking, History, Settings înainte ca fundația (DSP + baseline + experiment central) să fie solidă.
- Decizie: tratăm proiectul ca **MVP tăiat strict** + listă explicită de "later phases", nu ca un singur bloc monolitic de cerințe.

---

## 3. Problem Definition

Formulare corectă (per secțiunea 46 din cerința ta):

> **AI-powered vibration anomaly detection for predictive maintenance** — nu "failure prediction", nu "Remaining Useful Life", până nu implementăm efectiv degradation tracking.

Formulare metodologică (per secțiunea 5):

> Antrenare unsupervised/semi-supervised (modelul învață doar din date etichetate "normal"), evaluare supervizată (folosim label-urile reale ale dataset-ului ca ground truth la testare).

Această distincție se documentează explicit în README, secțiune dedicată, ca să nu pară o contradicție cuiva care citește "aveți labels, dar ziceți unsupervised?".

---

## 4. Dataset Research

### MAFAULDA (Machinery Fault Database, UFRJ)

- **Creator:** Laboratório de Sinais, Multimídia e Telecomunicações, UFRJ (Rio de Janeiro).
- **Disponibil:** public, pagina oficială UFRJ + oglindă pe Kaggle ("Machinery Fault Dataset").
- **Conținut:** 1951 serii temporale multivariate, achiziționate pe un simulator SpectraQuest Alignment-Balance-Vibration (ABVT). 6 stări: normal, dezechilibru, dezaliniere orizontală, dezaliniere verticală, defect rulment interior (underhang), defect rulment exterior (overhang).
- **Senzori:** accelerometru triaxial pe rulmentul underhang + 3 accelerometre industriale (axial/radial/tangențial) pe rulmentul overhang + tahometru + microfon → 8 coloane per fișier.
- **Sampling rate:** 50 kHz, ferestre de 5 secunde per fișier (250k eșantioane/canal).
- **Format:** CSV — direct utilizabil cu pandas, fără conversie MATLAB.
- **Probleme:** volum mare per fișier (~13GB tot dataset-ul); rotation frequency variază între fișiere, trebuie documentat ca variabilă de control.
- **Potrivire DSP:** foarte bună — semnal continuu, multi-canal, permite filtrare/FFT/spectrogramă pe fiecare canal separat.
- **Potrivire ML/DL:** bună — suficiente ferestre per clasă pentru antrenare Isolation Forest și Autoencoder.
- **Data leakage risk:** ridicat dacă split-ul se face naiv pe ferestre din același fișier (vezi secțiunea 8). Trebuie split la nivel de fișier/rulare.

**Dataset Audit Protocol — obligatoriu înainte de orice preprocessing (Phase 1.5, vezi secțiunea 33):**

Nu putem defini corect strategia de split (secțiunea 8) fără să înțelegem exact structura reală a datelor. Înainte de a scrie o singură linie de preprocessing, facem un audit dedicat, cu output documentat:

```
Dataset MAFAULDA (fișiere brute)
        ↓
Câte fișiere / recordings există, și cum sunt denumite?
        ↓
Ce reprezintă fiecare coloană/canal, exact, per fișier?
        ↓
Cum se identifică unic un "recording" (nume fișier → stare + condiție de operare)?
        ↓
Ce turații/rotation frequencies apar, și cum variază între fișiere din aceeași clasă?
        ↓
Există replicări ale aceleiași condiții (mai multe fișiere pentru aceeași stare+turație)?
        ↓
Distribuția claselor: e echilibrată? Câte fișiere per clasă?
        ↓
Ce poate ajunge legitim în train vs. test fără a încălca split-ul per fișier?
```

Rezultatul acestui audit (nu presupuneri) alimentează direct deciziile din secțiunea 8 (Data Leakage Strategy) și din Phase 2. Dacă auditul arată, de exemplu, că anumite clase au foarte puține fișiere, asta trebuie știut înainte de a proiecta split-ul train/val/test, nu descoperit după.

### CWRU Bearing Dataset (Case Western Reserve University)

- **Creator:** Case Western Reserve University Bearing Data Center.
- **Disponibil:** public, fișiere `.mat`.
- **Conținut:** defecte induse artificial (EDM) în rulmenți — bilă, inelul interior, inelul exterior, la diametre de 0.007"–0.021"/0.040", sub sarcini 0-3 HP.
- **Sampling rate:** 12 kHz și 48 kHz.
- **Format:** MATLAB `.mat` — necesită `scipy.io.loadmat`, mai puțin prietenos decât CSV.
- **Probleme:** un singur tip de mașină/rulment, mediu de laborator strict controlat; e cel mai citat benchmark din literatură, ceea ce e un avantaj de recunoaștere, dar și un semnal că modelele pot fi supra-optimizate pentru el.
- **Potrivire pentru proiect:** excelent ca **dataset secundar de validare** — testăm dacă pipeline-ul nostru (antrenat conceptual pe logica MAFAULDA) generalizează pe un dataset diferit ca proveniență, senzori și tip de fault.

### Al treilea dataset analizat: NASA IMS Bearing Dataset

- **Conținut:** rulmenți rulați până la defect real (run-to-failure), util pentru degradation tracking / RUL.
- **Motiv respingere pentru MVP:** etichetarea normal/anomaly nu e binară dată — trebuie definită de noi ("de la ce punct temporal considerăm degradare?"), ceea ce introduce subiectivitate greu de justificat într-un MVP. Fișierele sunt mari și greu de manevrat. **Rămâne candidat pentru extensia "Remaining Useful Life"** (secțiunea 45), nu pentru MVP.

---

## 5. Dataset Comparison Table

| Criteriu | MAFAULDA | CWRU | NASA IMS |
|---|---|---|---|
| Format | CSV | `.mat` | ASCII, mare |
| Canale | 8 (multi-senzor) | 1-3 per fișier | 4 canale |
| Sampling rate | 50 kHz | 12/48 kHz | 20 kHz |
| Clase | 6 (echilibrat pe tip de fault) | 4 (normal + 3 fault) | Nu are labels discrete |
| Etichetare | Clară, per fișier | Clară, per fișier | Ambiguă (necesită definire proprie) |
| Leakage risk | Ridicat dacă split naiv | Ridicat dacă split naiv | N/A pentru clasificare |
| Potrivire DSP | Foarte bună | Foarte bună | Bună, dar scop diferit (RUL) |
| Potrivire MVP | **Primary** | **Secondary validation** | Respins pentru MVP |
| Licență/uz | Public, research use | Public, research use | Public, research use |

---

## 6. Recommended Primary Dataset

**MAFAULDA.** Motiv: multi-canal + 6 clase variate (nu doar bearing fault, ci și imbalance/misalignment) dă un anomaly score cu gradație reală, CSV nativ simplifică ingestion, iar bogăția de senzori susține paginile Dataset/Signal Analysis din cerințele UI.

## 7. Recommended Secondary Dataset

**CWRU.** Folosit exclusiv pentru validare cross-dataset în faza finală (Phase 13/later) — nu pentru dezvoltare MVP. Scop: demonstrarea (sau infirmarea onestă) a generalizării pipeline-ului.

---

## 8. Data Leakage Strategy

**Decizie:** split-ul se face la nivel de **fișier de înregistrare** (recording), nu la nivel de fereastră individuală.

- **Motiv:** ferestre consecutive din același fișier MAFAULDA sunt înalt corelate (aceeași rulare, aceeași condiție de operare, zgomot de fond identic). Un split random pe ferestre ar lăsa modelul să "recunoască" caracteristici specifice rulării, nu caracteristici generale ale stării de fault.
- **Alternativă respinsă:** split random pe ferestre (comun în multe tutoriale online). Trade-off: ar da metrici artificial umflate (F1 fals-optimist), inutilizabile ca dovadă reală de competență.
- **Implementare:** `train/val/test` se atribuie per fișier (per `recording_id`), înainte de windowing. Normalizarea (mean/std) se fit-uiește **doar** pe train, apoi se aplică pe val/test — niciodată invers.
- **Overlap în windowing:** overlap-ul (ex. 50%) e permis **în interiorul** unei rulări alocate unui singur split, nu între ferestre care ar ajunge în split-uri diferite.
- **Documentare:** secțiune dedicată "Data Leakage Prevention" în README, cu diagrama exactă a split-ului.

---

## 9. Product Vision

Aplicație web care simulează o platformă reală de condition monitoring: utilizatorul selectează un dataset/semnal, îl explorează în timp/frecvență, rulează detecția de anomalii, vede scorul și explicația, compară modele. Nu e un notebook cu grafice statice — fiecare interacțiune (zoom, selectare interval, schimbare parametri filtru) recalculează sau re-randează live.

---

## 10. System Architecture

```
React (Vite/TS) ──HTTP/JSON──> FastAPI ──> Services layer
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
            signal_processing/            features/                  ml/
         (preprocessing, filtering,   (time-domain,           (isolation_forest,
          fft, psd, spectrogram)       frequency-domain)        autoencoder, scoring)
                    │                         │                         │
                    └─────────────────────────┴─────────────────────────┘
                                              ▼
                                   models/ (fișiere .pkl / .pt salvate pe disc)
                                              ▼
                                   SQLite (metadata, experiment runs, predictions)
```

Fiecare layer e testabil izolat (pytest), fără dependență de FastAPI — DSP-ul și feature engineering-ul sunt funcții pure NumPy/SciPy.

---

## 11. Backend Architecture

Structura propusă de tine (secțiunea 27) e corectă și o păstrăm aproape identic, cu o singură adăugire: `app/api/schemas/` separat de orice model de domeniu, ca validarea Pydantic a request/response-urilor să nu se amestece cu logica de business.

```
backend/
└── app/
    ├── api/
    │   ├── routes/        (health, datasets, signals, models, experiments)
    │   └── schemas/        (Pydantic request/response models)
    ├── core/               (config.py, logging.py)
    ├── datasets/           (loader.py, validators.py)
    ├── signal_processing/  (preprocessing, filtering, fft, psd, spectrogram, windowing)
    ├── features/           (time_domain, frequency_domain, extractor)
    ├── ml/                 (isolation_forest, autoencoder, training, inference, scoring, evaluation)
    ├── services/           (orchestrare — combină layerele de mai sus)
    └── main.py
```

**Decision → Reason → Alternative → Trade-off:**
- **Decision:** logica de business stă în `services/`, route handlers sunt subțiri (doar validare + apel service).
- **Reason:** testabilitate — poți testa `signal_analysis_service.py` fără să pornești serverul HTTP.
- **Alternative:** logică direct în route handlers (comun în tutoriale FastAPI simple).
- **Trade-off:** un layer în plus de indirecție, dar câștigi teste unitare rapide și separare clară responsabilitate.

---

## 12. Frontend Architecture

Structura din cerința ta (secțiunea 29) e adoptată integral. O singură notă: `components/charts/` va conține wrapper-e subțiri peste Plotly (ex. `TimeSeriesChart.tsx`, `FFTChart.tsx`) ca să nu repetăm configurare Plotly în fiecare pagină.

---

## 13. DSP Architecture

Pipeline: `raw signal → preprocessing (detrend, normalize) → windowing → filtering (opțional, configurabil) → FFT/PSD/STFT`.

- **Filtering:** Butterworth low-pass/high-pass/band-pass, ordin și frecvențe de tăiere configurabile prin request, nu hard-codate. Aplicăm `filtfilt` (zero-phase) ca să evităm distorsiunea de fază care ar deplasa artificial evenimentele în timp — critic pentru vibration analysis unde timing-ul defectelor contează.
- **FFT:** simplu, rapid, bun pentru identificarea frecvențelor dominante pe un segment static.
- **Welch PSD:** preferat față de FFT simplu când vrem o estimare mai robustă la zgomot (mediază peste mai multe ferestre suprapuse) — explicăm trade-off-ul rezoluție-frecvență vs. varianță-redusă în UI, nu doar în cod.
- **STFT/Spectrogramă:** pentru a vedea evoluția frecvențelor în timp, cu trade-off explicit window size (rezoluție frecvență) vs. hop length (rezoluție timp), configurabil din UI (DSP Lab).

---

## 14. Feature Engineering Architecture

Modul separat, fiecare feature = funcție pură cu test dedicat.

**Time-domain:** mean, std, variance, RMS, peak, peak-to-peak, skewness, kurtosis, crest factor.
**Frequency-domain:** dominant frequency, spectral centroid, spectral bandwidth, spectral energy, spectral entropy, energie pe benzi de frecvență.

Pentru fiecare, documentăm (per secțiunea 15 din cerința ta): formulă, interpretare fizică, relevanță pentru fault detection (ex: kurtosis crește la impact-uri scurte, tipice pentru defecte de rulment; crest factor detectează vârfuri fără să crească media). Extensibilitatea vine din faptul că `extractor.py` iterează un registry de funcții — adăugarea unui feature nou nu modifică restul codului.

---

## 15. ML Architecture — Isolation Forest

Pipeline: `normal windows → feature extraction → feature matrix → IsolationForest.fit() → anomaly score (raw) → normalizare → threshold → NORMAL/WARNING/ANOMALY`.

**Decision → Reason → Alternative → Trade-off:**
- **Decision:** Isolation Forest ca baseline.
- **Reason:** nu necesită presupuneri de distribuție, e rapid de antrenat, funcționează bine pe feature vectors de dimensiune moderată, e ușor de explicat la interviu (izolează observații prin partiționare random — anomaliile se izolează în mai puține tăieturi).
- **Alternative:** One-Class SVM, Local Outlier Factor.
- **Trade-off:** Isolation Forest e mai rapid și scalează mai bine decât One-Class SVM, dar e mai puțin sensibil la anomalii "locale" într-un cluster dens decât LOF — acceptabil pentru un baseline, discutat explicit ca limitare.

---

## 16. Autoencoder Architecture

**Decizia A vs B (secțiunea 19 din cerința ta):**

| | Raw waveform segments | Feature vectors |
|---|---|---|
| Interpretabilitate | Scăzută (ce a "văzut" rețeaua?) | Ridicată (poți lega reconstruction error de feature specific) |
| Complexitate arhitectură | Mai mare (conv1d layers) | Mică (dense layers) |
| Cost computațional | Mai mare | Mic |
| Legătură cu DSP | Slabă — ignoră tot ce am construit la features | Puternică — validează pipeline-ul DSP |
| Relevanță pentru experimentul central | Nu contribuie direct | Contribuie direct |

**Recomandare: Autoencoder pe feature vectors** pentru MVP, ca variantă principală — se leagă direct de experimentul central (secțiunea 21) unde comparăm raw vs. DSP+features. **Autoencoder pe raw waveform** rămâne un experiment opțional ulterior (secțiunea 45), nu obligatoriu pentru MVP, tocmai ca să nu dublăm scope-ul pe o comparație care nu e centrală poveștii proiectului.

Arhitectură: encoder 2-3 straturi dense (dimensiune input = nr. features → bottleneck mic), decoder simetric. Suficient de simplă încât să poți desena arhitectura pe o coală de hârtie într-un interviu.

---

## 17. Anomaly Scoring

- **Raw model score:** ieșirea brută (decision_function pentru Isolation Forest, MSE de reconstrucție pentru Autoencoder) — scale diferite, necomparabile direct.
- **Normalized anomaly score:** min-max sau percentile-scaling pe baza distribuției scorurilor din **validation set** (nu din test set — altfel leakage), mapat la [0, 1].
- **Threshold:** calibrat pe validation set folosind percentila scorurilor pe date normale (ex. percentila 95) sau prin analiză ROC/Precision-Recall dacă avem suficiente exemple de fault în validation. Pragurile NORMAL/WARNING/ANOMALY (0.30/0.70) din cerința inițială sunt **puncte de plecare ilustrative**, recalibrate empiric — nu adevăr universal, exact cum ai cerut.

**Precizare importantă de interpretare:** faptul că dataset-ul are 6 clase/stări distincte **nu înseamnă automat** că anomaly score-ul modelului reprezintă o gradație fizică a severității defectului. Un scor de 0.6 nu înseamnă "60% defect" — modelul nu are noțiune de severitate mecanică reală, doar de cât de departe e un semnal față de distribuția "normalului" învățat la antrenare. Formularea corectă, folosită consecvent în UI și README:

> **Anomaly score = cât de neobișnuit (out-of-distribution) este semnalul analizat față de comportamentul normal învățat de model, nu o măsură calibrată de severitate fizică a defectului.**

Cele 6 clase din MAFAULDA sunt folosite pentru **evaluare** (verificăm dacă scorurile modelului separă corect normal de diversele tipuri de fault) și pentru **vizualizare** (PCA 3D, secțiunea 21), nu ca etichetă de severitate pe care modelul o "știe" în mod direct.

---

## 18. Evaluation Strategy

Metrici: Precision, Recall, F1, ROC-AUC, PR-AUC (mai relevant decât ROC-AUC dacă clasele sunt dezechilibrate), confusion matrix, FPR, FNR, timp de antrenare, timp de inferență.

**De ce nu doar accuracy:** dacă 90% din ferestre sunt "normal", un model care prezice mereu "normal" are 90% accuracy fiind complet inutil. F1/PR-AUC forțează să vedem echilibrul real precision-recall.

---

## 19. Central DSP vs Raw Experiment

Aceasta rămâne **piesa centrală a proiectului**, cum ai cerut explicit. Reformulată ca matrice experimentală clară, cu un protocol explicit de comparație corectă (fair comparison), nu ca o singură comparație A/B ambiguă.

### Problema de fond, dacă nu tratăm dimensionalitatea explicit

O fereastră de semnal brut la 50 kHz poate avea sute sau mii de eșantioane; un feature vector DSP are 20-30 valori. Dacă comparăm direct `raw (mii de dimensiuni) → Isolation Forest` cu `features (~25 dimensiuni) → Isolation Forest`, orice diferență de performanță poate proveni **din dimensionalitate**, nu din faptul că DSP "extrage informație mai bună". Cele două variabile (reprezentare vs. dimensionalitate) sunt confundate dacă nu sunt separate explicit — acesta e riscul metodologic real al experimentului, și trebuie tratat ca atare, nu ignorat.

### Matricea experimentală

| Reprezentare input | Model | Rol în experiment |
|---|---|---|
| Raw waveform (dimensionalitate controlată — vezi mai jos) | Isolation Forest | Experiment A |
| DSP → Feature vectors (~20-30 dim) | Isolation Forest | Experiment B |
| DSP → Feature vectors (~20-30 dim) | Autoencoder | Experiment C |
| Raw waveform necomprimat | Autoencoder | Later phase (secțiunea 16/38), nu MVP |

Experimentele A și B sunt comparabile direct (același model — Isolation Forest — variabilă unică fiind reprezentarea). Experimentul C arată dacă trecerea la un model de deep learning, pe aceeași reprezentare DSP, aduce ceva față de un model clasic. Varianta "Autoencoder pe raw waveform necomprimat" rămâne opțională, later, tocmai pentru că ar introduce simultan două variabile noi (model + reprezentare) față de baseline — nu contribuie la o comparație curată.

### Protocolul de tratare a dimensionalității pentru brațul Raw (Experiment A)

Ca Experimentul A să fie o comparație corectă cu Experimentul B, dimensionalitatea input-ului brut trebuie tratată explicit, documentat, cu o singură metodă aleasă și justificată — nu lăsată "cum iese":

- **Opțiunea 1 — downsampling controlat:** reducem fereastra brută la un număr fix de eșantioane (ex. prin subsampling sau decimation cu anti-aliasing), comparabil ca ordin de mărime cu ce e computațional tratabil, dar documentăm explicit că această reducere pierde informație de frecvență înaltă (efect Nyquist), spre deosebire de feature-urile DSP care condensează informație fără acest compromis brutal.
- **Opțiunea 2 — PCA pe raw waveform**, reducând la același număr de componente cât are feature vector-ul DSP (~20-30), ca dimensionalitatea să fie literalmente egală între cele două brațe. Aceasta izolează cel mai curat variabila "reprezentare" de variabila "dimensionalitate".
- **Decizie:** folosim **Opțiunea 2 (PCA pe raw, la aceeași dimensionalitate ca feature vector-ul DSP)** ca protocol principal, tocmai pentru că egalizează dimensionalitatea între brațe și lasă drept variabilă reală doar tipul de reprezentare (raw comprimat generic vs. features inginerite cu semnificație fizică). Documentăm explicit în raportul experimentului că PCA pe raw e ea însăși o formă de "feature extraction" generică, nesupervizată — comparația reală devine *"features generice (PCA) vs. features inginerite cu semnificație fizică (DSP)"*, formulare mult mai precisă decât "raw vs DSP".

### Ipoteză, control, rezultat

- **Ipoteză:** feature engineering bazat pe DSP (cu semnificație fizică — RMS, kurtosis, frecvență dominantă etc.) separă mai bine normal/anomaly decât o reprezentare generică de aceeași dimensionalitate (PCA pe raw).
- **Control:** același split train/val/test (per fișier, fără leakage — secțiunea 8), același algoritm ML, aceeași dimensionalitate a input-ului între brațele A și B, aceleași metrici.
- **Rezultat:** `TO BE MEASURED` pentru toate cele 3 experimente din matrice — nu presupunem răspunsul. Dacă Experimentul A (PCA pe raw) e comparabil sau mai bun decât B, îl raportăm onest și discutăm de ce (posibil: componentele principale ale semnalului brut captează deja varianța dominantă legată de fault, caz în care concluzia corectă e că valoarea DSP stă mai mult în *interpretabilitate* decât în performanță brută — rezultat la fel de valid și de interesant la interviu ca un "DSP câștigă clar").

---

## 20. 2D Visualization Strategy

Line charts (raw/filtered overlay), FFT magnitude spectrum, PSD (Welch), spectrogramă (heatmap 2D cu colorbar — timp x frecvență x magnitudine ca și culoare), distribuții de features (histograme per clasă), confusion matrix, curbe ROC și Precision-Recall, anomaly score timeline. Toate interactive via Plotly (zoom, hover, pan, selectare interval).

---

## 21. 3D Visualization Strategy

Aici aplic scepticismul discutat înainte de blueprint.

- **3D time-frequency surface (waterfall):** evaluăm, dar **nu promitem** că intră în MVP. O spectrogramă 2D cu colorbar e, în practica reală de vibration analysis, adesea mai ușor de citit decât un surface plot 3D rotativ (ocluzie, unghi de vizualizare, dificultate de a citi valori exacte). O includem doar dacă, după ce avem spectrograma 2D funcțională, testăm empiric că 3D adaugă informație vizuală reală (ex. pentru a compara mai multe semnale simultan pe aceeași axă de timp). Altfel, rămâne "later phase", nu MVP.
- **3D PCA feature space:** aici 3D chiar aduce valoare — separarea vizuală normal/warning/anomaly/clase de fault în spațiul primelor 3 componente principale e greu de reprodus la fel de clar în 2D cu multe clase suprapuse. **Aceasta intră în MVP** (Phase 12), cu mențiunea explicită în UI că PCA e folosit pentru vizualizare/reducere dimensională, nu ca dovadă de separabilitate perfectă.
- **3D health/equipment trajectory:** rămâne idee pentru "future extensions" (secțiunea 38), condiționată de date longitudinale reale (MAFAULDA nu are degradare progresivă documentată per mașină).

---

## 22. Database Strategy

**Decizie: SQLite pentru MVP, nu PostgreSQL.**

- **Reason:** aplicație single-user, locală, fără concurență reală de scriere. SQLite elimină nevoia de docker-compose, migrations separate, connection pooling — infrastructură care nu aduce valoare la acest stadiu.
- **Alternative:** PostgreSQL de la început.
- **Trade-off:** PostgreSQL ar fi "mai production-like", dar ar adăuga complexitate operațională (server separat, credentials, backup) fără beneficiu funcțional pentru un proiect de portofoliu local. Documentăm explicit în README calea de upgrade (SQLAlchemy cu driver swap la Postgres, dacă proiectul ar trebui vreodată multi-user).

Entități (SQLAlchemy models): `datasets`, `signals`, `signal_windows`, `features`, `models`, `model_runs`, `predictions`, `experiments`, `experiment_results`.

---

## 23. API Design

Structura din cerința ta (secțiunea 28) e bună. O ajustare: separăm clar endpoint-urile de "calcul pe cerere" (FFT, filtrare — stateless, rapide) de cele de "operații pe modele" (train, predict — pot dura mai mult, candidate pentru background tasks mai târziu).

```
GET  /api/health
GET  /api/datasets
POST /api/datasets/upload
GET  /api/datasets/{id}
GET  /api/signals
GET  /api/signals/{id}
POST /api/signals/analyze
POST /api/signals/fft
POST /api/signals/psd
POST /api/signals/spectrogram
POST /api/dsp/filter
POST /api/features/extract
POST /api/models/train
GET  /api/models
GET  /api/models/{id}
POST /api/models/predict
GET  /api/models/{id}/performance
POST /api/experiments/run
GET  /api/experiments
GET  /api/experiments/{id}
```

---

## 24. Testing Strategy

- **Unit:** fiecare funcție DSP (preprocessing, filtering, FFT, PSD, spectrogram, windowing) și fiecare feature, testate cu semnale sintetice cunoscute (ex: sinusoidă pură de frecvență cunoscută → verificăm că FFT identifică exact acea frecvență).
- **Integration:** pipeline complet dataset → DSP → features → model → predicție, pe un subset mic de date reale.
- **API tests:** FastAPI `TestClient`, validare status codes + scheme de răspuns.
- **ML tests:** reproducibilitate (același seed → același rezultat), save/load model → predicții identice, validare shape-uri input/output, comportament threshold la limite.

---

## 25. Reproducibility

Fiecare experiment rulat produce o înregistrare cu: dataset folosit, strategia de split, configurație windowing (size, overlap), configurație preprocessing, lista de features, model + hiperparametri, threshold-ul folosit, metricile rezultate, seed. Stocat ca JSON + rând în `experiments`/`experiment_results` (SQLite). Seeds fixate (`numpy`, `torch`, `sklearn`) în `core/config.py`.

---

## 26. Security

Pentru MVP: variabile de mediu (`.env`, niciodată commit-uite), validare Pydantic strictă pe toate input-urile API, limită de dimensiune fișier la upload, whitelist de extensii acceptate (`.csv`), fără execuție de cod arbitrar la parsare. Documentăm explicit în README ce **nu** e rezolvat (autentificare, rate limiting, HTTPS) — nu pretindem production-grade security.

---

## 27. Performance

Nu optimizăm prematur. Punctele care merită atenție doar dacă devin bottleneck real, măsurat: FFT pe semnale foarte lungi (batch pe ferestre, nu pe semnalul întreg), calcul spectrogramă (cost STFT crește cu overlap), antrenare Autoencoder (batching standard PyTorch). Caching de rezultate DSP pentru același semnal+parametri, doar dacă UI-ul devine vizibil lent la testare manuală.

---

## 28. Deployment

MVP: strict local development (`uvicorn` + `vite dev`). Docker rămâne "later phase" — introdus doar când vrem să simplificăm onboarding-ul pentru cineva care clonează repo-ul, nu ca cerință de la început.

---

## 29. Repository Structure

```
signal-anomaly-platform/
├── backend/
│   ├── pyproject.toml
│   └── uv.lock
├── frontend/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
├── notebooks/
├── scripts/
├── docs/
├── .env.example
├── .gitignore
└── README.md
```

**Corecție (2026-09-06):** varianta inițială a acestei secțiuni plasa `pyproject.toml`/`uv.lock` la rădăcina repo-ului — copiat neadaptat din structura de referință a cererii inițiale, fără să reflecte decizia deja luată în secțiunea 11 (Backend Architecture) ca tot codul Python să trăiască izolat în `backend/`. Corectat aici ca să fie consistent cu TASK 1.2 din backlog (singura sursă care a fost și implementată). Motiv: `backend/` (Python/uv) și `frontend/` (Node/npm) sunt ecosisteme de dependențe complet separate; un `pyproject.toml` la rădăcină ar sugera greșit o dependență comună care nu există.

Am eliminat folderul `experiments/` separat de top-level (secțiunea 35 din cerința ta) — experiment metadata trăiește în SQLite + `models/` (checkpoints), nu ca fișiere libere pe disc, ca să evităm duplicare de sursă-a-adevărului.

---

## 30. Design System

Dark-first, o singură paletă de accent (nu multiple culori "vibrante" concurente), typography cu maxim 2 familii de fonturi (ex. Inter pentru UI, JetBrains Mono pentru valori numerice/cod), spacing pe grid consistent (4px/8px), borders subtile în loc de umbre puternice, badge-uri de status cu culoare semantică fixă (verde/galben/roșu pentru NORMAL/WARNING/ANOMALY, consistent peste tot). Tokens-urile se definesc **înainte** de prima componentă UI, nu improvizate pe parcurs.

---

## 31. UI Pages

Dashboard, Signals, Signal Analysis, DSP Lab, Anomaly Detection, Models, Experiments, Dataset, History, Settings — toate din cerința ta, dar prioritizate diferit în roadmap (secțiunea 32).

---

## 32. Roadmap (rezumat, detaliat în secțiunea 33)

Phase 0 (acest document) → Phase 1 (setup) → **Phase 1.5 (Dataset Audit — obligatorie, vezi secțiunea 4)** → Phase 2 (dataset integration, pe baza auditului) → Phase 3 (DSP foundation) → Phase 4 (FFT/PSD/spectrogram) → Phase 5 (features) → Phase 6 (Isolation Forest) → Phase 7 (Autoencoder) → Phase 8 (model comparison IF vs AE pe features) → **Phase 9 (matricea experimentală DSP vs Raw — Experimente A/B/C, secțiunea 19)** → Phase 10 (FastAPI) → Phase 11 (Frontend core) → Phase 12 (explainability + PCA 3D + polish) → Phase 13 (evaluare finală + CWRU validation + docs).

Fiecare săgeată de mai sus e un **gate**: trecerea la faza următoare are loc doar după ce criteriul de succes al fazei curente (secțiunea 33) e demonstrat și confirmat de tine, nu după un număr fix de sesiuni.

---

## 33. Phase-by-Phase Deliverables

| Fază | Deliverable | Criteriu succes |
|---|---|---|
| 1 | Repo, uv, FastAPI skeleton, Vite skeleton | ambele servere pornesc, `pytest` rulează |
| 1.5 | **Dataset Audit** — inventar fișiere, canale, recordings, turații, distribuție clase | document de audit scris, pe baza căruia se ia decizia finală de split (secțiunea 8) |
| 2 | Loader + validator MAFAULDA subset, pe baza auditului | semnal accesibil programatic cu metadata validată |
| 3 | Preprocessing + filtering + windowing, teste | semnal original vs. procesat comparabil, teste verzi |
| 4 | FFT + Welch PSD + STFT/spectrogramă | frecvență dominantă corect identificată pe semnal sintetic cunoscut |
| 5 | Feature extractor complet, teste per feature | matrice de features reproductibilă |
| 6 | Isolation Forest antrenat pe features, threshold calibrat | metrici reale (nu inventate) pe test set |
| 7 | Autoencoder pe feature vectors, antrenat | reconstruction error separă vizibil normal de fault pe validation |
| 8 | Tabel comparativ IF vs AE (ambele pe features) | metrici complete, fără presupuneri anterioare |
| 9 | **Matricea experimentală A/B/C** (raw-PCA vs features, IF vs AE) rulată și documentată | rezultat + interpretare pentru fiecare celulă, indiferent de cine "câștigă" |
| 10 | API completă, documentată (OpenAPI) | toate endpoint-urile testate cu `TestClient` |
| 11 | Dashboard, Signal Analysis, DSP Lab, Models funcționale | workflow complet click-through fără erori |
| 12 | Explainability (indicatori text), PCA 3D, polish vizual | anomaly explanation bazată pe valori reale calculate |
| 13 | CWRU validation, README complet, screenshots | proiect prezentabil pe GitHub/LinkedIn |

---

## 34. Risks

- **Tehnic:** Autoencoder pe feature vectors cu puține exemple de fault per clasă poate avea validation set prea mic pentru un threshold stabil — mitigare: raportăm interval de încredere / sensibilitate a threshold-ului, nu un singur număr rigid.
- **Metodologic:** riscul de a interpreta greșit rezultatul experimentului central dacă split-ul are dezechilibru de clase între train/val/test — mitigare: verificăm distribuția claselor per split înainte de a trage concluzii.
- **De scope:** tentația de a construi History/Settings/Experiments page înainte ca experimentul central să fie solid — mitigare: ordinea din roadmap protejează explicit Phase 9 înaintea paginilor secundare de UI.

---

## 35. Technical Debt (acceptat conștient pentru MVP)

- Fără autentificare/multi-user (documentat, nu ascuns).
- Fără streaming real-time (semnale statice din dataset, nu senzor live).
- 3D waterfall time-frequency amânat, posibil niciodată implementat dacă nu aduce valoare reală față de 2D.
- MLflow: **decizie explicită de a NU-l introduce în MVP**. Reason: sistemul de reproducibility descris în secțiunea 25 (JSON + SQLite) acoperă nevoile unui singur developer care rulează experimente secvențial. Alternative: MLflow de la început. Trade-off: MLflow ar aduce UI de comparare experimente gata făcut, dar adaugă un serviciu în plus de rulat/înțeles/configurat pentru un beneficiu marginal la scara asta — se reevaluează dacă numărul de experimente rulate depășește ce poate fi urmărit confortabil manual.

---

## 36. Interview Questions (pregătire, se completează pe parcurs cu răspunsuri concrete din proiect)

- De ce Isolation Forest și nu Random Forest? (Random Forest e supervizat, are nevoie de ambele clase în volum — Isolation Forest izolează anomalii fără a necesita exemple de fault în antrenare.)
- De ce Autoencoder și ce înseamnă reconstruction error?
- Cum ai prevenit data leakage? (split per fișier, nu per fereastră)
- De ce FFT și ce e frecvența Nyquist?
- De ce Welch PSD în loc de FFT simplu?
- Ce se întâmplă când schimbi fereastra STFT?
- Cum ai calibrat threshold-ul de anomaly?
- De ce F1 nu e suficient, și de ce accuracy poate induce în eroare?
- Cum știi că DSP chiar ajută modelul? (experimentul central)
- Ce s-ar întâmpla dacă se schimbă condițiile de operare (altă mașină, altă turație)?

---

## 37. Learning Objectives

Înțelegerea reală (nu doar apel de funcție) a: Nyquist/aliasing, spectral leakage și de ce windowing-ul contează, trade-off rezoluție timp/frecvență în STFT, de ce Butterworth + `filtfilt` pentru zero-phase, cum izolează Isolation Forest anomalii, ce învață un Autoencoder și de ce reconstruction error e un proxy valid pentru anomalie, de ce PCA e doar pentru vizualizare aici, nu feature selection "magică".

---

## 38. Future Extensions

CWRU ca a doua sursă de validare (deja planificat pentru Phase 13), MLflow (dacă volumul de experimente o justifică), Docker + GitHub Actions, PostgreSQL (dacă devine multi-user), streaming semnal live (integrare ESP32 + senzor real), degradation tracking / Remaining Useful Life folosind NASA IMS, 3D time-frequency waterfall (dacă testarea empirică arată valoare reală față de 2D), cloud deployment.

---

## 39. Final Recommendation

### Ce construim în MVP
Dataset MAFAULDA (subset), pipeline DSP complet (preprocessing, filtering, FFT, PSD, spectrogram), feature engineering modular și testat, Isolation Forest + Autoencoder pe feature vectors, **experimentul central DSP vs Raw**, API FastAPI completă, frontend cu Dashboard/Signal Analysis/DSP Lab/Models funcționale, PCA 3D pentru feature space, SQLite pentru metadata.

### Ce NU construim în MVP
PostgreSQL, MLflow, Docker, autentificare, streaming real-time, 3D time-frequency waterfall, History page completă, Settings page completă, Autoencoder pe raw waveform, CWRU integration (amânat la finalul MVP-ului ca validare, nu ca parte a lui).

### Ce construim mai târziu
Toate cele de mai sus, plus RUL/degradation tracking, integrare senzor real (ESP32), cloud deployment.

### Cea mai importantă parte tehnică
Strategia de data leakage prevention (split per fișier, bazată pe auditul de dataset din Phase 1.5) — e diferența dintre un proiect care pare bun pe hârtie și unul care rezistă la întrebări tehnice serioase.

### Cel mai puternic experiment pentru CV/interviu
Matricea experimentală din secțiunea 19 (raw comprimat via PCA vs. features DSP, Isolation Forest vs. Autoencoder) — pentru că e o comparație controlată, cu dimensionalitate egalizată explicit între brațe, ipoteză declarată și rezultat măsurat pentru fiecare celulă, nu o singură comparație A/B presupusă corectă din start.

### Cel mai mare risc tehnic
Autoencoder cu date insuficiente per clasă de fault pentru un threshold stabil.

### Cel mai mare risc metodologic
Doi, la fel de importanți: (1) split incorect (leakage) care ar umfla artificial metricile — de aceea secțiunea 8 e tratată ca fundație, nu detaliu; (2) confundarea variabilei "reprezentare" cu variabila "dimensionalitate" în experimentul central, dacă brațul raw nu are dimensionalitatea egalizată explicit față de feature vectors (secțiunea 19).

### Cel mai mare risc de scope
Construirea paginilor UI secundare (History, Settings, 3D waterfall) înainte ca experimentul central și baseline-urile să fie solide.

### Ce trebuie să înveți înainte să începi implementarea
Nyquist/aliasing, spectral leakage, Butterworth + zero-phase filtering, mecanismul Isolation Forest, arhitectura de bază a unui Autoencoder și rolul reconstruction error, diferența unsupervised training / supervised evaluation, de ce split-ul per fișier previne leakage.

---

**Nu s-a implementat nimic din acest blueprint. Aștept confirmarea ta înainte de a trece la Phase 1.**