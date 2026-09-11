export interface HealthResponse {
  status: string
}

// Mirrors backend `app.ml.model_artifact.ModelType`/`ScoreDirection` (TASK 6.5/7.3)
// and `app.api.schemas.models` (TASK 10.1/10.5) exactly.
export type ModelType = 'isolation_forest' | 'autoencoder'
export type ScoreDirection = 'higher_is_more_anomalous' | 'lower_is_more_anomalous'

export interface ModelArtifactInfo {
  model_version: string
  feature_names: string[]
  feature_dimension: number
  threshold_method: string
  threshold_value: number
  score_direction: ScoreDirection
  training_split: string
  dataset_hash: string
  split_manifest_hash: string
  random_seed: number
  created_at: string
}

export interface ModelMetrics {
  precision: number
  recall: number
  f1: number
  roc_auc: number
  pr_auc: number
  confusion_matrix: number[][]
  fpr: number
  fnr: number
  inference_time: number
}

// GET /api/models, GET /api/models/{id}/performance
export interface ModelResponse {
  model_type: ModelType
  artifact: ModelArtifactInfo
  metrics: ModelMetrics
}

// GET /api/datasets
export interface DatasetSummaryResponse {
  id: number
  name: string
  signal_count: number
}

// Mirrors backend `app.models.signal.SignalLabel` exactly.
export type SignalLabel = 'normal' | 'imbalance' | 'horizontal-misalignment' | 'vertical-misalignment'

// GET /api/datasets/{id} — mirrors `app.api.schemas.datasets.DatasetDetailResponse`
// exactly. No `duration` field exists on this real schema — see Dataset.tsx
// for how it's derived from `samples_per_signal`/`sampling_rate` instead.
export interface DatasetDetailResponse {
  id: number
  name: string
  signal_count: number
  sampling_rate: number
  channels: number[]
  samples_per_signal: number
  labels: SignalLabel[]
  missing_values: number
}

// GET /api/models/sample-signal
export interface SampleSignalResponse {
  recording_id: string
  label: SignalLabel
  channel: number
  sampling_rate: number
  signal: number[]
}

export type PredictionStatus = 'NORMAL' | 'WARNING' | 'ANOMALY'

export interface PredictRequest {
  model_type: ModelType
  signal: number[]
  sampling_rate: number
}

// Mirrors backend `app.api.schemas.models.PredictionDirection` exactly.
export type FeatureDeviationDirection = 'above' | 'below'

// TASK 12.1 — one feature's real, computed deviation from the real
// normal-validation baseline (`app.services.explanation_service.explain`).
// `deviation_percent` is `null` when `baseline_value` is 0 (undefined ratio)
// — never a fabricated/Infinity/NaN percentage.
export interface FeatureExplanation {
  feature: string
  current_value: number
  baseline_value: number
  deviation: number
  deviation_percent: number | null
  direction: FeatureDeviationDirection
  std_deviations: number
}

// POST /api/models/predict
export interface PredictResponse {
  anomaly_score: number
  status: PredictionStatus
  explanation: string
  explanations: FeatureExplanation[]
}

// POST /api/fft
export interface FFTRequest {
  signal: number[]
  sampling_rate: number
}

export interface FFTResponse {
  frequencies: number[]
  magnitude: number[]
  dominant_frequency: number
}

// POST /api/psd
export interface PSDRequest {
  signal: number[]
  sampling_rate: number
  nperseg: number
  noverlap: number
}

export interface PSDResponse {
  frequencies: number[]
  psd: number[]
}

// POST /api/spectrogram
export interface SpectrogramRequest {
  signal: number[]
  sampling_rate: number
  window_size: number
  hop_length: number
}

export interface SpectrogramResponse {
  frequencies: number[]
  times: number[]
  values: number[][]
}

// POST /api/features/extract
export interface FeatureExtractionRequest {
  signal: number[]
  sampling_rate: number
  nperseg: number
  noverlap: number
}

export interface FeatureExtractionResponse {
  feature_names: string[]
  values: number[]
}

// Mirrors backend `app.signal_processing.filtering.FilterType`/`Cutoff` exactly.
export type FilterType = 'lowpass' | 'highpass' | 'bandpass'

// POST /api/dsp/filter
export interface FilterRequest {
  signal: number[]
  sampling_rate: number
  cutoff: number | [number, number]
  order: number
  btype: FilterType
}

export interface FilterResponse {
  filtered_signal: number[]
}

// GET /api/experiments — mirrors `app.api.schemas.experiments.ExperimentResponse`
// exactly. `representation`/`model` are plain strings on this real schema (not
// the `ModelType` union `ModelResponse.model_type` uses), so callers should
// not assume a fixed, closed set of values.
export interface ExperimentResponse {
  experiment: 'A' | 'B' | 'C'
  experiment_id: string
  representation: string
  model: string
  dataset_hash: string
  split_manifest_hash: string
  precision: number
  recall: number
  f1: number
  roc_auc: number
  pr_auc: number
  fpr: number
  fnr: number
  confusion_matrix: number[][]
  inference_time: number
}

// GET /api/experiments/{id} — mirrors `ExperimentDetailResponse`. `metrics`
// reuses `ModelMetrics` (the same `evaluate()`-shaped schema TASK 10.5 already
// defines) rather than a second copy.
export interface ExperimentDetailResponse {
  experiment_id: string
  representation: string
  model: string
  dataset_hash: string
  split_manifest_hash: string
  feature_dimension: number
  preprocessing_config: Record<string, unknown>
  feature_set: Record<string, unknown> | string[]
  model_config: Record<string, unknown>
  random_seed: number
  threshold_method: string
  threshold_value: number
  metrics: ModelMetrics
  timestamp: string
}

// GET /api/experiments/pca-visualization — mirrors `PCAVisualizationResponse`
// (TASK 12.2) exactly. One point per real MAFAULDA window, projected through
// Experiment A's already-fitted PCA (TASK 9.1/9.2) — never recomputed here.
export interface PCAPoint {
  recording_id: string
  label: SignalLabel
  pc1: number
  pc2: number
  pc3: number
}

export interface PCAVisualizationResponse {
  points: PCAPoint[]
  explained_variance_ratio: [number, number, number]
  total_components: number
}

const API_URL = import.meta.env.VITE_API_URL

// FastAPI's own error contract for 4xx/5xx is `{"detail": "..."}` (TASK 10.7) —
// surface that real message when present, instead of just the HTTP status.
async function errorMessageFor(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json()
    if (body && typeof body === 'object' && 'detail' in body && typeof body.detail === 'string') {
      return body.detail
    }
  } catch {
    // response body wasn't JSON — fall through to the generic message
  }
  return `Backend responded with HTTP ${response.status}`
}

async function apiGet<T>(path: string): Promise<T> {
  if (!API_URL) {
    throw new Error('VITE_API_URL is not configured')
  }

  const response = await fetch(`${API_URL}${path}`)

  if (!response.ok) {
    throw new Error(await errorMessageFor(response))
  }

  return (await response.json()) as T
}

async function apiPost<T>(path: string, body: unknown): Promise<T> {
  if (!API_URL) {
    throw new Error('VITE_API_URL is not configured')
  }

  const response = await fetch(`${API_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!response.ok) {
    throw new Error(await errorMessageFor(response))
  }

  return (await response.json()) as T
}

export function getHealth(): Promise<HealthResponse> {
  return apiGet<HealthResponse>('/api/health')
}

export function getModels(): Promise<ModelResponse[]> {
  return apiGet<ModelResponse[]>('/api/models')
}

export function getDatasets(): Promise<DatasetSummaryResponse[]> {
  return apiGet<DatasetSummaryResponse[]>('/api/datasets')
}

export function getDataset(datasetId: number): Promise<DatasetDetailResponse> {
  return apiGet<DatasetDetailResponse>(`/api/datasets/${datasetId}`)
}

export function getSampleSignal(label?: SignalLabel): Promise<SampleSignalResponse> {
  const query = label ? `?label=${encodeURIComponent(label)}` : ''
  return apiGet<SampleSignalResponse>(`/api/models/sample-signal${query}`)
}

export function predict(request: PredictRequest): Promise<PredictResponse> {
  return apiPost<PredictResponse>('/api/models/predict', request)
}

export function runFFT(request: FFTRequest): Promise<FFTResponse> {
  return apiPost<FFTResponse>('/api/fft', request)
}

export function runPSD(request: PSDRequest): Promise<PSDResponse> {
  return apiPost<PSDResponse>('/api/psd', request)
}

export function runSpectrogram(request: SpectrogramRequest): Promise<SpectrogramResponse> {
  return apiPost<SpectrogramResponse>('/api/spectrogram', request)
}

export function extractFeatures(request: FeatureExtractionRequest): Promise<FeatureExtractionResponse> {
  return apiPost<FeatureExtractionResponse>('/api/features/extract', request)
}

export function runFilter(request: FilterRequest): Promise<FilterResponse> {
  return apiPost<FilterResponse>('/api/dsp/filter', request)
}

export function getExperiments(): Promise<ExperimentResponse[]> {
  return apiGet<ExperimentResponse[]>('/api/experiments')
}

export function getExperiment(experimentId: string): Promise<ExperimentDetailResponse> {
  return apiGet<ExperimentDetailResponse>(`/api/experiments/${encodeURIComponent(experimentId)}`)
}

export function getPCAVisualization(): Promise<PCAVisualizationResponse> {
  return apiGet<PCAVisualizationResponse>('/api/experiments/pca-visualization')
}
