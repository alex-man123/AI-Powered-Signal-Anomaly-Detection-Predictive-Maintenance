import type { ModelType, PredictionStatus, SignalLabel } from '@/services/api'

export const MODEL_LABELS: Record<ModelType, string> = {
  isolation_forest: 'Isolation Forest',
  autoencoder: 'Autoencoder',
}

export const STATUS_BADGE_CLASS: Record<PredictionStatus, string> = {
  NORMAL: 'bg-normal/15 text-normal',
  WARNING: 'bg-warning/15 text-warning',
  ANOMALY: 'bg-anomaly/15 text-anomaly',
}

// Display-only prettification of the real `app.models.signal.SignalLabel`
// values (TASK 12.2) — every real class the dataset defines, never a subset.
export const SIGNAL_LABEL_LABELS: Record<SignalLabel, string> = {
  normal: 'Normal',
  imbalance: 'Imbalance',
  'horizontal-misalignment': 'Horizontal Misalignment',
  'vertical-misalignment': 'Vertical Misalignment',
}
