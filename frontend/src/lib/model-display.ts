import type { ModelType, PredictionStatus } from '@/services/api'

export const MODEL_LABELS: Record<ModelType, string> = {
  isolation_forest: 'Isolation Forest',
  autoencoder: 'Autoencoder',
}

export const STATUS_BADGE_CLASS: Record<PredictionStatus, string> = {
  NORMAL: 'bg-normal/15 text-normal',
  WARNING: 'bg-warning/15 text-warning',
  ANOMALY: 'bg-anomaly/15 text-anomaly',
}
