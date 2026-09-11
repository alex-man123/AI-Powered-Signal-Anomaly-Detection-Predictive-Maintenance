import { EmptyState } from '@/components/ui/EmptyState'
import { ErrorState } from '@/components/ui/ErrorState'
import { LoadingState } from '@/components/ui/LoadingState'
import type { FeatureExplanation } from '@/services/api'

// TASK 12.1 — presentation only. Every value here comes straight from the
// backend's real `explanations` list (`app.services.explanation_service`,
// itself a real per-feature deviation from a real normal-validation
// baseline) — nothing is recomputed, re-derived, or invented in this
// component. Reuses TASK 11.9's shared Loading/Error/Empty shells so a
// caller (or a future standalone use of this list) gets the same states
// every other data-driven part of the app already uses.
export type ExplanationListState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'empty' }
  | { status: 'success'; data: FeatureExplanation[] }

function featureLabel(feature: string): string {
  return feature.replaceAll('_', ' ')
}

// Deliberately never phrased as physical damage/failure probability (AC2) —
// only ever "N% above/below the normal baseline", the real statistical
// deviation the backend computed.
function formatDeviation(item: FeatureExplanation): string {
  if (item.deviation_percent === null) {
    const sign = item.deviation >= 0 ? '+' : ''
    return `${sign}${item.deviation.toFixed(3)} vs. a zero baseline`
  }
  const sign = item.deviation_percent >= 0 ? '+' : ''
  return `${sign}${item.deviation_percent.toFixed(1)}% vs. normal baseline`
}

function ExplanationRow({ item }: { item: FeatureExplanation }) {
  // The bar is a magnitude indicator, not a severity/damage meter — same
  // accent color regardless of direction, capped visually at 100% while the
  // real (possibly larger) number is still shown as text.
  const barPercent =
    item.deviation_percent === null
      ? Math.min(100, (item.std_deviations / 10) * 100)
      : Math.min(100, Math.abs(item.deviation_percent))

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-sm font-medium capitalize">{featureLabel(item.feature)}</span>
        <span className="font-mono text-sm text-accent">{formatDeviation(item)}</span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-muted">
        <div className="h-full rounded-full bg-accent" style={{ width: `${barPercent}%` }} />
      </div>
    </div>
  )
}

export function ExplanationList({ state }: { state: ExplanationListState }) {
  if (state.status === 'loading') {
    return <LoadingState size="sm" message="Loading feature deviations…" />
  }
  if (state.status === 'error') {
    return <ErrorState size="sm" message={state.message} />
  }
  if (state.status === 'empty') {
    return <EmptyState size="sm" message="No significant feature deviations detected." />
  }

  return (
    <div className="flex flex-col gap-3">
      {state.data.map((item) => (
        <ExplanationRow key={item.feature} item={item} />
      ))}
    </div>
  )
}
