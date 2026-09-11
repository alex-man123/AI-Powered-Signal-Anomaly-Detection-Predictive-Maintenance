import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { EmptyState } from '@/components/ui/EmptyState'
import { ErrorState } from '@/components/ui/ErrorState'
import { LoadingState } from '@/components/ui/LoadingState'
import { STATUS_BADGE_CLASS } from '@/lib/model-display'
import { cn } from '@/lib/utils'
import type { PredictResponse, PredictionStatus } from '@/services/api'

// Blueprint section 17's own required, verbatim interpretation — "used
// consistently in the UI and README" per that section — never reworded into
// a physical-severity/confidence framing.
const ANOMALY_SCORE_DEFINITION =
  "Anomaly score = how unusual (out-of-distribution) this signal is compared to the normal behavior the model learned — not a calibrated measure of physical fault severity."

// A short, status-specific reading of that same definition (AC3) — never a
// bare number with no interpretation, and never phrased as a fault
// probability.
const STATUS_INTERPRETATION: Record<PredictionStatus, string> = {
  NORMAL: 'This signal is close to the normal behavior the model learned.',
  WARNING: 'This score is moderately unusual compared to the normal baseline the model learned.',
  ANOMALY: 'This score indicates a high level of unusualness compared to the normal baseline the model learned.',
}

const STATUS_METER_FILL_CLASS: Record<PredictionStatus, string> = {
  NORMAL: 'bg-normal',
  WARNING: 'bg-warning',
  ANOMALY: 'bg-anomaly',
}

export interface AnomalyPanelData {
  /** Human-readable label of the model this result came from (e.g. "Isolation Forest") — always the real active model, never hardcoded. */
  modelLabel: string
  prediction: PredictResponse
  threshold: number
  thresholdMethod: string
}

export type AnomalyPanelState =
  | { status: 'empty'; message: string }
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'success'; data: AnomalyPanelData }

// --- Feature indicators, parsed from the backend's own real explanation ---
//
// `POST /api/models/predict` does not return raw per-feature values — only a
// prose `explanation` string, already grounded in real computed features and
// a real per-feature baseline (see `model_service._build_explanation`). This
// parses that real sentence for a nicer per-feature layout; it never decides
// which features are "notable" or invents a threshold — that judgment (a
// >= 2 std-deviation cutoff) is already made by the backend. If the sentence
// doesn't match the expected shape (e.g. a future backend wording change),
// this falls back to rendering it as plain text rather than guessing.
interface FeatureDeviation {
  name: string
  value: string
  stdDeviations: number
  direction: 'above' | 'below'
}

type ParsedExplanation =
  | { kind: 'text'; text: string }
  | { kind: 'deviations'; items: FeatureDeviation[] }

const DEVIATION_PREFIX = 'Notable feature deviations from the real normal-validation baseline: '
const DEVIATION_ITEM_PATTERN =
  /^(\w+)=(\S+) \((\d+(?:\.\d+)?) std (above|below) the real normal-validation baseline\)$/

function parseExplanation(explanation: string): ParsedExplanation {
  if (!explanation.startsWith(DEVIATION_PREFIX)) {
    return { kind: 'text', text: explanation }
  }

  const body = explanation.slice(DEVIATION_PREFIX.length).replace(/\.$/, '')
  const items: FeatureDeviation[] = []

  for (const part of body.split('; ')) {
    const match = DEVIATION_ITEM_PATTERN.exec(part)
    if (!match) {
      return { kind: 'text', text: explanation }
    }
    const [, name, value, std, direction] = match
    items.push({ name, value, stdDeviations: Number(std), direction: direction as 'above' | 'below' })
  }

  return items.length > 0 ? { kind: 'deviations', items } : { kind: 'text', text: explanation }
}

function FeatureIndicators({ explanation }: { explanation: string }) {
  const parsed = parseExplanation(explanation)

  if (parsed.kind === 'text') {
    return <p className="text-sm text-muted-foreground">{parsed.text}</p>
  }

  return (
    <div className="grid gap-3 sm:grid-cols-3">
      {parsed.items.map((item) => (
        <div key={item.name} className="rounded-lg bg-muted/50 p-3">
          <p className="text-xs text-muted-foreground capitalize">{item.name.replaceAll('_', ' ')}</p>
          <p className="mt-0.5 font-mono text-sm">{item.value}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            {item.stdDeviations.toFixed(1)} std {item.direction} the learned normal baseline
          </p>
        </div>
      ))}
    </div>
  )
}

function ScoreMeter({ score, threshold, status }: { score: number; threshold: number; status: PredictionStatus }) {
  const scorePercent = Math.min(100, Math.max(0, score * 100))
  const thresholdPercent = Math.min(100, Math.max(0, threshold * 100))

  return (
    <div>
      <div className="relative h-2.5 w-full rounded-full bg-muted">
        <div
          className={cn('h-full rounded-full transition-[width]', STATUS_METER_FILL_CLASS[status])}
          style={{ width: `${scorePercent}%` }}
        />
        <div
          className="absolute top-1/2 h-4 w-0.5 -translate-y-1/2 rounded-full bg-foreground/50"
          style={{ left: `${thresholdPercent}%` }}
          title={`Threshold: ${threshold.toFixed(4)}`}
        />
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
        <span>0.0</span>
        <span>1.0</span>
      </div>
    </div>
  )
}

// TASK 11.6 — presentation-only: receives the real prediction result (and
// the real threshold/method it was evaluated against) as props from the
// parent page, which owns the actual `POST /api/models/predict` call and the
// active-model selection (TASK 11.5). No scoring/business logic lives here.
export function AnomalyPanel({ state }: { state: AnomalyPanelState }) {
  return (
    <Card className="w-full max-w-2xl">
      <CardHeader>
        <CardTitle>Anomaly Score</CardTitle>
        <CardDescription>{ANOMALY_SCORE_DEFINITION}</CardDescription>
      </CardHeader>

      <CardContent>
        {state.status === 'empty' && <EmptyState size="sm" message={state.message} />}

        {state.status === 'loading' && <LoadingState message="Running anomaly analysis…" />}

        {state.status === 'error' && <ErrorState message={state.message} />}

        {state.status === 'success' && (
          <div className="flex flex-col gap-5">
            <p className="text-xs text-muted-foreground">{state.data.modelLabel}</p>

            <div className="flex flex-wrap items-end gap-3">
              <span className="font-mono text-4xl font-semibold">
                {state.data.prediction.anomaly_score.toFixed(4)}
              </span>
              <Badge
                variant="outline"
                className={cn('border-transparent', STATUS_BADGE_CLASS[state.data.prediction.status])}
              >
                {state.data.prediction.status}
              </Badge>
            </div>

            <ScoreMeter
              score={state.data.prediction.anomaly_score}
              threshold={state.data.threshold}
              status={state.data.prediction.status}
            />

            <p className="text-xs text-muted-foreground">
              Threshold in use: <span className="font-mono text-foreground">{state.data.threshold.toFixed(4)}</span>{' '}
              ({state.data.thresholdMethod})
            </p>

            <p className="text-sm">{STATUS_INTERPRETATION[state.data.prediction.status]}</p>

            <div>
              <p className="mb-2 text-xs font-medium text-muted-foreground">Signal indicators</p>
              <FeatureIndicators explanation={state.data.prediction.explanation} />
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
