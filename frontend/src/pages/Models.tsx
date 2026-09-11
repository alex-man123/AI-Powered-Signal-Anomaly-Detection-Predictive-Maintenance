import { Check, CircleCheck } from 'lucide-react'
import { useEffect, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { EmptyState } from '@/components/ui/EmptyState'
import { ErrorState } from '@/components/ui/ErrorState'
import { LoadingState } from '@/components/ui/LoadingState'
import { useActiveModel } from '@/context/ActiveModelContext'
import { MODEL_LABELS } from '@/lib/model-display'
import { cn } from '@/lib/utils'
import { getModels, type ModelResponse } from '@/services/api'

type Loadable<T> =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'success'; data: T }

function useModels(): [Loadable<ModelResponse[]>, () => void] {
  const [state, setState] = useState<Loadable<ModelResponse[]>>({ status: 'loading' })
  const [reloadToken, setReloadToken] = useState(0)

  useEffect(() => {
    let cancelled = false

    getModels()
      .then((models) => {
        if (!cancelled) setState({ status: 'success', data: models })
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setState({
            status: 'error',
            message: error instanceof Error ? error.message : 'Failed to load models',
          })
        }
      })

    return () => {
      cancelled = true
    }
  }, [reloadToken])

  function reload() {
    setState({ status: 'loading' })
    setReloadToken((token) => token + 1)
  }

  return [state, reload]
}

function Metric({ label, value, suffix }: { label: string; value: number; suffix?: string }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-mono text-sm">
        {Number.isFinite(value) ? value.toFixed(4) : 'N/A'}
        {suffix ? ` ${suffix}` : ''}
      </p>
    </div>
  )
}

// Cell labels reflect the real, documented contract of
// `app.ml.evaluation.evaluate()`: confusion_matrix = [[TN, FP], [FN, TP]],
// with 1 = anomaly / 0 = normal — not a guessed layout.
function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  const [[tn, fp], [fn, tp]] = matrix

  return (
    <div className="grid grid-cols-3 gap-1 text-center text-xs">
      <div />
      <div className="self-end pb-1 text-muted-foreground">Pred. normal</div>
      <div className="self-end pb-1 text-muted-foreground">Pred. anomaly</div>

      <div className="flex items-center justify-end pr-1.5 text-muted-foreground">Actual normal</div>
      <div className="rounded-md bg-muted/60 py-2 font-mono">{tn}</div>
      <div className="rounded-md bg-muted/60 py-2 font-mono">{fp}</div>

      <div className="flex items-center justify-end pr-1.5 text-muted-foreground">Actual anomaly</div>
      <div className="rounded-md bg-muted/60 py-2 font-mono">{fn}</div>
      <div className="rounded-md bg-muted/60 py-2 font-mono">{tp}</div>
    </div>
  )
}

function ModelCard({
  model,
  isActive,
  onSelect,
}: {
  model: ModelResponse
  isActive: boolean
  onSelect: () => void
}) {
  const { artifact, metrics } = model
  const label = MODEL_LABELS[model.model_type] ?? model.model_type

  return (
    <Card className={cn(isActive ? 'ring-2 ring-accent' : 'ring-1 ring-foreground/10')}>
      <CardHeader>
        <div className="flex items-center justify-between gap-2">
          <CardTitle>{label}</CardTitle>
          {isActive && (
            <Badge variant="outline" className="gap-1 border-transparent bg-accent/15 text-accent">
              <CircleCheck className="size-3" />
              Active
            </Badge>
          )}
        </div>
        <CardDescription className="flex items-center gap-1.5">
          <Check className="size-3.5 text-normal" />
          Available
        </CardDescription>
      </CardHeader>

      <CardContent className="flex flex-col gap-4">
        <div className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
          <Metric label="F1" value={metrics.f1} />
          <Metric label="Precision" value={metrics.precision} />
          <Metric label="Recall" value={metrics.recall} />
          <Metric label="ROC-AUC" value={metrics.roc_auc} />
          <Metric label="PR-AUC" value={metrics.pr_auc} />
          <Metric label="FPR" value={metrics.fpr} />
          <Metric label="FNR" value={metrics.fnr} />
          <Metric label="Inference time" value={metrics.inference_time} suffix="s" />
        </div>

        <div>
          <p className="mb-1.5 text-xs text-muted-foreground">Confusion matrix</p>
          <ConfusionMatrix matrix={metrics.confusion_matrix} />
        </div>

        <div className="flex flex-wrap gap-x-6 gap-y-2 text-xs text-muted-foreground">
          <span>
            Threshold{' '}
            <span className="font-mono text-foreground">{artifact.threshold_value.toFixed(4)}</span> (
            {artifact.threshold_method})
          </span>
          <span>
            <span className="font-mono text-foreground">{artifact.feature_dimension}</span> features
          </span>
        </div>

        <Button type="button" variant={isActive ? 'secondary' : 'default'} aria-pressed={isActive} onClick={onSelect}>
          {isActive ? 'Selected' : 'Select model'}
        </Button>
      </CardContent>
    </Card>
  )
}

export default function Models() {
  const [state, reload] = useModels()
  const { activeModel, setActiveModel } = useActiveModel()

  // Once the real model list is in, resolve the active selection against it:
  // keep it if still valid, otherwise fall back to the first available real
  // model (covers first-ever visit, and a stale sessionStorage value from a
  // model that's no longer returned by the API). This is a genuine
  // synchronization with data outside this component (the shared,
  // cross-page active-model context) — not something derivable during
  // render the way a purely-local state adjustment would be.
  useEffect(() => {
    if (state.status !== 'success') return
    const models = state.data
    if (models.length === 0) return

    const stillValid = activeModel !== null && models.some((model) => model.model_type === activeModel)
    if (!stillValid) {
      setActiveModel(models[0].model_type)
    }
  }, [state, activeModel, setActiveModel])

  return (
    <div>
      <h1 className="text-xl font-heading font-semibold">Models</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Compare the trained models and choose which one Anomaly Detection uses.
      </p>

      {state.status === 'loading' && <LoadingState message="Loading models…" className="mt-6" />}

      {state.status === 'error' && (
        <ErrorState message={`Failed to load models — ${state.message}`} onRetry={reload} className="mt-6" />
      )}

      {state.status === 'success' && state.data.length === 0 && (
        <EmptyState
          title="No models available"
          message="No trained models are currently registered by the backend."
          className="mt-6"
        />
      )}

      {state.status === 'success' && state.data.length > 0 && (
        <div className="mt-6 grid gap-4 sm:grid-cols-2">
          {state.data.map((model) => (
            <ModelCard
              key={model.model_type}
              model={model}
              isActive={model.model_type === activeModel}
              onSelect={() => setActiveModel(model.model_type)}
            />
          ))}
        </div>
      )}
    </div>
  )
}
