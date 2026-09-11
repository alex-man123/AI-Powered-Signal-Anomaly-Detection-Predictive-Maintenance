import { useEffect, useState } from 'react'

import { SampleClassSelector } from '@/components/SampleClassSelector'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { EmptyState } from '@/components/ui/EmptyState'
import { ErrorState } from '@/components/ui/ErrorState'
import { LoadingState } from '@/components/ui/LoadingState'
import { useActiveSampleClass } from '@/context/ActiveSampleClassContext'
import { MODEL_LABELS, STATUS_BADGE_CLASS } from '@/lib/model-display'
import {
  getDatasets,
  getModels,
  getSampleSignal,
  predict,
  type ModelResponse,
  type ModelType,
  type PredictResponse,
  type SampleSignalResponse,
  type SignalLabel,
} from '@/services/api'

type DashboardData = {
  models: ModelResponse[]
  signalsAnalyzed: number
  sample: SampleSignalResponse
  predictions: Record<ModelType, PredictResponse>
}

type State =
  | { state: 'loading' }
  | { state: 'error'; message: string }
  | { state: 'success'; data: DashboardData }

function useDashboardData(sampleClass: SignalLabel | null): [State, () => void] {
  const [state, setState] = useState<State>({ state: 'loading' })
  const [reloadToken, setReloadToken] = useState(0)

  useEffect(() => {
    let cancelled = false

    async function load() {
      const [models, datasets, sample] = await Promise.all([
        getModels(),
        getDatasets(),
        getSampleSignal(sampleClass ?? undefined),
      ])

      const predictionEntries = await Promise.all(
        models.map(
          async (model) =>
            [
              model.model_type,
              await predict({
                model_type: model.model_type,
                signal: sample.signal,
                sampling_rate: sample.sampling_rate,
              }),
            ] as const,
        ),
      )

      return {
        models,
        // GET /api/datasets's real signal_count, summed across every dataset
        // this project has (currently exactly one: MAFAULDA).
        signalsAnalyzed: datasets.reduce((sum, dataset) => sum + dataset.signal_count, 0),
        sample,
        predictions: Object.fromEntries(predictionEntries) as Record<ModelType, PredictResponse>,
      }
    }

    load()
      .then((data) => {
        if (!cancelled) setState({ state: 'success', data })
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setState({
            state: 'error',
            message: error instanceof Error ? error.message : 'Failed to load dashboard data',
          })
        }
      })

    return () => {
      cancelled = true
    }
  }, [reloadToken, sampleClass])

  function reload() {
    // Reset to loading from the event that triggered the refetch, not from
    // inside the effect itself.
    setState({ state: 'loading' })
    setReloadToken((token) => token + 1)
  }

  return [state, reload]
}

export default function Dashboard() {
  const { activeSampleClass } = useActiveSampleClass()
  const [state, reload] = useDashboardData(activeSampleClass)

  return (
    <div>
      <h1 className="text-xl font-heading font-semibold">Dashboard</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Overview of the trained models and the dataset they were evaluated on.
      </p>

      <SampleClassSelector className="mt-4" />

      {state.state === 'loading' && <LoadingState message="Loading dashboard data…" className="mt-6" />}

      {state.state === 'error' && (
        <ErrorState
          message={`Failed to load dashboard data — ${state.message}`}
          onRetry={reload}
          className="mt-6"
        />
      )}

      {state.state === 'success' && (
        <div className="mt-6 flex flex-col gap-6">
          <Card className="w-full max-w-xs">
            <CardHeader>
              <CardTitle>Signals analyzed</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="font-mono text-2xl font-semibold">{state.data.signalsAnalyzed}</p>
              <p className="mt-1 text-xs text-muted-foreground">
                Total signal count across every dataset (GET /api/datasets).
              </p>
            </CardContent>
          </Card>

          {state.data.models.length === 0 ? (
            <EmptyState
              title="No models available"
              message="No trained models are currently registered by the backend."
            />
          ) : (
            <>
              <p className="text-xs text-muted-foreground">
                Anomaly scores below are computed just now by running a real validation-set signal (
                <code>{state.data.sample.recording_id}</code>, labeled "{state.data.sample.label}") through each
                model — this project has no live sensor feed, only static dataset signals.
              </p>

              <div className="grid gap-4 sm:grid-cols-2">
                {state.data.models.map((model) => {
                  const prediction = state.data.predictions[model.model_type]
                  return (
                    <Card key={model.model_type}>
                      <CardHeader>
                        <CardTitle>{MODEL_LABELS[model.model_type]}</CardTitle>
                      </CardHeader>
                      <CardContent className="flex flex-col gap-4">
                        <div>
                          <p className="text-xs text-muted-foreground">Threshold in use</p>
                          <p className="font-mono text-lg">{model.artifact.threshold_value.toFixed(4)}</p>
                          <p className="mt-0.5 text-xs text-muted-foreground">
                            {model.artifact.threshold_method} · {model.artifact.score_direction.replaceAll('_', ' ')}
                          </p>
                        </div>

                        <div>
                          <p className="text-xs text-muted-foreground">Anomaly score / status</p>
                          <div className="mt-1 flex items-center gap-2">
                            <span className="font-mono text-lg">{prediction.anomaly_score.toFixed(4)}</span>
                            <Badge variant="outline" className={STATUS_BADGE_CLASS[prediction.status]}>
                              {prediction.status}
                            </Badge>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )
                })}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
