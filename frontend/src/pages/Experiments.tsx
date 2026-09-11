import { useEffect, useMemo, useState } from 'react'

import { PCA3DPlot, type PCA3DPlotState } from '@/components/charts/PCA3DPlot'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { EmptyState } from '@/components/ui/EmptyState'
import { ErrorState } from '@/components/ui/ErrorState'
import { LoadingState } from '@/components/ui/LoadingState'
import { MODEL_LABELS } from '@/lib/model-display'
import { cn } from '@/lib/utils'
import {
  getExperiment,
  getExperiments,
  getPCAVisualization,
  type ExperimentDetailResponse,
  type ExperimentResponse,
} from '@/services/api'

type Loadable<T> =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'success'; data: T }

function useExperiments(): [Loadable<ExperimentResponse[]>, () => void] {
  const [state, setState] = useState<Loadable<ExperimentResponse[]>>({ status: 'loading' })
  const [reloadToken, setReloadToken] = useState(0)

  useEffect(() => {
    let cancelled = false

    getExperiments()
      .then((experiments) => {
        if (!cancelled) setState({ status: 'success', data: experiments })
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setState({
            status: 'error',
            message: error instanceof Error ? error.message : 'Failed to load experiments',
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

function usePCAVisualization(): PCA3DPlotState {
  const [state, setState] = useState<PCA3DPlotState>({ status: 'loading' })

  useEffect(() => {
    let cancelled = false

    getPCAVisualization()
      .then((data) => {
        if (!cancelled) setState({ status: 'success', data })
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setState({
            status: 'error',
            message: error instanceof Error ? error.message : 'Failed to load PCA visualization',
          })
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  return state
}

// Real, currently-known representation values (`raw_pca`, `dsp_features`) —
// display-only prettification. Any other real value the API returns falls
// back to the raw string rather than being hidden or guessed at.
const REPRESENTATION_LABELS: Record<string, string> = {
  raw_pca: 'Raw + PCA',
  dsp_features: 'DSP Features',
}

function representationLabel(representation: string): string {
  return REPRESENTATION_LABELS[representation] ?? representation
}

function modelLabel(model: string): string {
  return model in MODEL_LABELS ? MODEL_LABELS[model as keyof typeof MODEL_LABELS] : model
}

function Metric({ label, value, highlight }: { label: string; value: number; highlight?: boolean }) {
  return (
    <div>
      <p className="text-[11px] text-muted-foreground">{label}</p>
      <p className={cn('font-mono text-sm', highlight && 'font-semibold text-accent')}>{value.toFixed(4)}</p>
    </div>
  )
}

// Cell labels reflect the real, documented contract of
// `app.ml.evaluation.evaluate()`: confusion_matrix = [[TN, FP], [FN, TP]],
// with 1 = anomaly / 0 = normal.
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

function KeyValueList({ data }: { data: Record<string, unknown> }) {
  const entries = Object.entries(data)
  if (entries.length === 0) return <p className="text-xs text-muted-foreground">—</p>

  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs sm:grid-cols-3">
      {entries.map(([key, value]) => (
        <div key={key}>
          <dt className="text-muted-foreground capitalize">{key.replaceAll('_', ' ')}</dt>
          <dd className="font-mono text-foreground">
            {value === null ? 'null' : typeof value === 'object' ? JSON.stringify(value) : String(value)}
          </dd>
        </div>
      ))}
    </dl>
  )
}

function FeatureSetDisplay({ featureSet }: { featureSet: Record<string, unknown> | string[] }) {
  if (Array.isArray(featureSet)) {
    return (
      <div className="flex flex-wrap gap-1.5">
        {featureSet.map((name) => (
          <Badge key={name} variant="outline">
            {name.replaceAll('_', ' ')}
          </Badge>
        ))}
      </div>
    )
  }
  return <KeyValueList data={featureSet} />
}

function ExperimentDetails({ state }: { state: Loadable<ExperimentDetailResponse> }) {
  if (state.status === 'loading') {
    return <LoadingState size="sm" message="Loading experiment details…" />
  }
  if (state.status === 'error') {
    return <ErrorState size="sm" message={state.message} />
  }

  const detail = state.data

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
        <Metric label="Precision" value={detail.metrics.precision} />
        <Metric label="Recall" value={detail.metrics.recall} />
        <Metric label="FPR" value={detail.metrics.fpr} />
        <Metric label="FNR" value={detail.metrics.fnr} />
      </div>

      <div>
        <p className="mb-1.5 text-xs text-muted-foreground">Confusion matrix</p>
        <ConfusionMatrix matrix={detail.metrics.confusion_matrix} />
      </div>

      <div className="flex flex-wrap gap-x-6 gap-y-2 text-xs text-muted-foreground">
        <span>
          Feature dimension <span className="font-mono text-foreground">{detail.feature_dimension}</span>
        </span>
        <span>
          Random seed <span className="font-mono text-foreground">{detail.random_seed}</span>
        </span>
        <span>
          Threshold <span className="font-mono text-foreground">{detail.threshold_value.toFixed(4)}</span> (
          {detail.threshold_method})
        </span>
        <span>
          Inference time <span className="font-mono text-foreground">{detail.metrics.inference_time.toFixed(4)} s</span>
        </span>
        <span>
          Generated <span className="font-mono text-foreground">{detail.timestamp}</span>
        </span>
      </div>

      <div>
        <p className="mb-1.5 text-xs text-muted-foreground">Feature set</p>
        <FeatureSetDisplay featureSet={detail.feature_set} />
      </div>

      <div>
        <p className="mb-1.5 text-xs text-muted-foreground">Preprocessing config</p>
        <KeyValueList data={detail.preprocessing_config} />
      </div>

      <div>
        <p className="mb-1.5 text-xs text-muted-foreground">Model config</p>
        <KeyValueList data={detail.model_config} />
      </div>
    </div>
  )
}

function MatrixCell({
  experiment,
  isBestF1,
  isBestPrAuc,
  isExpanded,
  onToggleDetails,
}: {
  experiment: ExperimentResponse
  isBestF1: boolean
  isBestPrAuc: boolean
  isExpanded: boolean
  onToggleDetails: () => void
}) {
  return (
    <div
      className={cn(
        'flex h-full flex-col gap-3 rounded-lg border p-3 text-left transition-colors',
        isExpanded ? 'border-accent bg-accent/5' : 'border-border hover:border-foreground/20',
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <Badge variant="outline">Experiment {experiment.experiment}</Badge>
        <span className="font-mono text-[10px] text-muted-foreground">{experiment.experiment_id}</span>
      </div>

      <div className="grid grid-cols-3 gap-x-2 gap-y-2">
        <Metric label="F1" value={experiment.f1} highlight={isBestF1} />
        <Metric label="ROC-AUC" value={experiment.roc_auc} />
        <Metric label="PR-AUC" value={experiment.pr_auc} highlight={isBestPrAuc} />
      </div>

      <Button
        type="button"
        size="sm"
        variant="outline"
        aria-expanded={isExpanded}
        onClick={onToggleDetails}
        className="mt-auto"
      >
        {isExpanded ? 'Hide details' : 'View details'}
      </Button>
    </div>
  )
}

// A stable (never-recreated) reference for the "no data yet" case, so
// downstream `useMemo`s keyed on `experiments` don't see a new array identity
// on every render.
const EMPTY_EXPERIMENTS: ExperimentResponse[] = []

function EmptyCell() {
  return (
    <div className="flex h-full min-h-[132px] items-center justify-center rounded-lg border border-dashed border-border text-xs text-muted-foreground">
      N/A
    </div>
  )
}

export default function Experiments() {
  const [state, reload] = useExperiments()
  const pcaState = usePCAVisualization()
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [detailsById, setDetailsById] = useState<Record<string, Loadable<ExperimentDetailResponse>>>({})

  const experiments = state.status === 'success' ? state.data : EMPTY_EXPERIMENTS

  const representations = useMemo(() => {
    const ordered: string[] = []
    for (const experiment of experiments) {
      if (!ordered.includes(experiment.representation)) ordered.push(experiment.representation)
    }
    return ordered
  }, [experiments])

  const models = useMemo(() => {
    const ordered: string[] = []
    for (const experiment of experiments) {
      if (!ordered.includes(experiment.model)) ordered.push(experiment.model)
    }
    return ordered
  }, [experiments])

  const bestF1 = experiments.length > 0 ? Math.max(...experiments.map((experiment) => experiment.f1)) : null
  const bestPrAuc = experiments.length > 0 ? Math.max(...experiments.map((experiment) => experiment.pr_auc)) : null

  function toggleDetails(experimentId: string) {
    setExpandedId((current) => (current === experimentId ? null : experimentId))

    setDetailsById((current) => {
      if (current[experimentId]) return current

      getExperiment(experimentId)
        .then((detail) => {
          setDetailsById((prev) => ({ ...prev, [experimentId]: { status: 'success', data: detail } }))
        })
        .catch((error: unknown) => {
          setDetailsById((prev) => ({
            ...prev,
            [experimentId]: {
              status: 'error',
              message: error instanceof Error ? error.message : 'Failed to load experiment details',
            },
          }))
        })

      return { ...current, [experimentId]: { status: 'loading' } }
    })
  }

  const expandedExperiment = expandedId !== null ? experiments.find((e) => e.experiment_id === expandedId) : undefined

  return (
    <div>
      <h1 className="text-xl font-heading font-semibold">Experiments</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Representation × Model comparison for the real A/B/C experiment matrix.
      </p>

      {state.status === 'loading' && <LoadingState message="Loading experiments…" className="mt-6" />}

      {state.status === 'error' && (
        <ErrorState message={`Failed to load experiments — ${state.message}`} onRetry={reload} className="mt-6" />
      )}

      {state.status === 'success' && experiments.length === 0 && (
        <EmptyState
          title="No experiments available"
          message="No experiment results are currently registered by the backend."
          className="mt-6"
        />
      )}

      {state.status === 'success' && experiments.length > 0 && (
        <div className="mt-6 flex flex-col gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Representation × Model</CardTitle>
              <CardDescription>
                Highlighted values mark the highest real F1 / PR-AUC among the experiments shown — computed from
                this response, not a fixed winner.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full min-w-[560px] border-separate border-spacing-2">
                  <thead>
                    <tr>
                      <th className="w-40 text-left text-xs font-medium text-muted-foreground"> </th>
                      {models.map((model) => (
                        <th key={model} className="p-2 text-left text-sm font-semibold">
                          {modelLabel(model)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {representations.map((representation) => (
                      <tr key={representation}>
                        <th scope="row" className="p-2 text-left align-top text-sm font-semibold">
                          {representationLabel(representation)}
                        </th>
                        {models.map((model) => {
                          const experiment = experiments.find(
                            (candidate) => candidate.representation === representation && candidate.model === model,
                          )
                          return (
                            <td key={model} className="min-w-[220px] p-0 align-top">
                              {experiment ? (
                                <MatrixCell
                                  experiment={experiment}
                                  isBestF1={bestF1 !== null && experiment.f1 === bestF1}
                                  isBestPrAuc={bestPrAuc !== null && experiment.pr_auc === bestPrAuc}
                                  isExpanded={expandedId === experiment.experiment_id}
                                  onToggleDetails={() => toggleDetails(experiment.experiment_id)}
                                />
                              ) : (
                                <EmptyCell />
                              )}
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {expandedExperiment && (
            <Card>
              <CardHeader>
                <CardTitle>
                  Experiment {expandedExperiment.experiment} — {expandedExperiment.experiment_id}
                </CardTitle>
                <CardDescription>
                  {representationLabel(expandedExperiment.representation)} → {modelLabel(expandedExperiment.model)}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ExperimentDetails state={detailsById[expandedExperiment.experiment_id] ?? { status: 'loading' }} />
              </CardContent>
            </Card>
          )}
        </div>
      )}

      <Card className="mt-6">
        <CardHeader>
          <CardTitle>PCA 3D Feature Space</CardTitle>
          <CardDescription>
            Real MAFAULDA windows projected through Experiment A's fitted PCA (PC1/PC2/PC3), colored by real class.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <PCA3DPlot state={pcaState} />
        </CardContent>
      </Card>
    </div>
  )
}
