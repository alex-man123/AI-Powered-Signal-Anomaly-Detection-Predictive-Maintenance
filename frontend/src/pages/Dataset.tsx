import { Activity, Clock, Hash, Layers, ShieldCheck } from 'lucide-react'
import { useEffect, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { EmptyState } from '@/components/ui/EmptyState'
import { ErrorState } from '@/components/ui/ErrorState'
import { LoadingState } from '@/components/ui/LoadingState'
import { cn } from '@/lib/utils'
import { getDataset, getDatasets, type DatasetDetailResponse, type DatasetSummaryResponse } from '@/services/api'

type Loadable<T> =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'success'; data: T }

function useDatasetList(): [Loadable<DatasetSummaryResponse[]>, () => void] {
  const [state, setState] = useState<Loadable<DatasetSummaryResponse[]>>({ status: 'loading' })
  const [reloadToken, setReloadToken] = useState(0)

  useEffect(() => {
    let cancelled = false

    getDatasets()
      .then((datasets) => {
        if (!cancelled) setState({ status: 'success', data: datasets })
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setState({
            status: 'error',
            message: error instanceof Error ? error.message : 'Failed to load datasets',
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

function useDatasetDetail(datasetId: number | null): [Loadable<DatasetDetailResponse> | null, () => void] {
  const [state, setState] = useState<Loadable<DatasetDetailResponse> | null>(null)
  const [reloadToken, setReloadToken] = useState(0)

  useEffect(() => {
    if (datasetId === null) return

    let cancelled = false
    setState({ status: 'loading' })

    getDataset(datasetId)
      .then((detail) => {
        if (!cancelled) setState({ status: 'success', data: detail })
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setState({
            status: 'error',
            message: error instanceof Error ? error.message : 'Failed to load dataset detail',
          })
        }
      })

    return () => {
      cancelled = true
    }
  }, [datasetId, reloadToken])

  function reload() {
    setState({ status: 'loading' })
    setReloadToken((token) => token + 1)
  }

  return [datasetId === null ? null : state, reload]
}

// Presentation-only formatting — the underlying real values are never
// changed, only reformatted for readability (task section 11).
function formatCompactNumber(value: number): string {
  return value.toLocaleString(undefined, { notation: 'compact', maximumFractionDigits: 1 })
}

function formatHz(hz: number): string {
  if (hz >= 1000) {
    return `${(hz / 1000).toLocaleString(undefined, { maximumFractionDigits: 2 })} kHz`
  }
  return `${hz.toLocaleString()} Hz`
}

function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${seconds % 1 === 0 ? seconds : seconds.toFixed(3)}s`
  }
  const totalSeconds = Math.round(seconds)
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const secs = totalSeconds % 60
  return [hours ? `${hours}h` : null, minutes ? `${minutes}m` : null, `${secs}s`].filter(Boolean).join(' ')
}

function MetadataCard({
  icon: Icon,
  label,
  value,
  caption,
  unavailable,
}: {
  icon: typeof Hash
  label: string
  value: string
  caption?: string
  unavailable?: boolean
}) {
  return (
    <Card>
      <CardContent className="flex flex-col gap-1.5 pt-4">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Icon className="size-3.5" />
          {label}
        </div>
        <p className={cn('font-mono text-2xl font-semibold', unavailable && 'text-muted-foreground')}>{value}</p>
        {caption && <p className="text-xs text-muted-foreground">{caption}</p>}
      </CardContent>
    </Card>
  )
}

function DatasetInformation({ dataset }: { dataset: DatasetDetailResponse }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Dataset information</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-5">
        <div>
          <p className="mb-1 text-xs text-muted-foreground">Dataset source</p>
          <p className="text-sm">
            <span className="font-mono">{dataset.name}</span>{' '}
            <span className="text-muted-foreground">(id {dataset.id})</span>
          </p>
        </div>

        <div>
          <p className="mb-1 text-xs text-muted-foreground">Recording information</p>
          <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm">
            <span>
              <span className="font-mono">{dataset.signal_count.toLocaleString()}</span> recordings
            </span>
            <span>
              <span className="font-mono">{dataset.samples_per_signal.toLocaleString()}</span> samples per recording
            </span>
            <span>
              channel indices{' '}
              <span className="font-mono">
                0–{dataset.channels.length - 1} ({dataset.channels.length} total)
              </span>
            </span>
          </div>
        </div>

        {dataset.labels.length > 0 && (
          <div>
            <p className="mb-1.5 text-xs text-muted-foreground">Classes / labels</p>
            <div className="flex flex-wrap gap-1.5">
              {dataset.labels.map((label) => (
                <Badge key={label} variant="outline">
                  {label}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default function Dataset() {
  const [listState, reloadList] = useDatasetList()
  const [selectedId, setSelectedId] = useState<number | null>(null)

  const datasets = listState.status === 'success' ? listState.data : []

  // Default to the first dataset once the list loads — no selector UI is
  // built when there is nothing to choose between (task section 4).
  if (listState.status === 'success' && selectedId === null && datasets.length > 0) {
    setSelectedId(datasets[0].id)
  }

  const [detailState, reloadDetail] = useDatasetDetail(selectedId)
  const selectedSummary = datasets.find((dataset) => dataset.id === selectedId)

  return (
    <div>
      <h1 className="text-xl font-heading font-semibold">Dataset</h1>
      <p className="mt-1 text-sm text-muted-foreground">Real dataset metadata, sourced from the dataset API.</p>

      {listState.status === 'loading' && <LoadingState message="Loading dataset…" className="mt-6" />}

      {listState.status === 'error' && (
        <ErrorState
          message={`Unable to load dataset information — ${listState.message}`}
          onRetry={reloadList}
          retryLabel="Try again"
          className="mt-6"
        />
      )}

      {listState.status === 'success' && datasets.length === 0 && (
        <EmptyState
          title="No datasets available"
          message="No datasets are currently registered by the backend."
          className="mt-6"
        />
      )}

      {listState.status === 'success' && datasets.length > 0 && (
        <div className="mt-6 flex flex-col gap-6">
          {datasets.length > 1 && (
            <div className="flex flex-wrap gap-1.5">
              {datasets.map((dataset) => (
                <Button
                  key={dataset.id}
                  type="button"
                  size="sm"
                  variant={dataset.id === selectedId ? 'default' : 'outline'}
                  onClick={() => setSelectedId(dataset.id)}
                >
                  {dataset.name}
                </Button>
              ))}
            </div>
          )}

          <div>
            <h2 className="font-heading text-lg font-semibold capitalize">{selectedSummary?.name}</h2>
            <p className="text-sm text-muted-foreground">
              {selectedSummary?.signal_count.toLocaleString()} real recordings
            </p>
          </div>

          {detailState?.status === 'loading' && <LoadingState message="Loading dataset details…" />}

          {detailState?.status === 'error' && (
            <ErrorState
              message={`Unable to load dataset information — ${detailState.message}`}
              onRetry={reloadDetail}
              retryLabel="Try again"
            />
          )}

          {detailState?.status === 'success' && (
            <>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
                <MetadataCard
                  icon={Hash}
                  label="Samples"
                  value={formatCompactNumber(detailState.data.signal_count * detailState.data.samples_per_signal)}
                  caption={`${detailState.data.signal_count.toLocaleString()} recordings × ${detailState.data.samples_per_signal.toLocaleString()} samples`}
                />
                <MetadataCard
                  icon={Activity}
                  label="Sampling rate"
                  value={formatHz(detailState.data.sampling_rate)}
                  caption={`${detailState.data.sampling_rate.toLocaleString()} Hz`}
                />
                <MetadataCard
                  icon={Clock}
                  label="Duration"
                  value={formatDuration(detailState.data.samples_per_signal / detailState.data.sampling_rate)}
                  caption="per recording (samples ÷ sampling rate)"
                />
                <MetadataCard icon={Layers} label="Channels" value={String(detailState.data.channels.length)} />
                <MetadataCard
                  icon={ShieldCheck}
                  label="Missing values"
                  value={detailState.data.missing_values.toLocaleString()}
                  caption={`across ${detailState.data.signal_count.toLocaleString()} recordings`}
                />
              </div>

              <DatasetInformation dataset={detailState.data} />
            </>
          )}
        </div>
      )}
    </div>
  )
}
