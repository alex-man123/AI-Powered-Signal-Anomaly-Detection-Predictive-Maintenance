import type { ReactNode } from 'react'
import { useEffect, useState } from 'react'

import { FrequencySpectrumChart } from '@/components/charts/FrequencySpectrumChart'
import { SpectrogramChart } from '@/components/charts/SpectrogramChart'
import { TimeSeriesChart } from '@/components/charts/TimeSeriesChart'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { EmptyState } from '@/components/ui/EmptyState'
import { ErrorState } from '@/components/ui/ErrorState'
import { LoadingState } from '@/components/ui/LoadingState'
import { Tabs, TabsList, TabsPanel, TabsTab } from '@/components/ui/tabs'
import { MODEL_LABELS, STATUS_BADGE_CLASS } from '@/lib/model-display'
import { cn } from '@/lib/utils'
import {
  extractFeatures,
  getModels,
  getSampleSignal,
  predict,
  runFFT,
  runPSD,
  runSpectrogram,
  type FeatureExtractionResponse,
  type FFTResponse,
  type ModelResponse,
  type ModelType,
  type PredictResponse,
  type PSDResponse,
  type SampleSignalResponse,
  type SpectrogramResponse,
} from '@/services/api'

// Same real Welch-PSD/windowing config `app.services.model_service` itself
// uses for feature extraction/calibration (TASK 10.5) — reused here rather
// than picking arbitrary parameters for the DSP endpoints.
const DSP_NPERSEG = 256
const DSP_NOVERLAP = 128
const SPECTROGRAM_WINDOW_SIZE = 256
const SPECTROGRAM_HOP_LENGTH = 128

type Loadable<T> =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'success'; data: T }

type AnalysisData = {
  signal: SampleSignalResponse
  models: ModelResponse[]
  fft: Loadable<FFTResponse>
  psd: Loadable<PSDResponse>
  spectrogram: Loadable<SpectrogramResponse>
  features: Loadable<FeatureExtractionResponse>
  predictions: Record<ModelType, Loadable<PredictResponse>>
}

type PageState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; data: AnalysisData }

function toLoadable<T>(result: PromiseSettledResult<T>): Loadable<T> {
  if (result.status === 'fulfilled') {
    return { status: 'success', data: result.value }
  }
  return {
    status: 'error',
    message: result.reason instanceof Error ? result.reason.message : 'Request failed',
  }
}

function useSignalAnalysisData(): [PageState, () => void] {
  const [state, setState] = useState<PageState>({ status: 'loading' })
  const [reloadToken, setReloadToken] = useState(0)

  useEffect(() => {
    let cancelled = false

    async function load(): Promise<AnalysisData> {
      // The recording/channel selection this task asks to persist across
      // tabs: fetched once, here, and never touched again by tab switches —
      // see the page component below.
      const [signal, models] = await Promise.all([getSampleSignal(), getModels()])

      const [fftResult, psdResult, spectrogramResult, featuresResult, ...predictionResults] =
        await Promise.allSettled([
          runFFT({ signal: signal.signal, sampling_rate: signal.sampling_rate }),
          runPSD({
            signal: signal.signal,
            sampling_rate: signal.sampling_rate,
            nperseg: DSP_NPERSEG,
            noverlap: DSP_NOVERLAP,
          }),
          runSpectrogram({
            signal: signal.signal,
            sampling_rate: signal.sampling_rate,
            window_size: SPECTROGRAM_WINDOW_SIZE,
            hop_length: SPECTROGRAM_HOP_LENGTH,
          }),
          extractFeatures({
            signal: signal.signal,
            sampling_rate: signal.sampling_rate,
            nperseg: DSP_NPERSEG,
            noverlap: DSP_NOVERLAP,
          }),
          ...models.map((model) =>
            predict({ model_type: model.model_type, signal: signal.signal, sampling_rate: signal.sampling_rate }),
          ),
        ])

      const predictions = Object.fromEntries(
        models.map((model, index) => [model.model_type, toLoadable(predictionResults[index])]),
      ) as Record<ModelType, Loadable<PredictResponse>>

      return {
        signal,
        models,
        fft: toLoadable(fftResult),
        psd: toLoadable(psdResult),
        spectrogram: toLoadable(spectrogramResult),
        features: toLoadable(featuresResult),
        predictions,
      }
    }

    load()
      .then((data) => {
        if (!cancelled) setState({ status: 'ready', data })
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setState({
            status: 'error',
            message: error instanceof Error ? error.message : 'Failed to load signal analysis data',
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

function LoadableSection<T>({ loadable, render }: { loadable: Loadable<T>; render: (data: T) => ReactNode }) {
  if (loadable.status === 'loading') {
    return <LoadingState size="sm" />
  }
  if (loadable.status === 'error') {
    return <ErrorState size="sm" message={loadable.message} />
  }
  return <>{render(loadable.data)}</>
}

function Meta({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className={cn('mt-0.5 text-sm', mono && 'font-mono')}>{value}</p>
    </div>
  )
}

function TimeTab({ signal }: { signal: SampleSignalResponse }) {
  const durationSeconds = signal.signal.length / signal.sampling_rate

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardContent className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
          <Meta label="Recording ID" value={signal.recording_id} mono />
          <Meta label="Label" value={signal.label} />
          <Meta label="Channel" value={String(signal.channel)} />
          <Meta label="Sampling rate" value={`${signal.sampling_rate.toLocaleString()} Hz`} />
          <Meta label="Samples" value={signal.signal.length.toLocaleString()} />
          <Meta label="Duration" value={`${durationSeconds.toFixed(4)} s`} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Waveform</CardTitle>
          <CardDescription>Drag to zoom, double-click to reset, hover for exact values.</CardDescription>
        </CardHeader>
        <CardContent>
          <TimeSeriesChart signal={signal.signal} samplingRate={signal.sampling_rate} />
        </CardContent>
      </Card>
    </div>
  )
}

function FrequencyTab({ fft }: { fft: Loadable<FFTResponse> }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Frequency Spectrum (FFT)</CardTitle>
      </CardHeader>
      <CardContent>
        <LoadableSection
          loadable={fft}
          render={(data) => (
            <FrequencySpectrumChart
              frequencies={data.frequencies}
              magnitude={data.magnitude}
              dominantFrequency={data.dominant_frequency}
            />
          )}
        />
      </CardContent>
    </Card>
  )
}

function SpectrogramTab({ spectrogram }: { spectrogram: Loadable<SpectrogramResponse> }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Spectrogram</CardTitle>
        <CardDescription>
          Welch STFT, window {SPECTROGRAM_WINDOW_SIZE} samples, hop {SPECTROGRAM_HOP_LENGTH} samples.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <LoadableSection
          loadable={spectrogram}
          render={(data) => (
            <SpectrogramChart frequencies={data.frequencies} times={data.times} values={data.values} />
          )}
        />
      </CardContent>
    </Card>
  )
}

function FeaturesTab({ features }: { features: Loadable<FeatureExtractionResponse> }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Extracted Features</CardTitle>
        <CardDescription>Every feature currently registered in the backend's feature registry.</CardDescription>
      </CardHeader>
      <CardContent>
        <LoadableSection
          loadable={features}
          render={(data) =>
            data.feature_names.length === 0 ? (
              <EmptyState size="sm" message="The feature registry returned no features for this signal." />
            ) : (
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
                {data.feature_names.map((name, index) => (
                  <div key={name} className="rounded-lg bg-muted/50 p-3">
                    <p className="text-xs text-muted-foreground capitalize">{name.replaceAll('_', ' ')}</p>
                    <p className="mt-0.5 font-mono text-sm">{data.values[index].toFixed(4)}</p>
                  </div>
                ))}
              </div>
            )
          }
        />
      </CardContent>
    </Card>
  )
}

function AIAnalysisTab({
  models,
  predictions,
}: {
  models: ModelResponse[]
  predictions: Record<ModelType, Loadable<PredictResponse>>
}) {
  if (models.length === 0) {
    return <EmptyState title="No models available" message="No trained models are currently registered by the backend." />
  }

  return (
    <div className="grid gap-4 sm:grid-cols-2">
      {models.map((model) => (
        <Card key={model.model_type}>
          <CardHeader>
            <CardTitle>{MODEL_LABELS[model.model_type]}</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div>
              <p className="text-xs text-muted-foreground">Threshold in use</p>
              <p className="font-mono text-lg">{model.artifact.threshold_value.toFixed(4)}</p>
            </div>

            <LoadableSection
              loadable={predictions[model.model_type]}
              render={(prediction) => (
                <>
                  <div>
                    <p className="text-xs text-muted-foreground">Anomaly score / status</p>
                    <div className="mt-1 flex items-center gap-2">
                      <span className="font-mono text-lg">{prediction.anomaly_score.toFixed(4)}</span>
                      <Badge variant="outline" className={STATUS_BADGE_CLASS[prediction.status]}>
                        {prediction.status}
                      </Badge>
                    </div>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Explanation</p>
                    <p className="mt-0.5 text-sm text-foreground">{prediction.explanation}</p>
                  </div>
                </>
              )}
            />
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

export default function SignalAnalysis() {
  const [state, reload] = useSignalAnalysisData()

  return (
    <div>
      <h1 className="text-xl font-heading font-semibold">Signal Analysis</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Time, frequency, and AI-driven analysis of a real validation-set signal.
      </p>

      {state.status === 'loading' && <LoadingState message="Loading signal analysis data…" className="mt-6" />}

      {state.status === 'error' && (
        <ErrorState
          message={`Failed to load signal analysis data — ${state.message}`}
          onRetry={reload}
          className="mt-6"
        />
      )}

      {state.status === 'ready' && (
        <Tabs defaultValue="time" className="mt-6">
          <TabsList>
            <TabsTab value="time">Time</TabsTab>
            <TabsTab value="frequency">Frequency</TabsTab>
            <TabsTab value="spectrogram">Spectrogram</TabsTab>
            <TabsTab value="features">Features</TabsTab>
            <TabsTab value="ai">AI Analysis</TabsTab>
          </TabsList>

          <TabsPanel value="time">
            <TimeTab signal={state.data.signal} />
          </TabsPanel>
          <TabsPanel value="frequency">
            <FrequencyTab fft={state.data.fft} />
          </TabsPanel>
          <TabsPanel value="spectrogram">
            <SpectrogramTab spectrogram={state.data.spectrogram} />
          </TabsPanel>
          <TabsPanel value="features">
            <FeaturesTab features={state.data.features} />
          </TabsPanel>
          <TabsPanel value="ai">
            <AIAnalysisTab models={state.data.models} predictions={state.data.predictions} />
          </TabsPanel>
        </Tabs>
      )}
    </div>
  )
}
