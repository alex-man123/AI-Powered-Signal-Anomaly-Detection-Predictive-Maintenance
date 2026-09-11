import { Info } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { SampleClassSelector } from '@/components/SampleClassSelector'
import { RawVsFilteredChart } from '@/components/charts/RawVsFilteredChart'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ErrorState } from '@/components/ui/ErrorState'
import { Input } from '@/components/ui/input'
import { LoadingState } from '@/components/ui/LoadingState'
import { Slider } from '@/components/ui/slider'
import { useActiveSampleClass } from '@/context/ActiveSampleClassContext'
import { cn } from '@/lib/utils'
import {
  getSampleSignal,
  runFilter,
  type FilterResponse,
  type FilterType,
  type SampleSignalResponse,
  type SignalLabel,
} from '@/services/api'

const FILTER_TYPE_LABELS: Record<FilterType, string> = {
  lowpass: 'Lowpass',
  highpass: 'Highpass',
  bandpass: 'Bandpass',
}

// How long to wait after the last parameter change before calling the API —
// avoids firing a request per keystroke/drag while still satisfying AC1
// ("changing the cutoff triggers a new API call").
const FILTER_DEBOUNCE_MS = 350

type Loadable<T> =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'success'; data: T }

type PageState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; data: { signal: SampleSignalResponse } }

function useBaseSignal(sampleClass: SignalLabel | null): [PageState, () => void] {
  const [state, setState] = useState<PageState>({ status: 'loading' })
  const [reloadToken, setReloadToken] = useState(0)

  useEffect(() => {
    let cancelled = false

    getSampleSignal(sampleClass ?? undefined)
      .then((signal) => {
        if (!cancelled) setState({ status: 'ready', data: { signal } })
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setState({
            status: 'error',
            message: error instanceof Error ? error.message : 'Failed to load a signal',
          })
        }
      })

    return () => {
      cancelled = true
    }
  }, [reloadToken, sampleClass])

  function reload() {
    setState({ status: 'loading' })
    setReloadToken((token) => token + 1)
  }

  return [state, reload]
}

type FilterParams = {
  filterType: FilterType
  cutoffLow: number
  cutoffHigh: number
  order: number
}

function defaultParamsFor(nyquist: number): FilterParams {
  return {
    filterType: 'lowpass',
    cutoffLow: Math.max(1, Math.round(nyquist * 0.1)),
    cutoffHigh: Math.max(2, Math.round(nyquist * 0.3)),
    order: 4,
  }
}

// Client-side pre-validation — catches the four cases the task calls out
// before ever calling the API. Anything else (e.g. a signal too short for
// the requested filter) is still a real backend error, surfaced via its own
// message (see `errorMessageFor` in services/api.ts).
function validateFilterParams(params: FilterParams, nyquist: number): string | null {
  const { filterType, cutoffLow, cutoffHigh, order } = params

  if (!Number.isInteger(order) || order < 1) {
    return 'Filter order must be a positive integer.'
  }

  if (filterType === 'bandpass') {
    if (cutoffLow <= 0 || cutoffHigh <= 0) {
      return 'Cutoff frequencies must be greater than 0 Hz.'
    }
    if (cutoffLow >= cutoffHigh) {
      return 'Low cutoff must be less than high cutoff for a bandpass filter.'
    }
    if (cutoffHigh >= nyquist) {
      return `High cutoff must be below the Nyquist frequency (${nyquist.toLocaleString()} Hz).`
    }
    return null
  }

  if (cutoffLow <= 0) {
    return 'Cutoff frequency must be greater than 0 Hz.'
  }
  if (cutoffLow >= nyquist) {
    return `Cutoff frequency must be below the Nyquist frequency (${nyquist.toLocaleString()} Hz).`
  }
  return null
}

function NumberSliderField({
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  unit?: string
  onChange: (value: number) => void
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <label className="text-xs text-muted-foreground">{label}</label>
        <span className="font-mono text-sm">
          {value.toLocaleString()}
          {unit ? ` ${unit}` : ''}
        </span>
      </div>
      <div className="mt-1.5 flex items-center gap-3">
        <Slider className="flex-1" min={min} max={max} step={step} value={value} onValueChange={onChange} />
        <Input
          type="number"
          className="w-24 text-right"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => {
            const next = Number(event.target.value)
            if (Number.isFinite(next)) onChange(next)
          }}
        />
      </div>
    </div>
  )
}

function InfoRow({ term, value, description }: { term: string; value: string; description: string }) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <dt className="text-muted-foreground">{term}</dt>
        <dd className="font-mono text-foreground">{value}</dd>
      </div>
      <p className="mt-0.5 text-muted-foreground/80">{description}</p>
    </div>
  )
}

function DSPLabContent({ signal }: { signal: SampleSignalResponse }) {
  const nyquist = signal.sampling_rate / 2
  const [params, setParams] = useState<FilterParams>(() => defaultParamsFor(nyquist))
  const [filterState, setFilterState] = useState<Loadable<FilterResponse>>({ status: 'loading' })
  const [lastFiltered, setLastFiltered] = useState<number[] | null>(null)
  const requestIdRef = useRef(0)

  const validationMessage = validateFilterParams(params, nyquist)

  useEffect(() => {
    // Invalid params: nothing to fetch. `displayState` below already reflects
    // this from `validationMessage` directly, computed during render.
    if (validationMessage) return

    const requestId = ++requestIdRef.current
    setFilterState({ status: 'loading' })

    const timeoutId = setTimeout(() => {
      const cutoff: number | [number, number] =
        params.filterType === 'bandpass' ? [params.cutoffLow, params.cutoffHigh] : params.cutoffLow

      runFilter({
        signal: signal.signal,
        sampling_rate: signal.sampling_rate,
        cutoff,
        order: params.order,
        btype: params.filterType,
      })
        .then((data) => {
          if (requestIdRef.current !== requestId) return
          setFilterState({ status: 'success', data })
          setLastFiltered(data.filtered_signal)
        })
        .catch((error: unknown) => {
          if (requestIdRef.current !== requestId) return
          setFilterState({
            status: 'error',
            message: error instanceof Error ? error.message : 'Filtering failed',
          })
        })
    }, FILTER_DEBOUNCE_MS)

    return () => clearTimeout(timeoutId)
  }, [params.filterType, params.cutoffLow, params.cutoffHigh, params.order, validationMessage, signal])

  // Client-side validation overrides the async fetch state — computed
  // synchronously during render, not via an effect + setState.
  const displayState: Loadable<FilterResponse> = validationMessage
    ? { status: 'error', message: validationMessage }
    : filterState

  const cutoffDisplay =
    params.filterType === 'bandpass'
      ? `${params.cutoffLow.toLocaleString()}–${params.cutoffHigh.toLocaleString()} Hz`
      : `${params.cutoffLow.toLocaleString()} Hz`

  const cutoffMax = Math.max(2, Math.floor(nyquist) - 1)

  return (
    <div className="mt-6 grid gap-4 lg:grid-cols-3">
      <Card className="lg:col-span-1">
        <CardHeader>
          <CardTitle>Filter Configuration</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-5">
          <div>
            <p className="mb-1.5 text-xs text-muted-foreground">Filter type</p>
            <div className="flex flex-wrap gap-1.5">
              {(Object.keys(FILTER_TYPE_LABELS) as FilterType[]).map((type) => (
                <Button
                  key={type}
                  type="button"
                  size="sm"
                  variant={params.filterType === type ? 'default' : 'outline'}
                  onClick={() => setParams((current) => ({ ...current, filterType: type }))}
                >
                  {FILTER_TYPE_LABELS[type]}
                </Button>
              ))}
            </div>
          </div>

          {params.filterType === 'bandpass' ? (
            <>
              <NumberSliderField
                label="Low cutoff"
                unit="Hz"
                value={params.cutoffLow}
                min={1}
                max={cutoffMax}
                step={1}
                onChange={(value) => setParams((current) => ({ ...current, cutoffLow: value }))}
              />
              <NumberSliderField
                label="High cutoff"
                unit="Hz"
                value={params.cutoffHigh}
                min={1}
                max={cutoffMax}
                step={1}
                onChange={(value) => setParams((current) => ({ ...current, cutoffHigh: value }))}
              />
            </>
          ) : (
            <NumberSliderField
              label="Cutoff frequency"
              unit="Hz"
              value={params.cutoffLow}
              min={1}
              max={cutoffMax}
              step={1}
              onChange={(value) => setParams((current) => ({ ...current, cutoffLow: value }))}
            />
          )}

          <NumberSliderField
            label="Filter order"
            value={params.order}
            min={1}
            max={10}
            step={1}
            onChange={(value) => setParams((current) => ({ ...current, order: Math.round(value) }))}
          />

          {validationMessage && <ErrorState size="sm" message={validationMessage} />}

          <div className="rounded-lg bg-muted/50 p-3">
            <p className="mb-2 flex items-center gap-1.5 text-xs font-medium">
              <Info className="size-3.5" />
              DSP parameters
            </p>
            <dl className="flex flex-col gap-2.5 text-xs">
              <InfoRow
                term="Nyquist frequency"
                value={`${nyquist.toLocaleString()} Hz`}
                description={`The highest frequency this signal's ${signal.sampling_rate.toLocaleString()} Hz sampling rate can represent correctly.`}
              />
              <InfoRow
                term="Cutoff frequency"
                value={cutoffDisplay}
                description="The frequency at which the filter's transition begins."
              />
              <InfoRow
                term="Filter order"
                value={String(params.order)}
                description="Controls how steep the filter's rolloff is."
              />
              <InfoRow
                term="Filter type"
                value={FILTER_TYPE_LABELS[params.filterType]}
                description="Which frequency band the filter keeps."
              />
            </dl>
          </div>
        </CardContent>
      </Card>

      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>Raw vs Filtered</CardTitle>
          {displayState.status === 'loading' && <LoadingState size="sm" message="Updating…" />}
          {displayState.status === 'error' && <ErrorState size="sm" message={displayState.message} />}
        </CardHeader>
        <CardContent>
          <div className={cn('transition-opacity', displayState.status !== 'success' && 'opacity-60')}>
            <RawVsFilteredChart
              rawSignal={signal.signal}
              filteredSignal={lastFiltered}
              samplingRate={signal.sampling_rate}
            />
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default function DSPLab() {
  const { activeSampleClass } = useActiveSampleClass()
  const [state, reload] = useBaseSignal(activeSampleClass)

  return (
    <div>
      <h1 className="text-xl font-heading font-semibold">DSP Lab</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Configure a Butterworth filter and compare it against a real validation-set signal.
      </p>

      <SampleClassSelector className="mt-4" />

      {state.status === 'loading' && <LoadingState message="Loading signal…" className="mt-6" />}

      {state.status === 'error' && (
        <ErrorState message={`Failed to load a signal — ${state.message}`} onRetry={reload} className="mt-6" />
      )}

      {state.status === 'ready' && <DSPLabContent signal={state.data.signal} />}
    </div>
  )
}
