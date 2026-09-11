import { useEffect, useRef, useState } from 'react'

import { AnomalyPanel, type AnomalyPanelState } from '@/components/anomaly/AnomalyPanel'
import { useActiveModel } from '@/context/ActiveModelContext'
import { MODEL_LABELS } from '@/lib/model-display'
import { getModels, getSampleSignal, predict, type ModelResponse, type PredictResponse } from '@/services/api'

type Loadable<T> =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'success'; data: T }

function useModels(): Loadable<ModelResponse[]> {
  const [state, setState] = useState<Loadable<ModelResponse[]>>({ status: 'loading' })

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
  }, [])

  return state
}

// Re-runs the real `POST /api/models/predict` call whenever the shared
// active-model selection (TASK 11.5) changes — that context is the only
// source of truth for which model this page uses; nothing is duplicated
// here.
function usePrediction(modelType: ModelResponse['model_type'] | null): Loadable<PredictResponse> | null {
  const [state, setState] = useState<Loadable<PredictResponse> | null>(null)
  const requestIdRef = useRef(0)

  useEffect(() => {
    // Nothing to fetch — the render-time return below already reflects this
    // (`modelType === null ? null : state`) without touching state here.
    if (modelType === null) return

    const requestId = ++requestIdRef.current
    setState({ status: 'loading' })

    getSampleSignal()
      .then((signal) => predict({ model_type: modelType, signal: signal.signal, sampling_rate: signal.sampling_rate }))
      .then((prediction) => {
        if (requestIdRef.current !== requestId) return
        setState({ status: 'success', data: prediction })
      })
      .catch((error: unknown) => {
        if (requestIdRef.current !== requestId) return
        setState({
          status: 'error',
          message: error instanceof Error ? error.message : 'Anomaly analysis failed',
        })
      })
  }, [modelType])

  return modelType === null ? null : state
}

export default function AnomalyDetection() {
  const { activeModel } = useActiveModel()
  const modelsState = useModels()
  const predictionState = usePrediction(activeModel)

  const activeModelDetails =
    modelsState.status === 'success' ? modelsState.data.find((model) => model.model_type === activeModel) : undefined

  const panelState: AnomalyPanelState = (() => {
    if (activeModel === null) {
      return { status: 'empty', message: 'Select a model on the Models page to run anomaly analysis.' }
    }
    if (modelsState.status === 'error') {
      return { status: 'error', message: modelsState.message }
    }
    if (predictionState?.status === 'error') {
      return { status: 'error', message: predictionState.message }
    }
    if (modelsState.status === 'success' && predictionState?.status === 'success' && activeModelDetails) {
      return {
        status: 'success',
        data: {
          modelLabel: MODEL_LABELS[activeModel] ?? activeModel,
          prediction: predictionState.data,
          threshold: activeModelDetails.artifact.threshold_value,
          thresholdMethod: activeModelDetails.artifact.threshold_method,
        },
      }
    }
    return { status: 'loading' }
  })()

  return (
    <div>
      <h1 className="text-xl font-heading font-semibold">Anomaly Detection</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Real-time read of how unusual the current signal is, using the model selected on the Models page.
      </p>

      <div className="mt-6">
        <AnomalyPanel state={panelState} />
      </div>
    </div>
  )
}
