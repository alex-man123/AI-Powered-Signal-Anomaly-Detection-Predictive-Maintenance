import { useMemo } from 'react'
import Plot from 'react-plotly.js'
import type { Data } from 'plotly.js'

import { baseLayout, getChartColors, plotConfig } from '@/components/charts/chart-theme'
import { EmptyState } from '@/components/ui/EmptyState'
import { ErrorState } from '@/components/ui/ErrorState'
import { LoadingState } from '@/components/ui/LoadingState'
import { SIGNAL_LABEL_LABELS } from '@/lib/model-display'
import type { PCAVisualizationResponse, SignalLabel } from '@/services/api'

// TASK 12.2 — real per-window PC1/PC2/PC3 coordinates + real class label
// (`GET /api/experiments/pca-visualization`, `app.services.pca_service`),
// projected through Experiment A's already-fitted PCA (TASK 9.1/9.2). Every
// point here is one real MAFAULDA window; nothing is generated or sampled to
// "fill out" the plot.
export type PCA3DPlotState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'empty'; message: string }
  | { status: 'success'; data: PCAVisualizationResponse }

// Fixed, disclosed order over the app's own already-existing 4-way
// categorical tokens (`--chart-1`..`--chart-4`) — never a random/generated
// color, and never reused per-render.
const LABEL_ORDER: SignalLabel[] = ['normal', 'imbalance', 'horizontal-misalignment', 'vertical-misalignment']

function PCA3DPlotChart({ data }: { data: PCAVisualizationResponse }) {
  const colors = useMemo(() => getChartColors(), [])

  const layout = useMemo(() => {
    const base = baseLayout(colors)
    const axisCommon = {
      gridcolor: colors.border,
      zeroline: false,
      linecolor: colors.border,
      tickfont: { color: colors.mutedForeground, size: 10 },
      backgroundcolor: 'transparent',
    }

    return {
      ...base,
      scene: {
        xaxis: { ...axisCommon, title: { text: 'PC1' } },
        yaxis: { ...axisCommon, title: { text: 'PC2' } },
        zaxis: { ...axisCommon, title: { text: 'PC3' } },
      },
      showlegend: true,
      legend: {
        orientation: 'h' as const,
        y: 1.08,
        x: 0,
        font: { color: colors.mutedForeground, size: 11 },
      },
      margin: { l: 0, r: 0, t: 36, b: 0 },
    }
  }, [colors])

  const traces = useMemo<Data[]>(() => {
    return LABEL_ORDER.map((label, index) => {
      const pointsForLabel = data.points.filter((point) => point.label === label)

      return {
        type: 'scatter3d',
        mode: 'markers',
        name: `${SIGNAL_LABEL_LABELS[label]} (${pointsForLabel.length})`,
        x: pointsForLabel.map((point) => point.pc1),
        y: pointsForLabel.map((point) => point.pc2),
        z: pointsForLabel.map((point) => point.pc3),
        text: pointsForLabel.map((point) => point.recording_id),
        marker: { color: colors.categorical[index], size: 3, opacity: 0.8 },
        hovertemplate:
          `${SIGNAL_LABEL_LABELS[label]}` +
          '<br>Recording: %{text}' +
          '<br>PC1 = %{x:.3f}, PC2 = %{y:.3f}, PC3 = %{z:.3f}<extra></extra>',
      }
    }).filter((trace) => (trace.x as number[]).length > 0)
  }, [data.points, colors])

  const [pc1Ratio, pc2Ratio, pc3Ratio] = data.explained_variance_ratio

  return (
    <div>
      <Plot
        data={traces}
        layout={layout}
        config={plotConfig}
        style={{ width: '100%', height: '480px' }}
        useResizeHandler
      />
      <p className="mt-2 text-xs text-muted-foreground">
        PC1/PC2/PC3 explain {(pc1Ratio * 100).toFixed(1)}%, {(pc2Ratio * 100).toFixed(1)}%, and{' '}
        {(pc3Ratio * 100).toFixed(1)}% of variance respectively, out of {data.total_components} total components.
        PCA is used here for dimensionality reduction / 3D visualization only — it does not prove or guarantee that
        these classes are perfectly separable in the original feature space, and visual overlap between clusters does
        not by itself mean the model fails to detect faults.
      </p>
    </div>
  )
}

export function PCA3DPlot({ state }: { state: PCA3DPlotState }) {
  if (state.status === 'loading') {
    return <LoadingState message="Loading PCA feature space…" />
  }
  if (state.status === 'error') {
    return <ErrorState message={state.message} />
  }
  if (state.status === 'empty') {
    return <EmptyState message={state.message} />
  }
  if (state.data.points.length === 0) {
    return <EmptyState message="No PCA visualization data is currently available." />
  }

  return <PCA3DPlotChart data={state.data} />
}
