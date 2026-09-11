import { useMemo } from 'react'
import Plot from 'react-plotly.js'

import { baseLayout, getChartColors, plotConfig } from '@/components/charts/chart-theme'

interface TimeSeriesChartProps {
  signal: number[]
  samplingRate: number
}

// AC1: zoom, pan, hover and reset-zoom are all Plotly's own built-in
// behavior (drag-to-zoom, double-click-to-reset, the mode bar) — nothing
// here reimplements them.
export function TimeSeriesChart({ signal, samplingRate }: TimeSeriesChartProps) {
  const colors = useMemo(() => getChartColors(), [])
  const time = useMemo(() => signal.map((_, index) => index / samplingRate), [signal, samplingRate])

  const layout = useMemo(
    () => ({
      ...baseLayout(colors),
      xaxis: {
        ...baseLayout(colors).xaxis,
        title: { text: 'Time (s)' },
        showspikes: true,
        spikemode: 'across' as const,
        spikethickness: 1,
        spikedash: 'solid' as const,
        spikecolor: colors.mutedForeground,
      },
      yaxis: { ...baseLayout(colors).yaxis, title: { text: 'Amplitude' } },
      hovermode: 'x' as const,
    }),
    [colors],
  )

  return (
    <Plot
      data={[
        {
          x: time,
          y: signal,
          type: 'scattergl',
          mode: 'lines',
          line: { color: colors.accent, width: 2 },
          hovertemplate: 't = %{x:.4f}s<br>amplitude = %{y:.4f}<extra></extra>',
        },
      ]}
      layout={layout}
      config={plotConfig}
      style={{ width: '100%', height: '360px' }}
      useResizeHandler
    />
  )
}
