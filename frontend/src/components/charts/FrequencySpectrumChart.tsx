import { useMemo } from 'react'
import Plot from 'react-plotly.js'

import { baseLayout, getChartColors, plotConfig } from '@/components/charts/chart-theme'

interface FrequencySpectrumChartProps {
  frequencies: number[]
  magnitude: number[]
  dominantFrequency: number
}

export function FrequencySpectrumChart({ frequencies, magnitude, dominantFrequency }: FrequencySpectrumChartProps) {
  const colors = useMemo(() => getChartColors(), [])

  const layout = useMemo(
    () => ({
      ...baseLayout(colors),
      xaxis: { ...baseLayout(colors).xaxis, title: { text: 'Frequency (Hz)' } },
      yaxis: { ...baseLayout(colors).yaxis, title: { text: 'Magnitude' } },
      hovermode: 'x' as const,
      shapes: [
        {
          type: 'line' as const,
          x0: dominantFrequency,
          x1: dominantFrequency,
          y0: 0,
          y1: 1,
          yref: 'paper' as const,
          line: { color: colors.warning, width: 1, dash: 'dash' as const },
        },
      ],
      annotations: [
        {
          x: dominantFrequency,
          y: 1,
          yref: 'paper' as const,
          yanchor: 'bottom' as const,
          text: `Dominant: ${dominantFrequency.toFixed(1)} Hz`,
          showarrow: false,
          font: { color: colors.warning, size: 11 },
        },
      ],
    }),
    [colors, dominantFrequency],
  )

  return (
    <Plot
      data={[
        {
          x: frequencies,
          y: magnitude,
          type: 'scattergl',
          mode: 'lines',
          line: { color: colors.accent, width: 2 },
          hovertemplate: 'f = %{x:.2f} Hz<br>magnitude = %{y:.4f}<extra></extra>',
        },
      ]}
      layout={layout}
      config={plotConfig}
      style={{ width: '100%', height: '360px' }}
      useResizeHandler
    />
  )
}
