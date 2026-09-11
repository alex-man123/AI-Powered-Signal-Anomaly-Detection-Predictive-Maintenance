import { useMemo } from 'react'
import Plot from 'react-plotly.js'

import { accentSequentialColorscale, baseLayout, getChartColors, plotConfig } from '@/components/charts/chart-theme'

interface SpectrogramChartProps {
  frequencies: number[]
  times: number[]
  values: number[][]
}

export function SpectrogramChart({ frequencies, times, values }: SpectrogramChartProps) {
  const colors = useMemo(() => getChartColors(), [])

  const layout = useMemo(
    () => ({
      ...baseLayout(colors),
      xaxis: { ...baseLayout(colors).xaxis, title: { text: 'Time (s)' } },
      yaxis: { ...baseLayout(colors).yaxis, title: { text: 'Frequency (Hz)' } },
    }),
    [colors],
  )

  return (
    <Plot
      data={[
        {
          x: times,
          y: frequencies,
          z: values,
          type: 'heatmap',
          colorscale: accentSequentialColorscale(colors),
          colorbar: {
            title: { text: 'Magnitude', side: 'right' },
            tickfont: { color: colors.mutedForeground, size: 11 },
            outlinewidth: 0,
          },
          hoverongaps: false,
          hovertemplate: 't = %{x:.4f}s<br>f = %{y:.1f} Hz<br>magnitude = %{z:.4f}<extra></extra>',
        },
      ]}
      layout={layout}
      config={plotConfig}
      style={{ width: '100%', height: '360px' }}
      useResizeHandler
    />
  )
}
