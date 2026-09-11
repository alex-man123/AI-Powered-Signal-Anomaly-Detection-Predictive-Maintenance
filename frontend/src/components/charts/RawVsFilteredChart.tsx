import { useMemo } from 'react'
import Plot from 'react-plotly.js'
import type { Data } from 'plotly.js'

import { baseLayout, getChartColors, plotConfig } from '@/components/charts/chart-theme'

interface RawVsFilteredChartProps {
  rawSignal: number[]
  filteredSignal: number[] | null
  samplingRate: number
}

// Two series (raw vs filtered) — a categorical/identity comparison, not a
// magnitude scale, so each gets its own fixed color and a legend (raw is the
// de-emphasized reference line, filtered is the highlighted result).
export function RawVsFilteredChart({ rawSignal, filteredSignal, samplingRate }: RawVsFilteredChartProps) {
  const colors = useMemo(() => getChartColors(), [])
  const time = useMemo(() => rawSignal.map((_, index) => index / samplingRate), [rawSignal, samplingRate])

  const layout = useMemo(() => {
    const base = baseLayout(colors)
    return {
      ...base,
      xaxis: {
        ...base.xaxis,
        title: { text: 'Time (s)' },
        showspikes: true,
        spikemode: 'across' as const,
        spikethickness: 1,
        spikedash: 'solid' as const,
        spikecolor: colors.mutedForeground,
      },
      yaxis: { ...base.yaxis, title: { text: 'Amplitude' } },
      hovermode: 'x' as const,
      showlegend: true,
      legend: {
        orientation: 'h' as const,
        y: 1.15,
        x: 0,
        font: { color: colors.mutedForeground, size: 11 },
      },
      margin: { ...base.margin, t: 36 },
    }
  }, [colors])

  const data = useMemo<Data[]>(() => {
    const traces: Data[] = [
      {
        x: time,
        y: rawSignal,
        type: 'scattergl',
        mode: 'lines',
        name: 'Raw',
        line: { color: colors.mutedForeground, width: 1.5 },
        opacity: 0.55,
        hovertemplate: 't = %{x:.4f}s<br>raw = %{y:.4f}<extra></extra>',
      },
    ]

    if (filteredSignal) {
      traces.push({
        x: time,
        y: filteredSignal,
        type: 'scattergl',
        mode: 'lines',
        name: 'Filtered',
        line: { color: colors.accent, width: 2 },
        hovertemplate: 't = %{x:.4f}s<br>filtered = %{y:.4f}<extra></extra>',
      })
    }

    return traces
  }, [time, rawSignal, filteredSignal, colors])

  return (
    <Plot data={data} layout={layout} config={plotConfig} style={{ width: '100%', height: '420px' }} useResizeHandler />
  )
}
