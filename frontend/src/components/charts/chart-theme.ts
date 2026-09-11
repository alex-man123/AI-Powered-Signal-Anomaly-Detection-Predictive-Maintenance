import type { Config, Layout } from 'plotly.js'

// Resolves this app's own CSS design tokens (index.css's oklch() custom
// properties) into rgb()/rgba() strings — Plotly (in particular `scattergl`'s
// WebGL renderer, and colorscale interpolation) parses colors with its own
// vendored color library, which does not understand the modern CSS Color 4
// oklch() function, so the literal token string can't be handed to it as-is.
// Converts with the standard OKLab<->linear-sRGB matrices (Björn Ottosson),
// rather than hand-maintaining a second copy of the palette in hex/rgb that
// could drift from the real tokens in index.css.
function parseOklch(value: string): { l: number; c: number; h: number; alpha: number } | null {
  const match = value.match(/oklch\(\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s*\/\s*([\d.]+)%?)?\s*\)/)
  if (!match) return null
  const [, l, c, h, alpha] = match
  return { l: Number(l), c: Number(c), h: Number(h), alpha: alpha === undefined ? 1 : Number(alpha) / 100 }
}

function oklchToRgbString(value: string): string {
  const parsed = parseOklch(value)
  if (!parsed) return value // already rgb()/hex/named — pass through unchanged
  const { l, c, h, alpha } = parsed

  const hRad = (h * Math.PI) / 180
  const a = c * Math.cos(hRad)
  const b = c * Math.sin(hRad)

  const l_ = l + 0.3963377774 * a + 0.2158037573 * b
  const m_ = l - 0.1055613458 * a - 0.0638541728 * b
  const s_ = l - 0.0894841775 * a - 1.2914855480 * b

  const l3 = l_ ** 3
  const m3 = m_ ** 3
  const s3 = s_ ** 3

  const rLin = 4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
  const gLin = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
  const bLin = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3

  const gamma = (x: number) => (x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(Math.max(x, 0), 1 / 2.4) - 0.055)
  const toByte = (x: number) => Math.round(Math.min(1, Math.max(0, gamma(x))) * 255)

  const r = toByte(rLin)
  const g = toByte(gLin)
  const bC = toByte(bLin)

  return alpha < 1 ? `rgba(${r}, ${g}, ${bC}, ${alpha})` : `rgb(${r}, ${g}, ${bC})`
}

function cssVar(name: string): string {
  const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return oklchToRgbString(raw)
}

export interface ChartColors {
  accent: string
  foreground: string
  mutedForeground: string
  border: string
  surface: string
  normal: string
  warning: string
  anomaly: string
  categorical: [string, string, string, string]
}

export function getChartColors(): ChartColors {
  return {
    accent: cssVar('--color-accent'),
    foreground: cssVar('--foreground'),
    mutedForeground: cssVar('--muted-foreground'),
    border: cssVar('--border'),
    surface: cssVar('--color-surface'),
    normal: cssVar('--color-normal'),
    warning: cssVar('--color-warning'),
    anomaly: cssVar('--color-anomaly'),
    // The app's own already-defined `--chart-1`..`--chart-4` design tokens
    // (index.css), in their existing fixed order — a real, pre-existing
    // 4-way categorical set, not a new palette invented for one chart.
    categorical: [
      cssVar('--color-chart-1'),
      cssVar('--color-chart-2'),
      cssVar('--color-chart-3'),
      cssVar('--color-chart-4'),
    ],
  }
}

// One hue (the app's own accent), light->dark per the sequential-magnitude
// rule — anchored to the surface color at the low end since this app is
// dark-only (the ramp's anchor "flips" relative to a light-mode chart).
export function accentSequentialColorscale(colors: ChartColors): [number, string][] {
  return [
    [0, colors.surface],
    [1, colors.accent],
  ]
}

export function baseLayout(colors: ChartColors): Partial<Layout> {
  const axisCommon = {
    gridcolor: colors.border,
    gridwidth: 1,
    zeroline: false,
    linecolor: colors.border,
    tickfont: { color: colors.mutedForeground, size: 11 },
    titlefont: { color: colors.mutedForeground, size: 12 },
  }

  return {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: colors.mutedForeground, family: 'JetBrains Mono Variable, ui-monospace, monospace', size: 12 },
    margin: { l: 56, r: 24, t: 16, b: 44 },
    xaxis: axisCommon,
    yaxis: axisCommon,
    hoverlabel: {
      bgcolor: colors.surface,
      bordercolor: colors.border,
      font: { color: colors.foreground, family: 'JetBrains Mono Variable, ui-monospace, monospace' },
    },
    showlegend: false,
  }
}

export const plotConfig: Partial<Config> = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ['lasso2d', 'select2d'],
}
