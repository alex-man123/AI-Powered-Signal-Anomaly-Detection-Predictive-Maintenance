# Design System — TASK 1.5

> Source of truth for design tokens. Defined once in `frontend/src/index.css`, consumed
> everywhere else via Tailwind utilities or `var(--token-name)`. Never hard-code a color or
> font in a component — see [Usage Rules](#usage-rules).

## Design Principles

- **Dark-first** — the application ships one palette (dark). No light theme, no
  theme-switching infrastructure exists yet (per blueprint section 30 and TASK 1.5 scope).
- **Industrial / professional** — muted neutral backgrounds, restrained saturation, no
  decorative gradients.
- **Minimal** — one accent color, not a rainbow of competing brand colors.
- **Data-focused** — the type system reserves a monospace face specifically for numeric/signal
  values, so measurements are always visually distinguishable from prose.
- **Consistent** — status colors (NORMAL/WARNING/ANOMALY) are semantic tokens with a single
  definition, used identically everywhere they appear.

## Color Palette

All values are defined once, in `frontend/src/index.css`, inside the `@theme inline` block
(so Tailwind also generates matching utility classes).

| Token | Purpose | Value (OKLCH) | Tailwind utility |
|---|---|---|---|
| `--color-bg` | Main application background | `oklch(0.16 0.005 260)` | `bg-background` |
| `--color-surface` | Cards, panels, sections, containers | `oklch(0.22 0.006 260)` | `bg-card` / `bg-popover` |
| `--color-accent` | The single principal accent — buttons, active states, links, selected elements, highlights, focus rings | `oklch(0.72 0.15 231)` | `bg-primary` / `text-primary` / `ring-*` |
| `--color-normal` | Status: NORMAL | `oklch(0.75 0.16 145)` | `bg-normal` / `text-normal` |
| `--color-warning` | Status: WARNING | `oklch(0.80 0.15 85)` | `bg-warning` / `text-warning` |
| `--color-anomaly` | Status: ANOMALY | `oklch(0.70 0.19 25)` | `bg-anomaly` / `text-anomaly` (also aliased to shadcn's `--destructive`, so `bg-destructive`/`text-destructive` render the same color) |

### Why these values

- Background and surface use a near-black neutral with a faint blue-gray tint (industrial,
  not pure black) and a small lightness step between them (0.16 → 0.22) so cards read as
  a distinct layer without a harsh border.
- The accent is a cyan-blue, chosen deliberately far in hue from the three status colors
  (green/amber/red) so it never gets confused with a status signal — the accent is a UI/brand
  color, status colors are semantic and reserved exclusively for NORMAL/WARNING/ANOMALY.
- Status colors follow the conventional green/amber/red mapping, picked at a lightness
  (~0.7–0.8) that reads clearly as text on the dark background/surface (their primary
  documented usage — see `App.tsx`'s smoke test, which renders them as `text-normal` /
  `text-warning` / `text-anomaly`).

### Accessibility (manual review, no external library)

- Main text (`--foreground`, `oklch(0.96 ...)`) on `--color-bg`/`--color-surface`
  (`0.16`/`0.22`): very large lightness gap → high contrast, safe for body text.
- Secondary text (`--muted-foreground`, `oklch(0.68 ...)`) on `--color-surface` (`0.22`):
  large lightness gap → safe for secondary/label text.
- Accent/status colors (`0.70–0.80` lightness) used as **text** on `--color-bg`/`--color-surface`
  (`0.16`/`0.22`): large lightness gap → readable. Do not use these same values as small
  low-contrast decoration on light surfaces — they are calibrated for dark backgrounds only.

### shadcn / Base UI compatibility

TASK 1.4 (`shadcn init`) generated its own CSS-variable system (`--background`, `--card`,
`--primary`, `--accent`, `--destructive`, `--border`, `--chart-1..5`, `--sidebar-*`, plus a
light `:root` / dark `.dark` split). TASK 1.5 does **not** run a second, competing token
system alongside it. Instead:

- The light `:root` / `.dark` class split was collapsed into a single dark `:root` — this
  project has no light theme and no theme-toggle infrastructure, so keeping an inert,
  never-applied light variant around would just be dead code.
- shadcn's own tokens are **aliased** to the tokens above wherever the concept matches 1:1,
  instead of duplicating raw color values:
  - `--background: var(--color-bg)`, `--card` / `--popover: var(--color-surface)`
  - `--primary` / `--ring` / `--sidebar-primary` / `--sidebar-ring` / `--chart-1: var(--color-accent)`
  - `--destructive: var(--color-anomaly)` (so shadcn's built-in "destructive" button/badge
    variant automatically renders as our anomaly color, for free)
  - `--chart-2` / `--chart-3` / `--chart-4` map to normal / warning / anomaly respectively
- **Deliberate exception:** shadcn's own `--accent` variable (a subtle, neutral hover-tint
  background used e.g. behind ghost-button hover states) is **not** aliased to
  `--color-accent` (our brand accent). They are different concepts — one is a strong brand
  color, the other a barely-there neutral highlight — and forcing them to share a value would
  make one of the two meanings wrong. `--color-accent` instead drives `--primary`/`--ring`,
  which is what buttons and focus states actually use.
- No shadcn components (`button.tsx`, `card.tsx`, `badge.tsx`) were modified — they already
  consume `bg-primary`, `bg-card`, `bg-destructive`, etc., so they picked up the new palette
  automatically once the underlying variables changed.

## Typography

| Token | Family | Used for |
|---|---|---|
| `--font-sans` | Inter Variable (self-hosted via `@fontsource-variable/inter`) | UI, headings, body text, labels, buttons |
| `--font-mono` | JetBrains Mono Variable (self-hosted via `@fontsource-variable/jetbrains-mono`) | Signal values, metrics, numerical data, technical/code-like content |

`html { @apply font-sans; }` sets Inter as the document-wide default. Apply `font-mono`
explicitly (as a Tailwind utility class) wherever a value is a measurement, timestamp, ID,
signal reading, or other technical/numeric content — never apply it to prose.

Fonts are self-hosted (not loaded from a third-party CDN like Google Fonts), consistent with
how the Vite scaffold was already set up in TASK 1.4.

## Spacing

The project uses **Tailwind's default spacing scale as-is** (base unit `--spacing: 0.25rem`
= 4px, every utility a multiple of it) — no custom override was introduced, because
Tailwind's default scale already *is* the exact progression this task asked for:

| Tailwind step | Value |
|---|---|
| `1` | 4px |
| `2` | 8px |
| `3` | 12px |
| `4` | 16px |
| `6` | 24px |
| `8` | 32px |
| `12` | 48px |
| `16` | 64px |

Use these steps (`p-4`, `gap-6`, `px-8`, …) rather than arbitrary pixel values in future
components.

## Usage Rules

- Status colors (NORMAL/WARNING/ANOMALY) are **semantic tokens**, defined once in
  `frontend/src/index.css`. Never hard-code a status color inline in a component.
- There is exactly **one** principal accent color (`--color-accent` → `bg-primary`/`text-primary`).
  Do not introduce a second "brand" accent color without a documented reason.
- Components consume tokens through Tailwind utility classes
  (`bg-background`, `bg-card`, `bg-primary`, `text-normal`, `text-warning`, `text-anomaly`,
  `font-sans`, `font-mono`) — never raw hex/rgb/hsl values.
- `--font-mono` (JetBrains Mono) is reserved for signal values, metrics, and other numeric/
  technical data. `--font-sans` (Inter) is the default for everything else.
- This document, not a component file, is the place to look up a token's intended meaning
  before using it.
