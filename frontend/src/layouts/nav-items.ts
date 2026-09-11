import {
  AudioWaveform,
  Beaker,
  Brain,
  Database,
  FlaskConical,
  LayoutDashboard,
  Radio,
  TriangleAlert,
  type LucideIcon,
} from 'lucide-react'

export type NavItem = {
  path: string
  label: string
  icon: LucideIcon
}

// Single source of truth for the app's top-level pages — consumed by both the
// router config (App.tsx) and the sidebar (AppShell.tsx) so the two can't drift.
export const navItems: NavItem[] = [
  { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/signals', label: 'Signals', icon: Radio },
  { path: '/signal-analysis', label: 'Signal Analysis', icon: AudioWaveform },
  { path: '/dsp-lab', label: 'DSP Lab', icon: FlaskConical },
  { path: '/anomaly-detection', label: 'Anomaly Detection', icon: TriangleAlert },
  { path: '/models', label: 'Models', icon: Brain },
  { path: '/experiments', label: 'Experiments', icon: Beaker },
  { path: '/dataset', label: 'Dataset', icon: Database },
]
