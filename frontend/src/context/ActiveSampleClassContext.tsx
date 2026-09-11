import { createContext, useContext, useState, type ReactNode } from 'react'

import type { SignalLabel } from '@/services/api'

// The single source of truth for "which real class's sample signal is
// currently shown", shared between the selector rendered on Dashboard,
// Signal Analysis, DSP Lab, and Anomaly Detection — all four previously
// always fetched the same hardcoded real `normal` window
// (`GET /api/models/sample-signal` with no `label`). Backed by
// sessionStorage so the choice survives a reload within the same tab/
// session, mirroring `ActiveModelContext`'s own convention.
const STORAGE_KEY = 'activeSampleClass'

const REAL_SIGNAL_LABELS: SignalLabel[] = ['normal', 'imbalance', 'horizontal-misalignment', 'vertical-misalignment']

function isSignalLabel(value: string): value is SignalLabel {
  return (REAL_SIGNAL_LABELS as string[]).includes(value)
}

function readStoredActiveSampleClass(): SignalLabel | null {
  try {
    const stored = sessionStorage.getItem(STORAGE_KEY)
    return stored !== null && isSignalLabel(stored) ? stored : null
  } catch {
    // sessionStorage unavailable (private browsing, disabled storage, etc.) —
    // fall back to no persisted selection; in-memory state still works for
    // the current page life.
    return null
  }
}

interface ActiveSampleClassContextValue {
  /** The currently selected real class, or `null` for the backend's own
   * default (`normal`) — never a value outside the 4 real `SignalLabel`s. */
  activeSampleClass: SignalLabel | null
  setActiveSampleClass: (label: SignalLabel) => void
}

const ActiveSampleClassContext = createContext<ActiveSampleClassContextValue | null>(null)

export function ActiveSampleClassProvider({ children }: { children: ReactNode }) {
  const [activeSampleClass, setActiveSampleClassState] = useState<SignalLabel | null>(() =>
    readStoredActiveSampleClass(),
  )

  function setActiveSampleClass(label: SignalLabel) {
    setActiveSampleClassState(label)
    try {
      sessionStorage.setItem(STORAGE_KEY, label)
    } catch {
      // Persistence is a nice-to-have; the in-memory selection above still
      // works for the rest of this page session.
    }
  }

  return (
    <ActiveSampleClassContext.Provider value={{ activeSampleClass, setActiveSampleClass }}>
      {children}
    </ActiveSampleClassContext.Provider>
  )
}

export function useActiveSampleClass(): ActiveSampleClassContextValue {
  const context = useContext(ActiveSampleClassContext)
  if (!context) {
    throw new Error('useActiveSampleClass must be used within an ActiveSampleClassProvider')
  }
  return context
}
