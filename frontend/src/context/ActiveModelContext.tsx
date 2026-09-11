import { createContext, useContext, useState, type ReactNode } from 'react'

import type { ModelType } from '@/services/api'

// TASK 11.5 AC2 — the single source of truth for "which model is active",
// shared between the Models page (which sets it) and Anomaly Detection
// (which consumes it). Backed by sessionStorage so the choice survives a
// reload within the same tab/session, per the task's explicit requirement.
const STORAGE_KEY = 'activeModel'

// Mirrors `ModelType` (`isolation_forest` | `autoencoder`) without hardcoding
// a second copy of that literal union elsewhere.
function isModelType(value: string): value is ModelType {
  return value === 'isolation_forest' || value === 'autoencoder'
}

function readStoredActiveModel(): ModelType | null {
  try {
    const stored = sessionStorage.getItem(STORAGE_KEY)
    return stored !== null && isModelType(stored) ? stored : null
  } catch {
    // sessionStorage unavailable (private browsing, disabled storage, etc.) —
    // fall back to no persisted selection; in-memory state still works for
    // the current page life.
    return null
  }
}

interface ActiveModelContextValue {
  /** The currently active model, or `null` if none has been selected yet
   * (e.g. first visit, or the previously-selected model is no longer
   * returned by the API — the Models page resolves that fallback once it
   * has the real model list). */
  activeModel: ModelType | null
  setActiveModel: (model: ModelType) => void
}

const ActiveModelContext = createContext<ActiveModelContextValue | null>(null)

export function ActiveModelProvider({ children }: { children: ReactNode }) {
  const [activeModel, setActiveModelState] = useState<ModelType | null>(() => readStoredActiveModel())

  function setActiveModel(model: ModelType) {
    setActiveModelState(model)
    try {
      sessionStorage.setItem(STORAGE_KEY, model)
    } catch {
      // Persistence is a nice-to-have; the in-memory selection above still
      // works for the rest of this page session.
    }
  }

  return (
    <ActiveModelContext.Provider value={{ activeModel, setActiveModel }}>{children}</ActiveModelContext.Provider>
  )
}

export function useActiveModel(): ActiveModelContextValue {
  const context = useContext(ActiveModelContext)
  if (!context) {
    throw new Error('useActiveModel must be used within an ActiveModelProvider')
  }
  return context
}
