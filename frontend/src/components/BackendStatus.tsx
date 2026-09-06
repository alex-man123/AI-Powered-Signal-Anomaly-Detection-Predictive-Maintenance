import { useEffect, useState } from 'react'

import { getHealth } from '@/services/api'

type Status =
  | { state: 'loading' }
  | { state: 'success'; value: string }
  | { state: 'error'; message: string }

// TASK 1.6 connectivity smoke test — proves the frontend can reach the real backend
// over HTTP. Not a real application page.
export function BackendStatus() {
  const [status, setStatus] = useState<Status>({ state: 'loading' })

  useEffect(() => {
    let cancelled = false

    getHealth()
      .then((health) => {
        if (!cancelled) {
          setStatus({ state: 'success', value: health.status })
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setStatus({
            state: 'error',
            message: error instanceof Error ? error.message : 'Backend unavailable',
          })
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  if (status.state === 'loading') {
    return <p className="text-muted-foreground text-sm">Checking backend...</p>
  }

  if (status.state === 'error') {
    return <p className="text-anomaly text-sm">Backend unavailable — {status.message}</p>
  }

  return <p className="text-normal text-sm">Backend status: {status.value}</p>
}
