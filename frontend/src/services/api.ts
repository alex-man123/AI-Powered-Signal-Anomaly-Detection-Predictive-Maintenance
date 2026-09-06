export interface HealthResponse {
  status: string
}

const API_URL = import.meta.env.VITE_API_URL

export async function getHealth(): Promise<HealthResponse> {
  if (!API_URL) {
    throw new Error('VITE_API_URL is not configured')
  }

  const response = await fetch(`${API_URL}/api/health`)

  if (!response.ok) {
    throw new Error(`Backend responded with HTTP ${response.status}`)
  }

  return (await response.json()) as HealthResponse
}
