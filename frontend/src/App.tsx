import Plot from 'react-plotly.js'

import { BackendStatus } from '@/components/BackendStatus'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

// TASK 1.4/1.5/1.6 setup smoke test only — verifies Tailwind, shadcn/ui, Plotly, the
// design tokens, and backend connectivity are wired correctly. Real dashboard UI belongs
// to later phases.
function App() {
  return (
    <div className="bg-background flex min-h-screen items-center justify-center p-8">
      <Card className="w-full max-w-sm">
        <CardHeader>
          <CardTitle>Frontend setup smoke test</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col items-start gap-4">
          <Badge className="font-mono">TASK 1.6</Badge>
          <Button>shadcn Button</Button>
          <div className="font-mono flex gap-4 text-sm">
            <span className="text-normal">NORMAL</span>
            <span className="text-warning">WARNING</span>
            <span className="text-anomaly">ANOMALY</span>
          </div>
          <BackendStatus />
          <div className="h-0 w-0 overflow-hidden" aria-hidden="true">
            <Plot data={[]} layout={{}} />
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default App
