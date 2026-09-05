import Plot from 'react-plotly.js'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

// TASK 1.4 setup smoke test only — verifies Tailwind, shadcn/ui and Plotly are wired
// correctly. Real dashboard UI belongs to later phases.
function App() {
  return (
    <div className="bg-slate-900 flex min-h-screen items-center justify-center p-8">
      <Card className="w-full max-w-sm">
        <CardHeader>
          <CardTitle>Frontend setup smoke test</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col items-start gap-4">
          <Badge>TASK 1.4</Badge>
          <Button>shadcn Button</Button>
          <div className="h-0 w-0 overflow-hidden" aria-hidden="true">
            <Plot data={[]} layout={{}} />
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default App
