import { Navigate, Route, Routes } from 'react-router'

import { AppShell } from '@/layouts/AppShell'
import AnomalyDetection from '@/pages/AnomalyDetection'
import Dashboard from '@/pages/Dashboard'
import Dataset from '@/pages/Dataset'
import DSPLab from '@/pages/DSPLab'
import Experiments from '@/pages/Experiments'
import Models from '@/pages/Models'
import NotFound from '@/pages/NotFound'
import SignalAnalysis from '@/pages/SignalAnalysis'
import Signals from '@/pages/Signals'

function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<Dashboard />} />
        <Route path="signals" element={<Signals />} />
        <Route path="signal-analysis" element={<SignalAnalysis />} />
        <Route path="dsp-lab" element={<DSPLab />} />
        <Route path="anomaly-detection" element={<AnomalyDetection />} />
        <Route path="models" element={<Models />} />
        <Route path="experiments" element={<Experiments />} />
        <Route path="dataset" element={<Dataset />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}

export default App
