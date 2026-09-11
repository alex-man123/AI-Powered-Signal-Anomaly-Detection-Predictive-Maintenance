import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router'
import './index.css'
import App from './App.tsx'
import { ActiveModelProvider } from './context/ActiveModelContext.tsx'
import { ActiveSampleClassProvider } from './context/ActiveSampleClassContext.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <ActiveModelProvider>
        <ActiveSampleClassProvider>
          <App />
        </ActiveSampleClassProvider>
      </ActiveModelProvider>
    </BrowserRouter>
  </StrictMode>,
)
