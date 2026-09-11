import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router'
import './index.css'
import App from './App.tsx'
import { ActiveModelProvider } from './context/ActiveModelContext.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <ActiveModelProvider>
        <App />
      </ActiveModelProvider>
    </BrowserRouter>
  </StrictMode>,
)
