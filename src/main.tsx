import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'
import Studio from './Studio'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {location.pathname.startsWith('/studio') ? <Studio /> : <App />}
  </StrictMode>,
)
