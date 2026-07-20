import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { App } from './App'
import { loadStudioOverview } from './api/studioApi'
import './styles.css'

async function bootstrap() {
  const root = document.getElementById('root')
  if (!root) throw new Error('root element not found')
  const demo = new URLSearchParams(window.location.search).get('demo') === '1'
  const overview = await loadStudioOverview(fetch, { demo })
  createRoot(root).render(
    <StrictMode>
      <App initialOverview={overview} />
    </StrictMode>,
  )
}

void bootstrap()
