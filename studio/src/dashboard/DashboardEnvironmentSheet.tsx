import { Cpu, Server, X } from 'lucide-react'
import { useEffect, useRef } from 'react'

import type { DashboardEnvironmentView } from './dashboardCockpitModel'

interface DashboardEnvironmentSheetProps {
  open: boolean
  environment: DashboardEnvironmentView
  onClose: () => void
}

export function DashboardEnvironmentSheet({ open, environment, onClose }: DashboardEnvironmentSheetProps) {
  const closeRef = useRef<HTMLButtonElement | null>(null)
  useEffect(() => {
    if (!open) return undefined
    closeRef.current?.focus()
    const handle = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', handle)
    return () => window.removeEventListener('keydown', handle)
  }, [open, onClose])
  if (!open) return null
  return (
    <div className="dashboard-sheet-layer" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
      <aside className="dashboard-sheet" role="dialog" aria-modal="true" aria-labelledby="dashboard-environment-title">
        <header><div><span>RUNTIME CONTEXT</span><h2 id="dashboard-environment-title">Environment</h2></div><button ref={closeRef} type="button" aria-label="Environmentを閉じる" onClick={onClose}><X size={18} aria-hidden="true" /></button></header>
        <div className="dashboard-environment-identity">
          <article><Server size={18} aria-hidden="true" /><span>GPU</span><strong>{environment.gpuName}</strong></article>
          <article><Cpu size={18} aria-hidden="true" /><span>CUDA</span><strong className={environment.cudaReady ? 'text-positive' : 'text-danger'}>{environment.cudaReady ? 'READY' : 'OFFLINE'}</strong></article>
          <article><span>PY</span><span>Python</span><strong>{environment.pythonVersion}</strong></article>
        </div>
        <div className="dashboard-environment-metrics">
          {environment.metrics.map((item) => <article key={item.label}><div><span>{item.label}</span><strong>{item.value.toFixed(0)}%</strong></div><div className="dashboard-meter"><i style={{ width: `${Math.min(100, Math.max(0, item.value))}%` }} /></div><small>{item.detail}</small></article>)}
          {environment.metrics.length === 0 ? <p>Runtime metricは取得できません。</p> : null}
        </div>
      </aside>
    </div>
  )
}
