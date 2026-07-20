import { Cpu, UserRound } from 'lucide-react'

import type { RuntimeSource } from '../data/types'

interface TopBarProps {
  cudaReady: boolean
  gpuName: string
  pythonVersion: string
  source: RuntimeSource
  error: string | null
}

export function TopBar({ cudaReady, gpuName, pythonVersion, source, error }: TopBarProps) {
  const sourceLabel = source === 'live' ? 'LIVE' : source === 'demo' ? 'DEMO DATA' : 'OFFLINE'
  return (
    <header className="topbar">
      <div className="topbar-group">
        <div className="topbar-chip">
          <Cpu size={15} aria-hidden="true" />
          CUDA <strong className={cudaReady ? 'text-positive' : 'text-danger'}>{cudaReady ? 'READY' : 'OFFLINE'}</strong>
        </div>
        <div className="topbar-chip"><span>GPU</span><strong>{gpuName}</strong></div>
        <div className="topbar-chip"><span>Python</span><strong>{pythonVersion}</strong></div>
      </div>
      <div className="topbar-group topbar-group--right">
        <span className={`demo-badge runtime-source runtime-source--${source}`} title={error ?? undefined}>{sourceLabel}</span>
        <div className="topbar-user"><UserRound size={16} aria-hidden="true" /> researcher</div>
      </div>
    </header>
  )
}
