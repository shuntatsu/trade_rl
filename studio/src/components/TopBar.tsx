import { Cpu, Thermometer, UserRound } from 'lucide-react'

interface TopBarProps {
  cudaReady: boolean
  gpuName: string
  pythonVersion: string
  source: 'api' | 'demo'
}

export function TopBar({ cudaReady, gpuName, pythonVersion, source }: TopBarProps) {
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
        {source === 'demo' ? <span className="demo-badge">DEMO DATA</span> : null}
        <div className="topbar-chip"><Thermometer size={14} aria-hidden="true" /><strong className="text-positive">42°C</strong></div>
        <div className="topbar-user"><UserRound size={16} aria-hidden="true" /> researcher</div>
      </div>
    </header>
  )
}
