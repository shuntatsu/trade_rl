import type { ReactNode } from 'react'

import { Sidebar, type WorkspaceId } from './Sidebar'
import { StatusBar } from './StatusBar'
import { TopBar } from './TopBar'

interface AppShellProps {
  active: WorkspaceId
  onSelect: (id: WorkspaceId) => void
  source: 'api' | 'demo'
  cudaReady: boolean
  gpuName: string
  pythonVersion: string
  children: ReactNode
}

export function AppShell({
  active,
  onSelect,
  source,
  cudaReady,
  gpuName,
  pythonVersion,
  children,
}: AppShellProps) {
  return (
    <div className="app-shell">
      <Sidebar active={active} onSelect={onSelect} />
      <TopBar cudaReady={cudaReady} gpuName={gpuName} pythonVersion={pythonVersion} source={source} />
      <main className="main-workspace">{children}</main>
      <StatusBar />
    </div>
  )
}
