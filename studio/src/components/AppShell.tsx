import type { ReactNode } from 'react'

import type { RuntimeSource } from '../data/types'
import { Sidebar, type WorkspaceId } from './Sidebar'
import { StatusBar } from './StatusBar'
import { TopBar } from './TopBar'

interface AppShellProps {
  active: WorkspaceId
  onSelect: (id: WorkspaceId) => void
  source: RuntimeSource
  sourceError: string | null
  cudaReady: boolean
  gpuName: string
  pythonVersion: string
  children: ReactNode
}

export function AppShell({
  active,
  onSelect,
  source,
  sourceError,
  cudaReady,
  gpuName,
  pythonVersion,
  children,
}: AppShellProps) {
  return (
    <div className="app-shell">
      <Sidebar active={active} onSelect={onSelect} />
      <TopBar cudaReady={cudaReady} gpuName={gpuName} pythonVersion={pythonVersion} source={source} error={sourceError} />
      <main className="main-workspace">{children}</main>
      <StatusBar />
    </div>
  )
}
