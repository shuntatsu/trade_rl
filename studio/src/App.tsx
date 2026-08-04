import { useEffect, useState } from 'react'

import { loadStudioOverview } from './api/studioApi'
import { AppShell } from './components/AppShell'
import type { WorkspaceId } from './components/Sidebar'
import type { StudioOverviewResult } from './data/types'
import { ComparePage } from './pages/ComparePage'
import { DashboardPage } from './pages/DashboardPage'
import { DataLabPage } from './pages/DataLabPage'
import { EvidencePage } from './pages/EvidencePage'
import { ExperimentsPage } from './pages/ExperimentsPage'
import { LiveTrainingPage } from './pages/LiveTrainingPage'
import { RunCenterPage } from './pages/RunCenterPage'
import { ServingPage } from './pages/ServingPage'
import { WorkspacePage } from './pages/WorkspacePage'
import { readWorkspace, replaceParams } from './state/urlState'

interface AppProps {
  initialOverview: StudioOverviewResult
}

const workspaceMeta: Record<Exclude<WorkspaceId, 'dashboard' | 'data' | 'experiments' | 'runs' | 'live' | 'compare' | 'evidence' | 'serving'>, { title: string; description: string }> = {
  settings: { title: '設定', description: 'ローカルUIと実行環境の設定を管理します。' },
}

export function App({ initialOverview }: AppProps) {
  const [active, setActive] = useState<WorkspaceId>(() => readWorkspace(window.location.search))
  const [overviewResult, setOverviewResult] = useState(initialOverview)
  const { overview, source, error } = overviewResult
  useEffect(() => {
    if (initialOverview.source === 'demo') return undefined
    let activeRequest = true
    let timer: number | undefined
    const refresh = async () => {
      const result = await loadStudioOverview()
      if (activeRequest) setOverviewResult(result)
      if (activeRequest) timer = window.setTimeout(() => void refresh(), 500)
    }
    timer = window.setTimeout(() => void refresh(), 500)
    return () => {
      activeRequest = false
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [initialOverview.source])
  const select = (workspace: WorkspaceId) => {
    setActive(workspace)
    replaceParams({ workspace })
  }

  return (
    <AppShell
      active={active}
      onSelect={select}
      source={source}
      sourceError={error}
      cudaReady={overview.system.cudaReady}
      gpuName={overview.system.gpuName}
      pythonVersion={overview.system.pythonVersion}
    >
      {active === 'dashboard' ? <DashboardPage overview={overview} /> : null}
      {active === 'data' ? <DataLabPage /> : null}
      {active === 'experiments' ? <ExperimentsPage /> : null}
      {active === 'runs' ? <RunCenterPage /> : null}
      {active === 'live' ? <LiveTrainingPage /> : null}
      {active === 'compare' ? <ComparePage /> : null}
      {active === 'evidence' ? <EvidencePage /> : null}
      {active === 'serving' ? <ServingPage /> : null}
      {active === 'settings' ? <WorkspacePage {...workspaceMeta[active]} /> : null}
    </AppShell>
  )
}
