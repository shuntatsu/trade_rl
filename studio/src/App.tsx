import { useState } from 'react'

import { AppShell } from './components/AppShell'
import type { WorkspaceId } from './components/Sidebar'
import type { StudioOverviewResult } from './data/types'
import { ComparePage } from './pages/ComparePage'
import { DashboardPage } from './pages/DashboardPage'
import { DataLabPage } from './pages/DataLabPage'
import { EvidencePage } from './pages/EvidencePage'
import { ExperimentsPage } from './pages/ExperimentsPage'
import { RunCenterPage } from './pages/RunCenterPage'
import { ServingPage } from './pages/ServingPage'
import { WorkspacePage } from './pages/WorkspacePage'
import { readWorkspace, replaceParams } from './state/urlState'

interface AppProps {
  initialOverview: StudioOverviewResult
}

const workspaceMeta: Record<Exclude<WorkspaceId, 'dashboard' | 'data' | 'experiments' | 'runs' | 'compare' | 'evidence' | 'serving'>, { title: string; description: string }> = {
  settings: { title: '設定', description: 'ローカルUIと実行環境の設定を管理します。' },
}

export function App({ initialOverview }: AppProps) {
  const [active, setActive] = useState<WorkspaceId>(() => readWorkspace(window.location.search))
  const { overview, source, error } = initialOverview
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
      {active === 'compare' ? <ComparePage /> : null}
      {active === 'evidence' ? <EvidencePage /> : null}
      {active === 'serving' ? <ServingPage /> : null}
      {active === 'settings' ? <WorkspacePage {...workspaceMeta[active]} /> : null}
    </AppShell>
  )
}
