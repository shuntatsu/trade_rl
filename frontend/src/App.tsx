import { lazy, Suspense, useEffect, useState } from 'react'

import { AppShell } from './components/AppShell'
import type { WorkspaceId } from './components/Sidebar'
import type { StudioOverviewResult } from './data/types'
import { DashboardPage, type DashboardFreshness } from './pages/DashboardPage'
import { pushWorkspace, readWorkspace } from './state/urlState'
import { useStudioOverviewPolling } from './state/useStudioOverviewPolling'

const ComparePage = lazy(async () => ({
  default: (await import('./pages/ComparePage')).ComparePage,
}))
const DataLabPage = lazy(async () => ({
  default: (await import('./pages/DataLabPage')).DataLabPage,
}))
const EvidencePage = lazy(async () => ({
  default: (await import('./pages/EvidencePage')).EvidencePage,
}))
const ExperimentsPage = lazy(async () => ({
  default: (await import('./pages/ExperimentsPage')).ExperimentsPage,
}))
const LiveTrainingPage = lazy(async () => ({
  default: (await import('./pages/LiveTrainingPage')).LiveTrainingPage,
}))
const RunCenterPage = lazy(async () => ({
  default: (await import('./pages/RunCenterPage')).RunCenterPage,
}))
const ServingPage = lazy(async () => ({
  default: (await import('./pages/ServingPage')).ServingPage,
}))
const WorkspacePage = lazy(async () => ({
  default: (await import('./pages/WorkspacePage')).WorkspacePage,
}))

interface AppProps {
  initialOverview: StudioOverviewResult
}

const workspaceMeta: Record<Exclude<WorkspaceId, 'dashboard' | 'data' | 'experiments' | 'runs' | 'live' | 'compare' | 'evidence' | 'serving'>, { title: string; description: string }> = {
  settings: { title: '設定', description: 'ローカルUIと実行環境の設定を管理します。' },
}

function dashboardFreshness(source: StudioOverviewResult['source']): DashboardFreshness {
  if (source === 'live') return 'LIVE'
  if (source === 'stale') return 'STALE'
  if (source === 'demo') return 'DEMO'
  return 'OFFLINE'
}

export function App({ initialOverview }: AppProps) {
  const [active, setActive] = useState<WorkspaceId>(() => readWorkspace(window.location.search))
  const overviewResult = useStudioOverviewPolling(initialOverview)
  const { overview, source, error } = overviewResult

  useEffect(() => {
    const restore = () => setActive(readWorkspace(window.location.search))
    window.addEventListener('popstate', restore)
    return () => window.removeEventListener('popstate', restore)
  }, [])

  const select = (workspace: WorkspaceId) => {
    setActive(workspace)
    pushWorkspace(workspace)
  }
  const drillThrough = (workspace: Exclude<WorkspaceId, 'dashboard' | 'experiments' | 'serving' | 'settings'>, params: Record<string, string>) => {
    setActive(workspace)
    pushWorkspace(workspace, params)
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
      <Suspense fallback={<div role="status">ワークスペースを読み込んでいます…</div>}>
        {active === 'dashboard' ? <DashboardPage overview={overview} freshness={dashboardFreshness(source)} sourceError={error} onNavigate={drillThrough} /> : null}
        {active === 'data' ? <DataLabPage /> : null}
        {active === 'experiments' ? <ExperimentsPage /> : null}
        {active === 'runs' ? <RunCenterPage /> : null}
        {active === 'live' ? <LiveTrainingPage /> : null}
        {active === 'compare' ? <ComparePage /> : null}
        {active === 'evidence' ? <EvidencePage /> : null}
        {active === 'serving' ? <ServingPage /> : null}
        {active === 'settings' ? <WorkspacePage {...workspaceMeta[active]} /> : null}
      </Suspense>
    </AppShell>
  )
}
