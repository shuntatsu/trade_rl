import { useState } from 'react'

import { AppShell } from './components/AppShell'
import type { WorkspaceId } from './components/Sidebar'
import type { StudioOverviewResult } from './data/types'
import { DashboardPage } from './pages/DashboardPage'
import { DataLabPage } from './pages/DataLabPage'
import { ExperimentsPage } from './pages/ExperimentsPage'
import { RunCenterPage } from './pages/RunCenterPage'
import { WorkspacePage } from './pages/WorkspacePage'

interface AppProps {
  initialOverview: StudioOverviewResult
}

const workspaceMeta: Record<Exclude<WorkspaceId, 'dashboard' | 'data' | 'experiments' | 'runs'>, { title: string; description: string }> = {
  compare: { title: '比較', description: 'run、baseline、コスト条件の差を比較します。' },
  evidence: { title: 'Evidence Explorer', description: 'datasetからreleaseまでの証拠連鎖を確認します。' },
  serving: { title: 'Serving Monitor', description: 'paper serving状態と推論結果を監視します。' },
  settings: { title: '設定', description: 'ローカルUIと実行環境の設定を管理します。' },
}

export function App({ initialOverview }: AppProps) {
  const [active, setActive] = useState<WorkspaceId>('dashboard')
  const { overview, source } = initialOverview

  return (
    <AppShell
      active={active}
      onSelect={setActive}
      source={source}
      cudaReady={overview.system.cudaReady}
      gpuName={overview.system.gpuName}
      pythonVersion={overview.system.pythonVersion}
    >
      {active === 'dashboard' ? <DashboardPage overview={overview} /> : null}
      {active === 'data' ? <DataLabPage /> : null}
      {active === 'experiments' ? <ExperimentsPage /> : null}
      {active === 'runs' ? <RunCenterPage /> : null}
      {active !== 'dashboard' && active !== 'data' && active !== 'experiments' && active !== 'runs' ? (
        <WorkspacePage {...workspaceMeta[active]} />
      ) : null}
    </AppShell>
  )
}
