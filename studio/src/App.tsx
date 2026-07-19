import { useState } from 'react'

import { AppShell } from './components/AppShell'
import type { WorkspaceId } from './components/Sidebar'
import type { StudioOverviewResult } from './data/types'
import { DashboardPage } from './pages/DashboardPage'
import { WorkspacePage } from './pages/WorkspacePage'

interface AppProps {
  initialOverview: StudioOverviewResult
}

const workspaceMeta: Record<Exclude<WorkspaceId, 'dashboard'>, { title: string; description: string }> = {
  data: { title: 'Data Lab', description: '市場データと特徴量の品質を一画面で確認します。' },
  experiments: { title: '実験', description: '学習条件を組み立て、検証してから実行します。' },
  runs: { title: 'Run Center', description: 'seed、fold、checkpoint、ログを監視します。' },
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
      {active === 'dashboard' ? (
        <DashboardPage overview={overview} />
      ) : (
        <WorkspacePage {...workspaceMeta[active]} />
      )}
    </AppShell>
  )
}
