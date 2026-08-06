import type { WorkspaceId } from '../components/Sidebar'
import type { DashboardStageKey } from '../dashboard/dashboardCockpitModel'

const WORKSPACES: WorkspaceId[] = ['dashboard', 'data', 'experiments', 'runs', 'live', 'compare', 'evidence', 'serving', 'settings']
const DASHBOARD_STAGES: DashboardStageKey[] = ['data', 'training', 'evaluation', 'evidence', 'release']

export function readWorkspace(search: string): WorkspaceId {
  const value = new URLSearchParams(search).get('workspace')
  return WORKSPACES.includes(value as WorkspaceId) ? value as WorkspaceId : 'dashboard'
}

export function readParam(search: string, key: string): string | null {
  return new URLSearchParams(search).get(key)
}

export function replaceParams(updates: Record<string, string | null>): void {
  const url = new URL(window.location.href)
  Object.entries(updates).forEach(([key, value]) => {
    if (value) url.searchParams.set(key, value)
    else url.searchParams.delete(key)
  })
  window.history.replaceState(null, '', url)
}

export interface DashboardSelection {
  stage: DashboardStageKey | null
  decision: string | null
}

export function readDashboardSelection(search: string): DashboardSelection {
  const params = new URLSearchParams(search)
  const rawStage = params.get('stage')
  const stage = DASHBOARD_STAGES.includes(rawStage as DashboardStageKey) ? rawStage as DashboardStageKey : null
  const decision = params.get('decision') || null
  return { stage, decision }
}

export function replaceDashboardSelection(selection: DashboardSelection): void {
  replaceParams({ stage: selection.stage, decision: selection.decision })
}

export function pushWorkspace(workspace: WorkspaceId, params: Record<string, string> = {}): void {
  const url = new URL(window.location.href)
  url.searchParams.set('workspace', workspace)
  url.searchParams.delete('stage')
  url.searchParams.delete('decision')
  Object.entries(params).forEach(([key, value]) => url.searchParams.set(key, value))
  window.history.pushState(null, '', url)
}
