import type { WorkspaceId } from '../components/Sidebar'

const WORKSPACES: WorkspaceId[] = ['dashboard', 'data', 'experiments', 'runs', 'live', 'compare', 'evidence', 'serving', 'settings']

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
