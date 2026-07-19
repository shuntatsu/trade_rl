import { demoOverview } from '../data/demoOverview'
import type { StudioOverview, StudioOverviewResult } from '../data/types'

function isStudioOverview(value: unknown): value is StudioOverview {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as Partial<StudioOverview>
  return Boolean(
    candidate.system &&
      candidate.latestDataset &&
      Array.isArray(candidate.activeJobs) &&
      Array.isArray(candidate.runs) &&
      Array.isArray(candidate.alerts) &&
      Array.isArray(candidate.equity) &&
      Array.isArray(candidate.stability) &&
      candidate.assessment,
  )
}

export async function loadStudioOverview(
  fetcher: typeof fetch = fetch,
): Promise<StudioOverviewResult> {
  try {
    const response = await fetcher('/api/studio/overview')
    if (!response.ok) return { source: 'demo', overview: demoOverview }
    const payload: unknown = await response.json()
    if (!isStudioOverview(payload)) return { source: 'demo', overview: demoOverview }
    return { source: 'api', overview: payload }
  } catch {
    return { source: 'demo', overview: demoOverview }
  }
}
