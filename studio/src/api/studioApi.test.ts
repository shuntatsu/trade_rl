import { describe, expect, it } from 'vitest'

import { demoOverview } from '../data/demoOverview'
import { loadStudioOverview } from './studioApi'

describe('loadStudioOverview', () => {
  it('returns API data for a successful response', async () => {
    const result = await loadStudioOverview(async () =>
      new Response(JSON.stringify(demoOverview), { status: 200 }),
    )

    expect(result.source).toBe('api')
    expect(result.overview.latestDataset.name).toBe(demoOverview.latestDataset.name)
  })

  it('falls back to deterministic demo data when the request fails', async () => {
    const result = await loadStudioOverview(async () => {
      throw new Error('offline')
    })

    expect(result).toEqual({ source: 'demo', overview: demoOverview })
  })

  it('falls back when the response shape is invalid', async () => {
    const result = await loadStudioOverview(async () =>
      new Response(JSON.stringify({ status: 'ok' }), { status: 200 }),
    )

    expect(result.source).toBe('demo')
  })
})
