import { describe, expect, it } from 'vitest'

import { offlineOverview } from '../data/offlineOverview'
import { demoOverview } from '../data/demoOverview'
import { applyPollingFailure, applyPollingResult, initialPollingState } from './studioOverviewPollingState'

describe('Studio overview polling state', () => {
  it('retains last-known-good data as STALE after a failed refresh', () => {
    const live = initialPollingState({ source: 'live', overview: demoOverview, error: null }, 100)
    const stale = applyPollingResult(live, { source: 'offline', overview: offlineOverview, error: 'offline' }, 200)
    expect(stale.source).toBe('stale')
    expect(stale.overview).toBe(demoOverview)
    expect(stale.lastSuccessfulResponseAt).toBe(100)
  })

  it('replaces stale data only after a successful refresh', () => {
    const stale = { source: 'stale' as const, overview: demoOverview, error: 'offline', lastSuccessfulResponseAt: 100 }
    const next = { ...demoOverview, activeJobs: [] }
    const live = applyPollingResult(stale, { source: 'live', overview: next, error: null }, 250)
    expect(live.source).toBe('live')
    expect(live.overview).toBe(next)
    expect(live.lastSuccessfulResponseAt).toBe(250)
  })

  it('stays OFFLINE when no successful snapshot exists', () => {
    const initial = initialPollingState({ source: 'offline', overview: offlineOverview, error: 'offline' }, 100)
    expect(applyPollingFailure(initial, 'timeout').source).toBe('offline')
  })
})
