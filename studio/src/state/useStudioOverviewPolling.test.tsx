import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { demoOverview } from '../data/demoOverview'
import { offlineOverview } from '../data/offlineOverview'
import { useStudioOverviewPolling } from './useStudioOverviewPolling'

afterEach(() => vi.useRealTimers())

describe('useStudioOverviewPolling', () => {
  it('does not poll in demo mode', async () => {
    vi.useFakeTimers()
    const loader = vi.fn()
    renderHook(() => useStudioOverviewPolling({ source: 'demo', overview: demoOverview, error: null }, { loader, intervalMs: 10 }))
    await act(async () => vi.advanceTimersByTimeAsync(100))
    expect(loader).not.toHaveBeenCalled()
  })

  it('keeps live data when a refresh becomes offline', async () => {
    vi.useFakeTimers()
    const loader = vi.fn().mockResolvedValue({ source: 'offline', overview: offlineOverview, error: 'offline' })
    const { result } = renderHook(() => useStudioOverviewPolling({ source: 'live', overview: demoOverview, error: null }, { loader, intervalMs: 10, now: () => 100 }))
    await act(async () => vi.advanceTimersByTimeAsync(11))
    expect(result.current.source).toBe('stale')
    expect(result.current.overview).toBe(demoOverview)
  })
})
