import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { StudioApi } from '../api/studioApi'
import type { TrainingMetricsStatusResponse } from '../data/types'
import { useTrainingMetrics } from './useTrainingMetrics'

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  const promise = new Promise<T>((next) => { resolve = next })
  return { promise, resolve }
}

function metricsStatus(): TrainingMetricsStatusResponse {
  return {
    available: true,
    selectedSeed: 3,
    availableSeeds: [3],
    availableTags: ['train/learning_rate'],
    lastStep: 20,
    source: 'events',
    generation: 'a'.repeat(64),
  }
}

function api(): StudioApi {
  return {
    loadTrainingMetricsStatus: vi.fn(async () => metricsStatus()),
    loadTrainingMetricScalars: vi.fn(async (_jobId: string, tags: string[]) => ({
      seed: 3,
      series: tags.map((tag) => ({
        tag,
        displayName: tag,
        group: 'optimization' as const,
        unit: 'raw' as const,
        points: [{ step: 20, wallTime: 1, value: 0.0001 }],
      })),
      nextStep: 20,
      generation: 'a'.repeat(64),
      resetRequired: false,
    })),
  } as unknown as StudioApi
}

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('useTrainingMetrics', () => {
  it('loads status and finite scalar series for the selected seed', async () => {
    const source = api()
    const { result } = renderHook(() => useTrainingMetrics('job-1', 3, ['train/learning_rate'], source))
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.status?.selectedSeed).toBe(3)
    expect(result.current.series[0].points[0].step).toBe(20)
  })

  it('splits metric requests at the server eight-tag limit', async () => {
    const source = api()
    const tags = Array.from({ length: 14 }, (_, index) => `metric-${index}`)
    const { result } = renderHook(() => useTrainingMetrics('job-1', 3, tags, source))

    await waitFor(() => expect(result.current.loading).toBe(false))
    const loader = vi.mocked(source.loadTrainingMetricScalars!)
    expect(loader).toHaveBeenCalledTimes(2)
    expect(loader.mock.calls.map((call) => call[1].length)).toEqual([8, 6])
    expect(result.current.series).toHaveLength(14)
  })

  it('waits for a slow refresh before scheduling the next poll', async () => {
    vi.useFakeTimers()
    const statusRequest = deferred<TrainingMetricsStatusResponse>()
    const loadStatus = vi.fn().mockReturnValue(statusRequest.promise)
    const source = {
      ...api(),
      loadTrainingMetricsStatus: loadStatus,
    } as StudioApi

    const { unmount } = renderHook(() =>
      useTrainingMetrics('job-1', 3, ['train/learning_rate'], source))

    await act(async () => { await Promise.resolve() })
    expect(loadStatus).toHaveBeenCalledTimes(1)

    await act(async () => { vi.advanceTimersByTime(2_000) })
    expect(loadStatus).toHaveBeenCalledTimes(1)

    statusRequest.resolve(metricsStatus())
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
      await Promise.resolve()
    })

    await act(async () => { vi.advanceTimersByTime(1_999) })
    expect(loadStatus).toHaveBeenCalledTimes(1)
    await act(async () => {
      vi.advanceTimersByTime(1)
      await Promise.resolve()
    })
    expect(loadStatus).toHaveBeenCalledTimes(2)
    unmount()
  })
})
