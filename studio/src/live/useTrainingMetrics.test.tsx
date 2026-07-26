import { renderHook, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { StudioApi } from '../api/studioApi'
import { useTrainingMetrics } from './useTrainingMetrics'

function api(): StudioApi {
  return {
    loadTrainingMetricsStatus: vi.fn(async () => ({ available: true, selectedSeed: 3, availableSeeds: [3], availableTags: ['train/learning_rate'], lastStep: 20, source: 'events', generation: 'a'.repeat(64) })),
    loadTrainingMetricScalars: vi.fn(async () => ({ seed: 3, series: [{ tag: 'train/learning_rate', displayName: 'Learning rate', group: 'optimization', unit: 'rate', points: [{ step: 20, wallTime: 1, value: 0.0001 }] }], nextStep: 20, generation: 'a'.repeat(64), resetRequired: false })),
  } as unknown as StudioApi
}

describe('useTrainingMetrics', () => {
  it('loads status and finite scalar series for the selected seed', async () => {
    const source = api()
    const { result } = renderHook(() => useTrainingMetrics('job-1', 3, ['train/learning_rate'], source))
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.status?.selectedSeed).toBe(3)
    expect(result.current.series[0].points[0].step).toBe(20)
  })
})
