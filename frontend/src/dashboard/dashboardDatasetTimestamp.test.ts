import { describe, expect, it } from 'vitest'

import type { StudioOverview } from '../data/types'
import { buildDashboardCockpitModel } from './dashboardCockpitModel'

const updatedAt = '2026-08-06T00:00:00Z'

const overview: StudioOverview = {
  system: { gpuName: 'GPU', cudaReady: true, pythonVersion: '3.12', metrics: [] },
  latestDataset: {
    id: 'dataset-invalid',
    datasetId: 'd'.repeat(64),
    name: 'broken-market',
    relativePath: 'datasets/broken-market',
    market: 'spot',
    symbols: ['BTCUSDT'],
    timeframes: ['1h'],
    range: '2026',
    status: 'INVALID',
    featureCount: 1,
    barCount: 1,
    symbolCount: 1,
    updated: updatedAt,
    validationError: 'broken',
  },
  activeJobs: [],
  runs: [],
  alerts: [],
  evidence: {
    runResourceId: null,
    status: 'UNAVAILABLE',
    requiredCount: 0,
    verifiedCount: 0,
    blockerCount: 0,
    updatedAt: null,
  },
  equity: [],
  stability: [],
  assessment: { status: 'NO-GO', reasons: ['approval missing'] },
}

describe('Dashboard Dataset timestamps', () => {
  it('keeps an authoritative timestamp out of the human age field', () => {
    const model = buildDashboardCockpitModel(overview)
    const decision = model.decisions.find((item) => item.id === 'dataset:dataset-invalid:invalid')

    expect(decision?.occurredAt).toBe(updatedAt)
    expect(decision?.age).toBeNull()
  })
})
