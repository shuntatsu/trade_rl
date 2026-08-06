import { describe, expect, it } from 'vitest'

import type { DatasetSummary, RunSummary, StudioOverview } from '../data/types'
import { buildDashboardCockpitModel } from './dashboardCockpitModel'

const dataset = (status: 'VALID' | 'INVALID'): DatasetSummary => ({
  id: `dataset-${status.toLowerCase()}`,
  datasetId: 'd'.repeat(64),
  name: 'market',
  relativePath: 'datasets/market',
  market: 'spot',
  symbols: ['BTCUSDT'],
  timeframes: ['1h'],
  range: '2026',
  status,
  featureCount: 1,
  barCount: 1,
  symbolCount: 1,
  updated: 'now',
  validationError: status === 'INVALID' ? 'broken' : null,
})

const run = (status: 'VALID' | 'INVALID', id = `run-${status.toLowerCase()}`): RunSummary => ({
  id,
  runId: id,
  manifestDigest: null,
  relativePath: `research/${id}`,
  runKind: 'research_exploratory',
  algorithm: 'ppo',
  datasetId: 'd'.repeat(64),
  period: '2026',
  createdAt: '2026-08-06T00:00:00Z',
  completedAt: '2026-08-06T01:00:00Z',
  fileCount: 1,
  sharpe: 1.2,
  maxDrawdown: 0.1,
  totalReturn: 0.2,
  productionStatus: 'NO-GO',
  status,
  validationError: status === 'INVALID' ? 'tampered' : null,
})

const overview = (overrides: Partial<StudioOverview> = {}): StudioOverview => ({
  system: { gpuName: 'GPU', cudaReady: true, pythonVersion: '3.12', metrics: [] },
  latestDataset: dataset('VALID'),
  activeJobs: [],
  runs: [run('VALID')],
  alerts: [],
  evidence: { runResourceId: 'run-valid', status: 'VERIFIED', requiredCount: 4, verifiedCount: 4, blockerCount: 0, updatedAt: null },
  equity: [],
  stability: [],
  assessment: { status: 'NO-GO', reasons: ['approval missing'] },
  ...overrides,
})

describe('buildDashboardCockpitModel', () => {
  it('prioritizes invalid data before downstream blockers', () => {
    const model = buildDashboardCockpitModel(overview({ latestDataset: dataset('INVALID') }))
    expect(model.stages[0].state).toBe('BLOCKED')
    expect(model.primaryDecisionId).toBe('dataset:dataset-invalid:invalid')
  })

  it('keeps ready data while training is active', () => {
    const model = buildDashboardCockpitModel(overview({ activeJobs: [{ id: 'job-1', algorithm: 'ppo', phase: 'fold 2', seedProgress: 'seed 1/3', progress: 40 }] }))
    expect(model.stages.find((stage) => stage.key === 'data')?.state).toBe('READY')
    expect(model.stages.find((stage) => stage.key === 'training')?.state).toBe('ACTIVE')
  })

  it('aggregates Evidence blockers into one decision', () => {
    const model = buildDashboardCockpitModel(overview({ evidence: { runResourceId: 'run-valid', status: 'INVALID', requiredCount: 5, verifiedCount: 2, blockerCount: 3, updatedAt: null } }))
    expect(model.decisions.filter((item) => item.stage === 'evidence')).toHaveLength(1)
  })

  it('deduplicates backend alerts that match a derived decision', () => {
    const model = buildDashboardCockpitModel(overview({
      latestDataset: dataset('INVALID'),
      alerts: [{ id: 'dataset:dataset-invalid:invalid', level: 'warning', message: 'duplicate', age: 'now', occurredAt: null }],
    }))
    expect(model.decisions.filter((item) => item.id === 'dataset:dataset-invalid:invalid')).toHaveLength(1)
  })

  it('ranks Release and active Training before an informational valid result', () => {
    const model = buildDashboardCockpitModel(overview({ activeJobs: [{ id: 'job-1', algorithm: 'ppo', phase: 'running', seedProgress: 'seed 1', progress: 10 }] }))
    const ids = model.decisions.map((item) => item.id)
    expect(ids.indexOf('release:no-go')).toBeLessThan(ids.indexOf('job:job-1:active'))
    expect(ids.indexOf('job:job-1:active')).toBeLessThan(ids.indexOf('run:run-valid:valid'))
  })

  it('uses backend run order and never creates a GO state', () => {
    const first = run('VALID', 'run-first')
    const second = run('VALID', 'run-second')
    const model = buildDashboardCockpitModel(overview({ runs: [first, second] }))
    expect(model.latestResult?.resourceId).toBe(first.id)
    expect(model.productionStatus).toBe('NO-GO')
    expect(model.stages.find((stage) => stage.key === 'release')?.state).not.toBe('READY')
  })
})
