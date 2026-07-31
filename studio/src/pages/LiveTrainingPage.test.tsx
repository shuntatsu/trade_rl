import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { StudioApi } from '../api/studioApi'
import type { JobSummary, TrainingTelemetryRecord } from '../data/types'

vi.mock('../live/ResearchChartWorkspace', () => ({
  ResearchChartWorkspace: ({ records, onCommitRecord }: {
    records: TrainingTelemetryRecord[]
    onCommitRecord: (record: TrainingTelemetryRecord) => void
  }) => (
    <div role="img" aria-label="BTCUSDT 15m 市場・方策・学習・成績の同期チャート">
      <button
        type="button"
        disabled={!records[1]}
        onClick={() => records[1] && onCommitRecord(records[1])}
      >
        チャートでStep 64を選択
      </button>
    </div>
  ),
}))

import { LiveTrainingPage } from './LiveTrainingPage'

const job: JobSummary = {
  id: 'job-live', schemaVersion: 'studio_job_v2', kind: 'training', status: 'running', runId: 'btc-live-001',
  configResourceId: 'config-1', datasetResourceId: 'dataset-1', configDigest: 'c'.repeat(64), datasetId: 'd'.repeat(64),
  configPath: 'configs/training.json', datasetPath: 'datasets/btc', artifactRoot: 'research',
  submittedAt: '2026-07-21T08:00:00+00:00', ownerInstanceId: 'studio-1', startedAt: '2026-07-21T08:00:01+00:00',
  completedAt: null, pid: 42, pidStartToken: '1', exitCode: null, cancellable: true, error: null,
}

function telemetry(sequence: number, close: number, weight: number, seed = 7): TrainingTelemetryRecord {
  return {
    schemaVersion: 'training_telemetry_v1', sequence, recordedAt: `2026-07-21T08:0${sequence}:00+00:00`,
    globalStep: sequence * 32, environmentStep: sequence, seed, environmentId: 0, episodeId: null,
    eventType: sequence === 2 ? 'position' : 'rollout', marketIndex: 100 + sequence,
    marketTime: `2026-07-21T08:0${sequence}:00.000000000`, symbol: 'BTCUSDT', open: close - 20,
    high: close + 50, low: close - 60, close, action: [weight], executedTarget: [weight],
    weightsBefore: [sequence === 2 ? 0.1 : weight], weightsAfter: [weight], portfolioValue: 100_000 + sequence * 500,
    baselinePortfolioValue: 100_000 + sequence * 100, reward: sequence * 0.1, drawdown: sequence === 3 ? 0.0086 : 0.002,
    intervalCost: 2.5, intervalReturn: 0.001, riskReasons: [], emergencyDeleverage: false, terminated: false, truncated: false,
  }
}

function api(): StudioApi {
  const bySeed: Record<number, TrainingTelemetryRecord[]> = {
    7: [telemetry(1, 67_500, 0.1), telemetry(2, 67_842.3, 0.4), telemetry(3, 67_780, 0.4)],
    11: [telemetry(1, 67_500, 0, 11), telemetry(2, 67_100, -0.2, 11)],
  }
  return {
    loadDatasets: vi.fn().mockResolvedValue({ items: [], total: 0, invalid: 0 }),
    loadRuns: vi.fn().mockResolvedValue({ items: [], total: 0, invalid: 0 }),
    loadConfigs: vi.fn().mockResolvedValue({ items: [], total: 0, invalid: 0 }),
    loadJobs: vi.fn().mockResolvedValue({ items: [job], total: 1 }),
    submitTrainingJob: vi.fn().mockRejectedValue(new Error('not used')),
    cancelJob: vi.fn().mockRejectedValue(new Error('not used')),
    loadJobLog: vi.fn().mockRejectedValue(new Error('not used')),
    loadRunComparison: vi.fn().mockRejectedValue(new Error('not used')),
    loadEvidenceReport: vi.fn().mockRejectedValue(new Error('not used')),
    loadServingMonitor: vi.fn().mockRejectedValue(new Error('not used')),
    loadTelemetryStatus: vi.fn().mockImplementation((_jobId: string, seed: number | null = null) => {
      const selected = seed ?? 7
      const items = bySeed[selected] ?? []
      return Promise.resolve({ available: items.length > 0, selectedSeed: items.length > 0 ? selected : null,
        availableSeeds: [7, 11], recordCount: items.length, lastSequence: items.at(-1)?.sequence ?? 0,
        malformedLines: 0, sizeBytes: 2048, source: items.length ? `seed-${selected}.jsonl` : null,
        streamGeneration: '33333333-3333-4333-8333-333333333333' })
    }),
    loadTelemetryEvents: vi.fn().mockImplementation((_jobId: string, afterSequence = 0, _limit = 512, seed: number | null = null) => {
      const selected = seed ?? 7
      const items = (bySeed[selected] ?? []).filter((item) => item.sequence > afterSequence)
      return Promise.resolve({ seed: selected, items, nextSequence: items.at(-1)?.sequence ?? afterSequence,
        truncated: false, malformedLines: 0, sequenceGaps: [], streamGeneration: '33333333-3333-4333-8333-333333333333', resetRequired: false })
    }),
    loadCheckpointEvaluations: vi.fn().mockResolvedValue({ available: true, productionStatus: 'NO-GO', items: [
      { fold: 'fold-001', configuration: 'residual', seed: 7, policyDigest: 'a'.repeat(64), evaluationDigest: 'b'.repeat(64), score: Math.log1p(0.05), totalReturn: 0.05, finalist: true, checkpointRange: [120, 140], source: 'one.json' },
      { fold: 'fold-000', configuration: 'residual', seed: 7, policyDigest: 'e'.repeat(64), evaluationDigest: 'f'.repeat(64), score: Math.log1p(0.02), totalReturn: 0.02, finalist: true, checkpointRange: [100, 120], source: 'zero.json' },
      { fold: 'fold-000', configuration: 'residual', seed: 11, policyDigest: 'c'.repeat(64), evaluationDigest: 'd'.repeat(64), score: Math.log1p(-0.02), totalReturn: -0.02, finalist: true, checkpointRange: [100, 120], source: 'seed11.json' },
    ] }),
  }
}

describe('LiveTrainingPage research workspace', () => {
  it('renders Run as the only always-visible source selector and removes LIVE chrome', async () => {
    const user = userEvent.setup()
    render(<LiveTrainingPage api={api()} />)
    expect(await screen.findByRole('heading', { name: 'Live Training' })).toBeInTheDocument()
    expect(screen.getByLabelText('Live Training Run')).toBeInTheDocument()
    expect(screen.queryByLabelText('Live Training Seed')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Live Training Environment')).not.toBeInTheDocument()
    expect(screen.queryByText('LIVE')).not.toBeInTheDocument()
    expect(screen.getAllByRole('article')).toHaveLength(3)

    await user.click(screen.getByRole('button', { name: /対象を変更/ }))
    expect(screen.getByLabelText('Live Training Seed')).toBeInTheDocument()
    expect(screen.getByLabelText('Live Training Environment')).toBeInTheDocument()
  })

  it('keeps explicit checkpoint evidence and applies source changes atomically', async () => {
    const user = userEvent.setup()
    render(<LiveTrainingPage api={api()} />)
    const evidence = await screen.findByLabelText('Checkpoint evaluation evidence')
    await waitFor(() => expect(evidence).toHaveDisplayValue('fold-000 · residual · finalist'))
    await user.selectOptions(evidence, screen.getByRole('option', { name: 'fold-001 · residual · finalist' }))
    expect(screen.getByText(/\+5.00% · fold-001/)).toBeInTheDocument()

    const sourceButton = screen.getByRole('button', { name: /対象を変更/ })
    await waitFor(() => expect(sourceButton).toHaveAccessibleName(/Seed 7 · Env 0/))
    await user.click(sourceButton)
    await user.selectOptions(screen.getByLabelText('Live Training Seed'), '11')
    expect(sourceButton).toHaveAccessibleName(/Seed 7 · Env 0/)
    await user.click(screen.getByRole('button', { name: '対象を適用' }))
    await waitFor(() => expect(sourceButton).toHaveAccessibleName(/Seed 11 · Env 0/))
  })

  it('pauses and commits replay when the chart selects a record', async () => {
    const user = userEvent.setup()
    const runtimeApi = api()
    render(<LiveTrainingPage api={runtimeApi} />)
    await screen.findByRole('img', { name: /同期チャート/ })
    const chartSelection = screen.getByRole('button', { name: 'チャートでStep 64を選択' })
    await waitFor(() => expect(chartSelection).toBeEnabled())
    await user.click(chartSelection)

    const inspector = screen.getByRole('complementary', { name: '選択時点の研究データ' })
    await waitFor(() => expect(within(inspector).getByText('Step 64')).toBeInTheDocument())
    expect(screen.getByRole('button', { name: '再生' })).toBeInTheDocument()
    await waitFor(() => expect(runtimeApi.loadTelemetryEvents).toHaveBeenCalled())
  })
})
