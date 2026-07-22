import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { StudioApi } from '../api/studioApi'
import type { JobSummary, TrainingTelemetryRecord } from '../data/types'
import { LiveTrainingPage } from './LiveTrainingPage'

const job: JobSummary = {
  id: 'job-live',
  schemaVersion: 'studio_job_v2',
  kind: 'training',
  status: 'running',
  runId: 'btc-live-001',
  configResourceId: 'config-1',
  datasetResourceId: 'dataset-1',
  configDigest: 'c'.repeat(64),
  datasetId: 'd'.repeat(64),
  configPath: 'configs/training.json',
  datasetPath: 'datasets/btc',
  artifactRoot: 'research',
  submittedAt: '2026-07-21T08:00:00+00:00',
  ownerInstanceId: 'studio-1',
  startedAt: '2026-07-21T08:00:01+00:00',
  completedAt: null,
  pid: 42,
  pidStartToken: '1',
  exitCode: null,
  cancellable: true,
  error: null,
}

function telemetry(
  sequence: number,
  close: number,
  weight: number,
  seed = 7,
): TrainingTelemetryRecord {
  return {
    schemaVersion: 'training_telemetry_v1',
    sequence,
    recordedAt: `2026-07-21T08:0${sequence}:00+00:00`,
    globalStep: sequence * 32,
    environmentStep: sequence,
    seed,
    environmentId: 0,
    episodeId: 1,
    eventType: sequence === 2 ? 'position' : 'rollout',
    marketIndex: 100 + sequence,
    marketTime: `2026-07-21T08:0${sequence}:00.000000000`,
    symbol: 'BTCUSDT',
    open: close - 20,
    high: close + 50,
    low: close - 60,
    close,
    action: [weight],
    executedTarget: [weight],
    weightsBefore: [sequence === 2 ? 0.1 : weight],
    weightsAfter: [weight],
    portfolioValue: 100_000 + sequence * 500,
    baselinePortfolioValue: 100_000 + sequence * 100,
    reward: sequence * 0.1,
    drawdown: sequence === 3 ? 0.0086 : 0.002,
    intervalCost: 2.5,
    intervalReturn: 0.001,
    riskReasons: [],
    emergencyDeleverage: false,
    terminated: false,
    truncated: false,
  }
}

function api(): StudioApi {
  const bySeed: Record<number, TrainingTelemetryRecord[]> = {
    7: [
      telemetry(1, 67_500, 0.1),
      telemetry(2, 67_842.3, 0.4),
      telemetry(3, 67_780, 0.4),
    ],
    11: [
      telemetry(1, 67_500, 0, 11),
      telemetry(2, 67_100, -0.2, 11),
    ],
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
      return Promise.resolve({
        available: items.length > 0,
        selectedSeed: items.length > 0 ? selected : null,
        availableSeeds: [7, 11],
        recordCount: items.length,
        lastSequence: items.at(-1)?.sequence ?? 0,
        malformedLines: 0,
        sizeBytes: 2048,
        source: items.length > 0 ? `research/.staging/btc-live-001/seed-${selected}/telemetry/training-telemetry.jsonl` : null,
        streamGeneration: items.length > 0 ? '33333333-3333-4333-8333-333333333333' : null,
      })
    }),
    loadTelemetryEvents: vi.fn().mockImplementation((
      _jobId: string,
      afterSequence = 0,
      _limit = 512,
      seed: number | null = null,
    ) => {
      const selected = seed ?? 7
      const items = (bySeed[selected] ?? []).filter((item) => item.sequence > afterSequence)
      return Promise.resolve({
        seed: selected,
        items,
        nextSequence: items.at(-1)?.sequence ?? afterSequence,
        truncated: false,
        malformedLines: 0,
        sequenceGaps: [],
        streamGeneration: '33333333-3333-4333-8333-333333333333',
        resetRequired: false,
      })
    }),
    loadCheckpointEvaluations: vi.fn().mockResolvedValue({
      available: true,
      productionStatus: 'NO-GO',
      items: [
        {
          fold: 'fold-001',
          configuration: 'residual',
          seed: 7,
          policyDigest: 'a'.repeat(64),
          evaluationDigest: 'b'.repeat(64),
          score: Math.log1p(0.05),
          totalReturn: 0.05,
          finalist: true,
          checkpointRange: [120, 140],
          source: 'research/.staging/btc-live-001/fold-001/candidates/residual/checkpoint-selection.json',
        },
        {
          fold: 'fold-000',
          configuration: 'residual',
          seed: 7,
          policyDigest: 'e'.repeat(64),
          evaluationDigest: 'f'.repeat(64),
          score: Math.log1p(0.02),
          totalReturn: 0.02,
          finalist: true,
          checkpointRange: [100, 120],
          source: 'research/.staging/btc-live-001/fold-000/candidates/residual/checkpoint-selection.json',
        },
        {
          fold: 'fold-000',
          configuration: 'residual',
          seed: 11,
          policyDigest: 'c'.repeat(64),
          evaluationDigest: 'd'.repeat(64),
          score: Math.log1p(-0.02),
          totalReturn: -0.02,
          finalist: true,
          checkpointRange: [100, 120],
          source: 'research/.staging/btc-live-001/fold-000/candidates/residual/checkpoint-selection.json',
        },
      ],
    }),
  }
}

describe('LiveTrainingPage', () => {
  it('requires explicit fold selection instead of choosing the highest checkpoint score', async () => {
    const user = userEvent.setup()
    render(<LiveTrainingPage api={api()} />)

    expect(await screen.findByRole('heading', { name: 'Live Training' })).toBeInTheDocument()
    expect(screen.getByText('NO-GO')).toBeInTheDocument()
    expect(await screen.findByRole('img', { name: /BTCUSDT 市場リプレイ/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'バッファ再生' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'ローソク足ごと' })).toHaveAttribute('aria-pressed', 'true')
    expect(await screen.findByText(/ロング 40.0%/)).toBeInTheDocument()

    const evidence = screen.getByLabelText('Checkpoint evaluation evidence')
    await waitFor(() => expect(evidence).toHaveDisplayValue('fold-000 · residual · finalist'))
    expect(screen.getAllByText(/\+2.00%/).length).toBeGreaterThan(0)
    expect(screen.queryByText(/\+5.00% finalist/)).not.toBeInTheDocument()

    await user.selectOptions(evidence, screen.getByRole('option', { name: 'fold-001 · residual · finalist' }))

    expect(screen.getAllByText(/\+5.00%/).length).toBeGreaterThan(0)
    expect(screen.getByText(/fold-001 \[120, 140\)/)).toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('Live Training seed'), '11')

    expect(await screen.findByText(/ショート 20.0%/)).toBeInTheDocument()
    await waitFor(() => expect(screen.getByLabelText('Checkpoint evaluation evidence')).toHaveDisplayValue('fold-000 · residual · finalist'))
    expect(screen.getByText(/-2.00% finalist/)).toBeInTheDocument()
    expect(screen.getByText(/Seed 11 · exploration/)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'ほぼライブ' }))
    await user.click(screen.getByRole('button', { name: 'イベント圧縮' }))

    expect(screen.getByRole('button', { name: 'ほぼライブ' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'イベント圧縮' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('keeps receiving records while replay is paused and selects event positions', async () => {
    const user = userEvent.setup()
    const runtimeApi = api()
    render(<LiveTrainingPage api={runtimeApi} />)

    await screen.findByRole('img', { name: /BTCUSDT 市場リプレイ/ })
    await user.click(screen.getByRole('button', { name: '一時停止' }))
    await user.click(screen.getByRole('button', { name: /Step 64/ }))

    expect(screen.getByText('Replay Step 64')).toBeInTheDocument()
    await waitFor(() => expect(runtimeApi.loadTelemetryEvents).toHaveBeenCalled())
  })
})
