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

function telemetry(sequence: number, close: number, weight: number): TrainingTelemetryRecord {
  return {
    schemaVersion: 'training_telemetry_v1',
    sequence,
    recordedAt: `2026-07-21T08:0${sequence}:00+00:00`,
    globalStep: sequence * 32,
    environmentStep: sequence,
    seed: 7,
    environmentId: 0,
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
  const items = [
    telemetry(1, 67_500, 0.1),
    telemetry(2, 67_842.3, 0.4),
    telemetry(3, 67_780, 0.4),
  ]
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
    loadTelemetryStatus: vi.fn().mockResolvedValue({
      available: true,
      recordCount: items.length,
      lastSequence: items.at(-1)!.sequence,
      malformedLines: 0,
      sizeBytes: 2048,
      source: 'research/.staging/btc-live-001/seed-7/telemetry/training-telemetry.jsonl',
    }),
    loadTelemetryEvents: vi.fn().mockResolvedValue({
      items,
      nextSequence: items.at(-1)!.sequence,
      truncated: false,
      malformedLines: 0,
      sequenceGaps: [],
    }),
  }
}

describe('LiveTrainingPage', () => {
  it('renders the approved buffered chart-first replay and switches display modes', async () => {
    const user = userEvent.setup()
    render(<LiveTrainingPage api={api()} />)

    expect(await screen.findByRole('heading', { name: 'Live Training' })).toBeInTheDocument()
    expect(screen.getByText('NO-GO')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /BTCUSDT 市場リプレイ/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'バッファ再生' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'ローソク足ごと' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByText(/ロング 0.400/)).toBeInTheDocument()
    expect(screen.getByText(/\+1,500\.00/)).toBeInTheDocument()

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
