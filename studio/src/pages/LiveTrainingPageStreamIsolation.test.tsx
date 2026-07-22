import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { StudioApi } from '../api/studioApi'
import type { JobSummary, TrainingTelemetryRecord } from '../data/types'
import { LiveTrainingPage } from './LiveTrainingPage'

const mockedTelemetry = vi.hoisted(() => ({
  records: [] as TrainingTelemetryRecord[],
}))

vi.mock('../live/useTrainingTelemetry', () => ({
  useTrainingTelemetry: () => ({
    records: mockedTelemetry.records,
    status: {
      available: true,
      selectedSeed: 7,
      availableSeeds: [7],
      recordCount: mockedTelemetry.records.length,
      lastSequence: mockedTelemetry.records.at(-1)?.sequence ?? 0,
      malformedLines: 0,
      sizeBytes: 4_096,
      source: 'research/.staging/live/seed-7/telemetry/training-telemetry.jsonl',
      streamGeneration: '44444444-4444-4444-8444-444444444444',
    },
    connection: 'live',
    loading: false,
    error: null,
    refresh: vi.fn(),
  }),
}))

const job: JobSummary = {
  id: 'job-live',
  schemaVersion: 'studio_job_v2',
  kind: 'training',
  status: 'running',
  runId: 'stream-isolation',
  configResourceId: 'config-1',
  datasetResourceId: 'dataset-1',
  configDigest: 'c'.repeat(64),
  datasetId: 'd'.repeat(64),
  configPath: 'configs/training.json',
  datasetPath: 'datasets/btc',
  artifactRoot: 'research',
  submittedAt: '2026-07-22T12:00:00+00:00',
  ownerInstanceId: 'studio-1',
  startedAt: '2026-07-22T12:00:01+00:00',
  completedAt: null,
  pid: 42,
  pidStartToken: '1',
  exitCode: null,
  cancellable: true,
  error: null,
}

function telemetry(
  sequence: number,
  environmentId: number,
  episodeId: number,
  close: number,
  portfolioValue: number,
  riskReason: string,
): TrainingTelemetryRecord {
  return {
    schemaVersion: 'training_telemetry_v1',
    sequence,
    recordedAt: '2026-07-22T13:00:00+00:00',
    globalStep: sequence * 32,
    environmentStep: sequence,
    seed: 7,
    environmentId,
    episodeId,
    eventType: riskReason ? 'risk' : 'rollout',
    marketIndex: 100 + sequence,
    marketTime: '2026-07-22T12:55:00.000000000',
    symbol: environmentId === 0 ? 'BTCUSDT' : 'ETHUSDT',
    open: close - 1,
    high: close + 2,
    low: close - 2,
    close,
    action: [0.2],
    executedTarget: [0.2],
    weightsBefore: [0.1],
    weightsAfter: [0.2],
    portfolioValue,
    baselinePortfolioValue: portfolioValue - 50,
    reward: 0.1,
    drawdown: 0.01,
    intervalCost: 1,
    intervalReturn: 0.001,
    riskReasons: riskReason ? [riskReason] : [],
    emergencyDeleverage: false,
    terminated: false,
    truncated: false,
  }
}

function api(): StudioApi {
  const unused = vi.fn().mockRejectedValue(new Error('not used'))
  return {
    loadDatasets: unused,
    loadRuns: unused,
    loadConfigs: unused,
    loadJobs: vi.fn().mockResolvedValue({ items: [job], total: 1 }),
    submitTrainingJob: unused,
    cancelJob: unused,
    loadJobLog: unused,
    loadTelemetryStatus: unused,
    loadTelemetryEvents: unused,
    loadCheckpointEvaluations: vi.fn().mockResolvedValue({
      available: false,
      productionStatus: 'NO-GO',
      items: [],
    }),
    loadRunComparison: unused,
    loadEvidenceReport: unused,
    loadServingMonitor: unused,
  } as unknown as StudioApi
}

beforeEach(() => {
  mockedTelemetry.records = [
    telemetry(1, 0, 10, 100, 1_000, ''),
    telemetry(2, 1, 20, 500, 5_000, ''),
    telemetry(3, 0, 10, 110, 1_100, 'env-zero-old'),
    telemetry(4, 0, 11, 200, 2_000, ''),
    telemetry(5, 1, 20, 530, 5_300, 'env-one'),
    telemetry(6, 0, 11, 230, 2_300, 'env-zero-new'),
    telemetry(7, 1, 20, 560, 5_600, ''),
  ]
})

describe('LiveTrainingPage stream isolation', () => {
  it('binds replay metrics and events to one selected environment and episode', async () => {
    const user = userEvent.setup()
    render(<LiveTrainingPage api={api()} />)

    expect(await screen.findByRole('heading', { name: 'Live Training' })).toBeInTheDocument()
    const environment = screen.getByLabelText('Live Training environment')
    const episode = screen.getByLabelText('Live Training episode')

    await waitFor(() => {
      expect(environment).toHaveValue('1')
      expect(episode).toHaveValue('explicit:1:20')
      expect(screen.getByText('560 USDT')).toBeInTheDocument()
      expect(screen.getAllByText('+600.00 USDT').length).toBeGreaterThan(0)
    })
    expect(screen.getByText('3 / 7 records')).toBeInTheDocument()

    await user.selectOptions(environment, '0')
    await waitFor(() => {
      expect(episode).toHaveValue('explicit:0:11')
      expect(screen.getByText('230 USDT')).toBeInTheDocument()
      expect(screen.getAllByText('+300.00 USDT').length).toBeGreaterThan(0)
    })
    expect(screen.queryByText('env-one')).not.toBeInTheDocument()
    expect(screen.getByText('env-zero-new')).toBeInTheDocument()

    await user.selectOptions(episode, 'explicit:0:10')
    await waitFor(() => {
      expect(screen.getByText('110 USDT')).toBeInTheDocument()
      expect(screen.getAllByText('+100.00 USDT').length).toBeGreaterThan(0)
    })
    expect(screen.getByText('env-zero-old')).toBeInTheDocument()
    expect(screen.queryByText('env-zero-new')).not.toBeInTheDocument()
    expect(screen.getByText('2 / 7 records')).toBeInTheDocument()
  })
})
