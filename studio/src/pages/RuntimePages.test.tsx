import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { StudioApi } from '../api/studioApi'
import type { ConfigSummary, DatasetSummary, JobSummary } from '../data/types'
import { DataLabPage } from './DataLabPage'
import { ExperimentsPage } from './ExperimentsPage'
import { RunCenterPage } from './RunCenterPage'

const dataset: DatasetSummary = {
  id: 'dataset-1',
  name: 'btc-eth',
  relativePath: 'artifacts/datasets/btc-eth',
  market: 'continuous_24_7',
  symbols: ['BTCUSDT', 'ETHUSDT'],
  timeframes: ['15m', '1h', '4h', '1d'],
  range: '2026-01-01 → 2026-02-01',
  status: 'VALID',
  featureCount: 226,
  barCount: 744,
  symbolCount: 2,
  updated: '2026-07-19T00:00:00+00:00',
  validationError: null,
}

const config: ConfigSummary = {
  name: 'training.json',
  relativePath: 'configs/training.json',
  algorithm: 'ppo',
  status: 'VALID',
  validationError: null,
}

const job: JobSummary = {
  id: 'job-1',
  kind: 'training',
  status: 'running',
  runId: 'run-1',
  configPath: config.relativePath,
  datasetPath: dataset.relativePath,
  artifactRoot: 'artifacts/research',
  submittedAt: '2026-07-19T00:00:00+00:00',
  startedAt: '2026-07-19T00:00:01+00:00',
  completedAt: null,
  pid: 123,
  exitCode: null,
  error: null,
}

function api(overrides: Partial<StudioApi> = {}): StudioApi {
  return {
    loadDatasets: vi.fn().mockResolvedValue({ items: [dataset], total: 1, invalid: 0 }),
    loadRuns: vi.fn().mockResolvedValue({ items: [], total: 0, invalid: 0 }),
    loadConfigs: vi.fn().mockResolvedValue({ items: [config], total: 1, invalid: 0 }),
    loadJobs: vi.fn().mockResolvedValue({ items: [job], total: 1 }),
    submitTrainingJob: vi.fn().mockResolvedValue(job),
    cancelJob: vi.fn().mockResolvedValue({ ...job, status: 'cancelled' }),
    loadJobLog: vi.fn().mockResolvedValue({ jobId: job.id, lines: ['step 1', 'step 2'], truncated: false }),
    ...overrides,
  }
}

describe('DataLabPage', () => {
  it('shows validated datasets and their bounded detail view', async () => {
    render(<DataLabPage api={api()} />)

    expect(await screen.findByRole('button', { name: /btc-eth/i })).toBeInTheDocument()
    expect(screen.getByText('226')).toBeInTheDocument()
    expect(screen.getByText('BTCUSDT / ETHUSDT')).toBeInTheDocument()
    expect(screen.getAllByText('VALID')).toHaveLength(2)
  })
})

describe('ExperimentsPage', () => {
  it('submits an exploratory training job from validated config and dataset selections', async () => {
    const user = userEvent.setup()
    const runtimeApi = api()
    render(<ExperimentsPage api={runtimeApi} />)

    await screen.findByRole('option', { name: /training.json/i })
    const input = screen.getByLabelText('Run ID')
    await user.clear(input)
    await user.type(input, 'ui-training-001')
    await user.click(screen.getByRole('button', { name: '学習ジョブを開始' }))

    await waitFor(() => {
      expect(runtimeApi.submitTrainingJob).toHaveBeenCalledWith({
        configPath: config.relativePath,
        datasetPath: dataset.relativePath,
        runId: 'ui-training-001',
      })
    })
    expect(await screen.findByText(/job-1/)).toBeInTheDocument()
    expect(screen.getByText('NO-GO')).toBeInTheDocument()
  })
})

describe('RunCenterPage', () => {
  it('shows persistent logs and cancels an active job', async () => {
    const user = userEvent.setup()
    const runtimeApi = api()
    render(<RunCenterPage api={runtimeApi} />)

    await user.click(await screen.findByRole('button', { name: /job-1/i }))
    expect(await screen.findByLabelText('job log')).toHaveTextContent('step 2')
    await user.click(screen.getByRole('button', { name: '安全停止' }))

    await waitFor(() => expect(runtimeApi.cancelJob).toHaveBeenCalledWith('job-1'))
    expect(await screen.findAllByText('cancelled')).toHaveLength(3)
  })
})
