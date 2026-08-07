import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { JobSummary } from '../data/types'
import { DEFAULT_RESEARCH_CHART_LAYERS } from './researchChartModel'
import { ReplayToolbar, type ReplayToolbarProps } from './ReplayToolbar'

const job: JobSummary = {
  id: 'job-live',
  schemaVersion: 'studio_job_v2',
  kind: 'training',
  status: 'running',
  runId: 'btc-training-001',
  configResourceId: 'config-1',
  datasetResourceId: 'dataset-1',
  configDigest: 'c'.repeat(64),
  datasetId: 'd'.repeat(64),
  configPath: 'configs/training.json',
  datasetPath: 'datasets/btc',
  artifactRoot: 'research',
  submittedAt: '2026-07-31T08:00:00+00:00',
  ownerInstanceId: 'studio-1',
  startedAt: '2026-07-31T08:00:01+00:00',
  completedAt: null,
  pid: 42,
  pidStartToken: '1',
  exitCode: null,
  cancellable: true,
  error: null,
}

function props(overrides: Partial<ReplayToolbarProps> = {}): ReplayToolbarProps {
  return {
    jobs: [job],
    jobId: job.id,
    seeds: [7, 11],
    seed: 7,
    environments: [0, 1, 2],
    environmentId: 2,
    playing: true,
    speed: 4,
    followLatest: true,
    layers: DEFAULT_RESEARCH_CHART_LAYERS,
    hasRecords: true,
    onJobChange: vi.fn(),
    onSourceChange: vi.fn(),
    onTogglePlaying: vi.fn(),
    onFirst: vi.fn(),
    onPreviousEvent: vi.fn(),
    onNextEvent: vi.fn(),
    onLast: vi.fn(),
    onSpeedChange: vi.fn(),
    onFollowLatestChange: vi.fn(),
    onLayersChange: vi.fn(),
    onResetView: vi.fn(),
    ...overrides,
  }
}

describe('ReplayToolbar', () => {
  it('keeps Seed and Environment inside the source popover', async () => {
    const user = userEvent.setup()
    render(<ReplayToolbar {...props()} />)

    expect(screen.getByLabelText('Live Training Run')).toBeInTheDocument()
    expect(screen.queryByLabelText('Live Training Seed')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Live Training Environment')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /対象を変更/ }))

    expect(screen.getByLabelText('Live Training Seed')).toHaveDisplayValue('Seed 7')
    expect(screen.getByLabelText('Live Training Environment')).toHaveDisplayValue('Env 2')
    expect(screen.queryByText('LIVE')).not.toBeInTheDocument()
  })

  it('applies Seed and Environment atomically', async () => {
    const user = userEvent.setup()
    const onSourceChange = vi.fn()
    render(<ReplayToolbar {...props({ onSourceChange })} />)

    await user.click(screen.getByRole('button', { name: /対象を変更/ }))
    await user.selectOptions(screen.getByLabelText('Live Training Seed'), '11')
    await user.selectOptions(screen.getByLabelText('Live Training Environment'), '1')

    expect(onSourceChange).not.toHaveBeenCalled()
    await user.click(screen.getByRole('button', { name: '対象を適用' }))

    expect(onSourceChange).toHaveBeenCalledTimes(1)
    expect(onSourceChange).toHaveBeenCalledWith({ seed: 11, environmentId: 1 })
  })

  it('emits transport and speed callbacks', async () => {
    const user = userEvent.setup()
    const onFirst = vi.fn()
    const onPreviousEvent = vi.fn()
    const onTogglePlaying = vi.fn()
    const onNextEvent = vi.fn()
    const onLast = vi.fn()
    const onSpeedChange = vi.fn()

    render(<ReplayToolbar {...props({
      onFirst,
      onPreviousEvent,
      onTogglePlaying,
      onNextEvent,
      onLast,
      onSpeedChange,
    })} />)

    await user.click(screen.getByRole('button', { name: '先頭へ' }))
    await user.click(screen.getByRole('button', { name: '前のイベント' }))
    await user.click(screen.getByRole('button', { name: '一時停止' }))
    await user.click(screen.getByRole('button', { name: '次のイベント' }))
    await user.click(screen.getByRole('button', { name: '最新へ' }))
    await user.selectOptions(screen.getByLabelText('再生速度'), '8')

    expect(onFirst).toHaveBeenCalledTimes(1)
    expect(onPreviousEvent).toHaveBeenCalledTimes(1)
    expect(onTogglePlaying).toHaveBeenCalledTimes(1)
    expect(onNextEvent).toHaveBeenCalledTimes(1)
    expect(onLast).toHaveBeenCalledTimes(1)
    expect(onSpeedChange).toHaveBeenCalledWith(8)
  })

  it('toggles latest-follow and chart layers', async () => {
    const user = userEvent.setup()
    const onFollowLatestChange = vi.fn()
    const onLayersChange = vi.fn()
    render(<ReplayToolbar {...props({ onFollowLatestChange, onLayersChange })} />)

    await user.click(screen.getByRole('checkbox', { name: '最新へ追従' }))
    expect(onFollowLatestChange).toHaveBeenCalledWith(false)

    await user.click(screen.getByRole('button', { name: '表示項目' }))
    await user.click(screen.getByRole('checkbox', { name: 'Baseline' }))

    expect(onLayersChange).toHaveBeenCalledWith({
      ...DEFAULT_RESEARCH_CHART_LAYERS,
      baseline: false,
    })
  })

  it('disables transport when no replay records are available', () => {
    render(<ReplayToolbar {...props({ hasRecords: false })} />)

    expect(screen.getByRole('button', { name: '先頭へ' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '前のイベント' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '一時停止' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '次のイベント' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '最新へ' })).toBeDisabled()
  })
})
