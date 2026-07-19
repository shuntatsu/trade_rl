import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { StudioApi } from '../api/studioApi'
import type { RunComparison, RunSummary } from '../data/types'
import { ComparePage } from './ComparePage'

const runs: RunSummary[] = [
  {
    id: 'run-001', relativePath: 'research/runs/run-001', runKind: 'research_exploratory', algorithm: 'ppo',
    datasetId: 'dataset-1', period: '2026-01-01 — 2026-01-02', createdAt: '2026-01-01', completedAt: '2026-01-02',
    fileCount: 8, sharpe: 0.8, maxDrawdown: 0.1, totalReturn: 0.12, productionStatus: 'NO-GO', status: 'VALID', validationError: null,
  },
  {
    id: 'run-002', relativePath: 'research/runs/run-002', runKind: 'research_exploratory', algorithm: 'sac',
    datasetId: 'dataset-1', period: '2026-02-01 — 2026-02-02', createdAt: '2026-02-01', completedAt: '2026-02-02',
    fileCount: 8, sharpe: 1.1, maxDrawdown: 0.08, totalReturn: 0.18, productionStatus: 'NO-GO', status: 'VALID', validationError: null,
  },
]

const comparison: RunComparison = {
  leftRunId: 'run-001', rightRunId: 'run-002', productionStatus: 'NO-GO',
  metrics: [{ key: 'total_return', label: 'Total return', leftValue: 0.12, rightValue: 0.18, delta: 0.06, preference: 'higher' }],
  configDifferences: [{ path: 'training.algorithm', left: 'ppo', right: 'sac' }],
  folds: [{ label: 'Fold 1', leftSelectedReturn: 0.05, leftBaselineReturn: 0.03, rightSelectedReturn: 0.08, rightBaselineReturn: 0.03 }],
  wealth: [{ label: '0', left: 1, right: 1, leftBaseline: 1, rightBaseline: 1 }, { label: '1', left: 1.05, right: 1.08, leftBaseline: 1.03, rightBaseline: 1.03 }],
}

function api(): Pick<StudioApi, 'loadRuns' | 'loadRunComparison'> {
  return {
    loadRuns: vi.fn().mockResolvedValue({ items: runs, total: 2, invalid: 0 }),
    loadRunComparison: vi.fn().mockResolvedValue(comparison),
  }
}

describe('ComparePage', () => {
  it('compares two validated runs without exposing execution controls', async () => {
    const runtimeApi = api()
    render(<ComparePage api={runtimeApi} />)

    expect(await screen.findByText('Total return')).toBeInTheDocument()
    expect(screen.getByText('training.algorithm')).toBeInTheDocument()
    expect(screen.getByText('Fold 1')).toBeInTheDocument()
    expect(screen.getByText('NO-GO')).toBeInTheDocument()
    expect(screen.getByLabelText('comparison wealth chart')).toBeInTheDocument()
    expect(screen.queryByText(/注文|発注/)).not.toBeInTheDocument()
  })

  it('reloads the comparison when the selected right run changes', async () => {
    const user = userEvent.setup()
    const runtimeApi = api()
    render(<ComparePage api={runtimeApi} />)

    await screen.findByText('Total return')
    await user.selectOptions(screen.getByLabelText('Right run'), 'run-001')

    await waitFor(() => expect(runtimeApi.loadRunComparison).toHaveBeenLastCalledWith('run-001', 'run-001'))
  })
})
