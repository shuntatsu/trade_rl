import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { StudioApi } from '../api/studioApi'
import type { RunComparison, RunSummary } from '../data/types'
import { ComparePage } from './ComparePage'

const runs: RunSummary[] = [
  {
    id: 'run-111111111111111111111111', runId: 'run-001', manifestDigest: '1'.repeat(64), relativePath: 'research/runs/run-001', runKind: 'research_exploratory', algorithm: 'ppo',
    datasetId: 'dataset-1', period: '2026-01-01 — 2026-01-02', createdAt: '2026-01-01', completedAt: '2026-01-02',
    fileCount: 8, sharpe: 0.8, maxDrawdown: 0.1, totalReturn: 0.12, productionStatus: 'NO-GO', status: 'VALID', validationError: null,
  },
  {
    id: 'run-222222222222222222222222', runId: 'run-002', manifestDigest: '2'.repeat(64), relativePath: 'research/runs/run-002', runKind: 'research_exploratory', algorithm: 'sac',
    datasetId: 'dataset-1', period: '2026-02-01 — 2026-02-02', createdAt: '2026-02-01', completedAt: '2026-02-02',
    fileCount: 8, sharpe: 1.1, maxDrawdown: 0.08, totalReturn: 0.18, productionStatus: 'NO-GO', status: 'VALID', validationError: null,
  },
]

const comparison = {
  leftResourceId: runs[0].id, rightResourceId: runs[1].id, leftRunId: 'run-001', rightRunId: 'run-002', eligibility: { status: 'COMPARABLE', reasons: [], datasetId: 'dataset-1' }, productionStatus: 'NO-GO',
  metrics: [
    { key: 'total_return', label: 'Total return', leftValue: 0.12, rightValue: 0.18, delta: 0.06, preference: 'higher' },
    { key: 'total_cost', label: 'Total cost', leftValue: 0.006, rightValue: 0.009, delta: 0.003, preference: 'lower' },
  ],
  configDifferences: [{ path: 'training.algorithm', left: 'ppo', right: 'sac' }],
  folds: [{ label: 'Fold 1', leftSelectedReturn: 0.05, leftBaselineReturn: 0.03, rightSelectedReturn: 0.08, rightBaselineReturn: 0.03 }],
  wealth: [
    { label: 'start', foldIndex: null, left: 1, right: 1, leftBaseline: 1, rightBaseline: 1 },
    { label: '1', foldIndex: 0, left: 1.05, right: 1.08, leftBaseline: 1.03, rightBaseline: 1.03 },
  ],
} satisfies RunComparison & { wealth: Array<RunComparison['wealth'][number] & { foldIndex: number | null }> }

function api(): Pick<StudioApi, 'loadRuns' | 'loadRunComparison'> {
  return {
    loadRuns: vi.fn().mockResolvedValue({ items: runs, total: 2, invalid: 0 }),
    loadRunComparison: vi.fn().mockResolvedValue(comparison),
  }
}

beforeEach(() => {
  window.history.replaceState(null, '', '/?workspace=compare')
})

describe('ComparePage', () => {
  it('makes the synchronized interactive chart the primary comparison surface', async () => {
    render(<ComparePage api={api()} />)

    expect(await screen.findByRole('application', { name: 'Run comparison chart' })).toBeInTheDocument()
    expect(screen.getByLabelText('Cumulative wealth pane')).toBeInTheDocument()
    expect(screen.getByLabelText('Right minus Left pane')).toBeInTheDocument()
    expect(screen.getByText('NO-GO')).toBeInTheDocument()
    expect(screen.getByText(/no automatic winner/i)).toBeInTheDocument()
    expect(screen.queryByText('Decision metrics')).not.toBeInTheDocument()
    expect(screen.queryByText('Configuration diff')).not.toBeInTheDocument()
    expect(screen.queryByText('Fold returns')).not.toBeInTheDocument()
    expect(screen.queryByText(/注文|発注/)).not.toBeInTheDocument()
  })

  it('reloads the comparison when the selected right run changes', async () => {
    const user = userEvent.setup()
    const runtimeApi = api()
    render(<ComparePage api={runtimeApi} />)

    await screen.findByRole('application', { name: 'Run comparison chart' })
    await user.selectOptions(screen.getByLabelText('Right run'), runs[0].id)

    await waitFor(() => expect(runtimeApi.loadRunComparison).toHaveBeenLastCalledWith(runs[0].id, runs[0].id))
  })

  it('clears the old pair while a genuinely different pair is loading', async () => {
    let resolveSecond: ((value: RunComparison) => void) | null = null
    const second = new Promise<RunComparison>((resolve) => { resolveSecond = resolve })
    const runtimeApi: Pick<StudioApi, 'loadRuns' | 'loadRunComparison'> = {
      loadRuns: vi.fn().mockResolvedValue({ items: runs, total: 2, invalid: 0 }),
      loadRunComparison: vi.fn()
        .mockResolvedValueOnce(comparison)
        .mockReturnValueOnce(second),
    }
    const user = userEvent.setup()
    render(<ComparePage api={runtimeApi} />)
    await screen.findByText('run-001 ↔ run-002')

    await user.selectOptions(screen.getByLabelText('Right run'), runs[0].id)

    expect(screen.getByRole('status')).toHaveTextContent('比較を読み込み中です')
    expect(screen.queryByText('run-001 ↔ run-002')).not.toBeInTheDocument()
    expect(screen.queryByRole('application', { name: 'Run comparison chart' })).not.toBeInTheDocument()

    act(() => resolveSecond?.({ ...comparison, rightRunId: 'run-001', rightResourceId: runs[0].id }))
    await waitFor(() => expect(screen.getByText('run-001 ↔ run-001')).toBeInTheDocument())
  })

  it('opens metrics and configuration in an overlay inspector', async () => {
    const user = userEvent.setup()
    render(<ComparePage api={api()} />)
    await screen.findByRole('application', { name: 'Run comparison chart' })

    await user.click(screen.getByRole('button', { name: 'Details' }))
    expect(screen.getByRole('dialog', { name: 'Comparison inspector' })).toBeInTheDocument()
    expect(screen.getByRole('dialog', { name: 'Comparison inspector' })).not.toHaveAttribute('aria-modal')
    await user.click(screen.getByRole('tab', { name: 'metrics' }))
    expect(screen.getByText('Total return')).toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: 'config' }))
    expect(screen.getByText('training.algorithm')).toBeInTheDocument()
  })

  it('restores the selected point from browser history', async () => {
    render(<ComparePage api={api()} />)
    await screen.findByRole('application', { name: 'Run comparison chart' })

    act(() => {
      window.history.pushState(null, '', '/?workspace=compare&left=run-111111111111111111111111&right=run-222222222222222222222222&comparePoint=0')
      window.dispatchEvent(new PopStateEvent('popstate'))
    })

    await waitFor(() => expect(screen.getByText('Index start')).toBeInTheDocument())
  })

  it('ignores a stale comparison response after the pair changes', async () => {
    let resolveFirst: ((value: RunComparison) => void) | null = null
    const first = new Promise<RunComparison>((resolve) => { resolveFirst = resolve })
    const runtimeApi: Pick<StudioApi, 'loadRuns' | 'loadRunComparison'> = {
      loadRuns: vi.fn().mockResolvedValue({ items: runs, total: 2, invalid: 0 }),
      loadRunComparison: vi.fn()
        .mockReturnValueOnce(first)
        .mockResolvedValueOnce({ ...comparison, rightRunId: 'run-001', rightResourceId: runs[0].id }),
    }
    const user = userEvent.setup()
    render(<ComparePage api={runtimeApi} />)
    await waitFor(() => expect(runtimeApi.loadRunComparison).toHaveBeenCalledTimes(1))
    await user.selectOptions(screen.getByLabelText('Right run'), runs[0].id)
    await waitFor(() => expect(screen.getByText('run-001 ↔ run-001')).toBeInTheDocument())

    act(() => resolveFirst?.(comparison))
    expect(screen.getByText('run-001 ↔ run-001')).toBeInTheDocument()
  })
})
