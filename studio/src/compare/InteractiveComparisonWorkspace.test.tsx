import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { RunComparison } from '../data/types'
import { buildComparisonWorkspaceModel } from './comparisonWorkspaceModel'
import { InteractiveComparisonWorkspace } from './InteractiveComparisonWorkspace'

const comparison = {
  leftResourceId: 'left-resource',
  rightResourceId: 'right-resource',
  leftRunId: 'left-run',
  rightRunId: 'right-run',
  eligibility: {
    status: 'COMPARABLE',
    reasons: [],
    datasetId: 'd'.repeat(64),
  },
  metrics: [],
  configDifferences: [],
  folds: [
    {
      label: 'Fold 1',
      leftSelectedReturn: 0.02,
      leftBaselineReturn: 0.01,
      rightSelectedReturn: 0.04,
      rightBaselineReturn: 0.01,
    },
    {
      label: 'Fold 2',
      leftSelectedReturn: 0.03,
      leftBaselineReturn: 0.01,
      rightSelectedReturn: 0.05,
      rightBaselineReturn: 0.01,
    },
  ],
  wealth: [
    {
      label: 'start',
      foldIndex: null,
      left: 1,
      right: 1,
      leftBaseline: 1,
      rightBaseline: 1,
    },
    {
      label: '10',
      foldIndex: 0,
      left: 1.01,
      right: 1.02,
      leftBaseline: 1.005,
      rightBaseline: 1.005,
    },
    {
      label: '11',
      foldIndex: 0,
      left: 1.02,
      right: 1.04,
      leftBaseline: 1.01,
      rightBaseline: 1.01,
    },
    {
      label: '20',
      foldIndex: 1,
      left: 1.03,
      right: 1.06,
      leftBaseline: 1.015,
      rightBaseline: 1.015,
    },
    {
      label: '21',
      foldIndex: 1,
      left: 1.04,
      right: 1.08,
      leftBaseline: 1.02,
      rightBaseline: 1.02,
    },
  ],
  productionStatus: 'NO-GO',
} satisfies RunComparison & {
  wealth: Array<RunComparison['wealth'][number] & { foldIndex: number | null }>
}

beforeEach(() => {
  Object.defineProperty(globalThis, 'PointerEvent', {
    configurable: true,
    value: MouseEvent,
  })
})

function renderWorkspace() {
  const onCommitPoint = vi.fn()
  const onCommitRange = vi.fn()
  const view = render(
    <InteractiveComparisonWorkspace
      model={buildComparisonWorkspaceModel(comparison)}
      committedPoint={null}
      committedRange={null}
      onCommitPoint={onCommitPoint}
      onCommitRange={onCommitRange}
    />,
  )
  const surface = screen.getByRole('application', { name: 'Run comparison chart' })
  surface.getBoundingClientRect = () => ({
    x: 0,
    y: 0,
    left: 0,
    top: 0,
    right: 1000,
    bottom: 560,
    width: 1000,
    height: 560,
    toJSON: () => ({}),
  })
  Object.defineProperty(surface, 'setPointerCapture', {
    value: vi.fn(),
    configurable: true,
  })
  Object.defineProperty(surface, 'releasePointerCapture', {
    value: vi.fn(),
    configurable: true,
  })
  return { ...view, surface, onCommitPoint, onCommitRange }
}

describe('InteractiveComparisonWorkspace', () => {
  it('renders synchronized wealth and delta panes with direct labels', () => {
    renderWorkspace()
    expect(screen.getByLabelText('Cumulative wealth pane')).toBeInTheDocument()
    expect(screen.getByLabelText('Right minus Left pane')).toBeInTheDocument()
    const key = screen.getByLabelText('series key')
    expect(within(key).getByText('Right')).toBeInTheDocument()
    expect(screen.getByLabelText('latest series values')).toHaveTextContent('Left baseline')
  })

  it('commits a point on click and moves it with ArrowRight', () => {
    const { surface, onCommitPoint } = renderWorkspace()
    fireEvent.pointerDown(surface, {
      button: 0,
      pointerId: 1,
      clientX: 500,
      clientY: 200,
    })
    fireEvent.pointerUp(surface, {
      button: 0,
      pointerId: 1,
      clientX: 500,
      clientY: 200,
    })
    expect(onCommitPoint).toHaveBeenCalled()

    surface.focus()
    fireEvent.keyDown(surface, { key: 'ArrowRight' })
    expect(onCommitPoint).toHaveBeenCalledTimes(2)
  })

  it('creates a range only while Range mode is enabled', async () => {
    const user = userEvent.setup()
    const { surface, onCommitRange } = renderWorkspace()
    await user.click(screen.getByRole('button', { name: 'Range' }))
    fireEvent.pointerDown(surface, {
      button: 0,
      pointerId: 2,
      clientX: 250,
      clientY: 200,
    })
    fireEvent.pointerMove(surface, {
      pointerId: 2,
      clientX: 750,
      clientY: 200,
    })
    fireEvent.pointerUp(surface, {
      pointerId: 2,
      clientX: 750,
      clientY: 200,
    })
    expect(onCommitRange).toHaveBeenCalledWith(
      expect.objectContaining({
        start: expect.any(Number),
        end: expect.any(Number),
      }),
    )
  })

  it('zooms with the wheel and resets the visible range', async () => {
    const user = userEvent.setup()
    const { surface } = renderWorkspace()
    const before = surface.getAttribute('data-visible-range')
    fireEvent.wheel(surface, { deltaY: -100, clientX: 500 })
    expect(surface.getAttribute('data-visible-range')).not.toBe(before)
    await user.click(screen.getByRole('button', { name: 'Reset view' }))
    expect(surface.getAttribute('data-visible-range')).toBe('0:4')
  })

  it('selects a fold using its authoritative span', async () => {
    const user = userEvent.setup()
    const { onCommitRange } = renderWorkspace()
    await user.click(screen.getByRole('button', { name: /Fold 2/ }))
    expect(onCommitRange).toHaveBeenCalledWith({ start: 3, end: 4 })
  })
})
