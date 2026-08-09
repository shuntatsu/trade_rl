import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { TrainingMetricChart } from './TrainingMetricChart'

const series = {
  tag: 'train/learning_rate', displayName: 'Learning rate', group: 'optimization' as const, unit: 'rate' as const,
  points: [{ step: 10, wallTime: 1, value: 0.00012 }, { step: 20, wallTime: 2, value: 0.0001 }],
}

describe('TrainingMetricChart', () => {
  it('renders an accessible global-step SVG and supports keyboard selection', () => {
    const change = vi.fn()
    render(<TrainingMetricChart series={series} selectedStep={10} onSelectedStepChange={change} windowSize="all" />)
    const chart = screen.getByRole('img', { name: /Learning rate 学習ステップ推移/ })
    expect(screen.getAllByText(/Step 10/)).toHaveLength(2)
    fireEvent.keyDown(chart, { key: 'ArrowRight' })
    expect(change).toHaveBeenCalledWith(20)
  })

  it('renders empty and equal-valued series safely', () => {
    const { rerender } = render(<TrainingMetricChart series={null} selectedStep={null} onSelectedStepChange={() => undefined} windowSize="all" />)
    expect(screen.getByText('未出力')).toBeInTheDocument()
    rerender(<TrainingMetricChart series={{ ...series, points: [{ step: 1, wallTime: 1, value: 2 }] }} selectedStep={null} onSelectedStepChange={() => undefined} windowSize="all" />)
    expect(screen.getByRole('img')).toBeInTheDocument()
  })
})
