import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { BehaviorCloningProgressResponse } from '../data/types'
import { BehaviorCloningProgressPanel } from './BehaviorCloningProgressPanel'

const progress: BehaviorCloningProgressResponse = {
  schemaVersion: 'behavior_cloning_progress_v1', available: true, phase: 'training',
  epoch: 9, totalEpochs: 45, bestEpoch: 8, percent: 20, seed: 7,
  fold: 'fold-000', configuration: 'LAGRANGIAN_PPO', elapsedSeconds: 120,
  estimatedRemainingSeconds: 480, validationLoss: 0.125, gateLoss: 0.1,
  targetLoss: 0.02, composedLoss: 0.005, gatePrecision: 0.87,
  gateRecall: 0.61, activityRatio: 0.93, allHoldCollapse: false,
  allTradeCollapse: false, earlyStopping: false,
  updatedAt: '2026-08-02T14:00:00+00:00', source: 'progress.json',
}

describe('BehaviorCloningProgressPanel', () => {
  it('shows epoch, gate metrics, and identity', () => {
    render(<BehaviorCloningProgressPanel progress={progress} />)
    expect(screen.getByText('BC学習中')).toBeInTheDocument()
    expect(screen.getByText('9 / 45')).toBeInTheDocument()
    expect(screen.getByText('87.0%')).toBeInTheDocument()
    expect(screen.getByText('61.0%')).toBeInTheDocument()
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '20')
    expect(screen.getByText('fold-000 · LAGRANGIAN_PPO · seed-7')).toBeInTheDocument()
  })
})
