import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { TrainingTelemetryRecord } from '../data/types'
import { ResearchChartInspector } from './ResearchChartInspector'

function record(sequence: number, close: number): TrainingTelemetryRecord {
  return {
    schemaVersion: 'training_telemetry_v1', sequence, recordedAt: '2026-07-31T08:00:00+00:00', globalStep: sequence * 32,
    environmentStep: sequence, seed: 7, environmentId: 2, episodeId: 1, eventType: 'rollout', marketIndex: sequence,
    marketTime: '2026-07-31T08:00:00.000000000', symbol: 'BTCUSDT', open: close - 1, high: close + 2,
    low: close - 2, close, action: [0.4], executedTarget: [0.35], weightsBefore: [0.2], weightsAfter: [0.4],
    portfolioValue: 1_020, baselinePortfolioValue: 1_000, reward: 0.1, drawdown: 0.01, intervalCost: 2,
    intervalReturn: 0.001, riskReasons: [], emergencyDeleverage: false, terminated: false, truncated: false,
  }
}

describe('ResearchChartInspector', () => {
  it('shows hover preview before the committed record', () => {
    render(<ResearchChartInspector committed={record(1, 100)} preview={record(2, 200)} checkpoint={null}
      checkpointIdentity={null} checkpointOptions={[]} onCheckpointChange={vi.fn()}
      identityFor={() => ''} labelFor={() => ''} />)
    expect(screen.getByText('クロスヘア')).toBeInTheDocument()
    expect(screen.getByText(/Step 64/)).toBeInTheDocument()
    expect(screen.getByText(/199.00 \/ 202.00/)).toBeInTheDocument()
    expect(screen.getByText('ロング増し')).toBeInTheDocument()
    expect(screen.getByText(/\+0.200 → \+0.400/)).toBeInTheDocument()
  })
})
