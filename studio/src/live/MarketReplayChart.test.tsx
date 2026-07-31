import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { TrainingTelemetryRecord } from '../data/types'
import { MarketReplayChart } from './MarketReplayChart'

function positionRecord(): TrainingTelemetryRecord {
  return {
    schemaVersion: 'training_telemetry_v1',
    sequence: 1,
    recordedAt: '2026-07-31T08:00:00+00:00',
    globalStep: 32,
    environmentStep: 1,
    seed: 7,
    environmentId: 0,
    episodeId: 1,
    eventType: 'position',
    marketIndex: 100,
    marketTime: '2026-07-31T07:55:00.000000000',
    symbol: 'BTCUSDT',
    open: 67_500,
    high: 67_900,
    low: 67_300,
    close: 67_700,
    action: [0.1, 0.2],
    executedTarget: [0.1, 0.2],
    weightsBefore: [0.2, 0.1],
    weightsAfter: [0.1, 0.2],
    portfolioValue: 100_100,
    baselinePortfolioValue: 100_050,
    reward: 0.01,
    drawdown: 0.002,
    intervalCost: 1.5,
    intervalReturn: 0.001,
    riskReasons: [],
    emergencyDeleverage: false,
    terminated: false,
    truncated: false,
  }
}

describe('MarketReplayChart', () => {
  it('uses the displayed primary asset weight delta for marker direction', () => {
    const { container } = render(
      <MarketReplayChart records={[positionRecord()]} cursorSequence={1} compressed={false} />,
    )

    expect(container.querySelector('.live-marker--sell')).not.toBeNull()
    expect(container.querySelector('.live-marker--buy')).toBeNull()
  })
})
