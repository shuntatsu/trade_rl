import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { TrainingTelemetryRecord } from '../data/types'
import { MarketReplayChart } from './MarketReplayChart'

function telemetry(
  sequence: number,
  environmentId: number,
  episodeId: number,
  symbol: string,
  close: number,
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
    eventType: 'position',
    marketIndex: 100 + sequence,
    marketTime: '2026-07-22T12:55:00.000000000',
    symbol,
    open: close - 1,
    high: close + 2,
    low: close - 2,
    close,
    action: [0.4],
    executedTarget: [0.4],
    weightsBefore: [0.2],
    weightsAfter: [0.4],
    portfolioValue: 1_000 + sequence,
    baselinePortfolioValue: 1_000,
    reward: 0.1,
    drawdown: 0,
    intervalCost: 0,
    intervalReturn: 0.001,
    riskReasons: [],
    emergencyDeleverage: false,
    terminated: false,
    truncated: false,
  }
}

describe('MarketReplayChart track isolation', () => {
  it('renders only the track containing the replay cursor', () => {
    render(
      <MarketReplayChart
        records={[
          telemetry(1, 0, 10, 'BTCUSDT', 100),
          telemetry(2, 1, 20, 'ETHUSDT', 2_000),
          telemetry(3, 0, 10, 'BTCUSDT', 110),
        ]}
        cursorSequence={3}
        compressed={false}
      />,
    )

    const chart = screen.getByRole('img', { name: 'BTCUSDT 市場リプレイ' })
    const candles = [...chart.querySelectorAll('[data-sequence]')]

    expect(candles.map((node) => node.getAttribute('data-sequence'))).toEqual(['1', '3'])
    expect(candles.every((node) => node.getAttribute('data-track-key') === 'explicit:0:10')).toBe(true)
  })
})
