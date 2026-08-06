import { describe, expect, it } from 'vitest'

import type { TrainingTelemetryRecord } from '../data/types'
import { buildResearchChartData } from './researchChartModel'

function positionRecord(
  sequence: number,
  weightsBefore: number[],
  weightsAfter: number[],
  eventType: TrainingTelemetryRecord['eventType'] = 'position',
): TrainingTelemetryRecord {
  const marketTime = new Date(Date.parse('2026-08-01T00:00:00Z') + sequence * 15 * 60 * 1_000).toISOString()
  return {
    schemaVersion: 'training_telemetry_v1',
    sequence,
    recordedAt: marketTime,
    globalStep: sequence * 32,
    environmentStep: sequence,
    seed: 7,
    environmentId: 0,
    episodeId: 1,
    eventType,
    marketIndex: sequence,
    marketTime,
    symbol: 'BTCUSDT',
    open: 100 + sequence,
    high: 102 + sequence,
    low: 99 + sequence,
    close: 101 + sequence,
    action: weightsAfter,
    executedTarget: weightsAfter,
    weightsBefore,
    weightsAfter,
    portfolioValue: 1_000 + sequence,
    baselinePortfolioValue: 1_000,
    reward: 0,
    drawdown: 0,
    intervalCost: 0,
    intervalReturn: 0,
    riskReasons: [],
    emergencyDeleverage: false,
    terminated: false,
    truncated: false,
  }
}

describe('trade lifecycle background bands', () => {
  it('keeps ENTRY, REDUCE, and EXIT inside one closed trade band and preserves a later open trade', () => {
    const entryLong = positionRecord(1, [0], [0.6])
    const reduceLong = positionRecord(2, [0.6], [0.25])
    const exitLong = positionRecord(3, [0.25], [0])
    const entryShort = positionRecord(4, [0], [-0.4])
    const shortHold = positionRecord(5, [-0.4], [-0.4], 'rollout')

    const data = buildResearchChartData(
      [entryLong, reduceLong, exitLong, entryShort, shortHold],
      'BTCUSDT',
      '15m',
    )

    expect(data.tradeBands).toEqual([
      expect.objectContaining({
        id: 'trade-1',
        direction: 'long',
        status: 'closed',
        startSequence: 1,
        endSequence: 3,
      }),
      expect.objectContaining({
        id: 'trade-2',
        direction: 'short',
        status: 'open',
        startSequence: 4,
        endSequence: 5,
      }),
    ])

    const firstTrade = data.tradeBands[0]!
    const reduceTime = data.timeBySequence.get(2)!
    expect(reduceTime).toBeGreaterThan(firstTrade.startTime)
    expect(reduceTime).toBeLessThan(firstTrade.endTime)

    expect(data.markers.map(({ sequence, text }) => ({ sequence, text }))).toEqual([
      { sequence: 1, text: 'LONG' },
      { sequence: 2, text: 'REDUCE LONG' },
      { sequence: 3, text: 'CLOSE' },
      { sequence: 4, text: 'SHORT' },
    ])
  })
})
