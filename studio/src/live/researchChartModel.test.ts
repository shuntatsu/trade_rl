import { describe, expect, it } from 'vitest'

import type { TrainingTelemetryRecord } from '../data/types'
import {
  buildResearchChartData,
  nextEventIndex,
  positionTransition,
  previousEventIndex,
} from './researchChartModel'

function telemetry(overrides: Partial<TrainingTelemetryRecord> = {}): TrainingTelemetryRecord {
  return {
    schemaVersion: 'training_telemetry_v1',
    sequence: 1,
    recordedAt: '2026-07-31T08:05:00+00:00',
    globalStep: 32,
    environmentStep: 1,
    seed: 7,
    environmentId: 0,
    episodeId: 1,
    eventType: 'rollout',
    marketIndex: 100,
    marketTime: '2026-07-31T08:05:00.123456789',
    symbol: 'BTCUSDT',
    open: 100,
    high: 108,
    low: 96,
    close: 104,
    action: [0.2, 0.1],
    executedTarget: [0.18, 0.1],
    weightsBefore: [0.1, 0.1],
    weightsAfter: [0.2, 0.1],
    portfolioValue: 1_000,
    baselinePortfolioValue: 995,
    reward: 0.1,
    drawdown: 0.01,
    intervalCost: 1.5,
    intervalReturn: 0.001,
    riskReasons: [],
    emergencyDeleverage: false,
    terminated: false,
    truncated: false,
    ...overrides,
  }
}

function epochSeconds(value: string): number {
  return Math.floor(Date.parse(value) / 1_000)
}

describe('buildResearchChartData', () => {
  it('aggregates telemetry into hourly candles and keeps the latest metrics', () => {
    const first = telemetry()
    const second = telemetry({
      sequence: 2,
      globalStep: 64,
      recordedAt: '2026-07-31T08:50:00+00:00',
      marketTime: '2026-07-31T08:50:00.987654321',
      open: 104,
      high: 115,
      low: 90,
      close: 110,
      action: [0.45, 0.1],
      executedTarget: [0.35, 0.1],
      weightsBefore: [0.2, 0.1],
      weightsAfter: [0.4, 0.1],
      portfolioValue: 1_010,
      baselinePortfolioValue: 1_005,
      reward: 0.2,
      drawdown: 0.022,
      intervalCost: 2,
    })

    const result = buildResearchChartData([first, second], 'BTCUSDT', '1h')
    const bucket = Math.floor(epochSeconds('2026-07-31T08:00:00Z') / 3_600) * 3_600

    expect(result.candles).toEqual([
      { time: bucket, open: 100, high: 115, low: 90, close: 110 },
    ])
    expect(result.targetWeight).toEqual([{ time: bucket, value: 0.4 }])
    expect(result.executedWeight).toEqual([{ time: bucket, value: 0.35 }])
    expect(result.reward).toEqual([{ time: bucket, value: 0.2 }])
    expect(result.cost).toEqual([{ time: bucket, value: 2 }])
    expect(result.equity).toEqual([{ time: bucket, value: 1_010 }])
    expect(result.baseline).toEqual([{ time: bucket, value: 1_005 }])
    expect(result.drawdown).toEqual([{ time: bucket, value: -2.2 }])
    expect(result.recordByTime.get(bucket)?.sequence).toBe(2)
    expect(result.timeBySequence.get(1)).toBe(bucket)
    expect(result.timeBySequence.get(2)).toBe(bucket)
  })

  it('drops records with invalid timestamps instead of inventing chart times', () => {
    const invalid = telemetry({ marketTime: 'not-a-time', recordedAt: 'also-invalid' })

    const result = buildResearchChartData([invalid], 'BTCUSDT', '15m')

    expect(result.candles).toEqual([])
    expect(result.recordByTime.size).toBe(0)
    expect(result.timeBySequence.size).toBe(0)
  })

  it('does not create a directional marker when the primary asset delta is zero', () => {
    const secondaryAssetPosition = telemetry({
      eventType: 'position',
      weightsBefore: [0.2, 0.1],
      weightsAfter: [0.2, 0.35],
    })

    const result = buildResearchChartData([secondaryAssetPosition], 'BTCUSDT', '15m')

    expect(result.markers).toEqual([])
  })

  it('creates truthful LONG SHORT CLOSE reduction RISK and END markers', () => {
    const long = telemetry({
      eventType: 'position',
      sequence: 1,
      weightsBefore: [0, 0.1],
      weightsAfter: [0.2, 0.1],
    })
    const close = telemetry({
      eventType: 'position',
      sequence: 2,
      marketTime: '2026-07-31T08:20:00.000000000',
      weightsBefore: [0.4, 0.1],
      weightsAfter: [0, 0.1],
    })
    const short = telemetry({
      eventType: 'position',
      sequence: 3,
      marketTime: '2026-07-31T08:35:00.000000000',
      weightsBefore: [0, 0.1],
      weightsAfter: [-0.3, 0.1],
    })
    const cover = telemetry({
      eventType: 'position',
      sequence: 4,
      marketTime: '2026-07-31T08:50:00.000000000',
      weightsBefore: [-0.4, 0.1],
      weightsAfter: [-0.1, 0.1],
    })
    const risk = telemetry({
      eventType: 'risk',
      sequence: 5,
      marketTime: '2026-07-31T09:05:00.000000000',
      riskReasons: ['drawdown'],
    })
    const end = telemetry({
      eventType: 'episode_end',
      sequence: 6,
      marketTime: '2026-07-31T09:20:00.000000000',
      terminated: true,
    })

    const result = buildResearchChartData([long, close, short, cover, risk, end], 'BTCUSDT', '15m')

    expect(result.markers.map((marker) => ({ text: marker.text, sequence: marker.sequence }))).toEqual([
      { text: 'LONG', sequence: 1 },
      { text: 'CLOSE', sequence: 2 },
      { text: 'SHORT', sequence: 3 },
      { text: 'COVER SHORT', sequence: 4 },
      { text: 'RISK', sequence: 5 },
      { text: 'END', sequence: 6 },
    ])
    expect(result.recordBySequence.get(3)).toBe(short)
  })

  it('filters all chart series and markers to the selected symbol', () => {
    const btc = telemetry({ symbol: 'BTCUSDT', close: 100 })
    const eth = telemetry({
      symbol: 'ETHUSDT',
      sequence: 2,
      close: 2_000,
      eventType: 'position',
      marketTime: '2026-07-31T08:20:00.000000000',
    })

    const result = buildResearchChartData([btc, eth], 'BTCUSDT', '15m')

    expect(result.candles).toHaveLength(1)
    expect(result.candles[0]?.close).toBe(100)
    expect(result.markers).toEqual([])
    expect(result.symbols).toEqual(['BTCUSDT', 'ETHUSDT'])
  })
})

describe('event navigation', () => {
  const records = [
    telemetry({ sequence: 1, eventType: 'rollout' }),
    telemetry({ sequence: 2, eventType: 'position' }),
    telemetry({ sequence: 3, eventType: 'rollout' }),
    telemetry({ sequence: 4, eventType: 'risk' }),
    telemetry({ sequence: 5, eventType: 'episode_end' }),
  ]

  it('moves backward only to non-rollout events', () => {
    expect(previousEventIndex(records, 4)).toBe(3)
    expect(previousEventIndex(records, 3)).toBe(1)
    expect(previousEventIndex(records, 1)).toBe(1)
  })

  it('moves forward only to non-rollout events', () => {
    expect(nextEventIndex(records, 0)).toBe(1)
    expect(nextEventIndex(records, 1)).toBe(3)
    expect(nextEventIndex(records, 4)).toBe(4)
  })
})

describe('positionTransition', () => {
  it.each([
    [0, 0.3, 'LONG'],
    [-0.2, 0.3, 'LONG'],
    [0, -0.3, 'SHORT'],
    [0.2, -0.3, 'SHORT'],
    [0.3, 0, 'CLOSE'],
    [-0.3, 0, 'CLOSE'],
    [0.2, 0.4, 'ADD LONG'],
    [0.4, 0.2, 'REDUCE LONG'],
    [-0.2, -0.4, 'ADD SHORT'],
    [-0.4, -0.2, 'COVER SHORT'],
  ] as const)('classifies %s → %s as %s', (before, after, expected) => {
    expect(positionTransition(before, after)).toBe(expected)
  })
})
