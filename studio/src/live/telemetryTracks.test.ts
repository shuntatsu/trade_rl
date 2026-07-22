import { describe, expect, it } from 'vitest'

import type { TrainingTelemetryRecord } from '../data/types'
import { deriveTelemetryTracks } from './telemetryTracks'

function record(
  sequence: number,
  environmentId: number,
  overrides: Partial<TrainingTelemetryRecord> = {},
): TrainingTelemetryRecord {
  return {
    schemaVersion: 'training_telemetry_v1',
    sequence,
    recordedAt: `2026-07-23T00:00:${String(sequence).padStart(2, '0')}+00:00`,
    globalStep: sequence * 32,
    environmentStep: sequence,
    seed: 7,
    environmentId,
    eventType: 'rollout',
    marketIndex: 100 + sequence,
    marketTime: `2026-07-23T00:00:${String(sequence).padStart(2, '0')}.000000000`,
    symbol: 'BTCUSDT',
    open: 100,
    high: 101,
    low: 99,
    close: 100 + sequence,
    action: [0],
    executedTarget: [0],
    weightsBefore: [0],
    weightsAfter: [0],
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
    ...overrides,
  }
}

describe('deriveTelemetryTracks', () => {
  it('keeps interleaved vector environments in independent tracks', () => {
    const tracks = deriveTelemetryTracks([
      record(1, 0),
      record(2, 1),
      record(3, 0),
      record(4, 1),
    ])

    expect(tracks).toHaveLength(2)
    expect(tracks.find((track) => track.environmentId === 0)?.records.map((item) => item.sequence)).toEqual([1, 3])
    expect(tracks.find((track) => track.environmentId === 1)?.records.map((item) => item.sequence)).toEqual([2, 4])
  })

  it('starts a new current episode after a terminal record', () => {
    const tracks = deriveTelemetryTracks([
      record(1, 0),
      record(2, 0, { eventType: 'episode_end', terminated: true }),
      record(3, 0),
      record(4, 0),
    ])

    expect(tracks.map((track) => track.records.map((item) => item.sequence))).toEqual([[1, 2], [3, 4]])
    expect(tracks.at(-1)?.environmentId).toBe(0)
  })

  it('fails conservatively into a new episode when counters roll back', () => {
    const tracks = deriveTelemetryTracks([
      record(1, 0, { environmentStep: 10, marketIndex: 110 }),
      record(2, 0, { environmentStep: 11, marketIndex: 111 }),
      record(3, 0, { environmentStep: 1, marketIndex: 101 }),
    ])

    expect(tracks.map((track) => track.records.map((item) => item.sequence))).toEqual([[1, 2], [3]])
  })
})
