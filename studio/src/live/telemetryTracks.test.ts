import { describe, expect, it } from 'vitest'

import type { TrainingTelemetryRecord } from '../data/types'
import { deriveTelemetryTracks, selectTelemetryTrack } from './telemetryTracks'

function telemetry(
  sequence: number,
  overrides: Partial<TrainingTelemetryRecord> & { episodeId?: number | null } = {},
): TrainingTelemetryRecord {
  return {
    schemaVersion: 'training_telemetry_v1',
    sequence,
    recordedAt: '2026-07-22T12:00:00+00:00',
    globalStep: sequence * 32,
    environmentStep: sequence,
    seed: 7,
    environmentId: 0,
    episodeId: 10,
    eventType: 'rollout',
    marketIndex: 100 + sequence,
    marketTime: '2026-07-22T11:55:00.000000000',
    symbol: 'BTCUSDT',
    open: 100 + sequence,
    high: 101 + sequence,
    low: 99 + sequence,
    close: 100.5 + sequence,
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
    ...overrides,
  } as TrainingTelemetryRecord
}

describe('deriveTelemetryTracks', () => {
  it('keeps interleaved explicit environments and episodes independent', () => {
    const tracks = deriveTelemetryTracks([
      telemetry(1, { environmentId: 0, episodeId: 10 }),
      telemetry(2, { environmentId: 1, episodeId: 20, symbol: 'ETHUSDT' }),
      telemetry(3, { environmentId: 0, episodeId: 10 }),
      telemetry(4, { environmentId: 0, episodeId: 11 }),
    ])

    expect(tracks).toHaveLength(3)
    expect(tracks.find((track) => track.environmentId === 0 && track.episodeId === 10)?.records.map((item) => item.sequence)).toEqual([1, 3])
    expect(tracks.find((track) => track.environmentId === 1 && track.episodeId === 20)?.records.map((item) => item.sequence)).toEqual([2])
    expect(tracks.find((track) => track.environmentId === 0 && track.episodeId === 11)?.records.map((item) => item.sequence)).toEqual([4])
  })

  it('starts a legacy track after episode_end', () => {
    const tracks = deriveTelemetryTracks([
      telemetry(1, { episodeId: null, environmentStep: 1 }),
      telemetry(2, {
        episodeId: null,
        environmentStep: 2,
        eventType: 'episode_end',
        terminated: true,
      }),
      telemetry(3, { episodeId: null, environmentStep: 0 }),
    ])

    expect(tracks).toHaveLength(2)
    expect(tracks[0].records.map((item) => item.sequence)).toEqual([1, 2])
    expect(tracks[1].records.map((item) => item.sequence)).toEqual([3])
    expect(tracks.every((track) => track.inferred)).toBe(true)
  })

  it('starts a legacy track after environment step rollback', () => {
    const tracks = deriveTelemetryTracks([
      telemetry(1, { episodeId: null, environmentStep: 8, marketIndex: null }),
      telemetry(2, { episodeId: null, environmentStep: 0, marketIndex: null }),
    ])

    expect(tracks.map((track) => track.records.map((item) => item.sequence))).toEqual([[1], [2]])
  })

  it('starts a legacy track after market index rollback', () => {
    const tracks = deriveTelemetryTracks([
      telemetry(1, { episodeId: null, environmentStep: 8, marketIndex: 500 }),
      telemetry(2, { episodeId: null, environmentStep: 9, marketIndex: 100 }),
    ])

    expect(tracks.map((track) => track.records.map((item) => item.sequence))).toEqual([[1], [2]])
  })

  it('never merges explicit and legacy records', () => {
    const tracks = deriveTelemetryTracks([
      telemetry(1, { environmentId: 0, episodeId: 10 }),
      telemetry(2, { environmentId: 0, episodeId: null }),
    ])

    expect(tracks).toHaveLength(2)
    expect(tracks.some((track) => track.episodeId === 10 && !track.inferred)).toBe(true)
    expect(tracks.some((track) => track.episodeId === null && track.inferred)).toBe(true)
  })
})

describe('selectTelemetryTrack', () => {
  const records = [
    telemetry(1, { environmentId: 0, episodeId: 10 }),
    telemetry(2, { environmentId: 1, episodeId: 20 }),
    telemetry(3, { environmentId: 0, episodeId: 10 }),
    telemetry(4, { environmentId: 1, episodeId: 21 }),
  ]

  it('selects the track containing the cursor', () => {
    expect(selectTelemetryTrack(records, 3)?.records.map((item) => item.sequence)).toEqual([1, 3])
  })

  it('selects the newest track without a cursor match', () => {
    expect(selectTelemetryTrack(records, null)?.records.map((item) => item.sequence)).toEqual([4])
  })
})
