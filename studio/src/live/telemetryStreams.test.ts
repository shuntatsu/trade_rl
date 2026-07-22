import { describe, expect, it } from 'vitest'

import type { TrainingTelemetryRecord } from '../data/types'
import { currentEnvironmentEpisode, telemetryEnvironmentIds } from './telemetryStreams'

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

describe('telemetry stream selection', () => {
  it('lists vector environments independently and in stable order', () => {
    expect(telemetryEnvironmentIds([
      record(1, 2),
      record(2, 0),
      record(3, 2),
      record(4, 1),
    ])).toEqual([0, 1, 2])
  })

  it('returns only the selected environment current episode', () => {
    const records = [
      record(1, 0),
      record(2, 1),
      record(3, 0, { eventType: 'episode_end', terminated: true }),
      record(4, 1),
      record(5, 0),
      record(6, 0),
    ]

    expect(currentEnvironmentEpisode(records, 0).map((item) => item.sequence)).toEqual([5, 6])
    expect(currentEnvironmentEpisode(records, 1).map((item) => item.sequence)).toEqual([2, 4])
  })

  it('starts a new episode when environment counters roll back', () => {
    const records = [
      record(1, 0, { environmentStep: 10, marketIndex: 110 }),
      record(2, 0, { environmentStep: 11, marketIndex: 111 }),
      record(3, 0, { environmentStep: 1, marketIndex: 101 }),
    ]

    expect(currentEnvironmentEpisode(records, 0).map((item) => item.sequence)).toEqual([3])
  })

  it('returns no records when no environment is selected', () => {
    expect(currentEnvironmentEpisode([record(1, 0)], null)).toEqual([])
  })
})
