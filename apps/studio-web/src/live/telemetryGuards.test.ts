import { describe, expect, it } from 'vitest'

import type { TrainingTelemetryRecord } from '../data/types'
import { isTrainingTelemetryRecord } from './telemetryGuards'

function record(episodeId: number | null): TrainingTelemetryRecord {
  return {
    schemaVersion: 'training_telemetry_v1',
    sequence: 1,
    recordedAt: '2026-07-22T13:00:00+00:00',
    globalStep: 32,
    environmentStep: 1,
    seed: 7,
    environmentId: 0,
    episodeId,
    eventType: 'rollout',
    marketIndex: 101,
    marketTime: '2026-07-22T12:55:00.000000000',
    symbol: 'BTCUSDT',
    open: 100,
    high: 101,
    low: 99,
    close: 100.5,
    action: [0.2],
    executedTarget: [0.2],
    weightsBefore: [0.1],
    weightsAfter: [0.2],
    portfolioValue: 1_000,
    baselinePortfolioValue: 990,
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

describe('isTrainingTelemetryRecord episode identity', () => {
  it('accepts explicit and normalized legacy identities', () => {
    expect(isTrainingTelemetryRecord(record(4))).toBe(true)
    expect(isTrainingTelemetryRecord(record(null))).toBe(true)
  })

  it('rejects missing, negative, and boolean episode identities', () => {
    const missing = { ...record(null) } as Record<string, unknown>
    delete missing.episodeId

    expect(isTrainingTelemetryRecord(missing)).toBe(false)
    expect(isTrainingTelemetryRecord({ ...record(null), episodeId: -1 })).toBe(false)
    expect(isTrainingTelemetryRecord({ ...record(null), episodeId: true })).toBe(false)
  })
})
