import { renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { StudioApi } from '../api/studioApi'
import type {
  TelemetryEventsResponse,
  TelemetryStatusResponse,
  TrainingTelemetryRecord,
} from '../data/types'
import { isTelemetryEvents, isTelemetryStatus } from './telemetryGuards'
import { useTrainingTelemetry } from './useTrainingTelemetry'

const GENERATION_A = '11111111-1111-4111-8111-111111111111'
const GENERATION_B = '22222222-2222-4222-8222-222222222222'

function telemetry(sequence: number): TrainingTelemetryRecord {
  return {
    schemaVersion: 'training_telemetry_v1',
    sequence,
    recordedAt: '2026-07-22T11:00:00+00:00',
    globalStep: sequence * 32,
    environmentStep: sequence,
    seed: 7,
    environmentId: 0,
    episodeId: 1,
    eventType: 'rollout',
    marketIndex: 100 + sequence,
    marketTime: '2026-07-22T10:55:00.000000000',
    symbol: 'BTCUSDT',
    open: 67_500,
    high: 67_900,
    low: 67_400,
    close: 67_842.3,
    action: [0.4],
    executedTarget: [0.4],
    weightsBefore: [0.2],
    weightsAfter: [0.4],
    portfolioValue: 101_342.85,
    baselinePortfolioValue: 100_400,
    reward: 0.214,
    drawdown: 0.0086,
    intervalCost: 4.25,
    intervalReturn: 0.0012,
    riskReasons: [],
    emergencyDeleverage: false,
    terminated: false,
    truncated: false,
  }
}

function status(generation: string): TelemetryStatusResponse {
  return {
    available: true,
    selectedSeed: 7,
    availableSeeds: [7],
    recordCount: 1,
    lastSequence: 1,
    malformedLines: 0,
    sizeBytes: 1024,
    source: 'research/.staging/live/seed-7/telemetry/training-telemetry.jsonl',
    streamGeneration: generation,
  } as unknown as TelemetryStatusResponse
}

function page(
  generation: string,
  items: TrainingTelemetryRecord[],
  resetRequired = false,
): TelemetryEventsResponse {
  return {
    seed: 7,
    items,
    nextSequence: items.at(-1)?.sequence ?? 0,
    truncated: false,
    malformedLines: 0,
    sequenceGaps: [],
    streamGeneration: generation,
    resetRequired,
  } as unknown as TelemetryEventsResponse
}

function api(
  loadTelemetryStatus: ReturnType<typeof vi.fn>,
  loadTelemetryEvents: ReturnType<typeof vi.fn>,
): StudioApi {
  const unused = vi.fn().mockRejectedValue(new Error('not used'))
  return {
    loadDatasets: unused,
    loadRuns: unused,
    loadConfigs: unused,
    loadJobs: unused,
    submitTrainingJob: unused,
    cancelJob: unused,
    loadJobLog: unused,
    loadTelemetryStatus,
    loadTelemetryEvents,
    loadCheckpointEvaluations: unused,
    loadRunComparison: unused,
    loadEvidenceReport: unused,
    loadServingMonitor: unused,
  } as unknown as StudioApi
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('useTrainingTelemetry generation handling', () => {
  it('clears a stale cursor and immediately replays the replacement generation', async () => {
    const loadStatus = vi.fn().mockResolvedValue(status(GENERATION_B))
    const loadEvents = vi.fn()
      .mockResolvedValueOnce(page(GENERATION_B, [], true))
      .mockResolvedValueOnce(page(GENERATION_B, [telemetry(1)]))
    const runtimeApi = api(loadStatus, loadEvents)

    const { result, unmount } = renderHook(() =>
      useTrainingTelemetry('job-live', runtimeApi, 7))

    await waitFor(() => expect(loadEvents).toHaveBeenCalledTimes(2), { timeout: 300 })
    await waitFor(() =>
      expect(result.current.records.map((item) => item.sequence)).toEqual([1]))

    expect(loadEvents).toHaveBeenNthCalledWith(
      1,
      'job-live',
      0,
      512,
      7,
      null,
    )
    expect(loadEvents).toHaveBeenNthCalledWith(
      2,
      'job-live',
      0,
      512,
      7,
      GENERATION_B,
    )
    unmount()
  })

  it('discards a mixed status and events generation before publishing records', async () => {
    const loadStatus = vi.fn()
      .mockResolvedValueOnce(status(GENERATION_A))
      .mockResolvedValueOnce(status(GENERATION_B))
    const loadEvents = vi.fn()
      .mockResolvedValueOnce(page(GENERATION_B, [telemetry(99)]))
      .mockResolvedValueOnce(page(GENERATION_B, [telemetry(1)]))
    const runtimeApi = api(loadStatus, loadEvents)

    const { result, unmount } = renderHook(() =>
      useTrainingTelemetry('job-live', runtimeApi, 7))

    await waitFor(() => expect(loadEvents).toHaveBeenCalledTimes(2), { timeout: 300 })
    await waitFor(() =>
      expect(result.current.records.map((item) => item.sequence)).toEqual([1]))
    expect(result.current.records.some((item) => item.sequence === 99)).toBe(false)
    unmount()
  })
})

describe('telemetry generation guards', () => {
  it('rejects an invalid status generation', () => {
    expect(isTelemetryStatus({
      ...status(GENERATION_A),
      streamGeneration: 'not-a-generation',
    })).toBe(false)
  })

  it('rejects a non-boolean reset flag', () => {
    expect(isTelemetryEvents({
      ...page(GENERATION_A, []),
      resetRequired: 'yes',
    })).toBe(false)
  })
})
