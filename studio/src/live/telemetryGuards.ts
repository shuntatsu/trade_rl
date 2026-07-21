import type {
  TelemetryEventsResponse,
  TelemetryStatusResponse,
  TrainingTelemetryRecord,
} from '../data/types'
import { isRecord } from '../api/guards'

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)
const isNonNegativeInteger = (value: unknown): value is number =>
  isFiniteNumber(value) && Number.isInteger(value) && value >= 0
const isNullableNumber = (value: unknown): value is number | null =>
  value === null || isFiniteNumber(value)
const isNumberArray = (value: unknown): value is number[] =>
  Array.isArray(value) && value.every(isFiniteNumber)
const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === 'string')
const eventTypes = new Set([
  'rollout',
  'position',
  'risk',
  'episode_end',
  'checkpoint',
  'gap',
])

export function isTrainingTelemetryRecord(value: unknown): value is TrainingTelemetryRecord {
  if (!isRecord(value)) return false
  return value.schemaVersion === 'training_telemetry_v1'
    && isNonNegativeInteger(value.sequence) && value.sequence > 0
    && typeof value.recordedAt === 'string'
    && isNonNegativeInteger(value.globalStep)
    && isNonNegativeInteger(value.environmentStep)
    && isNonNegativeInteger(value.seed)
    && isNonNegativeInteger(value.environmentId)
    && typeof value.eventType === 'string' && eventTypes.has(value.eventType)
    && (value.marketIndex === null || isNonNegativeInteger(value.marketIndex))
    && (value.marketTime === null || typeof value.marketTime === 'string')
    && typeof value.symbol === 'string' && value.symbol.length > 0
    && isNullableNumber(value.open)
    && isNullableNumber(value.high)
    && isNullableNumber(value.low)
    && isNullableNumber(value.close)
    && isNumberArray(value.action)
    && isNumberArray(value.executedTarget)
    && isNumberArray(value.weightsBefore)
    && isNumberArray(value.weightsAfter)
    && isNullableNumber(value.portfolioValue)
    && isNullableNumber(value.baselinePortfolioValue)
    && isNullableNumber(value.reward)
    && isNullableNumber(value.drawdown)
    && isNullableNumber(value.intervalCost)
    && isNullableNumber(value.intervalReturn)
    && isStringArray(value.riskReasons)
    && typeof value.emergencyDeleverage === 'boolean'
    && typeof value.terminated === 'boolean'
    && typeof value.truncated === 'boolean'
}

export function isTelemetryStatus(value: unknown): value is TelemetryStatusResponse {
  if (!isRecord(value)) return false
  return typeof value.available === 'boolean'
    && isNonNegativeInteger(value.recordCount)
    && isNonNegativeInteger(value.lastSequence)
    && isNonNegativeInteger(value.malformedLines)
    && isNonNegativeInteger(value.sizeBytes)
    && (value.source === null || typeof value.source === 'string')
}

export function isTelemetryEvents(value: unknown): value is TelemetryEventsResponse {
  if (!isRecord(value) || !Array.isArray(value.items)) return false
  if (!value.items.every(isTrainingTelemetryRecord)) return false
  let previous = 0
  for (const item of value.items) {
    if (item.sequence <= previous) return false
    previous = item.sequence
  }
  return isNonNegativeInteger(value.nextSequence)
    && (value.items.length === 0 || value.nextSequence >= previous)
    && typeof value.truncated === 'boolean'
    && isNonNegativeInteger(value.malformedLines)
    && Array.isArray(value.sequenceGaps)
    && value.sequenceGaps.every((gap) =>
      Array.isArray(gap)
      && gap.length === 2
      && isNonNegativeInteger(gap[0])
      && isNonNegativeInteger(gap[1])
      && gap[0] <= gap[1])
}
