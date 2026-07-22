import type {
  CheckpointEvaluationItem,
  CheckpointEvaluationsResponse,
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
const generationPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
const isNullableGeneration = (value: unknown): value is string | null =>
  value === null || (typeof value === 'string' && generationPattern.test(value))
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
    && (value.selectedSeed === null || isNonNegativeInteger(value.selectedSeed))
    && Array.isArray(value.availableSeeds)
    && value.availableSeeds.every(isNonNegativeInteger)
    && new Set(value.availableSeeds).size === value.availableSeeds.length
    && isNonNegativeInteger(value.recordCount)
    && isNonNegativeInteger(value.lastSequence)
    && isNonNegativeInteger(value.malformedLines)
    && isNonNegativeInteger(value.sizeBytes)
    && (value.source === null || typeof value.source === 'string')
    && isNullableGeneration(value.streamGeneration)
}

export function isTelemetryEvents(value: unknown): value is TelemetryEventsResponse {
  if (!isRecord(value) || !Array.isArray(value.items)) return false
  if (!value.items.every(isTrainingTelemetryRecord)) return false
  if (value.seed !== null && !isNonNegativeInteger(value.seed)) return false
  let previous = 0
  for (const item of value.items) {
    if (item.sequence <= previous) return false
    if (value.seed !== null && item.seed !== value.seed) return false
    previous = item.sequence
  }
  return isNonNegativeInteger(value.nextSequence)
    && (value.items.length === 0 || value.nextSequence >= previous)
    && typeof value.truncated === 'boolean'
    && isNonNegativeInteger(value.malformedLines)
    && Array.isArray(value.sequenceGaps)
    && isNullableGeneration(value.streamGeneration)
    && typeof value.resetRequired === 'boolean'
    && (!value.resetRequired || (value.items.length === 0 && value.nextSequence === 0))
    && value.sequenceGaps.every((gap) =>
      Array.isArray(gap)
      && gap.length === 2
      && isNonNegativeInteger(gap[0])
      && isNonNegativeInteger(gap[1])
      && gap[0] <= gap[1])
}

function isCheckpointEvaluationItem(value: unknown): value is CheckpointEvaluationItem {
  if (!isRecord(value)) return false
  return typeof value.fold === 'string' && /^fold-\d+$/.test(value.fold)
    && typeof value.configuration === 'string' && value.configuration.length > 0
    && isNonNegativeInteger(value.seed)
    && typeof value.policyDigest === 'string' && /^[0-9a-f]{64}$/.test(value.policyDigest)
    && typeof value.evaluationDigest === 'string' && /^[0-9a-f]{64}$/.test(value.evaluationDigest)
    && isFiniteNumber(value.score)
    && isFiniteNumber(value.totalReturn) && value.totalReturn > -1
    && typeof value.finalist === 'boolean'
    && Array.isArray(value.checkpointRange)
    && value.checkpointRange.length === 2
    && isNonNegativeInteger(value.checkpointRange[0])
    && isNonNegativeInteger(value.checkpointRange[1])
    && value.checkpointRange[0] < value.checkpointRange[1]
    && typeof value.source === 'string' && value.source.length > 0
}

export function isCheckpointEvaluations(value: unknown): value is CheckpointEvaluationsResponse {
  return isRecord(value)
    && typeof value.available === 'boolean'
    && value.productionStatus === 'NO-GO'
    && Array.isArray(value.items)
    && value.items.every(isCheckpointEvaluationItem)
}
