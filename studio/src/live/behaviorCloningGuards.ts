import type { BehaviorCloningProgressResponse } from '../data/types'

const phases = new Set(['not_started', 'preparing', 'training', 'evaluating', 'passed', 'failed'])
const nullableNumber = (value: unknown): boolean => value === null || (typeof value === 'number' && Number.isFinite(value))
const nullableString = (value: unknown): boolean => value === null || typeof value === 'string'
const nullableBoolean = (value: unknown): boolean => value === null || typeof value === 'boolean'

export function isBehaviorCloningProgress(value: unknown): value is BehaviorCloningProgressResponse {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const item = value as Record<string, unknown>
  return item.schemaVersion === 'behavior_cloning_progress_v1'
    && typeof item.available === 'boolean'
    && phases.has(String(item.phase))
    && nullableNumber(item.epoch)
    && nullableNumber(item.totalEpochs)
    && nullableNumber(item.bestEpoch)
    && nullableNumber(item.percent)
    && nullableNumber(item.seed)
    && nullableString(item.fold)
    && nullableString(item.configuration)
    && nullableNumber(item.elapsedSeconds)
    && nullableNumber(item.estimatedRemainingSeconds)
    && nullableNumber(item.validationLoss)
    && nullableNumber(item.gateLoss)
    && nullableNumber(item.targetLoss)
    && nullableNumber(item.composedLoss)
    && nullableNumber(item.gatePrecision)
    && nullableNumber(item.gateRecall)
    && nullableNumber(item.activityRatio)
    && nullableBoolean(item.allHoldCollapse)
    && nullableBoolean(item.allTradeCollapse)
    && nullableBoolean(item.earlyStopping)
    && nullableString(item.updatedAt)
    && nullableString(item.source)
}
