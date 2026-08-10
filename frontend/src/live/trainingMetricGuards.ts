import type {
  TrainingMetricGroup,
  TrainingMetricSeries,
  TrainingMetricUnit,
  TrainingMetricsResponse,
  TrainingMetricsStatusResponse,
} from '../data/types'

export const TRAINING_METRIC_TAGS = [
  'train/learning_rate',
  'train/loss',
  'train/policy_gradient_loss',
  'train/value_loss',
  'train/entropy_loss',
  'train/approx_kl',
  'train/clip_fraction',
  'train/explained_variance',
  'trade_rl/reward_mean',
  'trade_rl/portfolio_value_mean',
  'trade_rl/baseline_portfolio_value_mean',
  'trade_rl/drawdown_mean',
  'trade_rl/interval_cost_mean',
  'trade_rl/reward_growth_raw_mean',
  'trade_rl/reward_absolute_component_mean',
  'trade_rl/reward_excess_component_mean',
  'trade_rl/reward_baseline_penalty_weighted_mean',
  'trade_rl/reward_drawdown_penalty_weighted_mean',
  'trade_rl/reward_projection_penalty_weighted_mean',
  'trade_rl/reward_terminal_penalty_weighted_mean',
  'trade_rl/reward_margin_penalty_weighted_mean',
  'trade_rl/reward_total_raw_mean',
  'trade_rl/rolling_growth_gap_mean',
  'trade_rl/action_abs_mean',
  'trade_rl/action_abs_max',
] as const

const tagSet = new Set<string>(TRAINING_METRIC_TAGS)
const groups = new Set<TrainingMetricGroup>(['optimization', 'policy', 'value', 'trading'])
const units = new Set<TrainingMetricUnit>(['raw', 'rate', 'percent', 'currency'])

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function nullableSeed(value: unknown): value is number | null {
  return value === null || (Number.isInteger(value) && finite(value) && value >= 0)
}

function stringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === 'string')
}

function isSeries(value: unknown): value is TrainingMetricSeries {
  if (!record(value) || !tagSet.has(String(value.tag))) return false
  if (typeof value.displayName !== 'string') return false
  if (!groups.has(value.group as TrainingMetricGroup) || !units.has(value.unit as TrainingMetricUnit)) return false
  if (!Array.isArray(value.points)) return false
  let previous = -1
  for (const point of value.points) {
    if (!record(point) || !Number.isInteger(point.step) || !finite(point.step) || point.step < 0) return false
    if (!finite(point.wallTime) || !finite(point.value) || point.step <= previous) return false
    previous = point.step
  }
  return true
}

export function isTrainingMetricsStatus(value: unknown): value is TrainingMetricsStatusResponse {
  if (!record(value) || typeof value.available !== 'boolean' || !nullableSeed(value.selectedSeed)) return false
  if (!Array.isArray(value.availableSeeds) || !value.availableSeeds.every((seed) => nullableSeed(seed) && seed !== null)) return false
  if (!stringArray(value.availableTags) || !value.availableTags.every((tag) => tagSet.has(tag))) return false
  if (new Set(value.availableTags).size !== value.availableTags.length) return false
  return Number.isInteger(value.lastStep) && finite(value.lastStep) && value.lastStep >= 0
    && (value.source === null || typeof value.source === 'string')
    && (value.generation === null || (typeof value.generation === 'string' && /^[0-9a-f]{64}$/.test(value.generation)))
}

export function isTrainingMetricsResponse(value: unknown): value is TrainingMetricsResponse {
  if (!record(value) || !nullableSeed(value.seed) || !Array.isArray(value.series)) return false
  if (!Number.isInteger(value.nextStep) || !finite(value.nextStep) || value.nextStep < 0) return false
  if (typeof value.resetRequired !== 'boolean') return false
  if (!(value.generation === null || (typeof value.generation === 'string' && /^[0-9a-f]{64}$/.test(value.generation)))) return false
  if (!value.series.every(isSeries)) return false
  return new Set(value.series.map((series) => series.tag)).size === value.series.length
}
