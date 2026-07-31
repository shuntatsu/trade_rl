import type { TrainingTelemetryRecord } from '../data/types'

export type ResearchTimeframe = '15m' | '1h' | '4h' | '1d'

export interface ResearchChartLayers {
  positionEvents: boolean
  riskEvents: boolean
  baseline: boolean
  executedWeight: boolean
  rewardCost: boolean
  drawdown: boolean
}

export const DEFAULT_RESEARCH_CHART_LAYERS: ResearchChartLayers = {
  positionEvents: true,
  riskEvents: true,
  baseline: true,
  executedWeight: true,
  rewardCost: true,
  drawdown: true,
}

export interface ResearchCandle {
  time: number
  open: number
  high: number
  low: number
  close: number
}

export interface ResearchLinePoint {
  time: number
  value: number
}

export type ResearchMarkerShape = 'arrowUp' | 'arrowDown' | 'circle' | 'square'
export type ResearchMarkerPosition = 'aboveBar' | 'belowBar'

export interface ResearchMarker {
  time: number
  position: ResearchMarkerPosition
  shape: ResearchMarkerShape
  color: string
  text: 'BUY' | 'SELL' | 'RISK' | 'END'
  sequence: number
}

export interface ResearchChartData {
  symbols: string[]
  candles: ResearchCandle[]
  targetWeight: ResearchLinePoint[]
  executedWeight: ResearchLinePoint[]
  reward: ResearchLinePoint[]
  cost: ResearchLinePoint[]
  equity: ResearchLinePoint[]
  baseline: ResearchLinePoint[]
  drawdown: ResearchLinePoint[]
  markers: ResearchMarker[]
  recordByTime: Map<number, TrainingTelemetryRecord>
  timeBySequence: Map<number, number>
}

const TIMEFRAME_SECONDS: Record<ResearchTimeframe, number> = {
  '15m': 15 * 60,
  '1h': 60 * 60,
  '4h': 4 * 60 * 60,
  '1d': 24 * 60 * 60,
}

interface TimedRecord {
  record: TrainingTelemetryRecord
  epochSeconds: number
  bucketTime: number
}

function finite(value: number | null | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function normalizedTimestamp(value: string): string {
  const milliseconds = value.replace(/(\.\d{3})\d+/, '$1')
  return /(?:Z|[+-]\d{2}:\d{2})$/.test(milliseconds) ? milliseconds : `${milliseconds}Z`
}

function epochSeconds(record: TrainingTelemetryRecord): number | null {
  for (const candidate of [record.marketTime, record.recordedAt]) {
    if (!candidate) continue
    const parsed = Date.parse(normalizedTimestamp(candidate))
    if (Number.isFinite(parsed)) return Math.floor(parsed / 1_000)
  }
  return null
}

function firstFinite(
  records: TrainingTelemetryRecord[],
  selector: (record: TrainingTelemetryRecord) => number | null | undefined,
): number | null {
  for (const record of records) {
    const value = selector(record)
    if (finite(value)) return value
  }
  return null
}

function lastFinite(
  records: TrainingTelemetryRecord[],
  selector: (record: TrainingTelemetryRecord) => number | null | undefined,
): number | null {
  for (let index = records.length - 1; index >= 0; index -= 1) {
    const value = selector(records[index]!)
    if (finite(value)) return value
  }
  return null
}

function extrema(
  records: TrainingTelemetryRecord[],
  selectors: Array<(record: TrainingTelemetryRecord) => number | null | undefined>,
  resolve: (values: number[]) => number,
): number | null {
  const values = records.flatMap((record) => selectors
    .map((selector) => selector(record))
    .filter(finite))
  return values.length === 0 ? null : resolve(values)
}

function linePoint(time: number, value: number | null): ResearchLinePoint[] {
  return value === null ? [] : [{ time, value }]
}

function markerFor(record: TrainingTelemetryRecord, time: number): ResearchMarker | null {
  if (record.eventType === 'position') {
    const delta = (record.weightsAfter[0] ?? 0) - (record.weightsBefore[0] ?? 0)
    if (delta > 0) {
      return {
        time,
        position: 'belowBar',
        shape: 'arrowUp',
        color: '#36e37d',
        text: 'BUY',
        sequence: record.sequence,
      }
    }
    if (delta < 0) {
      return {
        time,
        position: 'aboveBar',
        shape: 'arrowDown',
        color: '#ff5b63',
        text: 'SELL',
        sequence: record.sequence,
      }
    }
    return null
  }
  if (record.eventType === 'risk') {
    return {
      time,
      position: 'aboveBar',
      shape: 'circle',
      color: '#f3b33d',
      text: 'RISK',
      sequence: record.sequence,
    }
  }
  if (record.eventType === 'episode_end') {
    return {
      time,
      position: 'aboveBar',
      shape: 'square',
      color: '#4098ff',
      text: 'END',
      sequence: record.sequence,
    }
  }
  return null
}

export function buildResearchChartData(
  records: TrainingTelemetryRecord[],
  symbol: string,
  timeframe: ResearchTimeframe,
): ResearchChartData {
  const seconds = TIMEFRAME_SECONDS[timeframe]
  const symbols = [...new Set(records.map((record) => record.symbol).filter(Boolean))].sort()
  const timed: TimedRecord[] = []
  const timeBySequence = new Map<number, number>()

  for (const record of records) {
    if (record.symbol !== symbol) continue
    const time = epochSeconds(record)
    if (time === null) continue
    const bucketTime = Math.floor(time / seconds) * seconds
    timed.push({ record, epochSeconds: time, bucketTime })
    timeBySequence.set(record.sequence, bucketTime)
  }

  timed.sort((left, right) =>
    left.epochSeconds - right.epochSeconds
    || left.record.sequence - right.record.sequence)

  const buckets = new Map<number, TrainingTelemetryRecord[]>()
  for (const item of timed) {
    const bucket = buckets.get(item.bucketTime) ?? []
    bucket.push(item.record)
    buckets.set(item.bucketTime, bucket)
  }

  const candles: ResearchCandle[] = []
  const targetWeight: ResearchLinePoint[] = []
  const executedWeight: ResearchLinePoint[] = []
  const reward: ResearchLinePoint[] = []
  const cost: ResearchLinePoint[] = []
  const equity: ResearchLinePoint[] = []
  const baseline: ResearchLinePoint[] = []
  const drawdown: ResearchLinePoint[] = []
  const recordByTime = new Map<number, TrainingTelemetryRecord>()

  for (const [time, bucketRecords] of [...buckets.entries()].sort(([left], [right]) => left - right)) {
    const open = firstFinite(bucketRecords, (record) => record.open)
      ?? firstFinite(bucketRecords, (record) => record.close)
    const close = lastFinite(bucketRecords, (record) => record.close)
      ?? lastFinite(bucketRecords, (record) => record.open)
    const high = extrema(
      bucketRecords,
      [(record) => record.high, (record) => record.open, (record) => record.close],
      (values) => Math.max(...values),
    )
    const low = extrema(
      bucketRecords,
      [(record) => record.low, (record) => record.open, (record) => record.close],
      (values) => Math.min(...values),
    )

    if (open === null || close === null || high === null || low === null) continue

    candles.push({ time, open, high, low, close })
    targetWeight.push(...linePoint(time, lastFinite(bucketRecords, (record) => record.weightsAfter[0])))
    executedWeight.push(...linePoint(time, lastFinite(bucketRecords, (record) => record.executedTarget[0])))
    reward.push(...linePoint(time, lastFinite(bucketRecords, (record) => record.reward)))
    cost.push(...linePoint(time, lastFinite(bucketRecords, (record) => record.intervalCost)))
    equity.push(...linePoint(time, lastFinite(bucketRecords, (record) => record.portfolioValue)))
    baseline.push(...linePoint(time, lastFinite(bucketRecords, (record) => record.baselinePortfolioValue)))
    const latestDrawdown = lastFinite(bucketRecords, (record) => record.drawdown)
    drawdown.push(...linePoint(time, latestDrawdown === null ? null : -latestDrawdown * 100))
    recordByTime.set(time, bucketRecords.at(-1)!)
  }

  const markers = timed
    .map((item) => markerFor(item.record, item.bucketTime))
    .filter((marker): marker is ResearchMarker => marker !== null)
    .sort((left, right) => left.time - right.time || left.sequence - right.sequence)

  return {
    symbols,
    candles,
    targetWeight,
    executedWeight,
    reward,
    cost,
    equity,
    baseline,
    drawdown,
    markers,
    recordByTime,
    timeBySequence,
  }
}

function isEvent(record: TrainingTelemetryRecord): boolean {
  return record.eventType !== 'rollout'
}

export function previousEventIndex(records: TrainingTelemetryRecord[], cursor: number): number {
  for (let index = Math.min(cursor - 1, records.length - 1); index >= 0; index -= 1) {
    if (isEvent(records[index]!)) return index
  }
  return Math.max(0, Math.min(cursor, records.length - 1))
}

export function nextEventIndex(records: TrainingTelemetryRecord[], cursor: number): number {
  for (let index = Math.max(0, cursor + 1); index < records.length; index += 1) {
    if (isEvent(records[index]!)) return index
  }
  return Math.max(0, Math.min(cursor, records.length - 1))
}
