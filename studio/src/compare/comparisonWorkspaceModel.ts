import type {
  ComparisonMetric,
  ComparisonSeriesPoint,
  FoldComparison,
  RunComparison,
} from '../data/types'

export type ComparisonSeriesKey = 'left' | 'right' | 'leftBaseline' | 'rightBaseline'
export type ComparisonVerdict = 'improved' | 'worse' | 'tie' | 'unavailable'
export type ComparisonWinner = 'left' | 'right' | 'tie' | 'unknown'

export interface NumericDomain {
  minimum: number
  maximum: number
}

export interface FoldAwareComparisonPoint extends ComparisonSeriesPoint {
  foldIndex?: number | null
}

export interface ComparisonWorkspacePoint extends FoldAwareComparisonPoint {
  index: number
  foldIndex: number | null
  delta: number | null
}

export interface ComparisonFoldSpan extends FoldComparison {
  foldIndex: number
  startIndex: number
  endIndex: number
}

export interface ComparisonDirectLabel {
  key: ComparisonSeriesKey
  label: string
  value: number
  position: number
}

export interface ComparisonMetricVerdict extends ComparisonMetric {
  verdict: ComparisonVerdict
}

export interface ComparisonWorkspaceModel {
  leftRunId: string
  rightRunId: string
  points: ComparisonWorkspacePoint[]
  wealthDomain: NumericDomain
  deltaDomain: NumericDomain
  foldSpans: ComparisonFoldSpan[]
  directLabels: ComparisonDirectLabel[]
  metricVerdicts: ComparisonMetricVerdict[]
}

export interface ComparisonRangeSummary {
  startIndex: number
  endIndex: number
  startLabel: string
  endLabel: string
  leftReturn: number | null
  rightReturn: number | null
  relativeReturn: number | null
  finalDelta: number | null
  maximumGap: number | null
  winner: ComparisonWinner
}

const SERIES: Array<{ key: ComparisonSeriesKey; label: string }> = [
  { key: 'left', label: 'Left' },
  { key: 'right', label: 'Right' },
  { key: 'leftBaseline', label: 'Left baseline' },
  { key: 'rightBaseline', label: 'Right baseline' },
]

function finite(values: Array<number | null | undefined>): number[] {
  return values.filter(
    (value): value is number => typeof value === 'number' && Number.isFinite(value),
  )
}

function paddedDomain(
  values: number[],
  required: number,
  minimumSpan: number,
): NumericDomain {
  const minimumValue = Math.min(required, ...values)
  const maximumValue = Math.max(required, ...values)
  const rawSpan = maximumValue - minimumValue
  const span = Math.max(rawSpan, minimumSpan)
  const padding = span * 0.08
  const centre = (minimumValue + maximumValue) / 2
  if (rawSpan < minimumSpan) {
    return {
      minimum: centre - span / 2 - padding,
      maximum: centre + span / 2 + padding,
    }
  }
  return {
    minimum: minimumValue - padding,
    maximum: maximumValue + padding,
  }
}

function normalize(value: number, domain: NumericDomain): number {
  const span = domain.maximum - domain.minimum || 1
  return 1 - (value - domain.minimum) / span
}

function placeLabels(
  values: Array<{ key: ComparisonSeriesKey; label: string; value: number }>,
  domain: NumericDomain,
): ComparisonDirectLabel[] {
  const minimumGap = 0.075
  const ordered = values
    .map((item) => ({
      ...item,
      natural: Math.max(0, Math.min(1, normalize(item.value, domain))),
    }))
    .sort(
      (left, right) => left.natural - right.natural || left.key.localeCompare(right.key),
    )

  for (let index = 1; index < ordered.length; index += 1) {
    ordered[index].natural = Math.max(
      ordered[index].natural,
      ordered[index - 1].natural + minimumGap,
    )
  }
  const overflow = ordered.at(-1)?.natural ?? 0
  if (overflow > 1) {
    for (const item of ordered) item.natural -= overflow - 1
  }
  for (let index = ordered.length - 2; index >= 0; index -= 1) {
    ordered[index].natural = Math.min(
      ordered[index].natural,
      ordered[index + 1].natural - minimumGap,
    )
  }
  const underflow = ordered[0]?.natural ?? 0
  if (underflow < 0) {
    for (const item of ordered) item.natural -= underflow
  }

  return ordered.map(({ natural, ...item }) => ({
    ...item,
    position: Math.max(0, Math.min(1, natural)),
  }))
}

export function buildComparisonDirectLabels(
  point: ComparisonWorkspacePoint | null | undefined,
  domain: NumericDomain,
): ComparisonDirectLabel[] {
  if (!point) return []
  const values: Array<{
    key: ComparisonSeriesKey
    label: string
    value: number
  }> = []
  for (const { key, label } of SERIES) {
    const value = point[key]
    if (typeof value === 'number') values.push({ key, label, value })
  }
  return placeLabels(values, domain)
}

function metricVerdict(metric: ComparisonMetric): ComparisonVerdict {
  if (metric.delta === null || !Number.isFinite(metric.delta)) return 'unavailable'
  if (metric.delta === 0 || metric.preference === 'neutral') return 'tie'
  const improved = metric.preference === 'higher' ? metric.delta > 0 : metric.delta < 0
  return improved ? 'improved' : 'worse'
}

function foldSpans(
  points: ComparisonWorkspacePoint[],
  folds: FoldComparison[],
): ComparisonFoldSpan[] {
  const spans = new Map<number, { startIndex: number; endIndex: number }>()
  for (const point of points) {
    if (point.foldIndex === null) continue
    const current = spans.get(point.foldIndex)
    if (current) current.endIndex = point.index
    else spans.set(point.foldIndex, { startIndex: point.index, endIndex: point.index })
  }
  return [...spans.entries()]
    .sort(([left], [right]) => left - right)
    .map(([foldIndex, span]) => ({
      ...(folds[foldIndex] ?? {
        label: `Fold ${foldIndex + 1}`,
        leftSelectedReturn: null,
        leftBaselineReturn: null,
        rightSelectedReturn: null,
        rightBaselineReturn: null,
      }),
      foldIndex,
      ...span,
    }))
}

export function buildComparisonWorkspaceModel(
  comparison: RunComparison,
): ComparisonWorkspaceModel {
  const points = comparison.wealth.map((rawPoint, index) => {
    const point = rawPoint as FoldAwareComparisonPoint
    const foldIndex = Number.isInteger(point.foldIndex) && (point.foldIndex ?? -1) >= 0
      ? point.foldIndex ?? null
      : null
    return {
      ...rawPoint,
      index,
      foldIndex,
      delta: typeof point.left === 'number' && typeof point.right === 'number'
        ? point.right - point.left
        : null,
    }
  })
  const wealthValues = finite(
    points.flatMap((point) => [
      point.left,
      point.right,
      point.leftBaseline,
      point.rightBaseline,
    ]),
  )
  const deltaValues = finite(points.map((point) => point.delta))
  const wealthDomain = paddedDomain(wealthValues, 1, 0.01)
  const deltaDomain = paddedDomain(deltaValues, 0, 0.01)

  return {
    leftRunId: comparison.leftRunId,
    rightRunId: comparison.rightRunId,
    points,
    wealthDomain,
    deltaDomain,
    foldSpans: foldSpans(points, comparison.folds),
    directLabels: buildComparisonDirectLabels(points.at(-1), wealthDomain),
    metricVerdicts: comparison.metrics.map((metric) => ({
      ...metric,
      verdict: metricVerdict(metric),
    })),
  }
}

function periodReturn(start: number | null, end: number | null): number | null {
  return typeof start === 'number' && typeof end === 'number' && start !== 0
    ? end / start - 1
    : null
}

export function summarizeComparisonRange(
  model: ComparisonWorkspaceModel,
  firstIndex: number,
  secondIndex: number,
): ComparisonRangeSummary {
  const maximumIndex = Math.max(0, model.points.length - 1)
  const startIndex = Math.max(
    0,
    Math.min(maximumIndex, Math.min(firstIndex, secondIndex)),
  )
  const endIndex = Math.max(
    0,
    Math.min(maximumIndex, Math.max(firstIndex, secondIndex)),
  )
  const start = model.points[startIndex]
  const end = model.points[endIndex]
  const leftReturn = periodReturn(start?.left ?? null, end?.left ?? null)
  const rightReturn = periodReturn(start?.right ?? null, end?.right ?? null)
  const relativeReturn = leftReturn === null || rightReturn === null
    ? null
    : rightReturn - leftReturn
  const gaps = finite(
    model.points.slice(startIndex, endIndex + 1).map((point) => point.delta),
  ).map(Math.abs)
  const finalDelta = end?.delta ?? null
  const winner: ComparisonWinner = finalDelta === null
    ? 'unknown'
    : finalDelta > 0
      ? 'right'
      : finalDelta < 0
        ? 'left'
        : 'tie'

  return {
    startIndex,
    endIndex,
    startLabel: start?.label ?? '—',
    endLabel: end?.label ?? '—',
    leftReturn,
    rightReturn,
    relativeReturn,
    finalDelta,
    maximumGap: gaps.length ? Math.max(...gaps) : null,
    winner,
  }
}
