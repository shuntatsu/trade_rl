import { useMemo } from 'react'

import type { TrainingMetricPoint, TrainingMetricSeries } from '../data/types'

export type TrainingMetricWindow = 'all' | 256 | 512 | 1024

export interface TrainingMetricChartProps {
  series: TrainingMetricSeries | null
  selectedStep: number | null
  onSelectedStepChange: (step: number | null) => void
  windowSize: TrainingMetricWindow
}

function formatValue(value: number, unit: TrainingMetricSeries['unit']): string {
  if (unit === 'percent') return `${(value * 100).toLocaleString('ja-JP', { maximumFractionDigits: 3 })}%`
  if (unit === 'currency') return `${value.toLocaleString('ja-JP', { maximumFractionDigits: 2 })} USDT`
  if (unit === 'rate') return value.toExponential(3)
  return value.toLocaleString('ja-JP', { maximumFractionDigits: 5 })
}

function downsample(points: TrainingMetricPoint[], maximum = 1024): TrainingMetricPoint[] {
  if (points.length <= maximum) return points
  const bucketSize = Math.ceil(points.length / (maximum / 2))
  const selected: TrainingMetricPoint[] = []
  for (let start = 0; start < points.length; start += bucketSize) {
    const bucket = points.slice(start, start + bucketSize)
    let minimum = bucket[0]
    let maximumPoint = bucket[0]
    for (const point of bucket) {
      if (point.value < minimum.value) minimum = point
      if (point.value > maximumPoint.value) maximumPoint = point
    }
    selected.push(...(minimum.step <= maximumPoint.step ? [minimum, maximumPoint] : [maximumPoint, minimum]))
  }
  return selected.slice(0, maximum)
}

export function TrainingMetricChart({ series, selectedStep, onSelectedStepChange, windowSize }: TrainingMetricChartProps) {
  const source = series?.points ?? []
  const visible = windowSize === 'all' ? source : source.slice(-windowSize)
  const points = useMemo(() => downsample(visible), [visible])
  if (!series || points.length === 0) return <div className="training-chart-empty">未出力</div>

  const width = 720
  const height = 260
  const left = 64
  const right = 18
  const top = 18
  const bottom = 42
  const minStep = points[0].step
  const maxStep = points.at(-1)?.step ?? minStep
  const values = points.map((point) => point.value)
  const rawMin = Math.min(...values)
  const rawMax = Math.max(...values)
  const padding = rawMax === rawMin ? Math.max(Math.abs(rawMax) * 0.05, 1e-9) : (rawMax - rawMin) * 0.08
  const minValue = rawMin - padding
  const maxValue = rawMax + padding
  const x = (step: number) => left + (maxStep === minStep ? (width - left - right) / 2 : (step - minStep) / (maxStep - minStep) * (width - left - right))
  const y = (value: number) => top + (maxValue - value) / (maxValue - minValue) * (height - top - bottom)
  const path = points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${x(point.step)} ${y(point.value)}`).join(' ')
  const selectedIndex = selectedStep === null ? points.length - 1 : points.reduce((best, point, index) => Math.abs(point.step - selectedStep) < Math.abs(points[best].step - selectedStep) ? index : best, 0)
  const selected = points[selectedIndex]
  const chooseFromClientX = (clientX: number, target: SVGSVGElement) => {
    const rect = target.getBoundingClientRect()
    const local = rect.width > 0 ? (clientX - rect.left) / rect.width * width : 0
    const nearest = points.reduce((best, point, index) => Math.abs(x(point.step) - local) < Math.abs(x(points[best].step) - local) ? index : best, 0)
    onSelectedStepChange(points[nearest].step)
  }

  return (
    <div className="training-chart-shell">
      <div className="training-chart-summary">
        <strong>{series.displayName}</strong>
        <span>最新 {formatValue(source.at(-1)?.value ?? 0, series.unit)}</span>
        <span>min {formatValue(rawMin, series.unit)} / max {formatValue(rawMax, series.unit)}</span>
      </div>
      <svg
        className="training-metric-chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${series.displayName} 学習ステップ推移`}
        tabIndex={0}
        onPointerMove={(event) => chooseFromClientX(event.clientX, event.currentTarget)}
        onPointerLeave={() => onSelectedStepChange(null)}
        onKeyDown={(event) => {
          if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
          event.preventDefault()
          const next = Math.max(0, Math.min(points.length - 1, selectedIndex + (event.key === 'ArrowRight' ? 1 : -1)))
          onSelectedStepChange(points[next].step)
        }}
      >
        <title>{series.displayName}。横軸はglobal stepです。</title>
        {[0, 0.5, 1].map((ratio) => {
          const guideY = top + ratio * (height - top - bottom)
          const value = maxValue - ratio * (maxValue - minValue)
          return <g key={ratio}><line className="training-chart-guide" x1={left} x2={width - right} y1={guideY} y2={guideY} /><text x={left - 8} y={guideY + 4} textAnchor="end">{formatValue(value, series.unit)}</text></g>
        })}
        <text x={left} y={height - 12}>Step {minStep.toLocaleString('ja-JP')}</text>
        <text x={width - right} y={height - 12} textAnchor="end">Step {maxStep.toLocaleString('ja-JP')}</text>
        <path className="training-chart-line" d={path} />
        <line className="training-chart-crosshair" x1={x(selected.step)} x2={x(selected.step)} y1={top} y2={height - bottom} />
        <circle className="training-chart-point" cx={x(selected.step)} cy={y(selected.value)} r={4} />
      </svg>
      <div className="training-chart-tooltip">Step {selected.step.toLocaleString('ja-JP')} · {formatValue(selected.value, series.unit)}</div>
    </div>
  )
}
