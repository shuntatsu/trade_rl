import { Maximize2, MousePointer2, RotateCcw } from 'lucide-react'
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
  type WheelEvent,
} from 'react'

import type {
  ComparisonFoldSpan,
  ComparisonRangeSummary,
  ComparisonSeriesKey,
  ComparisonWorkspaceModel,
  NumericDomain,
} from './comparisonWorkspaceModel'
import {
  buildComparisonDirectLabels,
  summarizeComparisonRange,
} from './comparisonWorkspaceModel'

export interface ComparisonRangeSelection {
  start: number
  end: number
}

interface InteractiveComparisonWorkspaceProps {
  model: ComparisonWorkspaceModel
  committedPoint: number | null
  committedRange: ComparisonRangeSelection | null
  onCommitPoint: (index: number) => void
  onCommitRange: (range: ComparisonRangeSelection | null) => void
}

interface DragState {
  pointerId: number
  originX: number
  originIndex: number
  startVisible: number
  endVisible: number
  moved: boolean
}

const WIDTH = 1000
const HEIGHT = 560
const PLOT_LEFT = 58
const PLOT_RIGHT = 850
const PLOT_WIDTH = PLOT_RIGHT - PLOT_LEFT
const WEALTH_TOP = 28
const WEALTH_BOTTOM = 326
const DELTA_TOP = 370
const DELTA_BOTTOM = 492
const FOLD_TOP = 520
const FOLD_HEIGHT = 24
const FALLBACK_POINTER_ID = 1

const SERIES: Array<{ key: ComparisonSeriesKey; className: string }> = [
  {
    key: 'leftBaseline',
    className: 'comparison-line comparison-line--left-baseline',
  },
  {
    key: 'rightBaseline',
    className: 'comparison-line comparison-line--right-baseline',
  },
  { key: 'left', className: 'comparison-line comparison-line--left' },
  { key: 'right', className: 'comparison-line comparison-line--right' },
]

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value))
}

function finiteCoordinate(value: number | undefined, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function normalizedPointerId(value: number | undefined): number {
  return typeof value === 'number' && Number.isInteger(value)
    ? value
    : FALLBACK_POINTER_ID
}

function y(
  value: number,
  domain: NumericDomain,
  top: number,
  bottom: number,
): number {
  const span = domain.maximum - domain.minimum || 1
  return bottom - ((value - domain.minimum) / span) * (bottom - top)
}

function x(index: number, visibleStart: number, visibleEnd: number): number {
  const span = Math.max(1, visibleEnd - visibleStart)
  return PLOT_LEFT + ((index - visibleStart) / span) * PLOT_WIDTH
}

function pathFor(
  model: ComparisonWorkspaceModel,
  key: ComparisonSeriesKey | 'delta',
  visibleStart: number,
  visibleEnd: number,
  domain: NumericDomain,
  top: number,
  bottom: number,
): string {
  let path = ''
  let open = false
  for (const point of model.points) {
    if (point.index < visibleStart || point.index > visibleEnd) continue
    const value = key === 'delta' ? point.delta : point[key]
    if (typeof value !== 'number') {
      open = false
      continue
    }
    path += `${open ? ' L' : ' M'} ${x(point.index, visibleStart, visibleEnd).toFixed(2)} ${y(value, domain, top, bottom).toFixed(2)}`
    open = true
  }
  return path
}

function format(value: number | null, digits = 4): string {
  return value === null ? '—' : value.toFixed(digits)
}

function percent(value: number | null): string {
  return value === null
    ? '—'
    : `${value >= 0 ? '+' : ''}${(value * 100).toFixed(2)}%`
}

function selectedIndex(
  model: ComparisonWorkspaceModel,
  committedPoint: number | null,
  previewPoint: number | null,
): number {
  return clamp(
    previewPoint ?? committedPoint ?? model.points.length - 1,
    0,
    Math.max(0, model.points.length - 1),
  )
}

function summaryText(summary: ComparisonRangeSummary): string {
  const winner = summary.winner === 'right'
    ? 'RIGHT'
    : summary.winner === 'left'
      ? 'LEFT'
      : summary.winner === 'tie'
        ? 'TIE'
        : 'UNKNOWN'
  return `${summary.startLabel} → ${summary.endLabel} · relative ${percent(summary.relativeReturn)} · ${winner}`
}

function foldWidth(
  span: ComparisonFoldSpan,
  visibleStart: number,
  visibleEnd: number,
): { left: number; width: number } | null {
  const start = Math.max(span.startIndex, visibleStart)
  const end = Math.min(span.endIndex, visibleEnd)
  if (end < start) return null
  const left = x(start, visibleStart, visibleEnd)
  const right = x(end, visibleStart, visibleEnd)
  return { left, width: Math.max(4, right - left) }
}

export function InteractiveComparisonWorkspace({
  model,
  committedPoint,
  committedRange,
  onCommitPoint,
  onCommitRange,
}: InteractiveComparisonWorkspaceProps) {
  const maximumIndex = Math.max(0, model.points.length - 1)
  const [visible, setVisible] = useState<ComparisonRangeSelection>({
    start: 0,
    end: maximumIndex,
  })
  const [rangeMode, setRangeMode] = useState(false)
  const [previewPoint, setPreviewPoint] = useState<number | null>(null)
  const [draftRange, setDraftRange] = useState<ComparisonRangeSelection | null>(
    null,
  )
  const drag = useRef<DragState | null>(null)

  useEffect(() => {
    setVisible({ start: 0, end: maximumIndex })
    setPreviewPoint(null)
    setDraftRange(null)
    drag.current = null
  }, [maximumIndex, model.leftRunId, model.rightRunId])

  const requestedIndex = selectedIndex(model, committedPoint, previewPoint)
  const currentIndex = previewPoint !== null
    ? requestedIndex
    : committedPoint !== null
        && committedPoint >= visible.start
        && committedPoint <= visible.end
      ? committedPoint
      : visible.end
  const current = model.points[currentIndex]
  const effectiveRange = draftRange ?? committedRange
  const rangeSummary = useMemo(
    () => effectiveRange
      ? summarizeComparisonRange(model, effectiveRange.start, effectiveRange.end)
      : null,
    [effectiveRange, model],
  )
  const directLabels = useMemo(
    () => buildComparisonDirectLabels(
      model.points[visible.end],
      model.wealthDomain,
    ),
    [model.points, model.wealthDomain, visible.end],
  )

  const indexAtClientX = (
    rawClientX: number | undefined,
    target: HTMLElement,
  ): number => {
    const bounds = target.getBoundingClientRect()
    const left = finiteCoordinate(bounds.left)
    const renderedWidth = Math.max(1, finiteCoordinate(bounds.width, WIDTH))
    const fallbackClientX = left + (PLOT_LEFT / WIDTH) * renderedWidth
    const clientX = finiteCoordinate(rawClientX, fallbackClientX)
    const viewCoordinate = ((clientX - left) / renderedWidth) * WIDTH
    const plotCoordinate = clamp(viewCoordinate, PLOT_LEFT, PLOT_RIGHT)
    const ratio = (plotCoordinate - PLOT_LEFT) / PLOT_WIDTH
    return clamp(
      Math.round(
        visible.start + ratio * Math.max(1, visible.end - visible.start),
      ),
      visible.start,
      visible.end,
    )
  }

  const reset = () => {
    setVisible({ start: 0, end: maximumIndex })
    setPreviewPoint(null)
    setDraftRange(null)
  }

  const pointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || model.points.length === 0) return
    const pointerId = normalizedPointerId(event.pointerId)
    const originX = finiteCoordinate(event.clientX)
    const originIndex = indexAtClientX(event.clientX, event.currentTarget)
    try {
      event.currentTarget.setPointerCapture?.(pointerId)
    } catch {
      // Pointer capture can be unavailable in synthetic DOM environments.
    }
    drag.current = {
      pointerId,
      originX,
      originIndex,
      startVisible: visible.start,
      endVisible: visible.end,
      moved: false,
    }
    if (rangeMode) setDraftRange({ start: originIndex, end: originIndex })
  }

  const pointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const clientX = finiteCoordinate(event.clientX)
    const index = indexAtClientX(event.clientX, event.currentTarget)
    setPreviewPoint(index)
    const state = drag.current
    if (!state || state.pointerId !== normalizedPointerId(event.pointerId)) return
    if (Math.abs(clientX - state.originX) >= 4) state.moved = true
    if (rangeMode) {
      setDraftRange({ start: state.originIndex, end: index })
      return
    }
    if (!state.moved) return
    const bounds = event.currentTarget.getBoundingClientRect()
    const renderedWidth = Math.max(1, finiteCoordinate(bounds.width, WIDTH))
    const renderedPlotWidth = renderedWidth * (PLOT_WIDTH / WIDTH)
    const pointsPerPixel = Math.max(1, state.endVisible - state.startVisible)
      / Math.max(1, renderedPlotWidth)
    const shift = Math.round((state.originX - clientX) * pointsPerPixel)
    const span = state.endVisible - state.startVisible
    const start = clamp(
      state.startVisible + shift,
      0,
      Math.max(0, maximumIndex - span),
    )
    setVisible({ start, end: start + span })
  }

  const pointerUp = (event: PointerEvent<HTMLDivElement>) => {
    const state = drag.current
    const pointerId = normalizedPointerId(event.pointerId)
    if (!state || state.pointerId !== pointerId) return
    try {
      event.currentTarget.releasePointerCapture?.(pointerId)
    } catch {
      // Pointer capture can be unavailable in synthetic DOM environments.
    }
    const index = indexAtClientX(event.clientX, event.currentTarget)
    if (rangeMode) {
      const selection = {
        start: Math.min(state.originIndex, index),
        end: Math.max(state.originIndex, index),
      }
      setDraftRange(null)
      onCommitRange(selection)
    } else if (!state.moved) {
      onCommitPoint(index)
    }
    drag.current = null
  }

  const wheel = (event: WheelEvent<HTMLDivElement>) => {
    if (model.points.length <= 2) return
    event.preventDefault()
    const anchor = indexAtClientX(event.clientX, event.currentTarget)
    const currentSpan = Math.max(1, visible.end - visible.start)
    const minimumSpan = Math.min(2, maximumIndex)
    const nextSpan = clamp(
      Math.round(currentSpan * (event.deltaY < 0 ? 0.75 : 1.3)),
      minimumSpan,
      maximumIndex,
    )
    const ratio = currentSpan === 0
      ? 0.5
      : (anchor - visible.start) / currentSpan
    const start = clamp(
      Math.round(anchor - ratio * nextSpan),
      0,
      Math.max(0, maximumIndex - nextSpan),
    )
    setVisible({ start, end: start + nextSpan })
  }

  const keyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End', 'Escape'].includes(event.key)) {
      return
    }
    event.preventDefault()
    if (event.key === 'Escape') {
      onCommitRange(null)
      setDraftRange(null)
      setPreviewPoint(null)
      return
    }
    const base = committedPoint ?? maximumIndex
    const next = event.key === 'Home'
      ? 0
      : event.key === 'End'
        ? maximumIndex
        : clamp(
            base + (event.key === 'ArrowRight' ? 1 : -1),
            0,
            maximumIndex,
          )
    onCommitPoint(next)
  }

  const selectFold = (span: ComparisonFoldSpan) => {
    setVisible({
      start: span.startIndex,
      end: Math.min(
        maximumIndex,
        Math.max(span.startIndex + 1, span.endIndex),
      ),
    })
    onCommitRange({ start: span.startIndex, end: span.endIndex })
  }

  if (!model.points.length) {
    return (
      <div className="comparison-workspace-empty">
        比較可能なwealth系列がありません。
      </div>
    )
  }

  const rangeStart = effectiveRange
    ? Math.max(visible.start, Math.min(effectiveRange.start, effectiveRange.end))
    : null
  const rangeEnd = effectiveRange
    ? Math.min(visible.end, Math.max(effectiveRange.start, effectiveRange.end))
    : null
  const selectionVisible = rangeStart !== null
    && rangeEnd !== null
    && rangeStart <= rangeEnd
  const selectionLeft = selectionVisible
    ? x(rangeStart, visible.start, visible.end)
    : null
  const selectionRight = selectionVisible
    ? x(rangeEnd, visible.start, visible.end)
    : null

  return (
    <section
      className="comparison-workspace-shell"
      aria-label="Interactive run comparison"
    >
      <div className="comparison-workspace-toolbar">
        <div className="comparison-workspace-key" aria-label="series key">
          <span className="comparison-key comparison-key--left">Left</span>
          <span className="comparison-key comparison-key--right">Right</span>
          <span className="comparison-key comparison-key--baseline">
            Dashed = baseline
          </span>
        </div>
        <div className="comparison-workspace-actions">
          <button
            type="button"
            aria-pressed={rangeMode}
            onClick={() => setRangeMode((currentMode) => !currentMode)}
          >
            <Maximize2 size={14} aria-hidden="true" />Range
          </button>
          <button type="button" aria-label="Reset view" onClick={reset}>
            <RotateCcw size={14} aria-hidden="true" />Reset
          </button>
        </div>
      </div>

      <div
        className={`comparison-chart-surface${rangeMode ? ' comparison-chart-surface--range' : ''}`}
        role="application"
        aria-label="Run comparison chart"
        tabIndex={0}
        data-visible-range={`${visible.start}:${visible.end}`}
        onPointerDown={pointerDown}
        onPointerMove={pointerMove}
        onPointerLeave={() => {
          if (!drag.current) setPreviewPoint(null)
        }}
        onPointerUp={pointerUp}
        onPointerCancel={pointerUp}
        onWheel={wheel}
        onDoubleClick={reset}
        onKeyDown={keyDown}
      >
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <rect
            x={PLOT_LEFT}
            y={WEALTH_TOP}
            width={PLOT_WIDTH}
            height={WEALTH_BOTTOM - WEALTH_TOP}
            className="comparison-pane-background"
          />
          <rect
            x={PLOT_LEFT}
            y={DELTA_TOP}
            width={PLOT_WIDTH}
            height={DELTA_BOTTOM - DELTA_TOP}
            className="comparison-pane-background"
          />
          <g data-pane="wealth">
            <line
              x1={PLOT_LEFT}
              y1={y(1, model.wealthDomain, WEALTH_TOP, WEALTH_BOTTOM)}
              x2={PLOT_RIGHT}
              y2={y(1, model.wealthDomain, WEALTH_TOP, WEALTH_BOTTOM)}
              className="comparison-reference"
            />
            {SERIES.map((series) => (
              <path
                key={series.key}
                d={pathFor(
                  model,
                  series.key,
                  visible.start,
                  visible.end,
                  model.wealthDomain,
                  WEALTH_TOP,
                  WEALTH_BOTTOM,
                )}
                className={series.className}
                data-series={series.key}
              />
            ))}
          </g>
          <g data-pane="delta">
            <line
              x1={PLOT_LEFT}
              y1={y(0, model.deltaDomain, DELTA_TOP, DELTA_BOTTOM)}
              x2={PLOT_RIGHT}
              y2={y(0, model.deltaDomain, DELTA_TOP, DELTA_BOTTOM)}
              className="comparison-reference comparison-reference--zero"
            />
            <path
              d={pathFor(
                model,
                'delta',
                visible.start,
                visible.end,
                model.deltaDomain,
                DELTA_TOP,
                DELTA_BOTTOM,
              )}
              className="comparison-delta-line"
              data-series="delta"
            />
          </g>
          {selectionLeft !== null && selectionRight !== null ? (
            <rect
              x={Math.min(selectionLeft, selectionRight)}
              y={WEALTH_TOP}
              width={Math.max(2, Math.abs(selectionRight - selectionLeft))}
              height={DELTA_BOTTOM - WEALTH_TOP}
              className="comparison-range-band"
            />
          ) : null}
          <line
            x1={x(currentIndex, visible.start, visible.end)}
            y1={WEALTH_TOP}
            x2={x(currentIndex, visible.start, visible.end)}
            y2={DELTA_BOTTOM}
            className="comparison-crosshair"
          />
          <g data-pane="folds">
            {model.foldSpans.map((span) => {
              const geometry = foldWidth(span, visible.start, visible.end)
              return geometry ? (
                <rect
                  key={span.foldIndex}
                  x={geometry.left}
                  y={FOLD_TOP}
                  width={geometry.width}
                  height={FOLD_HEIGHT}
                  className={`comparison-fold-band comparison-fold-band--${span.foldIndex % 2}`}
                />
              ) : null
            })}
          </g>
        </svg>

        <div
          className="comparison-pane-label comparison-pane-label--wealth"
          aria-label="Cumulative wealth pane"
        >
          Cumulative wealth
        </div>
        <div
          className="comparison-pane-label comparison-pane-label--delta"
          aria-label="Right minus Left pane"
        >
          Right − Left
        </div>
        <div className="comparison-axis comparison-axis--wealth">
          <span>{model.wealthDomain.maximum.toFixed(3)}</span>
          <span>1.000</span>
          <span>{model.wealthDomain.minimum.toFixed(3)}</span>
        </div>
        <div className="comparison-axis comparison-axis--delta">
          <span>{model.deltaDomain.maximum.toFixed(3)}</span>
          <span>0.000</span>
          <span>{model.deltaDomain.minimum.toFixed(3)}</span>
        </div>
        <div
          className="comparison-direct-labels"
          aria-label="visible endpoint series values"
        >
          {directLabels.map((label) => (
            <span
              key={label.key}
              className={`comparison-direct-label comparison-direct-label--${label.key}`}
              style={{
                top: `${(
                  (WEALTH_TOP
                    + label.position * (WEALTH_BOTTOM - WEALTH_TOP))
                  / HEIGHT
                ) * 100}%`,
              }}
            >
              {label.label} {label.value.toFixed(4)}
            </span>
          ))}
        </div>
        <div className="comparison-point-readout" aria-live="polite">
          <MousePointer2 size={14} aria-hidden="true" />
          <strong>Index {current?.label ?? '—'}</strong>
          <span>Left {format(current?.left ?? null)}</span>
          <span>Right {format(current?.right ?? null)}</span>
          <span>Δ {format(current?.delta ?? null)}</span>
        </div>
        {rangeSummary ? (
          <div className="comparison-range-readout" aria-live="polite">
            {summaryText(rangeSummary)}
          </div>
        ) : null}
      </div>

      <div className="comparison-fold-strip" aria-label="Fold comparison strip">
        {model.foldSpans.map((span) => (
          <button
            key={span.foldIndex}
            type="button"
            onClick={() => selectFold(span)}
          >
            <strong>{span.label}</strong>
            <span>L {percent(span.leftSelectedReturn)}</span>
            <span>R {percent(span.rightSelectedReturn)}</span>
          </button>
        ))}
      </div>

      <table className="sr-only">
        <caption>Run comparison values by sealed evaluation index</caption>
        <thead>
          <tr>
            <th>Index</th>
            <th>Fold</th>
            <th>Left</th>
            <th>Right</th>
            <th>Left baseline</th>
            <th>Right baseline</th>
            <th>Right minus Left</th>
          </tr>
        </thead>
        <tbody>
          {model.points.map((point) => (
            <tr key={`${point.index}-${point.label}`}>
              <th>{point.label}</th>
              <td>{point.foldIndex ?? 'start'}</td>
              <td>{format(point.left)}</td>
              <td>{format(point.right)}</td>
              <td>{format(point.leftBaseline)}</td>
              <td>{format(point.rightBaseline)}</td>
              <td>{format(point.delta)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}
