import { X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import type { RunComparison } from '../data/types'
import type {
  ComparisonRangeSummary,
  ComparisonWorkspaceModel,
} from './comparisonWorkspaceModel'

export type ComparisonInspectorTab = 'selection' | 'metrics' | 'config'

interface ComparisonInspectorSheetProps {
  open: boolean
  comparison: RunComparison
  model: ComparisonWorkspaceModel
  selectedPoint: number | null
  rangeSummary: ComparisonRangeSummary | null
  onClose: () => void
}

function value(value: number | null, percent = false): string {
  if (value === null) return '—'
  return percent
    ? `${value >= 0 ? '+' : ''}${(value * 100).toFixed(2)}%`
    : value.toFixed(4)
}

export function ComparisonInspectorSheet({
  open,
  comparison,
  model,
  selectedPoint,
  rangeSummary,
  onClose,
}: ComparisonInspectorSheetProps) {
  const [tab, setTab] = useState<ComparisonInspectorTab>('selection')
  const closeRef = useRef<HTMLButtonElement | null>(null)

  useEffect(() => {
    if (!open) return
    setTab('selection')
    window.setTimeout(() => closeRef.current?.focus(), 0)
    const handle = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        onClose()
      }
    }
    window.addEventListener('keydown', handle)
    return () => window.removeEventListener('keydown', handle)
  }, [onClose, open])

  if (!open) return null
  const point = selectedPoint === null
    ? model.points.at(-1) ?? null
    : model.points[selectedPoint] ?? null

  return (
    <aside
      className="comparison-inspector"
      role="dialog"
      aria-labelledby="comparison-inspector-title"
    >
      <header>
        <div>
          <span>READ-ONLY DETAIL</span>
          <h2 id="comparison-inspector-title">Comparison inspector</h2>
        </div>
        <button
          ref={closeRef}
          type="button"
          aria-label="Comparison inspectorを閉じる"
          onClick={onClose}
        >
          <X size={17} aria-hidden="true" />
        </button>
      </header>
      <div
        className="comparison-inspector-tabs"
        role="tablist"
        aria-label="Comparison inspector tabs"
      >
        {(['selection', 'metrics', 'config'] as const).map((item) => (
          <button
            key={item}
            type="button"
            role="tab"
            aria-selected={tab === item}
            onClick={() => setTab(item)}
          >
            {item}
          </button>
        ))}
      </div>

      {tab === 'selection' ? (
        <section role="tabpanel" aria-label="Selection">
          {rangeSummary ? (
            <dl className="comparison-inspector-grid">
              <div><dt>Evaluation range</dt><dd>{rangeSummary.startLabel} → {rangeSummary.endLabel}</dd></div>
              <div><dt>Left return</dt><dd>{value(rangeSummary.leftReturn, true)}</dd></div>
              <div><dt>Right return</dt><dd>{value(rangeSummary.rightReturn, true)}</dd></div>
              <div><dt>Relative return</dt><dd>{value(rangeSummary.relativeReturn, true)}</dd></div>
              <div><dt>Maximum wealth gap</dt><dd>{value(rangeSummary.maximumGap)}</dd></div>
              <div><dt>End winner</dt><dd>{rangeSummary.winner.toUpperCase()}</dd></div>
            </dl>
          ) : point ? (
            <dl className="comparison-inspector-grid">
              <div><dt>Evaluation index</dt><dd>{point.label}</dd></div>
              <div><dt>Fold</dt><dd>{point.foldIndex === null ? 'Start' : `Fold ${point.foldIndex + 1}`}</dd></div>
              <div><dt>Left wealth</dt><dd>{value(point.left)}</dd></div>
              <div><dt>Right wealth</dt><dd>{value(point.right)}</dd></div>
              <div><dt>Right − Left</dt><dd>{value(point.delta)}</dd></div>
              <div><dt>Production</dt><dd>NO-GO</dd></div>
            </dl>
          ) : (
            <div className="runtime-empty">選択可能な比較点がありません。</div>
          )}
        </section>
      ) : null}

      {tab === 'metrics' ? (
        <section
          role="tabpanel"
          aria-label="Metrics"
          className="comparison-inspector-table-wrap"
        >
          <table className="audit-table">
            <thead><tr><th>Metric</th><th>Left</th><th>Right</th><th>Verdict</th></tr></thead>
            <tbody>
              {model.metricVerdicts.map((metric) => (
                <tr key={metric.key}>
                  <td>{metric.label}</td>
                  <td>{value(metric.leftValue, metric.key.includes('return') || metric.key.includes('drawdown') || metric.key.includes('cost'))}</td>
                  <td>{value(metric.rightValue, metric.key.includes('return') || metric.key.includes('drawdown') || metric.key.includes('cost'))}</td>
                  <td data-verdict={metric.verdict}>{metric.verdict}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : null}

      {tab === 'config' ? (
        <section
          role="tabpanel"
          aria-label="Config"
          className="comparison-inspector-table-wrap"
        >
          <table className="audit-table">
            <thead><tr><th>Path</th><th>Left</th><th>Right</th></tr></thead>
            <tbody>
              {comparison.configDifferences.map((item) => (
                <tr key={item.path}>
                  <td><code>{item.path}</code></td>
                  <td>{item.left ?? '—'}</td>
                  <td>{item.right ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!comparison.configDifferences.length ? (
            <div className="runtime-empty">設定差はありません。</div>
          ) : null}
        </section>
      ) : null}
    </aside>
  )
}
