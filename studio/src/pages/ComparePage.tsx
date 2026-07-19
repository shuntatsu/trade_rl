import { GitCompareArrows, RefreshCw, TrendingUp } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import { studioApi, type StudioApi } from '../api/studioApi'
import type { ComparisonSeriesPoint, RunComparison, RunSummary } from '../data/types'

type CompareApi = Pick<StudioApi, 'loadRuns' | 'loadRunComparison'>

interface ComparePageProps {
  api?: CompareApi
}

function percentage(value: number | null): string {
  return value === null ? '—' : `${(value * 100).toFixed(2)}%`
}

function metric(value: number | null, key: string): string {
  if (value === null) return '—'
  return key.includes('return') || key.includes('drawdown') || key.includes('cost') || key.includes('pnl')
    ? percentage(value)
    : value.toFixed(3)
}

function points(values: ComparisonSeriesPoint[], key: keyof ComparisonSeriesPoint): string {
  const numeric = values.map((item) => item[key]).filter((value): value is number => typeof value === 'number')
  if (!numeric.length) return ''
  const minimum = Math.min(...numeric)
  const maximum = Math.max(...numeric)
  const range = maximum - minimum || 1
  return values
    .map((item, index) => {
      const value = item[key]
      if (typeof value !== 'number') return null
      const x = values.length <= 1 ? 0 : (index / (values.length - 1)) * 100
      const y = 92 - ((value - minimum) / range) * 82
      return `${x.toFixed(2)},${y.toFixed(2)}`
    })
    .filter(Boolean)
    .join(' ')
}

function ComparisonChart({ values }: { values: ComparisonSeriesPoint[] }) {
  if (!values.length) return <div className="runtime-empty">比較可能な資産曲線がありません。</div>
  return (
    <svg className="audit-chart" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="comparison wealth chart" role="img">
      <line x1="0" y1="92" x2="100" y2="92" className="audit-chart__axis" />
      <polyline points={points(values, 'leftBaseline')} className="audit-chart__line audit-chart__line--left-baseline" />
      <polyline points={points(values, 'rightBaseline')} className="audit-chart__line audit-chart__line--right-baseline" />
      <polyline points={points(values, 'left')} className="audit-chart__line audit-chart__line--left" />
      <polyline points={points(values, 'right')} className="audit-chart__line audit-chart__line--right" />
    </svg>
  )
}

export function ComparePage({ api = studioApi }: ComparePageProps) {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [leftRunId, setLeftRunId] = useState('')
  const [rightRunId, setRightRunId] = useState('')
  const [comparison, setComparison] = useState<RunComparison | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  async function loadComparison(left: string, right: string) {
    if (!left || !right) return
    setLoading(true)
    setError(null)
    try {
      setComparison(await api.loadRunComparison(left, right))
    } catch (reason) {
      setComparison(null)
      setError(reason instanceof Error ? reason.message : 'run比較を取得できませんでした。')
    } finally {
      setLoading(false)
    }
  }

  async function refresh() {
    setLoading(true)
    setError(null)
    try {
      const response = await api.loadRuns()
      const valid = response.items.filter((item) => item.status === 'VALID')
      setRuns(valid)
      const left = valid.some((item) => item.id === leftRunId) ? leftRunId : valid[0]?.id ?? ''
      const right = valid.some((item) => item.id === rightRunId)
        ? rightRunId
        : valid[1]?.id ?? valid[0]?.id ?? ''
      setLeftRunId(left)
      setRightRunId(right)
      await loadComparison(left, right)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'run一覧を取得できませんでした。')
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
  }, [])

  const metricRows = comparison?.metrics ?? []
  const deltaClass = useMemo(() => (value: number | null) => {
    if (value === null || value === 0) return ''
    return value > 0 ? ' audit-positive' : ' audit-negative'
  }, [])

  return (
    <section className="runtime-page" aria-labelledby="compare-title">
      <header className="runtime-toolbar">
        <div>
          <span className="runtime-eyebrow">IMMUTABLE RUN REVIEW</span>
          <h1 id="compare-title">比較</h1>
          <p>検証済みrunの指標、設定差、fold安定性、累積wealthを同じ尺度で確認します。</p>
        </div>
        <div className="runtime-toolbar__actions">
          <span className="runtime-danger">NO-GO</span>
          <button type="button" className="runtime-button runtime-button--quiet" onClick={() => void refresh()} disabled={loading}>
            <RefreshCw size={14} aria-hidden="true" />再読込
          </button>
        </div>
      </header>

      <div className="audit-compare-grid">
        <section className="runtime-pane audit-selector-pane">
          <div className="runtime-pane__header"><strong>Run pair</strong><span>{runs.length} validated</span></div>
          <div className="audit-selectors">
            <label>Left run
              <select aria-label="Left run" value={leftRunId} onChange={(event) => {
                const value = event.target.value
                setLeftRunId(value)
                void loadComparison(value, rightRunId)
              }}>
                {runs.map((run) => <option key={run.id} value={run.id}>{run.id} · {run.algorithm}</option>)}
              </select>
            </label>
            <GitCompareArrows size={18} aria-hidden="true" />
            <label>Right run
              <select aria-label="Right run" value={rightRunId} onChange={(event) => {
                const value = event.target.value
                setRightRunId(value)
                void loadComparison(leftRunId, value)
              }}>
                {runs.map((run) => <option key={run.id} value={run.id}>{run.id} · {run.algorithm}</option>)}
              </select>
            </label>
          </div>
        </section>

        <section className="runtime-pane audit-metrics-pane">
          <div className="runtime-pane__header"><strong>Metric delta</strong><span>right − left</span></div>
          <div className="audit-metrics" aria-busy={loading}>
            {error ? <div className="runtime-error">{error}</div> : null}
            {!error && !metricRows.length && !loading ? <div className="runtime-empty">評価指標がありません。</div> : null}
            {metricRows.map((item) => (
              <article key={item.key}>
                <span>{item.label}</span>
                <div><strong>{metric(item.leftValue, item.key)}</strong><TrendingUp size={12} aria-hidden="true" /><strong>{metric(item.rightValue, item.key)}</strong></div>
                <small className={deltaClass(item.delta)}>{item.delta === null ? 'delta —' : `delta ${metric(item.delta, item.key)}`}</small>
              </article>
            ))}
          </div>
        </section>

        <section className="runtime-pane audit-chart-pane">
          <div className="runtime-pane__header"><strong>Cumulative wealth</strong><span>selected + baseline</span></div>
          <div className="audit-chart-wrap">
            <ComparisonChart values={comparison?.wealth ?? []} />
            <div className="audit-chart-legend"><span>Left</span><span>Right</span><span>Baselines</span></div>
          </div>
        </section>

        <section className="runtime-pane audit-config-pane">
          <div className="runtime-pane__header"><strong>Configuration diff</strong><span>{comparison?.configDifferences.length ?? 0}</span></div>
          <div className="audit-table-wrap">
            <table className="audit-table"><thead><tr><th>Path</th><th>Left</th><th>Right</th></tr></thead><tbody>
              {(comparison?.configDifferences ?? []).map((item) => <tr key={item.path}><td><code>{item.path}</code></td><td>{item.left ?? '—'}</td><td>{item.right ?? '—'}</td></tr>)}
            </tbody></table>
            {comparison && comparison.configDifferences.length === 0 ? <div className="runtime-empty">設定差はありません。</div> : null}
          </div>
        </section>

        <section className="runtime-pane audit-fold-pane">
          <div className="runtime-pane__header"><strong>Fold stability</strong><span>{comparison?.folds.length ?? 0}</span></div>
          <div className="audit-fold-list">
            {(comparison?.folds ?? []).map((fold) => (
              <article key={fold.label}><strong>{fold.label}</strong><span>L {percentage(fold.leftSelectedReturn)}</span><span>LB {percentage(fold.leftBaselineReturn)}</span><span>R {percentage(fold.rightSelectedReturn)}</span><span>RB {percentage(fold.rightBaselineReturn)}</span></article>
            ))}
            {comparison && comparison.folds.length === 0 ? <div className="runtime-empty">fold比較がありません。</div> : null}
          </div>
        </section>
      </div>
    </section>
  )
}
