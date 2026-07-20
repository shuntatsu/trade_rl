import { GitCompareArrows, RefreshCw } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { studioApi, type StudioApi } from '../api/studioApi'
import type { ComparisonMetric, ComparisonSeriesPoint, FoldComparison, RunComparison, RunSummary } from '../data/types'
import { readParam, replaceParams } from '../state/urlState'

type CompareApi = Pick<StudioApi, 'loadRuns' | 'loadRunComparison'>

interface ComparePageProps {
  api?: CompareApi
}

type SeriesKey = 'left' | 'right' | 'leftBaseline' | 'rightBaseline'
const SERIES: SeriesKey[] = ['leftBaseline', 'rightBaseline', 'left', 'right']

function percentage(value: number | null): string {
  return value === null ? '—' : `${(value * 100).toFixed(2)}%`
}

function metric(value: number | null, key: string): string {
  if (value === null) return '—'
  return key.includes('return') || key.includes('drawdown') || key.includes('cost') || key.includes('pnl')
    ? percentage(value)
    : value.toFixed(3)
}

function signedMetric(value: number | null, key: string): string {
  if (value === null) return '—'
  const rendered = metric(Math.abs(value), key)
  if (value === 0) return rendered
  return `${value > 0 ? '+' : '−'}${rendered}`
}

function sharedDomain(values: ComparisonSeriesPoint[]): { minimum: number; maximum: number } {
  const numeric = values.flatMap((item) => SERIES.map((key) => item[key])).filter((value): value is number => typeof value === 'number')
  if (!numeric.length) return { minimum: 0, maximum: 1 }
  const minimumValue = Math.min(1, ...numeric)
  const maximumValue = Math.max(1, ...numeric)
  const range = maximumValue - minimumValue || Math.max(Math.abs(maximumValue), 0.01)
  const padding = range * 0.08
  return { minimum: minimumValue - padding, maximum: maximumValue + padding }
}

function seriesPoints(
  values: ComparisonSeriesPoint[],
  key: SeriesKey,
  domain: { minimum: number; maximum: number },
): string {
  const range = domain.maximum - domain.minimum || 1
  return values
    .map((item, index) => {
      const value = item[key]
      if (typeof value !== 'number') return null
      const x = values.length <= 1 ? 8 : 8 + (index / (values.length - 1)) * 84
      const y = 90 - ((value - domain.minimum) / range) * 78
      return `${x.toFixed(2)},${y.toFixed(2)}`
    })
    .filter(Boolean)
    .join(' ')
}

function yCoordinate(value: number, domain: { minimum: number; maximum: number }): number {
  const range = domain.maximum - domain.minimum || 1
  return 90 - ((value - domain.minimum) / range) * 78
}

function ComparisonChart({ values }: { values: ComparisonSeriesPoint[] }) {
  if (!values.length) return <div className="runtime-empty">比較可能な資産曲線がありません。</div>
  const domain = sharedDomain(values)
  const referenceY = yCoordinate(1, domain)
  const final = values.at(-1)
  return (
    <div className="audit-chart-stack">
      <svg className="audit-chart" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="comparison wealth chart" role="img">
        <title>4系列を同一スケールで比較した累積wealth</title>
        <desc>実線は各run、破線は各runのbaselineです。基準wealth 1.00を水平線で示します。</desc>
        <line x1="8" y1={referenceY} x2="92" y2={referenceY} className="audit-chart__reference" />
        {SERIES.map((key) => (
          <polyline
            key={key}
            data-series={key}
            data-final={typeof final?.[key] === 'number' ? final[key] : undefined}
            points={seriesPoints(values, key, domain)}
            className={`audit-chart__line audit-chart__line--${key.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`)}`}
          />
        ))}
      </svg>
      <div className="audit-chart-scale" aria-hidden="true">
        <span>{domain.maximum.toFixed(2)}</span><span>1.00</span><span>{domain.minimum.toFixed(2)}</span>
      </div>
      <div className="audit-chart-legend" aria-label="chart legend">
        <span className="audit-chart-key audit-chart-key--left">Left run</span>
        <span className="audit-chart-key audit-chart-key--right">Right run</span>
        <span className="audit-chart-key audit-chart-key--baseline">Dashed = baseline</span>
      </div>
      <table className="sr-only">
        <caption>Final cumulative wealth values</caption>
        <tbody>
          {SERIES.map((key) => <tr key={key}><th>{key}</th><td>{typeof final?.[key] === 'number' ? final[key].toFixed(4) : 'missing'}</td></tr>)}
        </tbody>
      </table>
    </div>
  )
}

function metricOutcome(item: ComparisonMetric): { label: string; className: string } {
  if (item.delta === null || item.delta === 0 || item.preference === 'neutral') return { label: '同等', className: 'audit-neutral' }
  const improved = item.preference === 'higher' ? item.delta > 0 : item.delta < 0
  return improved ? { label: '改善', className: 'audit-positive' } : { label: '悪化', className: 'audit-negative' }
}

function FoldBar({ label, value, maximum }: { label: string; value: number | null; maximum: number }) {
  const normalized = value === null ? 0 : Math.min(Math.abs(value) / maximum, 1) * 50
  const direction = value !== null && value < 0 ? 'negative' : 'positive'
  return (
    <div className="audit-fold-value" aria-label={`${label} ${percentage(value)}`}>
      <span>{label}</span>
      <div className="audit-diverging-track" aria-hidden="true">
        <i className={`audit-diverging-bar audit-diverging-bar--${direction}`} style={{ width: `${normalized}%` }} />
      </div>
      <strong>{percentage(value)}</strong>
    </div>
  )
}

function foldMaximum(folds: FoldComparison[]): number {
  const values = folds.flatMap((fold) => [fold.leftSelectedReturn, fold.leftBaselineReturn, fold.rightSelectedReturn, fold.rightBaselineReturn])
    .filter((value): value is number => typeof value === 'number')
    .map(Math.abs)
  return Math.max(...values, 0.01)
}

export function ComparePage({ api = studioApi }: ComparePageProps) {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [leftRunId, setLeftRunId] = useState(() => readParam(window.location.search, 'left') ?? '')
  const [rightRunId, setRightRunId] = useState(() => readParam(window.location.search, 'right') ?? '')
  const [comparison, setComparison] = useState<RunComparison | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const requestSequence = useRef(0)

  async function loadComparison(left: string, right: string) {
    if (!left || !right) return
    const sequence = ++requestSequence.current
    setLoading(true)
    setError(null)
    replaceParams({ left, right })
    try {
      const value = await api.loadRunComparison(left, right)
      if (sequence === requestSequence.current) setComparison(value)
    } catch (reason) {
      if (sequence === requestSequence.current) {
        setComparison(null)
        setError(reason instanceof Error ? reason.message : 'run比較を取得できませんでした。')
      }
    } finally {
      if (sequence === requestSequence.current) setLoading(false)
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
      const right = valid.some((item) => item.id === rightRunId) ? rightRunId : valid[1]?.id ?? valid[0]?.id ?? ''
      setLeftRunId(left)
      setRightRunId(right)
      await loadComparison(left, right)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'run一覧を取得できませんでした。')
      setLoading(false)
    }
  }

  useEffect(() => { void refresh() }, [])

  const metricRows = comparison?.metrics ?? []
  const folds = comparison?.folds ?? []
  const maximumFoldReturn = useMemo(() => foldMaximum(folds), [folds])

  return (
    <section className="runtime-page" aria-labelledby="compare-title">
      <header className="runtime-toolbar">
        <div><span className="runtime-eyebrow">IMMUTABLE RUN REVIEW</span><h1 id="compare-title">比較</h1><p>右runが左runより改善したかを、同一尺度・同一方向の判定で確認します。</p></div>
        <div className="runtime-toolbar__actions"><span className="runtime-danger">NO-GO</span><button type="button" className="runtime-button runtime-button--quiet" onClick={() => void refresh()} disabled={loading}><RefreshCw size={14} aria-hidden="true" />再読込</button></div>
      </header>

      <div className="audit-compare-grid">
        <section className="runtime-pane audit-selector-pane">
          <div className="runtime-pane__header"><strong>Run pair</strong><span>{runs.length} validated</span></div>
          <div className="audit-selectors">
            <label>Left run<select aria-label="Left run" value={leftRunId} onChange={(event) => { const value = event.target.value; setLeftRunId(value); void loadComparison(value, rightRunId) }}>{runs.map((run) => <option key={run.id} value={run.id}>{run.runId} · {run.algorithm}</option>)}</select></label>
            <GitCompareArrows size={18} aria-hidden="true" />
            <label>Right run<select aria-label="Right run" value={rightRunId} onChange={(event) => { const value = event.target.value; setRightRunId(value); void loadComparison(leftRunId, value) }}>{runs.map((run) => <option key={run.id} value={run.id}>{run.runId} · {run.algorithm}</option>)}</select></label>
          </div>
          {comparison ? <div className={`audit-eligibility audit-eligibility--${comparison.eligibility.status.toLowerCase().replaceAll('_', '-')}`} aria-label="comparison eligibility"><strong>{comparison.eligibility.status}</strong><span>{comparison.leftRunId} ↔ {comparison.rightRunId}</span><small>{comparison.eligibility.reasons.join(' · ') || 'sealed evaluation identities are compatible'}</small></div> : null}
        </section>

        <section className="runtime-pane audit-metrics-pane">
          <div className="runtime-pane__header"><strong>Decision metrics</strong><span>right − left · preference aware</span></div>
          <div className="audit-metrics" aria-busy={loading}>
            {error ? <div className="runtime-error">{error}</div> : null}
            {!error && !metricRows.length && !loading ? <div className="runtime-empty">評価指標がありません。</div> : null}
            {metricRows.map((item) => {
              const outcome = metricOutcome(item)
              return <article key={item.key}>
                <span>{item.label}</span>
                <div className="audit-metric-values"><span><small>LEFT</small><strong>{metric(item.leftValue, item.key)}</strong></span><span><small>RIGHT</small><strong>{metric(item.rightValue, item.key)}</strong></span></div>
                <div className={`audit-metric-outcome ${outcome.className}`}><strong>{outcome.label}</strong><small>{signedMetric(item.delta, item.key)}</small></div>
              </article>
            })}
          </div>
        </section>

        <section className="runtime-pane audit-chart-pane">
          <div className="runtime-pane__header"><strong>Cumulative wealth</strong><span>shared y-scale</span></div>
          <div className="audit-chart-wrap"><ComparisonChart values={comparison?.wealth ?? []} /></div>
        </section>

        <section className="runtime-pane audit-config-pane">
          <div className="runtime-pane__header"><strong>Configuration diff</strong><span>{comparison?.configDifferences.length ?? 0}</span></div>
          <div className="audit-table-wrap"><table className="audit-table"><thead><tr><th>Path</th><th>Left</th><th>Right</th></tr></thead><tbody>{(comparison?.configDifferences ?? []).map((item) => <tr key={item.path}><td><code>{item.path}</code></td><td>{item.left ?? '—'}</td><td>{item.right ?? '—'}</td></tr>)}</tbody></table>{comparison && comparison.configDifferences.length === 0 ? <div className="runtime-empty">設定差はありません。</div> : null}</div>
        </section>

        <section className="runtime-pane audit-fold-pane">
          <div className="runtime-pane__header"><strong>Fold returns</strong><span>zero-centred</span></div>
          <div className="audit-fold-list">
            {folds.map((fold) => <article key={fold.label}><strong>{fold.label}</strong><FoldBar label="L" value={fold.leftSelectedReturn} maximum={maximumFoldReturn} /><FoldBar label="LB" value={fold.leftBaselineReturn} maximum={maximumFoldReturn} /><FoldBar label="R" value={fold.rightSelectedReturn} maximum={maximumFoldReturn} /><FoldBar label="RB" value={fold.rightBaselineReturn} maximum={maximumFoldReturn} /></article>)}
            {comparison && folds.length === 0 ? <div className="runtime-empty">fold比較がありません。</div> : null}
          </div>
        </section>
      </div>
    </section>
  )
}
