import { Activity, RefreshCw, ShieldAlert } from 'lucide-react'
import { useEffect, useState } from 'react'

import { studioApi, type StudioApi } from '../api/studioApi'
import type { ServingMonitorReport } from '../data/types'

type ServingApi = Pick<StudioApi, 'loadServingMonitor'>
interface ServingPageProps { api?: ServingApi }

function short(value: string | null): string {
  return value ? `${value.slice(0, 12)}…${value.slice(-8)}` : '—'
}

export function ServingPage({ api = studioApi }: ServingPageProps) {
  const [report, setReport] = useState<ServingMonitorReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  async function refresh() {
    setLoading(true)
    setError(null)
    try { setReport(await api.loadServingMonitor()) }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Serving状態を取得できませんでした。') }
    finally { setLoading(false) }
  }

  useEffect(() => { void refresh() }, [])

  return (
    <section className="runtime-page" aria-labelledby="serving-title">
      <header className="runtime-toolbar">
        <div><span className="runtime-eyebrow">PAPER INFERENCE OBSERVABILITY</span><h1 id="serving-title">Serving Monitor</h1><p>active bundle、identity closure、paper推論スナップショットを監視します。状態変更は行いません。</p></div>
        <div className="runtime-toolbar__actions"><span className="audit-readonly">READ ONLY</span><span className="runtime-danger">NO-GO</span><button type="button" className="runtime-button runtime-button--quiet" onClick={() => void refresh()} disabled={loading}><RefreshCw size={14} aria-hidden="true" />再読込</button></div>
      </header>

      <div className="audit-serving-grid" aria-busy={loading}>
        <section className="runtime-pane audit-serving-state">
          <div className="runtime-pane__header"><strong>Runtime state</strong><span>{report?.state ?? '—'}</span></div>
          <div className="audit-state-hero"><Activity size={34} aria-hidden="true" /><strong>{report?.state ?? 'LOADING'}</strong><span>{report?.productionStatus ?? 'NO-GO'}</span><small>{report?.validationError ?? 'local registry inspection'}</small></div>
        </section>

        <section className="runtime-pane audit-serving-identities">
          <div className="runtime-pane__header"><strong>Bound identities</strong><span>{report?.releaseAttestationPresent ? 'attested' : 'not attested'}</span></div>
          <div className="audit-identity-grid">
            <article><span>Bundle</span><code title={report?.activeBundleDigest ?? ''}>{short(report?.activeBundleDigest ?? null)}</code></article>
            <article><span>Dataset</span><code>{report?.datasetId ?? '—'}</code></article>
            <article><span>Policy</span><code title={report?.policyDigest ?? ''}>{short(report?.policyDigest ?? null)}</code></article>
            <article><span>Run kind</span><strong>{report?.runKind ?? '—'}</strong></article>
            <article><span>Action schema</span><strong>{report?.actionSchema ?? '—'}</strong></article>
            <article><span>Observation</span><strong>{report?.observationSchema ?? '—'}</strong></article>
          </div>
        </section>

        <section className="runtime-pane audit-serving-checks">
          <div className="runtime-pane__header"><strong>Validation checks</strong><span>{report?.checks.length ?? 0}</span></div>
          <div className="audit-check-list">
            {error ? <div className="runtime-error">{error}</div> : null}
            {(report?.checks ?? []).map((check) => <article key={check.key}><span className={`audit-check audit-check--${check.status.toLowerCase()}`}>{check.status}</span><div><strong>{check.label}</strong><small>{check.detail}</small></div></article>)}
            {!error && report?.checks.length === 0 ? <div className="runtime-empty">検証項目がありません。</div> : null}
          </div>
        </section>

        <section className="runtime-pane audit-paper-pane">
          <div className="runtime-pane__header"><strong>Paper inference snapshot</strong><span>{report?.paperSnapshot?.recordedAt ?? 'not present'}</span></div>
          {report?.paperSnapshot ? (() => {
            const entries = Object.entries(report.paperSnapshot.targetWeights)
            const nonCash = entries.filter(([symbol]) => symbol.toUpperCase() !== 'CASH')
            const gross = nonCash.reduce((total, [, weight]) => total + Math.abs(weight), 0)
            const net = nonCash.reduce((total, [, weight]) => total + weight, 0)
            const maximum = Math.max(...entries.map(([, weight]) => Math.abs(weight)), 0.01)
            return <div className="audit-paper-layout">
              <div className="audit-paper-stats"><article><span>Decision index</span><strong>{report.paperSnapshot.decisionIndex}</strong></article><article><span>Latency</span><strong>{report.paperSnapshot.latencyMs.toFixed(1)} ms</strong></article><article><span>Non-cash gross</span><strong>{(gross * 100).toFixed(2)}%</strong></article><article><span>Non-cash net</span><strong>{(net * 100).toFixed(2)}%</strong></article><article><span>Snapshot</span><code>{short(report.paperSnapshot.snapshotDigest)}</code></article></div>
              <div className="audit-weight-list">{entries.map(([symbol, weight]) => {
                const direction = weight < 0 ? 'short' : 'long'
                const width = Math.min(Math.abs(weight) / maximum, 1) * 50
                return <article key={symbol} aria-label={`${symbol} target weight ${(weight * 100).toFixed(2)}%`} data-direction={direction}><strong>{symbol}</strong><div className="audit-weight-track" aria-hidden="true"><i className={`audit-weight-bar audit-weight-bar--${direction}`} style={{ width: `${width}%` }} /></div><span>{(weight * 100).toFixed(2)}%</span></article>
              })}</div>
            </div>
          })() : <div className="runtime-empty"><ShieldAlert size={26} aria-hidden="true" />paper推論スナップショットはありません。</div>}
        </section>
      </div>
    </section>
  )
}
