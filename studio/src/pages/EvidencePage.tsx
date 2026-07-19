import { FileCheck2, RefreshCw, ShieldCheck } from 'lucide-react'
import { useEffect, useState } from 'react'

import { studioApi, type StudioApi } from '../api/studioApi'
import type { EvidenceNode, EvidenceReport, RunSummary } from '../data/types'

type EvidenceApi = Pick<StudioApi, 'loadRuns' | 'loadEvidenceReport'>

interface EvidencePageProps { api?: EvidenceApi }

function bytes(value: number): string {
  if (value < 1024) return `${value} B`
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / 1024 ** 2).toFixed(1)} MB`
}

export function EvidencePage({ api = studioApi }: EvidencePageProps) {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [runId, setRunId] = useState('')
  const [report, setReport] = useState<EvidenceReport | null>(null)
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  async function loadReport(selected: string) {
    if (!selected) return
    setLoading(true)
    setError(null)
    try {
      const value = await api.loadEvidenceReport(selected)
      setReport(value)
      setSelectedKey(value.nodes[0]?.key ?? null)
    } catch (reason) {
      setReport(null)
      setError(reason instanceof Error ? reason.message : 'Evidenceを取得できませんでした。')
    } finally {
      setLoading(false)
    }
  }

  async function refresh() {
    setLoading(true)
    setError(null)
    try {
      const response = await api.loadRuns()
      setRuns(response.items)
      const selected = response.items.some((item) => item.id === runId) ? runId : response.items[0]?.id ?? ''
      setRunId(selected)
      await loadReport(selected)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'run一覧を取得できませんでした。')
      setLoading(false)
    }
  }

  useEffect(() => { void refresh() }, [])

  const selected: EvidenceNode | null = report?.nodes.find((item) => item.key === selectedKey) ?? null

  return (
    <section className="runtime-page" aria-labelledby="evidence-title">
      <header className="runtime-toolbar">
        <div><span className="runtime-eyebrow">PROVENANCE AND CLOSURE</span><h1 id="evidence-title">Evidence Explorer</h1><p>run manifestを起点に、identity、authorization、artifact file closureを読み取り専用で検査します。</p></div>
        <div className="runtime-toolbar__actions">
          <label className="audit-run-select">Run<select value={runId} onChange={(event) => { setRunId(event.target.value); void loadReport(event.target.value) }}>{runs.map((run) => <option key={run.id}>{run.id}</option>)}</select></label>
          <span className="runtime-danger">NO-GO</span>
          <button type="button" className="runtime-button runtime-button--quiet" onClick={() => void refresh()} disabled={loading}><RefreshCw size={14} aria-hidden="true" />再読込</button>
        </div>
      </header>

      <div className="audit-evidence-grid">
        <section className="runtime-pane audit-chain-pane">
          <div className="runtime-pane__header"><strong>Evidence chain</strong><span>{report?.runKind ?? '—'}</span></div>
          <div className="audit-chain" aria-busy={loading}>
            {error ? <div className="runtime-error">{error}</div> : null}
            {(report?.nodes ?? []).map((node, index) => (
              <button key={node.key} type="button" className={`audit-chain-node${node.key === selectedKey ? ' audit-chain-node--selected' : ''}`} onClick={() => setSelectedKey(node.key)}>
                <span className={`audit-status audit-status--${node.status.toLowerCase()}`}>{node.status}</span>
                <strong>{node.label}</strong><small>{node.detail}</small>
                {index < (report?.nodes.length ?? 0) - 1 ? <i aria-hidden="true" /> : null}
              </button>
            ))}
            {!error && !loading && !report ? <div className="runtime-empty">Evidence reportがありません。</div> : null}
          </div>
        </section>

        <section className="runtime-pane audit-evidence-detail">
          <div className="runtime-pane__header"><strong>Evidence detail</strong><span>{selected?.required ? 'required' : 'optional'}</span></div>
          {selected ? <div className="audit-detail-stack">
            <article><span>Status</span><strong className={`audit-status-text audit-status-text--${selected.status.toLowerCase()}`}>{selected.status}</strong></article>
            <article><span>Artifact path</span><code>{selected.path ?? '—'}</code></article>
            <article><span>Digest</span><code>{selected.digest ?? '—'}</code></article>
            <article><span>Interpretation</span><p>{selected.detail}</p></article>
          </div> : <div className="runtime-empty">左からEvidenceを選択してください。</div>}
        </section>

        <section className="runtime-pane audit-integrity-pane">
          <div className="runtime-pane__header"><strong>File integrity</strong><span>{report?.files.status ?? '—'}</span></div>
          {report ? <div className="audit-integrity">
            <FileCheck2 size={28} aria-hidden="true" />
            <div><span>Verified files</span><strong>{report.files.verifiedCount} / {report.files.declaredCount}</strong></div>
            <div><span>Total size</span><strong>{bytes(report.files.totalSizeBytes)}</strong></div>
            <div><span>Run status</span><strong>{report.status}</strong></div>
          </div> : <div className="runtime-empty">—</div>}
        </section>

        <section className="runtime-pane audit-verdict-pane">
          <div className="runtime-pane__header"><strong>Research verdict</strong><span>read only</span></div>
          <div className="audit-verdict"><ShieldCheck size={32} aria-hidden="true" /><strong>{report?.status ?? '—'}</strong><span>{report?.productionStatus ?? 'NO-GO'}</span><p>{report?.validationError ?? 'Evidenceは研究監査用途です。release承認を代替しません。'}</p></div>
        </section>
      </div>
    </section>
  )
}
