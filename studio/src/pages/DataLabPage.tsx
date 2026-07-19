import { Database, RefreshCw, ShieldAlert, ShieldCheck } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import { studioApi, type StudioApi } from '../api/studioApi'
import type { DatasetSummary } from '../data/types'

interface DataLabPageProps {
  api?: StudioApi
}

const PAGE_SIZE = 8

function formatUpdated(value: string): string {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('ja-JP')
}

export function DataLabPage({ api = studioApi }: DataLabPageProps) {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [page, setPage] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  async function refresh() {
    setLoading(true)
    setError(null)
    try {
      const response = await api.loadDatasets()
      setDatasets(response.items)
      setSelectedId((current) =>
        current && response.items.some((item) => item.id === current)
          ? current
          : response.items[0]?.id ?? null,
      )
      setPage((current) => Math.min(current, Math.max(Math.ceil(response.items.length / PAGE_SIZE) - 1, 0)))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'データセットを取得できませんでした。')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
  }, [])

  const selected = datasets.find((item) => item.id === selectedId) ?? null
  const pageCount = Math.max(Math.ceil(datasets.length / PAGE_SIZE), 1)
  const visible = useMemo(
    () => datasets.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE),
    [datasets, page],
  )

  return (
    <section className="runtime-page" aria-labelledby="data-lab-title">
      <header className="runtime-toolbar">
        <div>
          <span className="runtime-eyebrow">DATA CATALOG</span>
          <h1 id="data-lab-title">Data Lab</h1>
          <p>正本artifactを検証し、特徴量・期間・識別情報を画面内で確認します。</p>
        </div>
        <div className="runtime-toolbar__actions">
          <span className="runtime-counter">{datasets.length} datasets</span>
          <button type="button" className="runtime-button runtime-button--quiet" onClick={() => void refresh()} disabled={loading}>
            <RefreshCw size={14} aria-hidden="true" />再読込
          </button>
        </div>
      </header>

      <div className="runtime-split">
        <div className="runtime-pane runtime-pane--list">
          <div className="runtime-pane__header">
            <strong>データセット</strong>
            <span>{page + 1} / {pageCount}</span>
          </div>
          <div className="runtime-list" aria-busy={loading}>
            {error ? <div className="runtime-error">{error}</div> : null}
            {!error && !loading && visible.length === 0 ? <div className="runtime-empty">artifactが見つかりません。</div> : null}
            {visible.map((dataset) => (
              <button
                type="button"
                key={dataset.id}
                className={`runtime-row${dataset.id === selectedId ? ' runtime-row--selected' : ''}`}
                onClick={() => setSelectedId(dataset.id)}
                aria-label={`${dataset.name} ${dataset.status}`}
              >
                <Database size={16} aria-hidden="true" />
                <span className="runtime-row__main">
                  <strong>{dataset.name}</strong>
                  <small>{dataset.symbols.join(' / ')} ・ {dataset.timeframes.join(' / ')}</small>
                </span>
                <span className={`runtime-badge runtime-badge--${dataset.status.toLowerCase()}`}>{dataset.status}</span>
              </button>
            ))}
          </div>
          <div className="runtime-pagination">
            <button type="button" onClick={() => setPage((value) => Math.max(value - 1, 0))} disabled={page === 0}>前へ</button>
            <button type="button" onClick={() => setPage((value) => Math.min(value + 1, pageCount - 1))} disabled={page >= pageCount - 1}>次へ</button>
          </div>
        </div>

        <div className="runtime-pane runtime-pane--detail">
          <div className="runtime-pane__header">
            <strong>検証詳細</strong>
            {selected ? (
              <span className={`runtime-badge runtime-badge--${selected.status.toLowerCase()}`}>
                {selected.status === 'VALID' ? <ShieldCheck size={12} aria-hidden="true" /> : <ShieldAlert size={12} aria-hidden="true" />}
                {selected.status}
              </span>
            ) : null}
          </div>
          {selected ? (
            <div className="runtime-detail-grid">
              <article className="runtime-hero-stat">
                <span>Features</span><strong>{selected.featureCount}</strong><small>{selected.barCount.toLocaleString()} bars</small>
              </article>
              <article><span>Symbols</span><strong>{selected.symbols.join(' / ')}</strong></article>
              <article><span>Market</span><strong>{selected.market}</strong></article>
              <article><span>Timeframes</span><strong>{selected.timeframes.join(' / ') || 'base only'}</strong></article>
              <article><span>Range</span><strong>{selected.range}</strong></article>
              <article><span>Artifact path</span><code>{selected.relativePath}</code></article>
              <article><span>Dataset identity</span><code>{selected.id}</code></article>
              <article><span>Updated</span><strong>{formatUpdated(selected.updated)}</strong></article>
              {selected.validationError ? <article className="runtime-validation-error"><span>Validation error</span><p>{selected.validationError}</p></article> : null}
            </div>
          ) : (
            <div className="runtime-empty">左からデータセットを選択してください。</div>
          )}
        </div>
      </div>
    </section>
  )
}
