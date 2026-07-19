import { RefreshCw, Square, TerminalSquare } from 'lucide-react'
import { useEffect, useState } from 'react'

import { studioApi, type StudioApi } from '../api/studioApi'
import type { JobSummary } from '../data/types'

interface RunCenterPageProps {
  api?: StudioApi
}

function canCancel(status: JobSummary['status']): boolean {
  return status === 'queued' || status === 'running'
}

export function RunCenterPage({ api = studioApi }: RunCenterPageProps) {
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [lines, setLines] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  async function refreshJobs() {
    setLoading(true)
    setError(null)
    try {
      const response = await api.loadJobs()
      setJobs(response.items)
      setSelectedId((current) =>
        current && response.items.some((item) => item.id === current)
          ? current
          : response.items[0]?.id ?? null,
      )
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'ジョブを取得できませんでした。')
    } finally {
      setLoading(false)
    }
  }

  async function loadLog(jobId: string) {
    setSelectedId(jobId)
    try {
      const response = await api.loadJobLog(jobId)
      setLines(response.lines)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'ログを取得できませんでした。')
    }
  }

  async function cancel(jobId: string) {
    setError(null)
    try {
      const updated = await api.cancelJob(jobId)
      setJobs((current) => current.map((item) => item.id === jobId ? updated : item))
      await loadLog(jobId)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'ジョブを停止できませんでした。')
    }
  }

  useEffect(() => {
    void refreshJobs()
  }, [])

  const selected = jobs.find((item) => item.id === selectedId) ?? null

  return (
    <section className="runtime-page" aria-labelledby="run-center-title">
      <header className="runtime-toolbar">
        <div>
          <span className="runtime-eyebrow">PROCESS SUPERVISOR</span>
          <h1 id="run-center-title">Run Center</h1>
          <p>永続化されたジョブ状態とログを確認し、所有プロセスだけを安全停止します。</p>
        </div>
        <button type="button" className="runtime-button runtime-button--quiet" onClick={() => void refreshJobs()} disabled={loading}>
          <RefreshCw size={14} aria-hidden="true" />再読込
        </button>
      </header>

      <div className="runtime-split runtime-split--runs">
        <div className="runtime-pane runtime-pane--list">
          <div className="runtime-pane__header"><strong>Jobs</strong><span>{jobs.length}</span></div>
          <div className="runtime-list" aria-busy={loading}>
            {error ? <div className="runtime-error">{error}</div> : null}
            {!error && !loading && jobs.length === 0 ? <div className="runtime-empty">実行履歴がありません。</div> : null}
            {jobs.map((job) => (
              <button
                type="button"
                key={job.id}
                className={`runtime-row runtime-row--job${job.id === selectedId ? ' runtime-row--selected' : ''}`}
                onClick={() => void loadLog(job.id)}
                aria-label={`${job.id} ${job.status}`}
              >
                <TerminalSquare size={16} aria-hidden="true" />
                <span className="runtime-row__main"><strong>{job.id}</strong><small>{job.runId} · {job.configPath}</small></span>
                <span className={`runtime-badge runtime-badge--job-${job.status}`}>{job.status}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="runtime-pane runtime-pane--log">
          <div className="runtime-pane__header">
            <strong>{selected?.id ?? 'Job log'}</strong>
            {selected && canCancel(selected.status) ? (
              <button type="button" className="runtime-button runtime-button--danger" onClick={() => void cancel(selected.id)}>
                <Square size={12} aria-hidden="true" />安全停止
              </button>
            ) : selected ? <span className={`runtime-badge runtime-badge--job-${selected.status}`}>{selected.status}</span> : null}
          </div>
          {selected ? (
            <div className="runtime-log-layout">
              <div className="runtime-log-meta">
                <span>run <code>{selected.runId}</code></span>
                <span>pid <code>{selected.pid ?? '—'}</code></span>
                <span>exit <code>{selected.exitCode ?? '—'}</code></span>
                <span>status <strong>{selected.status}</strong></span>
              </div>
              <pre aria-label="job log">{lines.length ? lines.join('\n') : 'ログを読み込んでいます…'}</pre>
              {selected.error ? <div className="runtime-error">{selected.error}</div> : null}
            </div>
          ) : <div className="runtime-empty">左からジョブを選択してください。</div>}
        </div>
      </div>
    </section>
  )
}
